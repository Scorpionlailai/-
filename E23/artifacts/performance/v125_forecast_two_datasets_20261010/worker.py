"""Immutable v125 repair ensemble -> original frozen, pure-MSE Stage2."""
from pathlib import Path
from time import perf_counter
ENTRY_STARTED=perf_counter()
import json,sys,csv,argparse,hashlib,statistics,dataclasses
Q=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('action',choices=['preflight','train','evaluate']);parser.add_argument('case');parser.add_argument('--selection',choices=['mae','mse'],default='mae');args=parser.parse_args()
ROOT=Path(args.case).resolve();p=json.loads((ROOT/'plan.json').read_text());S=Path(p['source']);sys.path[:0]=[str(S/'src'),str(S),str(ROOT)]
import torch
from weather_common import protocol_for
from integrations.e23.config import load_e23_settings
from integrations.e23 import train as tr
from integrations.e23.calibration_ensemble import build_calibration_repair,verify_ensemble_instance
from integrations.e23.forecast_sensitive import parameter_hash
from integrations.e23.repair_updates import build_stage2_optimizer,stage2_training_mode
from integrations.forecast_compare.protocol import iter_p4_batches
from integrations.forecast_compare.resources import TrainingResources,selected_training_resources

torch.set_num_threads(4);scene=p['scenarios'][0];protocol=protocol_for(p,scene)
settings=load_e23_settings(Path(p['config']),protocol=protocol,allow_variable_scaling=True)
parent=Path(p['stage1_selection']).resolve();record=json.loads(parent.read_text());spec=record['calibration_ensemble_spec'];assert spec['weights']==[.5,.5]
original_load=tr.load_stage1_selection;original_build=tr.build_forecast_model

def load_parent(path,protocol,settings,**kw):
 assert Path(path).resolve()==parent,'Only the fixed declared ensemble parent is authorized'
 kw['allow_calibration_ensemble']=True
 return original_load(path,protocol,settings,**kw)

def build(protocol,settings,*,device,**kw):
 assert kw.get('forecast_mixer','none')=='none' and kw.get('forecast_input','explicit_v1')=='explicit_v1' and kw.get('history_adapter','none')=='none'
 model=original_build(protocol,settings,device=device,**kw)
 assert parameter_hash(model.predictor)==p['predictor_initial_sha256']
 assert sum(x.numel() for x in model.predictor.parameters())==35427
 with torch.random.fork_rng(devices=[device.index or 0] if device.type=='cuda' else []):
  repair=build_calibration_repair(protocol,settings,spec,device=device)
 model.repair=repair;verify_ensemble_instance(model.repair,spec)
 return model
tr.load_stage1_selection=load_parent;tr.build_forecast_model=build

def restore(device):
 torch.manual_seed(2023);model=build(protocol,settings,device=device)
 payload=load_parent(parent,protocol,settings,device=device);model.repair.load_state_dict(payload['model_state'],strict=True)
 model.repair.historical.retrieval.load_state_dict(payload['retrieval_state']);memory=tr._new_memory(protocol);memory.load_state_dict(payload['memory_state']);model.repair.historical.refresh_snapshot(memory.snapshot())
 return model,memory,payload

def preflight():
 model,memory,payload=restore(torch.device('cpu'));optimizer=build_stage2_optimizer(model,mode='frozen_v1',learning_rate=.0001,weight_decay=0);stage2_training_mode(model,'frozen_v1')
 assert not model.repair.training and not any(x.requires_grad for x in model.repair.parameters())
 assert {id(x) for g in optimizer.param_groups for x in g['params']}=={id(x) for x in model.predictor.parameters() if x.requires_grad}
 full=next(iter_p4_batches(protocol,'validation',batch_size=128));b=dataclasses.replace(full,**{f.name:getattr(full,f.name)[:2] for f in dataclasses.fields(full) if isinstance(getattr(full,f.name),torch.Tensor)},refs=full.refs[:2])
 starts=tr._starts(b,stride=model.repair.stride,patches=model.repair.patch_count,device=torch.device('cpu'));before=parameter_hash(model.repair)
 forecast,repaired=model(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts)
 assert torch.equal(repaired.explicit.normalized_sequence[b.observed_mask],b.normalized_observed[b.observed_mask])
 loss=tr.stage2_objective(forecast,b.forecast_target,loss_mode='mse');assert torch.equal(loss,(forecast-b.forecast_target).square().mean());loss.backward();optimizer.step()
 assert before==parameter_hash(model.repair) and all(x.grad is None for x in model.repair.parameters())
 assert any(x.grad is not None and torch.isfinite(x.grad).all() for x in model.predictor.parameters())
 r=dict(status='passed',strict_ensemble_restore=True,predictor_initial_sha256=p['predictor_initial_sha256'],predictor_parameters=35427,repair_trainable_parameters=0,repair_gradients_absent=True,optimizer_predictor_only=True,repair_state_unchanged_after_backward_step=True,visible_values_exact=True,forecast_loss='pure_mse',source_entry=str(tr.__file__),parent_checkpoint_sha256=record['checkpoint_sha256'])
 (ROOT/'preflight.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)

def train():
 assert json.loads((ROOT/'preflight.json').read_text())['status']=='passed'
 device=torch.device('cuda:0')
 with TrainingResources(ROOT/'stage2',device=device,batch_size=128,model='E23-v125',started_at=ENTRY_STARTED) as resources:
  with resources.stage('forecast'):
   selection=tr.train_forecast(protocol,settings,stage1_manifest=parent,output=ROOT/'stage2',device=device,epochs=100,selection_metric='mae',forecast_mixer='none',forecast_input='explicit_v1',forecast_loss='mse',repair_retention='none',repair_update_policy='frozen_v1',forecast_padding='zero_invalid_v1',extra_mse_checkpoint=True,cache_fixed_validation=True)
  resources.selection_manifest=selection
 print(json.dumps(dict(status='completed',selection=str(selection))),flush=True)

@torch.no_grad()
def score(model,current,split,device):
 sums={k:0. for k in ['forecast_squared','forecast_absolute','repair96_squared','repair96_absolute','repair32_squared','repair32_absolute']};counts=dict(windows=0,forecast_coordinates=0,hidden96_coordinates=0,hidden32_coordinates=0)
 for cpu in iter_p4_batches(current,split,batch_size=128):
  b=cpu.to(device);starts=tr._starts(b,stride=model.repair.stride,patches=model.repair.patch_count,device=device)
  pred,out=model(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts);o=out.explicit;err=(pred-b.forecast_target).double()
  assert torch.equal(o.normalized_sequence[b.observed_mask],b.normalized_observed[b.observed_mask]);sums['forecast_squared']+=err.square().sum().item();sums['forecast_absolute']+=err.abs().sum().item();counts['forecast_coordinates']+=err.numel();counts['windows']+=len(b.refs)
  error=((o.normalized_sequence-b.normalized_target)*b.query_std[:,None]).double();hidden=~b.observed_mask & o.sequence_validity
  for length in [96,32]:
   e=error[:,-length:][hidden[:,-length:]];sums[f'repair{length}_squared']+=e.square().sum().item();sums[f'repair{length}_absolute']+=e.abs().sum().item();counts[f'hidden{length}_coordinates']+=e.numel()
 result=dict(**counts,forecast_mse=sums['forecast_squared']/counts['forecast_coordinates'],forecast_mae=sums['forecast_absolute']/counts['forecast_coordinates'])
 for length in [96,32]:
  for name,key in [('mse','squared'),('mae','absolute')]:result[f'repair{length}_{name}']=sums[f'repair{length}_{key}']/counts[f'hidden{length}_coordinates']
 return result

@torch.no_grad()
def evaluate():
 device=torch.device('cuda:0');selection=ROOT/'stage2/runs'/('e23-stage2.json' if args.selection=='mae' else 'e23-stage2-mse.json');r=json.loads(selection.read_text())
 payload=tr.load_stage2_selection(selection,protocol,settings,device=device);torch.manual_seed(2023);model=build(protocol,settings,device=device);model.load_state_dict(payload['model_state'],strict=True)
 assert r['repair_update_policy']=='frozen_v1' and r['repair_trainable_parameters']==0 and r['epochs_completed']==100
 model.repair.historical.retrieval.load_state_dict(payload['retrieval_state']);memory=tr._new_memory(protocol);memory.load_state_dict(payload['memory_state']);model.repair.historical.refresh_snapshot(memory.snapshot());model.eval().requires_grad_(False);del payload
 before=parameter_hash(model),memory.snapshot_hash(),tr._key_sha256(model.repair.historical.retrieval.state_dict());validation=score(model,protocol,'validation',device)
 assert abs(validation['forecast_mse']-r['best_val_mse'])<1e-6 and abs(validation['forecast_mae']-r['best_val_mae'])<1e-6
 if p['dataset']=='Weather':
  from weather_common import data
  dataset=data.read_manifest(Path(p['dataset_manifest']));views=[(dict(seed=i['seed'],mask_sha256=i['sha256']),data.with_test_mask(protocol,dataset,i)) for i in dataset['masks'] if i['split']=='test']
 else:
  from integrations.forecast_compare.test_mask_bank_v2 import load_test_bank
  views=[(dict(group_id=f'group_{i:02d}',mask_sha256=v.mask_sha256['test']),v) for i,v in enumerate(load_test_bank(protocol,Path(scene['test_bank'])))]
 assert len(views)==20;draws=[];started=perf_counter()
 for ident,view in views:
  row=dict(**ident,**score(model,view,'test',device));assert row['windows']==p['expected_test_windows'];draws.append(row);print(json.dumps(dict(selection=args.selection,**row)),flush=True)
 assert before==(parameter_hash(model),memory.snapshot_hash(),tr._key_sha256(model.repair.historical.retrieval.state_dict()))
 keys=['forecast_mse','forecast_mae','repair96_mse','repair96_mae','repair32_mse','repair32_mae'];summary={k:dict(mean=statistics.mean(d[k] for d in draws),mask_std_ddof1=statistics.stdev(d[k] for d in draws)) for k in keys}
 cost=selected_training_resources([*[Path(x) for x in p['stage1_sources']],selection]);stage2cost=selected_training_resources([selection]);assert cost['resource_status']==stage2cost['resource_status']=='measured_all_stages'
 out=ROOT/('evaluation_'+args.selection);out.mkdir(exist_ok=False)
 result=dict(dataset=p['dataset'],model='E23-v125',scene=scene['name'],selection_metric=args.selection,selection_manifest=str(selection),best_epoch=r['best_epoch'],draws=draws,summary=summary,validation=validation,training_resources=cost,stage2_training_resources=stage2cost,repair_frozen_verified=True,model_memory_keys_unchanged=True,test_inference_seconds=perf_counter()-started,training_seed=2023,mask_scope='20 development positions, one training seed')
 (out/'twenty-test-draws.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(status='completed',selection=args.selection,summary=summary)),flush=True)
if args.action=='preflight':preflight()
elif args.action=='train':train()
else:evaluate()
