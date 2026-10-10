"""Stage1 only: frozen v79 coarse repair and history, train a zero-init dense residual."""
from time import perf_counter
ENTRY_STARTED=perf_counter()
from pathlib import Path
import json,sys,hashlib,copy
import torch
from weather_common import ROOT,protocol_for
from integrations.e23.config import load_e23_settings
from integrations.e23.repair_validation_cache import FixedRepairValidationCache
from integrations.e23.train import (build_repair_model,load_stage1_selection,_new_memory,_starts,
 _validation_repair_metrics,_paired_payload,_atomic_torch_save,_atomic_json,_file_sha256,_identity,_revision,_memory_state_sha256)
from integrations.forecast_compare.protocol import iter_p4_batches
from integrations.forecast_compare.resources import TrainingResources
from stage1_iteration import repair_loss,selection_updates

def coarse_hash(model):
 d=hashlib.sha256()
 for name,value in model.state_dict().items():
  if not name.startswith('recent_mean_projector.'):
   d.update(name.encode());d.update(value.detach().cpu().contiguous().numpy().tobytes())
 return d.hexdigest()

def bank_content(memory):
 state=copy.deepcopy(memory.state_dict());state['snapshot']['epoch']=0
 return _memory_state_sha256(state)

def train(scene_name):
 torch.set_num_threads(4);torch.manual_seed(2023)
 plan=json.loads((ROOT/'plan.json').read_text());scene=next(s for s in plan['scenarios'] if s['name']==scene_name);entry=plan['models'][0]
 output=ROOT/entry['label']/scene_name/'stage1';assert not output.exists();device=torch.device(plan['device'])
 with TrainingResources(output,device=device,batch_size=128,model=entry['label'],started_at=ENTRY_STARTED) as resources:
  with resources.stage('repair_refinement'):
   protocol=protocol_for(plan,scene);settings=load_e23_settings(Path(entry['config']),protocol=protocol,allow_variable_scaling=True);parent_settings=load_e23_settings(Path(plan['parent_settings']),protocol=protocol,allow_variable_scaling=True)
   selection=Path(plan['parent_selection'])
   original=json.loads(selection.read_text());payload=load_stage1_selection(selection,protocol,parent_settings,device=device)
   model=build_repair_model(protocol,settings,device=device)
   incompatible=model.load_state_dict(payload['model_state'],strict=False)
   assert not incompatible.unexpected_keys and incompatible.missing_keys and all('.global_dense_refiner.' in k for k in incompatible.missing_keys)
   model.historical.retrieval.load_state_dict(payload['retrieval_state']);memory=_new_memory(protocol);memory.load_state_dict(payload['memory_state']);model.historical.refresh_snapshot(memory.snapshot())
   model.eval().requires_grad_(False);refiner=model.recent_mean_projector;refiner.requires_grad_(True)
   assert sum(p.numel() for p in model.parameters() if p.requires_grad)==entry['trainable_parameters']
   original_hash=coarse_hash(model);bank_hash=bank_content(memory)
   # Initial zero residual must exactly reproduce the parent validation score.
   cache=FixedRepairValidationCache(model,protocol,settings,coarse_hash)
   initial=cache.build(device=device)
   assert initial==cache.evaluate(device=device), "Initial cached metrics differ from full repair"
   assert abs(initial['global_mse']-original['validation_repair_metrics']['global_mse'])<1e-6
   optimizer=torch.optim.Adam(refiner.parameters(),lr=plan['actual_refinement_learning_rate'],weight_decay=0)
   base_epoch=int(payload['epoch']);best={'mse':float('inf'),'mae':float('inf')};history=[];start=perf_counter()
   checkpoints={'mse':output/'checkpoints/e23-stage1.pt','mae':output/'checkpoints/e23-stage1-mae.pt'}
   manifests={'mse':output/'runs/e23-stage1.json','mae':output/'runs/e23-stage1-mae.json'}
   choices={};weights=plan['refinement_objective_weights']
   for local_epoch in range(1,plan['refinement_epochs']+1):
    epoch=base_epoch+local_epoch
    # Advance paired state metadata only. Keep the original frozen key encoder and bank contents.
    retrieval=model.historical.retrieval;retrieval.ema_encoder.load_state_dict(retrieval.frozen_encoder.state_dict());retrieval.begin_epoch(epoch)
    model.eval();refiner.train()
    for index,cpu in enumerate(iter_p4_batches(protocol,'train',batch_size=128,epoch=local_epoch-1),1):
     b=cpu.to(device);starts=_starts(b,stride=model.stride,patches=model.patch_count,device=device)
     optimizer.zero_grad(set_to_none=True)
     out=model(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts,epoch=epoch,explore_variable=False,specialist_exposure=0).explicit
     loss=repair_loss(out.normalized_sequence,b.normalized_target,b.observed_mask,out.sequence_validity,b.query_std,weights)
     loss.backward();optimizer.step()
     if index==1 or index%10==0:print(f'stage=repair_refinement epoch={local_epoch}/100 batch={index}/{len(protocol.refs["train"])//128} loss={loss.item():.6f}',flush=True)
    memory.publish_epoch(epoch);model.historical.refresh_snapshot(memory.snapshot())
    metrics=cache.evaluate(device=device);history.append({'epoch':epoch,'refinement_epoch':local_epoch,'metrics':metrics})
    updates=selection_updates(metrics,best)
    if updates:
     paired=_paired_payload(model,memory,epoch=epoch,validation_mse=metrics['global_mse'],settings=settings,validation_metrics=metrics)
     for criterion in updates:
      _atomic_torch_save(paired,checkpoints[criterion])
      choices[criterion]={'epoch':epoch,'local_epoch':local_epoch,'metrics':dict(metrics)}
    assert coarse_hash(model)==original_hash and bank_content(memory)==bank_hash
    _atomic_json({'selection_metric_space':'train_standardized','initial_parent_metrics':initial,'epochs':history},output/'runs/e23-stage1-history.json')
    print(f'stage=repair_refinement epoch={local_epoch}/100 val_repair_mse={metrics["global_mse"]:.8f} val_repair_mae={metrics["global_mae"]:.8f} best_mse_epoch={choices["mse"]["local_epoch"]} best_mae_epoch={choices["mae"]["local_epoch"]}',flush=True)
   training_seconds=perf_counter()-start
   for criterion in ('mse','mae'):
    saved=torch.load(checkpoints[criterion],map_location='cpu',weights_only=False)
    choice=choices[criterion]
    assert saved['epoch']==choice['epoch'] and saved['validation_repair_metrics']==choice['metrics']
    record={'fixed_stage1_validation_cache':cache.identity(),'stage':'repair','best_epoch':choice['epoch'],'best_refinement_epoch':choice['local_epoch'],'epochs_completed':100,'best_val_repair_mse':choice['metrics']['global_mse'],'best_val_repair_mae':choice['metrics']['global_mae'],'checkpoint':str(checkpoints[criterion]),'checkpoint_sha256':_file_sha256(checkpoints[criterion]),'memory_sha256':saved['memory_sha256'],'key_sha256':saved['key_sha256'],'validation_repair_metrics':saved['validation_repair_metrics'],'code_revision':_revision(),'optimizer_state_restored':False,'refinement_objective_weights':weights,'refinement_loss':'explicit_weighted_full_and_last32_mse_mae','refinement_full_mae_units':'train_standardized','learning_rate':plan['actual_refinement_learning_rate'],'training_seconds':training_seconds,'selection':f'lowest fixed-validation hidden-coordinate global repair {criterion.upper()}','selection_metric':criterion,'primary':criterion==plan['stage1_primary_selection'],'stage1_training_batch_size':128,'validation_batch_size':128,'optimizer_updates':len(protocol.refs['train'])//128*100,'training_windows_per_epoch':len(protocol.refs['train'])//128*128,'stage1_aux_forecast':'none','coarse_parameters_frozen_verified':False,'expert_parameters_frozen_verified':True,'parent_mean_projector_trainable_parameters':plan['stage1_parent_mean_trainable_parameters'],'bank_contents_frozen_verified':True,'parent_selection':str(selection),'parent_checkpoint_sha256':original['checkpoint_sha256'],'parent_training_seconds':original['training_seconds'],'parent_epochs_completed':original['epochs_completed'],'total_stage1_epochs':original['epochs_completed']+100,'trainable_parameters':entry['trainable_parameters'],'frozen_expert_hash':original_hash,**_identity(protocol,settings)}
    if criterion=='mae':
     record.update(training_resource_parent_manifest=str(manifests['mse']),training_resource_parent_sha256=_file_sha256(manifests['mse']))
    _atomic_json(record,manifests[criterion])
   resources.selection_manifest=manifests['mse']
 print(json.dumps({'manifests':{k:str(v) for k,v in manifests.items()},'resources':str(resources.report_path)}),flush=True)

if __name__=='__main__':train(sys.argv[1])
