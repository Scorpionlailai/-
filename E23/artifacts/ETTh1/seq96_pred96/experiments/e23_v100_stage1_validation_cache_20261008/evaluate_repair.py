"""Identical full-history/last32 hidden-coordinate repair audit; no forecast head."""
import hashlib,json,os,subprocess
from pathlib import Path
import torch
from integrations.e23.config import load_e23_p4_protocol,load_e23_settings
from integrations.e23.train import load_stage1_selection,build_repair_model,_new_memory,_starts,_key_sha256
from integrations.e23.forecast_sensitive import parameter_hash
from integrations.forecast_compare.author_profiles import build_author_model
from integrations.forecast_compare.protocol import iter_p4_batches
from integrations.forecast_compare.test_mask_bank_v2 import load_test_bank
from integrations.forecast_compare.t1_p4 import t1_inputs,t1_clean_history
from malr.migration import MigrationCatalog

ROOT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')

def protocol_for(plan,scene):
 os.environ.pop('MALR_MISSING_SCENARIO',None)
 if scene['scenario_manifest']:os.environ['MALR_MISSING_SCENARIO']=scene['scenario_manifest']
 return load_e23_p4_protocol(catalog=MigrationCatalog.load('/home/xyw/paper/E22/migration/paper_catalog.jsonl',project='E23'),protocol_config=Path(plan['protocol_config']),mask_artifacts=Path(plan['mask_artifacts']),training_seed=2023)

@torch.inference_mode()
def score(model,protocol,device,kind,split):
 totals={k:[0.,0.,0] for k in ['all_hidden','last32_hidden']};windows=0
 for cpu in iter_p4_batches(protocol,split,batch_size=128):
  b=cpu.to(device);windows+=b.observed_mask.shape[0]
  if kind=='E23':
   starts=_starts(b,stride=model.stride,patches=model.patch_count,device=device)
   out=model(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts).explicit
   assert torch.equal(out.normalized_sequence[b.observed_mask],b.normalized_observed[b.observed_mask])
   error=(out.normalized_sequence-b.normalized_target)*b.query_std[:,None];hidden=~b.observed_mask & out.sequence_validity
  else:
   values,mask=t1_inputs(b);candidate=model(values,mask=mask).float();error=candidate-t1_clean_history(b);hidden=~b.observed_mask
  for name,e,h in [('all_hidden',error,hidden),('last32_hidden',error[:,-32:],hidden[:,-32:])]:
   r=e[h].double();assert torch.isfinite(r).all()
   totals[name][0]+=r.square().sum().item();totals[name][1]+=r.abs().sum().item();totals[name][2]+=r.numel()
 return dict(windows=windows,metrics={k:dict(mse=s/n,mae=a/n,hidden_coordinates=n) for k,(s,a,n) in totals.items()})

def evaluate(plan,scene,entry):
 device=torch.device(plan['device']);protocol=protocol_for(plan,scene)
 settings=load_e23_settings(Path(entry['config']),protocol=protocol)
 selection=ROOT/entry['label']/scene['name']/'stage1/runs/e23-stage1.json'
 record=json.loads(selection.read_text());payload=load_stage1_selection(selection,protocol,settings,device=device)
 model=build_repair_model(protocol,settings,device=device);model.load_state_dict(payload['model_state'],strict=True)
 model.historical.retrieval.load_state_dict(payload['retrieval_state']);memory=_new_memory(protocol);memory.load_state_dict(payload['memory_state']);model.historical.refresh_snapshot(memory.snapshot());model.eval().requires_grad_(False)
 before=(parameter_hash(model),memory.snapshot_hash(),_key_sha256(model.historical.retrieval.state_dict()))
 validation=score(model,protocol,device,'E23','validation')
 assert all(abs(validation['metrics']['all_hidden'][k]-record['validation_repair_metrics']['global_'+k])<1e-6 for k in ['mse','mae'])
 views=load_test_bank(protocol,Path(scene['test_bank']));tests=[]
 for index,view in enumerate(views):
  tests.append(dict(group_id=f'group_{index:02d}',mask_sha256=view.mask_sha256['test'],**score(model,view,device,'E23','test')))
 assert before==(parameter_hash(model),memory.snapshot_hash(),_key_sha256(model.historical.retrieval.state_dict()))
 own=dict(model=entry['label'],scene=scene['name'],selection=str(selection),checkpoint_sha256=record['checkpoint_sha256'],split_validation=validation,test_groups=tests,parameter_memory_key_state_unchanged=True,metric_space='train_standardized',stage2_training=False)
 del model,memory,payload;torch.cuda.empty_cache()
 cache=ROOT/'T1'/scene['name']/'repair-audit.json'
 if cache.exists():baseline=json.loads(cache.read_text())
 else:
  parent=json.loads(Path(scene['t1_repair_selection']).read_text());cp=Path(parent['checkpoint']);assert sha(cp)==parent['checkpoint_sha256']
  official=Path('/home/xyw/paper/E23')/parent['profile']['root'];git='/home/xyw/miniforge3/envs/malr/bin/git'
  assert subprocess.check_output([git,'-C',str(official),'rev-parse','HEAD'],text=True).strip()==parent['upstream_revision']
  tracked=subprocess.check_output([git,'-C',str(official),'ls-files','-z']).decode().split('\0');digest=hashlib.sha256()
  for name in sorted(tracked):
   p=official/name
   if p.suffix in ('.py','.yaml','.yml','.sh','.ini') and p.is_file():digest.update(name.encode());digest.update(p.read_bytes())
  assert digest.hexdigest()==parent['upstream_source_sha256']
  imputer=build_author_model('T1',parent['profile'],official,device)
  imputer.load_state_dict(torch.load(cp,map_location=device,weights_only=False)['model_state'],strict=True);imputer.eval().requires_grad_(False);before_t1=parameter_hash(imputer)
  baseline=dict(model='T1',scene=scene['name'],checkpoint_sha256=parent['checkpoint_sha256'],split_validation=score(imputer,protocol,device,'T1','validation'),test_groups=[])
  for index,view in enumerate(views):baseline['test_groups'].append(dict(group_id=f'group_{index:02d}',mask_sha256=view.mask_sha256['test'],**score(imputer,view,device,'T1','test')))
  assert before_t1==parameter_hash(imputer);baseline['state_unchanged']=True
  write(cache,baseline);del imputer;torch.cuda.empty_cache()
 for a,b in zip(tests,baseline['test_groups']):
  assert a['mask_sha256']==b['mask_sha256'] and a['windows']==b['windows']
  assert all(a['metrics'][k]['hidden_coordinates']==b['metrics'][k]['hidden_coordinates'] for k in a['metrics'])
 refs={x['scene']:x for x in json.loads((ROOT/'validation_references.json').read_text())};ref=refs[scene['name']]
 assert all(abs(baseline['split_validation']['metrics'][region][k]-ref['t1'][region][k])<1e-6 for region in ref['t1'] for k in ['mse','mae'])
 own['test_mean']={region:{k:sum(t['metrics'][region][k] for t in tests)/len(tests) for k in ['mse','mae']} for region in ['all_hidden','last32_hidden']}
 own['t1_test_mean']={region:{k:sum(t['metrics'][region][k] for t in baseline['test_groups'])/len(tests) for k in ['mse','mae']} for region in ['all_hidden','last32_hidden']}
 own['screening_passed_v79_validation']=all(validation['metrics'][region][k]<ref['v79'][region][k] for region in ref['v79'] for k in ['mse','mae'])
 own['beats_t1_validation_all_four_metrics']=all(validation['metrics'][region][k]<ref['t1'][region][k] for region in ref['t1'] for k in ['mse','mae'])
 own['beats_t1_test_all_four_metrics']=all(own['test_mean'][region][k]<own['t1_test_mean'][region][k] for region in own['test_mean'] for k in ['mse','mae'])
 write(ROOT/entry['label']/scene['name']/'repair-audit.json',own)
 return own

if __name__=='__main__':
 import sys
 torch.set_num_threads(4);plan=json.loads((ROOT/'plan.json').read_text());entry=next(x for x in plan['models'] if x['label']==sys.argv[1]);scene=next(x for x in plan['scenarios'] if x['name']==sys.argv[2])
 result=evaluate(plan,scene,entry);print(json.dumps({k:result[k] for k in ['model','scene','split_validation','test_mean','t1_test_mean','screening_passed_v79_validation','beats_t1_validation_all_four_metrics','beats_t1_test_all_four_metrics']}),flush=True)
