"""Oracle interventions on fixed validation inputs; no training or test access."""
from pathlib import Path
from time import perf_counter
from datetime import datetime,timezone
import os,json,csv,hashlib
import numpy as np
import torch
from integrations.e23.config import load_e23_settings
from integrations.e23.train import (_key_sha256,_new_memory,_starts,build_repair_model,build_forecast_model,load_stage1_selection,load_stage2_selection)
from integrations.forecast_compare.protocol import iter_p4_batches
from scripts.diagnose_e23_v30_stack import load_protocol
from scripts.run_e23_v30_trial import verify_freeze
from malr.data.patching import patchify_preserving_values

ROOT=Path.cwd();OUT=Path(__file__).resolve().parent
TRIALS=ROOT/'artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments'
FREEZE=TRIALS/'e23_v30_stacked_parallel/code-freeze.json'
VARIABLES=['HUFL','HULL','MUFL','MULL','LUFL','LULL','OT']
MODELS=[('v29_r1',TRIALS/'e23_v29_iterative_tv_h_r1',ROOT/'configs/e23_multi_expert_p4_width64_iterative_tv_h_r1_v29.yaml'),('v30_l4_g8',TRIALS/'e23_v30_stacked_parallel/l4_g8',ROOT/'configs/e23_multi_expert_p4_width64_stack_l4_g8_v30.yaml')]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def model_hash(state):
 digest=hashlib.sha256()
 for name,value in sorted(state.items()):
  array=value.detach().cpu().contiguous().numpy()
  digest.update(f'{name}:{array.dtype}:{array.shape}\n'.encode());digest.update(array.tobytes())
 return digest.hexdigest()

def load_models(protocol,device):
 models={};memories={};records={};before={}
 for label,root,config in MODELS:
  settings=load_e23_settings(config,protocol=protocol)
  for stage in ('stage1','stage2'):
   manifest=root/stage/'runs'/f'e23-{stage}.json';record=json.loads(manifest.read_text())
   loader=load_stage1_selection if stage=='stage1' else load_stage2_selection
   payload=loader(manifest,protocol,settings,device=device)
   model=(build_repair_model if stage=='stage1' else build_forecast_model)(protocol,settings,device=device)
   model.load_state_dict(payload['model_state'],strict=True);model.eval().requires_grad_(False)
   repair=model if stage=='stage1' else model.repair
   repair.historical.retrieval.load_state_dict(payload['retrieval_state'])
   memory=_new_memory(protocol);memory.load_state_dict(payload['memory_state']);repair.historical.refresh_snapshot(memory.snapshot())
   key=f'{label}_{stage}';models[key]=model;memories[key]=memory;records[key]=record
   before[key]=dict(model_sha=model_hash(model.state_dict()),key_sha=_key_sha256(repair.historical.retrieval.state_dict()),bank_sha=memory.snapshot_hash())
   assert before[key]['key_sha']==record['key_sha256']
   del payload
 return models,memories,records,before

@torch.inference_mode()
def run():
 start=perf_counter();torch.set_num_threads(4);device=torch.device('cuda:0')
 verify_freeze(TRIALS/'e23_v30_stacked_parallel');os.environ['MALR_E23_CODE_FREEZE']=str(FREEZE)
 protocol=load_protocol();models,memories,records,before=load_models(protocol,device)
 heads={label:models[f'{label}_stage2'].predictor for label,_,_ in MODELS}
 # Disjoint original-time regions; all overlapping patch copies are changed together.
 groups=[(c,t,t+16) for t in range(0,96,16) for c in range(7)]
 time_index=torch.arange(12,device=device)[:,None]*8+torch.arange(16,device=device)[None,:]
 region_masks=torch.zeros(42,7,12,16,device=device,dtype=torch.bool)
 for g,(c,left,right) in enumerate(groups):region_masks[g,c]=(time_index>=left)&(time_index<right)
 summaries=[];group_rows=[];arrays={};repair_sums={key:np.zeros((42,3),dtype=np.float64) for key in models}
 accum={(source,head):dict(base_sq=0.,base_abs=0.,oracle_sq=0.,oracle_abs=0.,gain=[],gain_mae=[],horizon_gain=[],counts=[]) for source in models for head in heads}
 windows=coordinates=0
 for batch_index,cpu in enumerate(iter_p4_batches(protocol,'validation',batch_size=128)):
  batch=cpu.to(device);B=len(batch.normalized_observed)
  clean=patchify_preserving_values(batch.normalized_target,batch.observed_mask,16,8)
  clean_values=clean.values.masked_fill(~clean.validity,0)
  timeline_hidden=~batch.observed_mask
  counts=torch.stack([timeline_hidden[:,left:right,c].sum(1) for c,left,right in groups],1)
  for source,model in models.items():
   repair=model if source.endswith('stage1') else model.repair
   starts=_starts(batch,stride=8,patches=12,device=device)
   output=repair(batch.normalized_observed,batch.observed_mask,batch.query_mean,batch.query_std,starts)
   x=output.explicit;hidden=(~x.observation_patches)&x.validity_patches;values=x.repaired_patches
   assert torch.equal(x.observation_patches,clean.effective_mask)
   assert torch.equal(x.validity_patches,clean.validity)
   assert torch.equal(values[x.observation_patches],clean_values[x.observation_patches])
   err=(x.normalized_sequence-batch.normalized_target)*batch.query_std[:,None,:]
   for g,(c,left,right) in enumerate(groups):
    e=err[:,left:right,c][timeline_hidden[:,left:right,c]]
    repair_sums[source][g]+=[e.square().double().sum().item(),e.abs().double().sum().item(),e.numel()]
   oracle=torch.where(hidden,clean_values,values)
   for head_name,head in heads.items():
    acc=accum[(source,head_name)]
    predict=lambda v,m=x.observation_patches,mean=batch.query_mean,std=batch.query_std,valid=x.validity_patches:head(v,m,mean,std,patch_validity=valid)
    baseline=predict(values);full_oracle=predict(oracle)
    if source.endswith('stage2') and source.startswith(head_name):
     direct,_=model(batch.normalized_observed,batch.observed_mask,batch.query_mean,batch.query_std,starts)
     torch.testing.assert_close(baseline,direct,rtol=0,atol=0)
    base_error=baseline-batch.forecast_target;oracle_error=full_oracle-batch.forecast_target
    base_mse=base_error.square().mean((1,2));base_mae=base_error.abs().mean((1,2))
    base_horizon=base_error.square().reshape(B,7,4,24).mean((1,3))
    acc['base_sq']+=base_error.square().double().sum().item();acc['base_abs']+=base_error.abs().double().sum().item()
    acc['oracle_sq']+=oracle_error.square().double().sum().item();acc['oracle_abs']+=oracle_error.abs().double().sum().item()
    gains=[];mae_gains=[];horizon_gains=[]
    for left in range(0,42,7):
     K=7;selected=region_masks[left:left+K,None]&hidden[None]
     variants=torch.where(selected,clean_values[None],values[None])
     assert torch.equal(variants.masked_select(~selected),values[None].expand_as(variants).masked_select(~selected))
     forecast=head(variants.reshape(K*B,7,12,16),x.observation_patches.repeat(K,1,1,1),batch.query_mean.repeat(K,1),batch.query_std.repeat(K,1),patch_validity=x.validity_patches.repeat(K,1,1,1)).reshape(K,B,7,96)
     errors=forecast-batch.forecast_target[None]
     gain=(base_mse[None]-errors.square().mean((2,3))).T
     mae_gain=(base_mae[None]-errors.abs().mean((2,3))).T
     hg=(base_horizon[None]-errors.square().reshape(K,B,7,4,24).mean((2,4))).permute(1,0,2)
     empty=counts[:,left:left+K]==0
     assert gain[empty].abs().max().item()<2e-5 if empty.any() else True
     gain=gain.masked_fill(empty,0);mae_gain=mae_gain.masked_fill(empty,0);hg=hg.masked_fill(empty[...,None],0)
     assert torch.isfinite(gain).all() and torch.isfinite(mae_gain).all() and torch.isfinite(hg).all()
     gains.append(gain.cpu().numpy());mae_gains.append(mae_gain.cpu().numpy());horizon_gains.append(hg.cpu().numpy())
    acc['gain'].append(np.concatenate(gains,axis=1));acc['gain_mae'].append(np.concatenate(mae_gains,axis=1));acc['horizon_gain'].append(np.concatenate(horizon_gains,axis=1));acc['counts'].append(counts.cpu().numpy())
  windows+=B;coordinates+=batch.forecast_target.numel()
  if batch_index%5==0:print(f'validation batch={batch_index+1}/21 windows={windows}',flush=True)
 assert windows==2688 and coordinates==1806336
 for (source,head),acc in accum.items():
  gains=np.concatenate(acc['gain']);mae_gains=np.concatenate(acc['gain_mae']);horizon=np.concatenate(acc['horizon_gain']);counts=np.concatenate(acc['counts'])
  repair_stats=repair_sums[source]
  arrays[f'{source}__{head}__gain']=gains;arrays[f'{source}__{head}__horizon_gain']=horizon
  reconstruction_mse=float(repair_stats[:,0].sum()/repair_stats[:,2].sum())
  metric=dict(source=source,predictor=head,repair_mse_global=reconstruction_mse,forecast_mse=acc['base_sq']/coordinates,forecast_mae=acc['base_abs']/coordinates,all_hidden_oracle_mse=acc['oracle_sq']/coordinates,all_hidden_oracle_mae=acc['oracle_abs']/coordinates,
   oracle_mse_gain=(acc['base_sq']-acc['oracle_sq'])/coordinates,positive_mean_regions=int((gains.mean(0)>0).sum()),regions=42,windows=windows,forecast_coordinates=coordinates)
  summaries.append(metric)
  if source.endswith('stage1'):assert abs(reconstruction_mse-records[source]['best_val_repair_mse'])<1e-6
  if source.endswith('stage2') and source.startswith(head):
   assert abs(metric['forecast_mse']-records[source]['best_val_mse'])<1e-6 and abs(metric['forecast_mae']-records[source]['best_val_mae'])<1e-6
  for g,(c,left,right) in enumerate(groups):
   active=counts[:,g]>0;positive=gains[active,g]>0
   group_rows.append(dict(source=source,predictor=head,variable=VARIABLES[c],history_start=left,history_end_exclusive=right,hidden_coordinates=int(repair_stats[g,2]),repair_mse_global=float(repair_stats[g,0]/repair_stats[g,2]),mean_forecast_mse_gain=float(gains[:,g].mean(dtype=np.float64)),mean_forecast_mae_gain=float(mae_gains[:,g].mean(dtype=np.float64)),positive_fraction_active=float(positive.mean()),active_windows=int(active.sum()),**{f'horizon_{i*24+1}_{(i+1)*24}_mse_gain':float(horizon[:,g,i].mean(dtype=np.float64)) for i in range(4)}))
 for source,model in models.items():
  repair=model if source.endswith('stage1') else model.repair
  after=dict(model_sha=model_hash(model.state_dict()),key_sha=_key_sha256(repair.historical.retrieval.state_dict()),bank_sha=memories[source].snapshot_hash())
  assert before[source]==after
 verify_freeze(TRIALS/'e23_v30_stacked_parallel')
 for name,rows in [('summary.csv',summaries),('position_interventions.csv',group_rows)]:
  with (OUT/name).open('x',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
 np.savez_compressed(OUT/'per_window_gains.npz',**arrays)
 result=dict(schema_version=1,completed_utc=datetime.now(timezone.utc).isoformat(),diagnostic_only=True,split='original_fixed_validation',training_or_checkpoint_selection=False,test_accessed=False,real_world_causality_claim=False,
  intervention='restore missing values in each variable x disjoint 16-step region to oracle history; fixed predictor, original masks, query mean/std, observed values and validity',
  region_count=42,windows=windows,metric_space='train_standardized',overlapping_windows_not_independent=True,positive_gain_definition='original forecast MSE minus oracle-region forecast MSE; negative values retained',
  oracle_predictions_not_deployable=True,oracle_not_a_guaranteed_error_upper_bound=True,predictors_cross_swapped_for_diagnosis_not_formal_benchmarks=True,
  source_csv_sha256=protocol.source_sha256,validation_mask_sha256=protocol.mask_sha256['validation'],code_freeze_sha256=sha(FREEZE),source_code_freeze_unchanged=True,
  checkpoint_sha256={key:r['checkpoint_sha256'] for key,r in records.items()},state_hashes_unchanged=before,
  device=str(device),gpu=torch.cuda.get_device_name(device),diagnostic_seconds=perf_counter()-start,peak_allocated_gib=torch.cuda.max_memory_allocated(device)/1024**3,peak_reserved_gib=torch.cuda.max_memory_reserved(device)/1024**3,summaries=summaries,
  script_sha256=sha(Path(__file__)))
 with (OUT/'diagnosis.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps(summaries,indent=2),flush=True)

if __name__=='__main__':run()
