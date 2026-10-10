"""Read-only validation diagnostics; oracle historical values are diagnostic only."""
from pathlib import Path
from dataclasses import replace
from time import perf_counter
import sys,json,functools,csv
BASE=Path('/home/xyw/paper/E23/artifacts/performance/e23_v103_dynamic_candidates_20261008')
sys.path.insert(0,str(BASE))
import train_v103_forecast as setup
import torch
from integrations.e23.train import load_stage2_selection,build_forecast_model,_new_memory,_starts
from integrations.e23.forecast_sensitive import parameter_hash
from integrations.forecast_compare.protocol import iter_p4_batches
from malr.data.patching import patchify_preserving_values
R=Path(__file__).resolve().parent
torch.set_num_threads(4);device=torch.device('cuda:0')
setup.freeze_check();protocol,settings=setup.protocol_settings();selection=setup.BASE/'stage2/runs/e23-stage2.json';payload=load_stage2_selection(selection,protocol,settings,device=device)
model=build_forecast_model(protocol,settings,device=device);model.load_state_dict(payload['model_state'],strict=True);model.repair.historical.retrieval.load_state_dict(payload['retrieval_state']);memory=_new_memory(protocol);memory.load_state_dict(payload['memory_state']);model.repair.historical.refresh_snapshot(memory.snapshot());model.eval().requires_grad_(False);del payload
before=parameter_hash(model);columns=json.loads(setup.MANIFEST.read_text())['columns']
conditions=('repaired','oracle_all_hidden','oracle_last32_hidden','oracle_first64_hidden','visible_boundary_guide')
stats={name:{k:torch.zeros(21,dtype=torch.float64,device=device) for k in ('forecast_squared','forecast_absolute','forecast_count','repair_squared','repair_absolute','repair_count','tail_squared','tail_absolute','tail_count')} for name in conditions}
started=perf_counter();cached=[]
with torch.no_grad():
 for index,cpu in enumerate(iter_p4_batches(protocol,'validation',batch_size=128),1):
  b=cpu.to(device);starts=_starts(b,stride=model.repair.stride,patches=model.repair.patch_count,device=device)
  out=model.repair(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts).explicit
  hidden=~b.observed_mask & out.sequence_validity;original=out.normalized_sequence
  T=original.shape[1];time=torch.arange(T,device=device)[None,:,None]
  left=torch.where(b.observed_mask,time,-1).cummax(1).values;right=torch.where(b.observed_mask,time,T).flip(1).cummin(1).values.flip(1)
  leftval=b.normalized_observed.gather(1,left.clamp_min(0));rightval=b.normalized_observed.gather(1,right.clamp_max(T-1))
  both=(left>=0)&(right<T);guide=leftval+(rightval-leftval)*(time-left)/(right-left).clamp_min(1)
  guide=torch.where(both,guide,torch.where(left>=0,leftval,torch.where(right<T,rightval,torch.zeros_like(guide))))
  for name in conditions:
   if name=='repaired':sequence=original
   elif name=='visible_boundary_guide':sequence=torch.where(hidden,guide,original)
   else:
    scope=hidden
    if name=='oracle_last32_hidden':scope=hidden & (time>=T-32)
    elif name=='oracle_first64_hidden':scope=hidden & (time<T-32)
    sequence=torch.where(scope,b.normalized_target,original)
   patch=patchify_preserving_values(sequence,out.sequence_observed,16,8,validity=out.sequence_validity)
   explicit=replace(out,normalized_sequence=sequence,repaired_patches=patch.values.masked_fill(~patch.validity,0))
   values,observed,valid=model.explicit_predictor_inputs(explicit)
   prediction=model.predictor(values,observed,b.query_mean,b.query_std,patch_validity=valid)
   error=(prediction-b.forecast_target).double().transpose(1,2)
   repair_error=((sequence-b.normalized_target)*b.query_std[:,None]).double()
   s=stats[name];s['forecast_squared']+=error.square().sum((0,1));s['forecast_absolute']+=error.abs().sum((0,1));s['forecast_count']+=error.shape[0]*error.shape[1]
   s['repair_squared']+=(repair_error.square()*hidden).sum((0,1));s['repair_absolute']+=(repair_error.abs()*hidden).sum((0,1));s['repair_count']+=hidden.sum((0,1))
   s['tail_squared']+=(repair_error[:,-32:].square()*hidden[:,-32:]).sum((0,1));s['tail_absolute']+=(repair_error[:,-32:].abs()*hidden[:,-32:]).sum((0,1));s['tail_count']+=hidden[:,-32:].sum((0,1))
   if name=='repaired':
    cached.append(tuple(v.detach().cpu() for v in (values,observed,valid,b.query_mean,b.query_std,b.forecast_target)))
  if index%5==0 or index==1:print('validation batch',index,'/41',flush=True)
assert parameter_hash(model)==before
summary={};rows=[]
for name,s in stats.items():
 summary[name]=dict(forecast_mse=float(s['forecast_squared'].sum()/s['forecast_count'].sum()),forecast_mae=float(s['forecast_absolute'].sum()/s['forecast_count'].sum()),repair96_mse=float(s['repair_squared'].sum()/s['repair_count'].sum()),repair96_mae=float(s['repair_absolute'].sum()/s['repair_count'].sum()),repair32_mse=float(s['tail_squared'].sum()/s['tail_count'].sum()),repair32_mae=float(s['tail_absolute'].sum()/s['tail_count'].sum()))
 for c,col in enumerate(columns):
  row=dict(condition=name,variable=col)
  for key,num,den in [('forecast_mse','forecast_squared','forecast_count'),('forecast_mae','forecast_absolute','forecast_count'),('repair96_mse','repair_squared','repair_count'),('repair96_mae','repair_absolute','repair_count'),('repair32_mse','tail_squared','tail_count'),('repair32_mae','tail_absolute','tail_count')]:row[key]=float(s[num][c]/s[den][c])
  rows.append(row)
(R/'diagnostic_validation.json').write_text(json.dumps(dict(summary=summary,validation_windows=len(protocol.refs['validation']),seconds=perf_counter()-started,weights_unchanged=True,scope='fixed validation only; original frozen trained predictor; oracle historical truth used only to diagnose sensitivity, not deployable accuracy; no causal identification',test_used=False),indent=2))
with (R/'diagnostic_per_variable.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
torch.save(dict(batches=cached,repair_sha256=parameter_hash(model.repair),selection=str(selection),mask_sha256=protocol.mask_sha256['validation'],scope='fixed validation only'),R/'v103_validation_predictor_inputs.pt')
print(json.dumps(summary,indent=2),flush=True)
# Inclusive module forward CUDA times and full repair forward/backward; no optimizer.
for name in ('front_context','point_candidate_readout','point_refiner','recent_mean_projector','point_router','overlap_confidence'):
 module=getattr(model.repair,name,None)
 if module is None:continue
 def pre(module,args,label=name):
  context=torch.profiler.record_function('repair_module::'+label);context.__enter__();module._profile_context=context
 def post(module,args,result):module._profile_context.__exit__(None,None,None)
 module.register_forward_pre_hook(pre);module.register_forward_hook(post)
for name in ('variable','historical','temporal'):
 module=getattr(model.repair,name)
 for method in ('prepare_sources','prepare_candidates','from_cache'):
  if not hasattr(module,method):continue
  original_method=getattr(module,method)
  def wrapper(*args,_method=original_method,_label=name+'.'+method,**kwargs):
   with torch.profiler.record_function('repair_module::'+_label):return _method(*args,**kwargs)
  setattr(module,method,wrapper)
model.repair.requires_grad_(True).train();batch=next(iter_p4_batches(protocol,'train',batch_size=128,epoch=0)).to(device)
starts=_starts(batch,stride=model.repair.stride,patches=model.repair.patch_count,device=device)
model.repair.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats(device)
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],profile_memory=True) as prof:
 with torch.profiler.record_function('repair_total_forward'):
  result=model.repair(batch.normalized_observed,batch.observed_mask,batch.query_mean,batch.query_std,starts,epoch=1,explore_variable=True)
  error=(result.explicit.normalized_sequence-batch.normalized_target)*batch.query_std[:,None]
  loss=error[~batch.observed_mask].square().mean()
 with torch.profiler.record_function('repair_total_backward'):loss.backward()
torch.cuda.synchronize(device)
profile_rows=[]
for event in prof.key_averages():
 if event.key.startswith('repair_'):
  profile_rows.append(dict(name=event.key,device_time_total_us=event.device_time_total,self_device_time_total_us=event.self_device_time_total,cpu_time_total_us=event.cpu_time_total,device_memory_usage=event.device_memory_usage))
(R/'module_profile.json').write_text(json.dumps(dict(rows=profile_rows,peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,batch_size=128,H_bank_loaded=True,optimizer=False,single_profiled_batch=True,scope='inclusive module times may overlap; not full epoch speed; trained v103 full H bank'),indent=2))
print('PROFILE',json.dumps(profile_rows),flush=True)
