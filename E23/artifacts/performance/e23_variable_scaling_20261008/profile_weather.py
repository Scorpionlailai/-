import os,sys,json,time,inspect,textwrap
from pathlib import Path
W=Path('/home/xyw/paper/E24');sys.path[:0]=[str(W/'src'),str(W)]
import torch
from integrations.e24.data import load_dataset_protocol
from integrations.e23.config import load_e23_settings
from integrations.e23.train import build_repair_model,_new_memory,_starts
from integrations.forecast_compare.protocol import iter_p4_batches
from malr.prototypes.tensor_index import TensorPrototypeIndex
R=Path(__file__).resolve().parent;device=torch.device('cuda:0');torch.set_num_threads(4)
manifest=W/'artifacts/Weather/seq96_pred96/sr010_bl8/shared_protocol_v1/dataset_manifest.json';identity=json.loads(manifest.read_text())
p=load_dataset_protocol(manifest,config_path=Path(identity['config_path']),artifacts=manifest.parent);settings=load_e23_settings(W/'configs/e24_v79.yaml',protocol=p)
record_path=W/'artifacts/Weather/seq96_pred96/sr010_bl8/e24_v79_weather_20261007/seed_2023/E24-v79/stage1/runs/e23-stage1.json';record=json.loads(record_path.read_text());payload=torch.load(record['checkpoint'],map_location='cpu',weights_only=False)
m=build_repair_model(p,settings,device=device);m.load_state_dict(payload['model_state']);m.historical.retrieval.load_state_dict(payload['retrieval_state']);bank=_new_memory(p);bank.load_state_dict(payload['memory_state']);m.historical.refresh_snapshot(bank.snapshot());m.eval()
batches=[b.to(device) for _,b in zip(range(3),iter_p4_batches(p,'validation',batch_size=128))]
def forward(b):
 return m(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,_starts(b,stride=m.stride,patches=m.patch_count,device=device)).explicit.normalized_sequence
timings={};hooks=[]
for name,mod in [('window_context',m.front_context),('historical_candidates',m.historical),('variable',m.variable),('temporal',m.temporal),('point_candidate_readout',m.point_candidate_readout),('recent_mean',m.recent_mean_projector)]:
 if mod is None:continue
 def before(mod,args,name=name):
  a=torch.cuda.Event(enable_timing=True);a.record();timings.setdefault(name,[]).append([a,None])
 def after(mod,args,out,name=name):
  e=torch.cuda.Event(enable_timing=True);e.record();timings[name][-1][1]=e
 hooks += [mod.register_forward_pre_hook(before),mod.register_forward_hook(after)]
for name,mod,method in [('historical_lookup',m.historical,'prepare_candidates'),('variable_selection',m.variable,'prepare_sources')]:
 original_method=getattr(mod,method)
 def wrapped(*args,_name=name,_method=original_method,**kwargs):
  a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();out=_method(*args,**kwargs);b.record();timings.setdefault(_name,[]).append([a,b]);return out
 setattr(mod,method,wrapped)
with torch.inference_mode():
 forward(batches[0]);torch.cuda.synchronize();timings.clear();torch.cuda.reset_peak_memory_stats();t=time.perf_counter();outputs=[forward(b) for b in batches];torch.cuda.synchronize();base_seconds=time.perf_counter()-t;base_peak=torch.cuda.max_memory_allocated()
base_components={k:sum(a.elapsed_time(b) for a,b in v)/len(v) for k,v in timings.items()}
for h in hooks:h.remove()
source=textwrap.dedent(inspect.getsource(TensorPrototypeIndex.query))
source=source.replace('member_score = (q[..., None, :] * chosen_keys).sum(dim=-1)','member_score = (q[..., None, :] * chosen_keys).sum(dim=-1)\n        del chosen_keys')
source=source.replace('chosen_raw = self.group_member_raw[best_global]','selected_raw = self.group_member_raw[best_global[..., None], order]')
source=source.replace('chosen_raw.gather(\n            2, order[..., None].expand(-1, -1, -1, self.patch_length),\n        )','selected_raw')
ns=dict(TensorPrototypeIndex.query.__globals__);exec(source,ns);original=TensorPrototypeIndex.query;TensorPrototypeIndex.query=ns['query'];(R/'query_optimized.py').write_text(source)
with torch.inference_mode():
 forward(batches[0]);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter();new_outputs=[forward(b) for b in batches];torch.cuda.synchronize();new_seconds=time.perf_counter()-t;new_peak=torch.cuda.max_memory_allocated()
result=dict(dataset='Weather',variables=21,batch_size=128,validation_batches=3,checkpoint=str(record_path),training=False,test_used=False,original_forward_ms=1000*base_seconds/3,optimized_forward_ms=1000*new_seconds/3,original_peak_allocated_gib=base_peak/2**30,optimized_peak_allocated_gib=new_peak/2**30,component_cuda_ms=base_components,exact_equal=[torch.equal(a,b) for a,b in zip(outputs,new_outputs)],max_absolute_difference=max((a-b).abs().max().item() for a,b in zip(outputs,new_outputs)),index_shapes=dict(group_member_keys=list(m.historical.index.group_member_keys.shape),padded_centers=list(m.historical.index.padded_centers.shape)),optimization='release gathered keys after reduction; gather only selected raw patches; FP64 and stable full sorting unchanged')
(R/'weather_profile.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
