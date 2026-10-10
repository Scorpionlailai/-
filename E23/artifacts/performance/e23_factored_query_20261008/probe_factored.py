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

import types,statistics,hashlib,importlib.util
spec=importlib.util.spec_from_file_location('malr.models.e23.factored_probe','/home/xyw/paper/E23/versions/e23-efficiency-factored-query-20261008/E23/src/malr/models/e23/point_candidates.py')
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
# Replace only the method, retaining the original parameter objects and weights.
m.point_candidate_readout.forward=types.MethodType(probe.PointCandidateReadout.forward,m.point_candidate_readout)
m.point_candidate_readout.factor_query=False
os.environ['TORCHINDUCTOR_CACHE_DIR']=str(R/'inductor_cache')
# Profile forward+backward on fixed validation INPUTS. No optimizer update or test evaluation.
m.eval()
for parameter in m.historical.key_encoder.parameters():parameter.requires_grad_(False)
def timed():
 times=[];peaks=[];out=None
 for repeat in range(5):
  b=batches[repeat%len(batches)];m.zero_grad(set_to_none=True);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
  out=forward(b);hidden=~b.observed_mask;error=(out-b.normalized_target)*b.query_std[:,None];error.square()[hidden].mean().backward()
  torch.cuda.synchronize();times.append(1000*(time.perf_counter()-t));peaks.append(torch.cuda.max_memory_allocated()/2**30)
 return dict(step_ms_median=statistics.median(times[1:]),step_ms_samples=times[1:],peak_allocated_gib=max(peaks)),out

base_stats,_=timed();m.zero_grad(set_to_none=True)
b=batches[0];old=forward(b);err=(old-b.normalized_target)*b.query_std[:,None];old_loss=err.square()[~b.observed_mask].mean();old_loss.backward();old_grads={name:param.grad.detach().cpu().clone() for name,param in m.named_parameters() if param.grad is not None};old=old.detach();m.zero_grad(set_to_none=True)
m.point_candidate_readout.factor_query=True
new=forward(b);err=(new-b.normalized_target)*b.query_std[:,None];new_loss=err.square()[~b.observed_mask].mean();new_loss.backward();new_grads={name:param.grad.detach().cpu().clone() for name,param in m.named_parameters() if param.grad is not None}
assert old_grads.keys()==new_grads.keys()
grad_max=max((old_grads[n]-new_grads[n]).abs().max().item() for n in old_grads)
grad_norm_diff=(sum((old_grads[n]-new_grads[n]).double().square().sum().item() for n in old_grads))**.5
grad_norm=(sum(old_grads[n].double().square().sum().item() for n in old_grads))**.5
new_stats,_=timed();m.zero_grad(set_to_none=True)
started=time.perf_counter();m.point_candidate_readout.compile(fullgraph=True,dynamic=False);forward(b).square().mean().backward();torch.cuda.synchronize();first_compiled=time.perf_counter()-started
compiled_stats,_=timed()
result=dict(dataset='Weather',variables=21,batch_size=128,mode='eval dropout, fixed validation inputs, no optimizer/no tests',original=base_stats,factored=new_stats,factored_compiled=compiled_stats,compile_first_step_seconds=first_compiled,normalized_max_diff=(old-new.detach()).abs().max().item(),global_max_diff=((old-new.detach())*b.query_std[:,None]).abs().max().item(),loss_difference=abs(old_loss.item()-new_loss.item()),gradient_max_abs_difference=grad_max,gradient_relative_l2_difference=grad_norm_diff/max(grad_norm,1e-12))
(R/'weather_factored_query_probe.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
