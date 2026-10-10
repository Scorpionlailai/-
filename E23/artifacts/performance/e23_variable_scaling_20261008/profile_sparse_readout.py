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

import types,statistics,hashlib
# Profile forward+backward on fixed validation INPUTS. No optimizer update or test evaluation.
m.eval()
for parameter in m.historical.key_encoder.parameters():parameter.requires_grad_(False)
originals=[];mask_fractions=[]
def enable_sparse():
 for parent,names in [(m.point_router,['score']),(m.overlap_confidence,['score']),(m.point_candidate_readout,['query','output'])]:
  if parent is None:continue
  old_parent=parent.forward
  def parent_forward(*args,_parent=parent,_old=old_parent,**kwargs):
   inputs=args[0];_parent.sparse_eligible=inputs.patch_validity & ~inputs.patch_mask
   return _old(*args,**kwargs)
  parent.forward=parent_forward;originals.append((parent,old_parent))
  for name in names:
   child=getattr(parent,name);old_child=child.forward
   def child_forward(features,_parent=parent,_old=old_child):
    eligible=_parent.sparse_eligible
    selected=_old(features[eligible])
    result=features.new_zeros(*features.shape[:-1],selected.shape[-1]);result[eligible]=selected
    return result
   child.forward=child_forward;originals.append((child,old_child))

def disable_sparse():
 for mod,method in originals:mod.forward=method
 originals.clear()
def timed():
 times=[];peaks=[];out=None
 for repeat in range(5):
  b=batches[repeat%len(batches)];m.zero_grad(set_to_none=True);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
  out=forward(b);hidden=~b.observed_mask;error=(out-b.normalized_target)*b.query_std[:,None];error.square()[hidden].mean().backward()
  torch.cuda.synchronize();times.append(1000*(time.perf_counter()-t));peaks.append(torch.cuda.max_memory_allocated()/2**30)
 return dict(step_ms_median=statistics.median(times[1:]),step_ms_samples=times[1:],peak_allocated_gib=max(peaks)),out
base_stats,_=timed();m.zero_grad(set_to_none=True)
b=batches[0];old=forward(b);err=(old-b.normalized_target)*b.query_std[:,None];old_loss=err.square()[~b.observed_mask].mean();old_loss.backward();old_grads={name:param.grad.detach().cpu().clone() for name,param in m.named_parameters() if param.grad is not None};old=old.detach();m.zero_grad(set_to_none=True)
enable_sparse();new=forward(b);err=(new-b.normalized_target)*b.query_std[:,None];new_loss=err.square()[~b.observed_mask].mean();new_loss.backward();new_grads={name:param.grad.detach().cpu().clone() for name,param in m.named_parameters() if param.grad is not None}
assert old_grads.keys()==new_grads.keys()
grad_max=max((old_grads[n]-new_grads[n]).abs().max().item() for n in old_grads)
grad_norm_diff=(sum((old_grads[n]-new_grads[n]).double().square().sum().item() for n in old_grads))**.5
grad_norm=(sum(old_grads[n].double().square().sum().item() for n in old_grads))**.5
new_stats,_=timed();result=dict(dataset='Weather',variables=21,batch_size=128,mode='fixed validation forward/backward; eval dropout; no optimizer updates; no tests',original=base_stats,sparse=new_stats,output_max_abs_difference=(old-new.detach()).abs().max().item(),loss_difference=abs(old_loss.item()-new_loss.item()),gradient_max_abs_difference=grad_max,gradient_relative_l2_difference=grad_norm_diff/max(grad_norm,1e-12),optimization='query/output/point-route/overlap MLPs only evaluate hidden valid positions; eight heads/candidates/weights unchanged; observed attention diagnostics can differ')
(R/'weather_sparse_profile.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
