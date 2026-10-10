import sys,json,time,statistics
from pathlib import Path
from types import SimpleNamespace
R=Path(__file__).resolve().parent;W=Path('/home/xyw/paper/E23/versions/e23-efficiency-validation-cache-20261008/E23');sys.path[:0]=[str(W/'src'),str(W)]
import torch
from malr.models.e23.variable import SparseVariableExpert
from shortlist_sources import shortlisted_sources
torch.set_num_threads(4);torch.manual_seed(2023);device=torch.device('cuda:0');rows=[]
for C in [7,21,64,128,321]:
 B,N,P=16,11,16;m=SparseVariableExpert(C,P,64,topk=4,selection='window_context_topk_v1',contextual=True).to(device);values=torch.randn(B,C,N,P,device=device);mask=torch.rand_like(values)>.73;valid=torch.ones_like(mask);x=SimpleNamespace(patch_values=values,patch_mask=mask,patch_validity=valid)
 def full():return m._contextual_sources(x,epoch=0,explore=False)
 def short():return shortlisted_sources(m,x,epoch=0,explore=False,cap=16,original=SparseVariableExpert._contextual_sources)
 def measure(f):
  with torch.inference_mode():
   f();torch.cuda.synchronize();times=[];torch.cuda.reset_peak_memory_stats()
   for _ in range(5):
    t=time.perf_counter();out=f();torch.cuda.synchronize();times.append((time.perf_counter()-t)*1000)
  return dict(ms=statistics.median(times),peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30),out
 a,aa=measure(full);b,bb=measure(short);rows.append(dict(variables=C,batch_size=B,patches=N,patch_length=P,cap=16,full=a,shortlist=b,speedup=a['ms']/b['ms'],selected_sources_equal=torch.equal(aa[0],bb[0]),top4_source_overlap=(aa[0][...,None]==bb[0][...,None,:]).any(-1).float().mean().item(),synthetic_inputs=True,trained_weights=False,full_model_or_training_speedup=False));del m,values,mask,valid,x;torch.cuda.empty_cache()
(R/'shortlist_gpu_scaling.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
