import json,time,gc
from pathlib import Path
import torch
from torch.nn import functional as F
from contrastive_rows import full_contrastive_rows
from malr.prototypes.retrieval import masked_contrastive_loss
root=Path(__file__).resolve().parent;torch.set_num_threads(4);torch.manual_seed(2023)
checks=[]
for n,d,rows,temp,dtype in [(1,16,1,1.,torch.float64),(19,16,7,.3,torch.float64),(257,64,32,1.,torch.float32)]:
 q=F.normalize(torch.randn(n,d,device='cuda',dtype=dtype),dim=-1).requires_grad_();k=F.normalize(torch.randn(n,d,device='cuda',dtype=dtype),dim=-1).requires_grad_()
 a=masked_contrastive_loss(q,k,temperature=temp);ga=torch.autograd.grad(a,q)[0]
 b=full_contrastive_rows(q,k,temperature=temp,rows=rows);gb,gk=torch.autograd.grad(b,(q,k),allow_unused=True)
 torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-6);torch.testing.assert_close(ga,gb,rtol=1e-5,atol=1e-7)
 assert gk is None
 checks.append(dict(n=n,temperature=temp,loss_absolute_difference=abs(a.item()-b.item()),gradient_max_abs_difference=(ga-gb).abs().max().item()))
profile=json.loads((root/'cost_profile.json').read_text());n=profile['damaged_patches']
q=F.normalize(torch.randn(n,64,device='cuda'),dim=-1).requires_grad_();k=F.normalize(torch.randn_like(q),dim=-1)
results=[]
for name,fn in [('dense',masked_contrastive_loss),('row_blocks',full_contrastive_rows)]:
 measurements=[]
 for i in range(4):
  q.grad=None;gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();base=torch.cuda.memory_allocated();t=time.perf_counter()
  loss=fn(q,k);loss.backward();torch.cuda.synchronize();elapsed=time.perf_counter()-t
  measurements.append(dict(seconds=elapsed,additional_peak_GiB=(torch.cuda.max_memory_allocated()-base)/2**30))
  if i==3:
   gradient=q.grad.detach().cpu();value=loss.item()
   if name=='dense':old_gradient=gradient;old_value=value
   else:
    torch.testing.assert_close(gradient,old_gradient,rtol=1e-4,atol=1e-8)
    checks.append(dict(n=n,loss_absolute_difference=abs(value-old_value),gradient_max_abs_difference=(gradient-old_gradient).abs().max().item()))
  del loss
 results.append(dict(name=name,warmup_discarded=1,measurements=measurements[1:]))
out=dict(checks=checks,damaged_patches=n,benchmarks=results,scope='retrieval contrastive loss only; synthetic normalized keys at actual Weather batch geometry',original_training_unchanged=True)
(root/'contrastive_benchmark.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
