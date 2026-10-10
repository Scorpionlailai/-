import sys,json
from pathlib import Path
from types import SimpleNamespace
R=Path(__file__).resolve().parent;W=Path('/home/xyw/paper/E23/versions/e23-efficiency-validation-cache-20261008/E23');sys.path[:0]=[str(W/'src'),str(W)]
import torch
from malr.models.e23.variable import SparseVariableExpert
from shortlist_sources import shortlisted_sources
torch.set_num_threads(4);torch.manual_seed(2023);checks=[]
for C in [7,21,64]:
 m=SparseVariableExpert(C,16,64,topk=4,selection='window_context_topk_v1',contextual=True);v=torch.randn(2,C,3,16);mask=torch.rand_like(v)>.57;valid=torch.ones_like(mask);valid[:,:,-1,-2:]=False
 x=SimpleNamespace(patch_values=v,patch_mask=mask,patch_validity=valid)
 for explore in [False,True]:
  a=shortlisted_sources(m,x,epoch=3,explore=explore,cap=16,original=SparseVariableExpert._contextual_sources)
  assert a[0].shape==(2,C,3,4) and not (a[0]==torch.arange(C)[None,:,None,None]).any()
  poison=SimpleNamespace(patch_values=torch.where(mask & valid,v,torch.full_like(v,1e6)),patch_mask=mask,patch_validity=valid)
  b=shortlisted_sources(m,poison,epoch=3,explore=explore,cap=16,original=SparseVariableExpert._contextual_sources);assert all(torch.equal(u,z) for u,z in zip(a,b))
  if C==7:
   full=m._contextual_sources(x,epoch=3,explore=explore);assert all(torch.equal(u,z) for u,z in zip(a,full))
  a[1].sum().backward();assert m.source_embedding.weight.grad is not None and torch.isfinite(m.source_embedding.weight.grad).all()
 checks.append(dict(variables=C,hidden_truth_ignored=True,self_sources_excluded=True,exploration_checked=True,prior_gradients_finite=True,fallback_exact=C==7))
print(json.dumps(checks,indent=2));(R/'shortlist_preflight.json').write_text(json.dumps(checks,indent=2))
