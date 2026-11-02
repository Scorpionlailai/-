"""No trained coarse model: exact zero or existing within-window neighbor guide."""
from pathlib import Path
import sys
import torch
from torch import nn
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'implementation'))
import bootstrap
from malr.data.normalization import masked_window_normalize,fit_train_variable_stats
from malr.data.patching import patchify_preserving_values
from malr.models.explicit_repair import ExplicitRepairBatch
from single import PrefixBatch,apply_head
from r2_variants import make_calibrator

def neighbor_initial(values,observed):
 visible=torch.where(observed,values,0.)
 index=torch.arange(values.shape[1],device=values.device)[None,:,None]
 left=torch.cummax(torch.where(observed,index,-1),dim=1).values
 right=torch.cummin(torch.where(observed,index,values.shape[1]).flip(1),dim=1).values.flip(1)
 lv=visible.gather(1,left.clamp_min(0));rv=visible.gather(1,right.clamp_max(values.shape[1]-1))
 span=(right-left).clamp_min(1)
 both=(left>=0)&(right<values.shape[1])&(right>left)
 linear=lv+(rv-lv)*(index-left)/span
 guide=torch.where(both,linear,torch.where(left>=0,lv,rv))
 return torch.where(observed,visible,guide)

class Initialization(nn.Module):
 def __init__(self,data,variant):
  super().__init__();self.data=data;self.variant=variant
  if variant not in ['S0','S1']:raise ValueError(variant)
 def forward(self,values,observed,starts):
  values=torch.where(observed,values,0.)
  return values if self.variant=='S0' else neighbor_initial(values,observed)
 @torch.no_grad()
 def extract(self,values,observed,starts):
  values=torch.where(observed,values,0.);initial=self(values,observed,starts)
  if not hasattr(self.data,'_reference_fallback'):
   self.data._reference_fallback=fit_train_variable_stats(torch.from_numpy(self.data.values[:self.data.bounds.train.end]),eps=0.)
  norm=masked_window_normalize(values,observed,self.data._reference_fallback,eps=1e-5)
  local=(initial-norm.mean[:,None])/norm.std[:,None]
  repaired=patchify_preserving_values(local,observed,16,8);obs=patchify_preserving_values(norm.student,observed,16,8)
  valid=torch.ones_like(observed,dtype=torch.bool);coverage=torch.zeros_like(values,dtype=torch.long)
  for i in range(repaired.values.shape[2]):coverage[:,i*8:min(i*8+16,96)]+=1
  explicit=ExplicitRepairBatch(local,initial,norm.student,observed,valid,~observed,repaired.values,obs.effective_mask,repaired.validity,coverage)
  return PrefixBatch(values,observed,explicit,None,None,norm.mean,norm.std)

class FineRecovery(nn.Module):
 def __init__(self,data,variant,seed=2023):
  super().__init__();self.initializer=Initialization(data,variant);self.head=make_calibrator(data.n_vars,'H_GLOBAL',seed)
 def forward(self,values,observed,starts):return apply_head(self.head,self.initializer.extract(values,observed,starts))
