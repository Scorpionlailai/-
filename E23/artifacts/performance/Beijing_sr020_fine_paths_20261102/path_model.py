"""Only change fine-repair input/output paths; frozen coarse model untouched."""
from pathlib import Path
import sys,json,torch
from torch import nn
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'implementation'))
import bootstrap
from r2_variants import make_calibrator
from single import FrozenRepair,apply_head
from t1_bridge import T1Repair,FrozenT1
from integrations.v125_ablation.model.global_dense import GlobalDenseRefiner
from variants import restore_retrieval,library_hash
from aligned_h_model import make_repair

class PathDense(GlobalDenseRefiner):
 def forward(self,normalized,explicit,mean,std):
  if normalized.shape[-1]!=self.variables:raise ValueError('variable count mismatch')
  valid=explicit.sequence_validity;observed=explicit.sequence_observed&valid
  coarse=torch.where(valid,normalized*std[:,None]+mean[:,None],0)
  safe=torch.where(observed,explicit.observed_sequence,0)
  visible=torch.where(observed,safe*std[:,None]+mean[:,None],0)
  guide=None
  if self.path_mode=='P2':suggested_delta=torch.zeros_like(coarse);support=torch.zeros_like(coarse)
  else:
   index=torch.arange(normalized.shape[1],device=normalized.device)[None,:,None]
   left=torch.cummax(torch.where(observed,index,-1),dim=1).values
   right=torch.cummin(torch.where(observed,index,normalized.shape[1]).flip(1),dim=1).values.flip(1)
   left_value=visible.gather(1,left.clamp_min(0));right_value=visible.gather(1,right.clamp_max(normalized.shape[1]-1))
   span=(right-left).clamp_min(1)
   both=(left>=0)&(right<normalized.shape[1])&(right>left)&valid
   linear=left_value+(right_value-left_value)*(index-left)/span
   has_left,has_right=left>=0,right<normalized.shape[1]
   one=(has_left^has_right)&valid&~observed
   distance=torch.where(has_left,index-left,right-index).clamp_min(0)
   guide=torch.where(both,linear,torch.where(has_left,left_value,right_value))
   support=(both.to(coarse.dtype)*torch.exp(-(span-1).to(coarse.dtype)/16)+one.to(coarse.dtype)*torch.exp(-distance.to(coarse.dtype)/16))
   suggested_delta=torch.where(both|one,guide-coarse.detach(),0)*support
  features=torch.cat((coarse.detach(),visible,observed.to(coarse.dtype),suggested_delta,support),-1)
  x=self.input(features)
  local=self.channel_mix(torch.nn.functional.gelu(self.depthwise(self.input_norm(x).transpose(1,2)))).transpose(1,2)
  x=x+local+self.position;token_valid=valid.any(-1);safe_padding=~token_valid;safe_padding=safe_padding.clone();safe_padding[:,0]&=token_valid.any(-1)
  x=self.temporal(x,src_key_padding_mask=safe_padding)
  hidden=valid&~observed;has_source=observed.flatten(1).any(-1)[:,None,None]
  if self.path_mode=='P3':additive=normalized
  else:
   delta=2*torch.tanh(self.output(x)/2);delta=torch.where(hidden&has_source,delta,0);additive=normalized+delta/std[:,None]
  if self.path_mode in ['P1','P2']:return additive
  gate=torch.tanh(self.guide_gate(x));active=hidden&observed.any(1,keepdim=True)
  correction=torch.where(active,gate*(guide-coarse.detach())/std[:,None],0)
  return additive+correction

def head(mode,variables=12,seed=2023):
 result=make_calibrator(variables,'H_GLOBAL',seed)
 dense=result.global_dense_refiner;dense.__class__=PathDense;dense.path_mode=mode
 if mode in ['P1','P2']:dense.guide_gate=None
 if mode=='P3':dense.output=None
 return result

def load_prefix(data,backbone,device,plan):
 if backbone=='R3':
  state=torch.load(plan['R3_main_checkpoint'],map_location=device,weights_only=False)
  repair=make_repair(data,'H_GLOBAL',2023).to(device);repair.load_state_dict(state['model'],strict=True);restore_retrieval(repair,state['retrieval']);repair.epoch=state['epoch'];repair.eval()
  assert library_hash(repair)==state['library_sha256']
  frozen=FrozenRepair(repair);metadata=dict(epoch=state['epoch'],checkpoint=plan['R3_main_checkpoint'],sha256=plan['R3_main_sha256'],validation=state['validation'])
 else:
  repair=T1Repair(data).to(device);repair.load_state_dict(torch.load(plan['T1_checkpoint'],map_location=device,weights_only=True),strict=True);repair.eval();frozen=FrozenT1(repair)
  metadata=dict(epoch=90,checkpoint=plan['T1_checkpoint'],sha256=plan['T1_sha256'])
 return frozen,metadata

class FineModel(nn.Module):
 def __init__(self,frozen,mode,seed=2023):
  super().__init__();self.frozen=frozen;self.head=head(mode,12,seed)
 def forward(self,values,observed,starts):return apply_head(self.head,self.frozen.extract(values,observed,starts))
