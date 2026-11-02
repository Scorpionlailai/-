"""E8: same candidate members, one candidate read per target patch and head."""
from math import sqrt
import torch
import variants
from aligned_h_model import make_repair as make_r3
from interfaces import DirectReadout
from integrations.v125_ablation.model.point_evidence import point_patches
from integrations.v125_ablation.model.coordinates import numeric_features
from integrations.v125_ablation.model.router import EXPERTS

class PatchSharedReadout(DirectReadout):
 def forward(self,inputs,router,evidence,decoder,points,visible_features,candidates):
  context=point_patches(points,inputs.patch_validity);visible=point_patches(visible_features,inputs.patch_validity)
  B,C,N,P,_=context.shape
  available=router.available[...,None,:].expand(B,C,N,P,4)
  anchor=self.latent_norm(inputs.base_latents)[...,None,:].expand(B,C,N,P,64)
  phase=torch.linspace(-1,1,P,device=context.device,dtype=context.dtype).expand(B,C,N,P)[...,None]
  numerical=numeric_features(candidates,inputs.query_mean,inputs.query_std,self.coordinates)[...,None]
  query=self.query(torch.cat((self.context_norm(context),anchor,visible,phase,numerical,available.to(context.dtype)),-1))
  # The original point query direct path is unchanged. Only the candidate read is pooled.
  hidden=inputs.patch_validity&~inputs.patch_mask
  pool_mask=torch.where(hidden.any(-1,keepdim=True),hidden,inputs.patch_validity)
  patch_query=torch.where(pool_mask[...,None],query,0).sum(-2)/pool_mask.sum(-1,keepdim=True).clamp_min(1)
  messages=[];masses=[];allowed=[];counts=[]
  for i,name in enumerate(EXPERTS):
   item=evidence[name];cand=item.candidates
   valid=cand.valid&item.available[...,None]&router.available[...,i+1,None]
   raw=torch.where(valid[...,None],cand.messages,0);quality=torch.where(valid[...,None],cand.quality,0)
   encoded=self.message_norm(raw)+self.quality[name](quality)+self.kind.weight[i]
   messages.append(encoded);masses.append(torch.where(valid,cand.weights,0));allowed.append(valid);counts.append(raw.shape[-2])
  encoded=torch.cat(messages,-2);valid=torch.cat(allowed,-1);mass=torch.cat(masses,-1)
  K=encoded.shape[-2];H=self.heads;D=self.width//H
  q=patch_query.reshape(B,C,N,1,H,D).transpose(-3,-2)
  k=self.key(encoded).reshape(B,C,N,K,H,D).transpose(-3,-2)
  v=self.value(encoded).reshape(B,C,N,K,H,D).transpose(-3,-2)
  logits=torch.matmul(q,k.transpose(-1,-2))/sqrt(D)
  logits=logits+mass.clamp_min(1e-8).log()[...,None,None,:]
  logits=logits.masked_fill(~valid[...,None,None,:],-torch.inf)
  null=torch.zeros(B,C,N,H,1,1,device=logits.device,dtype=logits.dtype)
  attention=torch.softmax(torch.cat((null,logits),-1),-1)
  shared=torch.matmul(attention[...,1:],v).transpose(-3,-2).reshape(B,C,N,1,64)
  readout=shared.expand(B,C,N,P,64)
  correction=self.output(torch.cat((query,readout),-1)).squeeze(-1)
  eligible=inputs.patch_validity&~inputs.patch_mask&valid.any(-1)[...,None]
  return torch.where(eligible,correction,0),{'weights':attention,'candidate_counts':tuple(counts),'valid':valid,'shared_readout':True,'query_pool':'valid_missing_mean_else_valid_mean'}

def make_repair(data,variant,seed=2023):
 if variant not in ['E0','E8']:raise ValueError(variant)
 model=make_r3(data,'H_GLOBAL',seed)
 if variant=='E8':model.core.point_candidate_readout.__class__=PatchSharedReadout
 return model

def effective_parameters(full):
 return sum(p.numel() for n,p in full.named_parameters() if not n.startswith(('frozen.repair.ema_encoder.','frozen.repair.core.historical.key_encoder.')))

def install():variants.make_repair=make_repair
