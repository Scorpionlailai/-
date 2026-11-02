"""Preregistered numeric/random selectors. No new learned parameters."""
from dataclasses import replace
from itertools import combinations
from math import sqrt
import torch
from integrations.v125_ablation.model.contracts import CandidateEvidence,ExpertEvidence

NAMES=('historical','variable','temporal')

def empty(inputs,q,slots=1):
 b=inputs.base_latents;shape=(*b.shape[:-1],slots)
 valid=torch.zeros(shape,device=b.device,dtype=torch.bool)
 return ExpertEvidence(torch.zeros_like(b),torch.zeros_like(b),valid.any(-1),{},CandidateEvidence(b.new_zeros(*shape,b.shape[-1]),b.new_zeros(*shape,q),b.new_zeros(shape),valid))

def starts_for(inputs):
 s=inputs.patch_starts
 return s[:,None].expand(*inputs.base_latents.shape[:-1]) if s.ndim==2 else s

def random_priority(target,variable,identity,epoch):
 # Stateless, batch-order-independent and identical across training seeds.
 variable=torch.as_tensor(variable,device=target.device,dtype=torch.long)
 x=(target.long()*1000003+variable.long()*9176+identity.long()*6361+int(epoch)*104729+2023)%2147483647
 for _ in range(3):x=(x*x+48271*x+12345)%2147483647
 return x.to(torch.float64)/2147483647.

def choose(scores,valid,count,capacity):
 k=min(capacity,scores.shape[-1])
 order=torch.argsort(scores.masked_fill(~valid,-torch.inf),dim=-1,descending=True,stable=True)[...,:k]
 selected=valid.gather(-1,order)&(torch.arange(k,device=scores.device)<count[...,None])
 return order,selected

def take(pool,order,valid):
 out={}
 for name,t in pool.items():
  if t is None:continue
  out[name]=t.gather(-2,order[...,None].expand(*order.shape,t.shape[-1])) if t.ndim==order.ndim+1 else t.gather(-1,order)
 out['valid']=valid
 return out

def numeric_score(target,target_mask,source,source_mask,fallback):
 common=target_mask[...,None,:]&source_mask
 t=torch.where(common,target[...,None,:],0);s=torch.where(common,source,0)
 tn=t.square().sum(-1);sn=s.square().sum(-1)
 enough=(common.sum(-1)>=2)&(tn>1e-12)&(sn>1e-12)
 cosine=(t*s).sum(-1)/(tn*sn).clamp_min(1e-24).sqrt()
 return torch.where(enough,cosine.clamp(-1,1).to(fallback.dtype)+2.,fallback-2.),enough

def window_pools(inputs):
 base=inputs.base_latents;B,C,N,D=base.shape;P=inputs.patch_values.shape[-1];dev=base.device
 observed=inputs.patch_mask&inputs.patch_validity
 local=torch.where(observed,inputs.patch_values,0)
 global_values=torch.where(observed,local*inputs.query_std[:,:,None,None]+inputs.query_mean[:,:,None,None],0)
 b=torch.arange(B,device=dev)[:,None,None,None]
 c=torch.arange(C,device=dev)[None,:,None,None]
 target=torch.arange(N,device=dev)[None,None,:,None]
 source=torch.arange(N,device=dev).view(1,1,1,N).expand(B,C,N,N)
 ci=c.expand_as(source);T={}
 T['local']=local[b,ci,source];T['raw']=global_values[b,ci,source];T['mask']=observed[b,ci,source]
 T['source_variable']=ci;T['source_patch']=source
 T['source_start']=starts_for(inputs).gather(-1,source.reshape(B,C,-1)).reshape_as(source)
 T['valid']=T['mask'].any(-1)&inputs.damaged[...,None]&(source!=target)
 offsets=torch.tensor((-1,0,1),device=dev)
 sv=torch.arange(C,device=dev).repeat_interleave(3).view(1,1,1,-1).expand(B,C,N,C*3)
 delta=offsets.repeat(C).view(1,1,1,-1).expand_as(sv)
 raw_ix=target+delta;ix=raw_ix.clamp(0,N-1)
 V={'local':local[b,sv,ix],'raw':global_values[b,sv,ix],'mask':observed[b,sv,ix], 'source_variable':sv,'source_patch':ix,'offset':delta}
 V['source_start']=inputs.patch_starts[b,ix] if inputs.patch_starts.ndim==2 else starts_for(inputs)[b,sv,ix]
 V['valid']=V['mask'].any(-1)&(raw_ix>=0)&(raw_ix<N)&(sv!=c)&inputs.damaged[...,None]
 return T,V,global_values,observed

def temporal_reference_count(inputs):
 B,C,N,D=inputs.base_latents.shape;dev=inputs.base_latents.device
 seen=(inputs.patch_mask&inputs.patch_validity).any(-1)
 pos=torch.arange(N,device=dev);sep=pos[:,None]-pos[None]
 counts=torch.zeros((B,C,N),device=dev,dtype=torch.long)
 for direction in [1,-1]:
  for lower,upper in zip((0,2,4,8,16),(2,4,8,16,N)):
   allowed=(direction*sep>lower)&(direction*sep<=upper)
   counts+=(seen[:,:,None,:]&allowed).any(-1)
 return counts*inputs.damaged

def temporal_selected(self,inputs,pool,source_context_allowed,condition):
 base=inputs.base_latents;observed=inputs.patch_mask&inputs.patch_validity
 safe=torch.where(observed,inputs.patch_values,0)
 enc=self.source_encoder(torch.cat((safe,observed.to(base.dtype)),-1))
 full=enc+self.source_context_projection(self.source_context_norm(base)) if self.contextual else enc
 B,C,N,K=pool['valid'].shape;b=torch.arange(B,device=base.device)[:,None,None,None];c=torch.arange(C,device=base.device)[None,:,None,None]
 source=full[b,c,pool['source_patch']]
 if source_context_allowed is not None:source=torch.where(source_context_allowed[...,None,None],source,enc[b,c,pool['source_patch']])
 query=self.query_projection(base+condition);key=self.key_projection(source)
 similarity=(query[...,None,:]*key).sum(-1)/sqrt(self.d_model)
 mask=pool['mask'];local=pool['local'];valid=pool['valid'];fraction=mask.float().mean(-1)
 common=mask&observed[...,None,:]
 error=(torch.where(common,local,0)-torch.where(common,safe[...,None,:],0)).abs().sum(-1)/common.sum(-1).clamp_min(1)
 consistency=torch.exp(-error)*common.any(-1)
 distance=(pool['source_patch']-torch.arange(N,device=base.device)[None,None,:,None]).abs()
 relative=distance.to(base.dtype)/max(N-1,1);islocal=(distance<=self.local_radius).to(base.dtype)
 features=torch.cat((base[...,None,:].expand_as(source),source,fraction[...,None],relative[...,None],consistency[...,None],islocal[...,None]),-1)
 message=self.message_projector(features)
 logits=similarity+.5*fraction-.05*relative
 q=torch.stack((fraction,relative,consistency,observed.float().mean(-1)[...,None].expand_as(fraction),islocal),-1)
 return evidence(inputs,message,q,logits,valid,pool)

def variable_selected(self,inputs,pool,source_context_allowed,condition):
 base=inputs.base_latents;observed=inputs.patch_mask&inputs.patch_validity
 valid=pool['valid'];source_mask=pool['mask'];visible=pool['local'];sv=pool['source_variable'];ix=pool['source_patch'];lag=pool['offset'].to(base.dtype)
 B,C,N,K=valid.shape;b=torch.arange(B,device=base.device)[:,None,None,None]
 fraction=source_mask.float().mean(-1);common=source_mask&observed[...,None,:]
 error=(torch.where(common,visible,0)-torch.where(common,inputs.patch_values[...,None,:],0)).abs().sum(-1)/common.sum(-1).clamp_min(1)
 consistency=torch.exp(-error)*common.any(-1)
 relation=(self.source_embedding.weight@self.target_embedding.weight.T/sqrt(self.d_model)).T
 target=torch.arange(C,device=base.device)[None,:,None,None].expand_as(sv)
 prior=relation[target,sv]
 logits=prior+.5*fraction+.5*consistency-.1*lag.abs()
 source_state=None
 if self.contextual:
  source_state=self.context_norm(base[b,sv,ix]);target_state=self.context_norm(base)+condition
  match=(self.context_query(target_state)[...,None,:]*self.context_key(source_state)).sum(-1)/sqrt(self.d_model)
  adjustment=.5*torch.tanh(match)
  if source_context_allowed is not None:adjustment=adjustment*source_context_allowed[...,None]
  logits=logits+adjustment
 features=torch.cat((visible,source_mask.to(base.dtype),base[...,None,:].expand(B,C,N,K,self.d_model),lag[...,None],fraction[...,None],consistency[...,None]),-1)
 message=self.message_projector(features)
 if source_state is not None:
  extra=self.source_context_projection(source_state)
  if source_context_allowed is not None:extra=extra*source_context_allowed[...,None,None]
  message=message+extra
 q=torch.stack((prior,fraction,common.float().mean(-1)*consistency,lag),-1)
 return evidence(inputs,message,q,logits,valid,pool)

def evidence(inputs,message,quality,logits,valid,provenance):
 weights=torch.softmax(logits.masked_fill(~valid,-1e9),-1)*valid
 weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-8)
 return ExpertEvidence(torch.zeros_like(inputs.base_latents),torch.zeros_like(inputs.base_latents),valid.any(-1),{k:v.detach() for k,v in provenance.items() if k not in ['raw','local','mask']},CandidateEvidence(message,quality,weights,valid))

def feasible_variable_mask(pool,reference,epoch,inputs):
 B,C,N,K=pool['valid'].shape;dev=reference.device
 available=pool['valid'].reshape(B,C,N,C,3).sum(-1)
 result=torch.zeros_like(pool['valid'])
 for target in range(C):
  others=[i for i in range(C) if i!=target]
  sets=[ids for k in range(1,min(4,C-1)+1) for ids in combinations(others,k)]
  padded=torch.tensor([list(ids)+[0]*(4-len(ids)) for ids in sets],device=dev)
  active=torch.tensor([[True]*len(ids)+[False]*(4-len(ids)) for ids in sets],device=dev)
  counts=available[:,target,:,padded]
  feasible=((counts>0)|~active).all(-1)&((counts*active).sum(-1)>=reference[:,target,:,None])
  desired=(available[:,target]>0).sum(-1).clamp_max(4)
  feasible=feasible&(active.sum(-1)[None,None,:]==desired[...,None])
  score=random_priority(starts_for(inputs)[:,target,:,None],target,torch.arange(len(sets),device=dev),epoch)
  selected=score.masked_fill(~feasible,-torch.inf).argmax(-1)
  chosen=padded[selected];use=active[selected]
  keep=((pool['source_variable'][:,target,:,:,None]==chosen[:,:,None,:])&use[:,:,None,:]).any(-1)
  result[:,target]=keep&pool['valid'][:,target]
  if ((reference[:,target]>0)&~feasible.any(-1)).any():raise RuntimeError('No variable subset can match reference count')
 return result

@torch.no_grad()
def history_pool(self,inputs,count,epoch,random=False,capacity=26):
 base=inputs.base_latents;B,C,N,D=base.shape;P=inputs.patch_values.shape[-1];dev=base.device
 shape=(B,C,N,capacity)
 pool={'raw':base.new_zeros(*shape,P),'local':base.new_zeros(*shape,P),'mask':torch.ones((*shape,P),device=dev,dtype=torch.bool),'valid':torch.zeros(shape,device=dev,dtype=torch.bool),'source_start':torch.zeros(shape,device=dev,dtype=torch.long),'group_id':torch.full(shape,-1,device=dev,dtype=torch.long),'group_support':torch.zeros(shape,device=dev,dtype=torch.long),'key_similarity':base.new_zeros(shape),'rank':torch.full(shape,-torch.inf,device=dev,dtype=torch.float64)}
 if self.index is None or not self.index.member_raw.numel():return pool
 index=self.index;observed=inputs.patch_mask&inputs.patch_validity
 global_values=torch.where(observed,inputs.patch_values*inputs.query_std[:,:,None,None]+inputs.query_mean[:,:,None,None],0)
 query_keys=self.retrieval.encode_frozen(global_values.reshape(-1,P),observed.reshape(-1,P)).reshape(B,C,N,-1)
 query_starts=starts_for(inputs)
 for c in range(C):
  members=torch.where(index.group_variables[index.member_groups]==c)[0]
  if not len(members):continue
  Q=B*N;q=global_values[:,c].reshape(Q,P);mask=observed[:,c].reshape(Q,P);starts=query_starts[:,c].reshape(Q)
  legal_target=inputs.damaged[:,c].reshape(Q)&mask.any(-1)
  best_score=torch.full((Q,capacity),-torch.inf,device=dev,dtype=torch.float64);best_ids=torch.zeros((Q,capacity),device=dev,dtype=torch.long)
  for first in range(0,len(members),256):
   ids=members[first:first+256];raw=index.member_raw[ids].to(base.dtype);ss=index.member_starts[ids]
   legal=(ss[None]+P<=starts[:,None])&legal_target[:,None]
   fallback=random_priority(starts[:,None],c,ss[None],epoch)
   if random:score=fallback
   else:
    tn=q.square().sum(-1,keepdim=True);sn=mask.to(base.dtype)@raw.square().T
    enough=(mask.sum(-1,keepdim=True)>=2)&(tn>1e-12)&(sn>1e-12)
    cosine=(q@raw.T)/(tn*sn).clamp_min(1e-24).sqrt()
    score=torch.where(enough,cosine.clamp(-1,1).to(torch.float64)+2.,fallback-2.)
   score=score.masked_fill(~legal,-torch.inf)
   scores=torch.cat((best_score,score),-1);identities=torch.cat((best_ids,ids[None].expand(Q,-1)),-1)
   order=torch.argsort(scores,dim=-1,descending=True,stable=True)[...,:capacity]
   best_score=scores.gather(-1,order);best_ids=identities.gather(-1,order)
  valid=torch.isfinite(best_score)&(torch.arange(capacity,device=dev)<count[:,c].reshape(Q,1))
  raw=index.member_raw[best_ids].to(base.dtype);memberkeys=index.member_keys[best_ids].to(base.dtype)
  keys=query_keys[:,c].reshape(Q,-1);keycos=(keys[:,None,:]*memberkeys).sum(-1)
  def store(name,value):pool[name][:,c]=value.reshape(B,N,*value.shape[1:])
  store('raw',raw);store('local',(raw-inputs.query_mean[:,c,None,None,None].expand(B,N,capacity,P).reshape(Q,capacity,P))/inputs.query_std[:,c,None,None,None].expand(B,N,capacity,P).reshape(Q,capacity,P))
  store('valid',valid);store('source_start',index.member_starts[best_ids]);store('group_id',index.group_ids[index.member_groups[best_ids]])
  store('group_support',index.group_support[index.member_groups[best_ids]]);store('key_similarity',keycos);store('rank',best_score)
 return pool

def historical_selected(self,inputs,pool,condition):
 base=inputs.base_latents;valid=pool['valid'];local=pool['local'];mask=inputs.patch_mask&inputs.patch_validity
 error=(torch.where(mask[...,None,:],local,0)-torch.where(mask,inputs.patch_values,0)[...,None,:]).abs().sum(-1)/mask.sum(-1,keepdim=True).clamp_min(1)
 sim=pool['key_similarity'];scores=sim.masked_fill(~valid,-1e9)
 top=torch.topk(scores,min(2,scores.shape[-1]),-1).values
 margin=(top[...,0]-top[...,1]).clamp_min(0) if top.shape[-1]>1 else torch.zeros_like(top[...,0])
 margin=torch.where(valid.sum(-1)>=2,margin,0)
 quality=torch.stack((sim,-error,margin[...,None].expand_as(sim),mask.float().mean(-1)[...,None].expand_as(sim),torch.log1p(pool['group_support'].to(base.dtype))),-1)
 quality=torch.where(valid[...,None],quality,0)
 logits=scores-error
 if self.contextual:
  query=self.context_query(self.context_norm(base)+condition);key=self.context_key(local)
  logits=logits+.5*torch.tanh((query[...,None,:]*key).sum(-1)/sqrt(self.d_model))
 features=torch.cat((local,base[...,None,:].expand(*local.shape[:-1],self.d_model),quality),-1)
 return evidence(inputs,self.message_projector(features),quality,logits,valid,pool)


def controlled_evidence(core,inputs,variant,reference_h,reference_v,source_allowed,epoch,specialist_exposure=0.,specialist_seed=0):
 T,V,target,observed=window_pools(inputs)
 counts={'temporal':temporal_reference_count(inputs),'variable':reference_v.valid.sum((-1,-2)),'historical':reference_h.valid.sum(-1)}
 start=starts_for(inputs)[...,None];variable=torch.arange(core.variable_count,device=start.device)[None,:,None,None]
 tp=random_priority(start,variable,T['source_start']+101,epoch)
 vp=random_priority(start,variable,V['source_variable']*100000+V['source_start']+503,epoch)
 if variant=='E4':
  order,valid=choose(tp,T['valid'],counts['temporal'],10);T=take(T,order,valid)
  allowed=feasible_variable_mask(V,counts['variable'],epoch,inputs)
  order,valid=choose(vp,allowed,counts['variable'],12);V=take(V,order,valid)
  H=history_pool(core.historical,inputs,counts['historical'],epoch,random=True,capacity=4)
 else:
  total=sum(counts.values())
  H=history_pool(core.historical,inputs,total,epoch,random=False,capacity=26)
  ts,_=numeric_score(target,observed,T['raw'],T['mask'],tp)
  vs,_=numeric_score(target,observed,V['raw'],V['mask'],vp)
  scores=torch.cat((ts,vs,H['rank']),-1);valid=torch.cat((T['valid'],V['valid'],H['valid']),-1)
  order,chosen=choose(scores,valid,total,26)
  keep=torch.zeros_like(valid).scatter(-1,order,chosen)
  first=T['valid'].shape[-1];second=first+V['valid'].shape[-1]
  selected=[]
  for pool,mask in [(T,keep[...,:first]),(V,keep[...,first:second]),(H,keep[...,second:])]:
   # Compact each source to the maximum selected count of that source in this batch.
   n=mask.sum(-1);cap=max(int(n.max()),1)
   order,yes=choose(mask.to(torch.float32),mask,n,cap);selected.append(take(pool,order,yes))
  T,V,H=selected
 exposure_allowed=exposure_selected=None
 if variant=='E5':
  from integrations.v125_ablation.model.exposure import sample_specialist_exposure
  available=torch.stack((H['valid'].any(-1),V['valid'].any(-1),T['valid'].any(-1)),-1)
  exposure_allowed,exposure_selected=sample_specialist_exposure(inputs.damaged,available,fraction=specialist_exposure,seed=specialist_seed) if specialist_exposure else (available,None)
  source_allowed=None if exposure_selected is None else ~exposure_selected
 out={'temporal':temporal_selected(core.temporal,inputs,T,source_allowed,core.candidate_condition),'variable':variable_selected(core.variable,inputs,V,source_allowed,core.candidate_condition),'historical':historical_selected(core.historical,inputs,H,core.candidate_condition)}
 actual={k:e.candidates.valid.sum(-1) for k,e in out.items()}
 if variant=='E4':
  for k in counts:assert torch.equal(actual[k],counts[k]),k
 else:assert torch.equal(sum(actual.values()),sum(counts.values()))
 trace={'reference_counts':{k:v.detach() for k,v in counts.items()},'actual_counts':{k:v.detach() for k,v in actual.items()},'selected':{k:{n:v.detach() for n,v in p.items() if n in ['source_variable','source_patch','source_start','valid','group_id']} for k,p in [('temporal',T),('variable',V),('historical',H)]}}
 return out,trace,exposure_allowed,exposure_selected
