"""Point interface changes; source encoding remains the sealed A definition."""
from dataclasses import replace
from math import sqrt
import torch
from torch import nn
import bootstrap
from integrations.v125_ablation.model.contracts import CandidateEvidence
from integrations.v125_ablation.model.point_candidates import PointCandidateReadout
from integrations.v125_ablation.model.point_evidence import OverlapConfidence,point_patches
from integrations.v125_ablation.model.router import EXPERTS
from integrations.v125_ablation.model.coordinates import numeric_features

def summarize_candidates(cand,valid):
    weights=torch.where(valid,cand.weights,0)
    present=valid.any(-1)
    if not torch.allclose(weights.sum(-1)[present],torch.ones_like(weights.sum(-1)[present]),atol=2e-6,rtol=2e-6):
        raise ValueError('source weights must already be normalized over valid candidates')
    raw=torch.where(valid[...,None],cand.messages,0)
    quality=torch.where(valid[...,None],cand.quality,0)
    return CandidateEvidence((weights[...,None]*raw).sum(-2,keepdim=True),
        (weights[...,None]*quality).sum(-2,keepdim=True),present.to(weights.dtype)[...,None],present[...,None])

class SummaryReadout(PointCandidateReadout):
    def forward(self,inputs,router,evidence,decoder,points,visible_features,candidates):
        compact={}
        for i,name in enumerate(EXPERTS):
            item=evidence[name];cand=item.candidates
            valid=cand.valid&item.available[...,None]&router.available[...,i+1,None]
            compact[name]=replace(item,candidates=summarize_candidates(cand,valid))
        return super().forward(inputs,router,compact,decoder,points,visible_features,candidates)

class DirectReadout(PointCandidateReadout):
    def __init__(self,template):
        # Reuse shared initialized operators; delete four query prior columns.
        nn.Module.__init__(self)
        for name in ('coordinates','width','heads','context_norm','latent_norm','message_norm','quality','kind','key','value','output','variable_readout'):
            setattr(self,name,getattr(template,name))
        with torch.random.fork_rng(devices=[]):first=nn.Linear(139,64)
        ix=list(range(135))+list(range(139,143))
        with torch.no_grad():first.weight.copy_(template.query[0].weight[:,ix]);first.bias.copy_(template.query[0].bias)
        self.query=nn.Sequential(first,template.query[1],template.query[2])

    def forward(self,inputs,router,evidence,decoder,points,visible_features,candidates):
        context=point_patches(points,inputs.patch_validity);visible=point_patches(visible_features,inputs.patch_validity)
        B,C,N,P,_=context.shape
        available=router.available[...,None,:].expand(B,C,N,P,4)
        anchor=self.latent_norm(inputs.base_latents)[...,None,:].expand(B,C,N,P,64)
        phase=torch.linspace(-1,1,P,device=context.device,dtype=context.dtype).expand(B,C,N,P)[...,None]
        numerical=numeric_features(candidates,inputs.query_mean,inputs.query_std,self.coordinates)[...,None]
        query=self.query(torch.cat((self.context_norm(context),anchor,visible,phase,numerical,available.to(context.dtype)),-1))
        messages=[];masses=[];allowed=[];counts=[]
        for i,name in enumerate(EXPERTS):
            item=evidence[name];cand=item.candidates
            valid=cand.valid&item.available[...,None]&router.available[...,i+1,None]
            raw=torch.where(valid[...,None],cand.messages,0);quality=torch.where(valid[...,None],cand.quality,0)
            encoded=self.message_norm(raw)+self.quality[name](quality)+self.kind.weight[i]
            messages.append(encoded);masses.append(torch.where(valid,cand.weights,0));allowed.append(valid);counts.append(raw.shape[-2])
        encoded=torch.cat(messages,-2);valid=torch.cat(allowed,-1);mass=torch.cat(masses,-1)
        K=encoded.shape[-2];H=self.heads;D=self.width//H
        q=query.reshape(B,C,N,P,H,D).transpose(-3,-2)
        k=self.key(encoded).reshape(B,C,N,K,H,D).transpose(-3,-2)
        v=self.value(encoded).reshape(B,C,N,K,H,D).transpose(-3,-2)
        logits=torch.matmul(q,k.transpose(-1,-2))/sqrt(D)
        logits=logits+mass.clamp_min(1e-8).log()[...,None,None,:]
        logits=logits.masked_fill(~valid[...,None,None,:],-torch.inf)
        null=torch.zeros(B,C,N,H,P,1,device=logits.device,dtype=logits.dtype)
        attention=torch.softmax(torch.cat((null,logits),-1),-1)
        readout=torch.matmul(attention[...,1:],v).transpose(-3,-2).reshape(B,C,N,P,64)
        correction=self.output(torch.cat((query,readout),-1)).squeeze(-1)
        eligible=inputs.patch_validity&~inputs.patch_mask&valid.any(-1)[...,None]
        return torch.where(eligible,correction,0),{'weights':attention,'candidate_counts':tuple(counts),'valid':valid}

class DirectConfidence(OverlapConfidence):
    """Same 84→32→1 network, with deterministic replacement of removed inputs."""
    def forward(self,inputs,router,decoder,points,visible_features,candidates):
        context=point_patches(points,inputs.patch_validity);visible=point_patches(visible_features,inputs.patch_validity)
        B,C,N,P,_=context.shape
        base=numeric_features(decoder(inputs.base_latents),inputs.query_mean,inputs.query_std,self.coordinates)
        available=router.available[...,None,:].expand(B,C,N,P,4)
        # No source deltas or alternate predictions are computed. Every available
        # slot carries the same base value; unavailable slots are exactly zero.
        alternatives=base[...,None].expand(B,C,N,P,4).masked_fill(~available,0)
        weights=available.to(context.dtype)/available.sum(-1,keepdim=True)
        phase=torch.linspace(-1,1,P,device=context.device,dtype=context.dtype).expand(B,C,N,P)[...,None]
        numerical=numeric_features(candidates,inputs.query_mean,inputs.query_std,self.coordinates)[...,None]
        disagreement=torch.full_like(numerical,.001) # Original sqrt(clamp(variance,1e-6)) at zero disagreement.
        features=torch.cat((self.context_norm(context),visible,phase,alternatives,weights,available.to(context.dtype),numerical,disagreement),-1)
        eligible=inputs.patch_validity&~inputs.patch_mask
        logits=(2*torch.tanh(self.score(features).squeeze(-1)/2)).masked_fill(~eligible,0)
        return logits.exp().masked_fill(~inputs.patch_validity,0)
