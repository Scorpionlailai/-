"""Source eligibility and content adapters; no final candidate normalization."""
from dataclasses import dataclass
from math import sqrt
from typing import Mapping
import torch
from torch import nn
import bootstrap
from integrations.v125_ablation.model.exposure import sample_specialist_exposure

@dataclass
class CandidateSet:
    content: torch.Tensor
    quality: torch.Tensor
    valid: torch.Tensor
    provenance: Mapping

class SourceAdapters(nn.Module):
    def __init__(self):
        super().__init__()
        self.variable=nn.Sequential(nn.Linear(96,64),nn.GELU(),nn.Linear(64,64))
        self.historical=nn.Sequential(nn.Linear(16,64),nn.GELU(),nn.Linear(64,64))

    def forward(self, inputs, temporal, variable, historical, *,epoch,explore,exposure,seed,disabled):
        hc=historical.prepare_candidates(inputs)
        vc=variable.prepare_sources(inputs,epoch=epoch,explore=explore)
        tc=temporal.prepare_sources(inputs)
        visible=tc.observed.any(-1)
        tav=(visible.sum(-1,keepdim=True)-visible.long()).gt(0)&inputs.damaged
        availability=torch.stack((hc.available,vc.available,tav),-1)
        for i,name in enumerate(('historical','variable','temporal')):
            if name in disabled:availability[...,i]=False
        allowed,selected=(sample_specialist_exposure(inputs.damaged,availability,fraction=exposure,seed=seed)
                          if exposure else (availability,None))
        source_allowed=None if selected is None else ~selected
        t=self.temporal_content(inputs,temporal,tc,source_allowed)
        t.valid=t.valid&allowed[...,2,None]
        if vc.empty_evidence is not None:
            v=CandidateSet(vc.empty_evidence.candidates.messages,vc.empty_evidence.candidates.quality,
                vc.empty_evidence.candidates.valid,vc.provenance)
        else:
            B,C,N,D=inputs.base_latents.shape
            bi=torch.arange(B,device=inputs.base_latents.device)[:,None,None,None,None]
            identities=variable.source_embedding(vc.source_ix)+variable.target_embedding.weight[None,:,None,None,None,:]
            h=self.variable(torch.cat((vc.visible_values,vc.source_observed.to(identities.dtype),identities),-1))
            source_state=variable.context_norm(inputs.base_latents[bi,vc.source_ix,vc.patch_ix])
            update=variable.source_context_projection(source_state)
            if source_allowed is not None:update=update*source_allowed[...,None,None,None]
            h=h+update
            valid=vc.valid&allowed[...,1,None,None]
            v=CandidateSet(h.flatten(-3,-2),vc.raw_quality.flatten(-3,-2),valid.flatten(-2),vc.provenance)
        hv=torch.where(hc.valid[...,None],hc.normalized_raw,0)
        h=CandidateSet(self.historical(hv),hc.raw_quality,hc.valid&allowed[...,0,None],hc.provenance)
        return {'historical':h,'variable':v,'temporal':t},selected

    @staticmethod
    def temporal_content(inputs,module,cache,source_allowed):
        base=inputs.base_latents;B,C,N,D=base.shape
        source=cache.source_encoded+module.source_context_projection(module.source_context_norm(base))
        query=module.query_projection(base)
        keys=module.key_projection(source)
        similarity=torch.einsum('bcnd,bcmd->bcnm',query,keys)/sqrt(D)
        if source_allowed is not None:
            original=torch.einsum('bcnd,bcmd->bcnm',query,module.key_projection(cache.source_encoded))/sqrt(D)
            similarity=torch.where(source_allowed[...,None],similarity,original)
        pos=torch.arange(N,device=base.device)
        separation=pos[:,None]-pos[None,:]
        signed=torch.tensor((1,-1),device=base.device)[:,None,None]*separation
        lower=torch.tensor((0,*module.boundaries),device=base.device)
        upper=torch.tensor((*module.boundaries,N),device=base.device)
        band=((signed[:,None]>lower[None,:,None,None])&(signed[:,None]<=upper[None,:,None,None])).reshape(-1,N,N)
        local=torch.tensor((True,*(False for _ in module.boundaries))*2,device=base.device)
        eligible=cache.observed.any(-1)[:,:,None,:]&inputs.damaged[...,None]
        valid_grid=eligible[:,:,None]&band[None,None]
        distance_rank=-signed[:,None].expand(-1,len(lower),-1,-1).reshape(-1,N,N).to(base.dtype)
        rank=torch.where(local[None,None,:,None,None],distance_rank[None,None],similarity[:,:,None])
        rank=rank.masked_fill(~valid_grid,-torch.inf)
        ix=rank.argmax(-1).permute(0,1,3,2)
        valid=valid_grid.any(-1).permute(0,1,3,2)
        bi=torch.arange(B,device=base.device)[:,None,None,None]
        ci=torch.arange(C,device=base.device)[None,:,None,None]
        h=source[bi,ci,ix]
        if source_allowed is not None:h=torch.where(source_allowed[...,None,None],h,cache.source_encoded[bi,ci,ix])
        values=cache.visible[bi,ci,ix];mask=cache.observed[bi,ci,ix]
        common=mask&cache.observed[...,None,:]
        error=((values-cache.visible[...,None,:]).abs()*common).sum(-1)/common.sum(-1).clamp_min(1)
        consistency=torch.exp(-error)*common.any(-1)
        fraction=mask.float().mean(-1)
        target_fraction=cache.observed.float().mean(-1)[...,None].expand_as(fraction)
        relative=(ix-pos[None,None,:,None]).abs().to(base.dtype)/max(N-1,1)
        islocal=local.to(base.dtype).view(1,1,1,-1).expand_as(fraction)
        quality=torch.stack((fraction,relative,consistency,target_fraction,islocal),-1)
        return CandidateSet(h,quality,valid,{'source_indices':ix.detach(),'candidate_valid':valid.detach()})
