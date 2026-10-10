"""Select independent groups, concatenate their deltas, and return the shared width."""
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class GroupFusionOutput:
    latents: Tensor
    group_weights: Tensor
    projected_deltas: Tensor


class StackedGroupFusion(nn.Module):
    def __init__(self,d_model:int,group_count:int):
        super().__init__()
        if min(d_model,group_count)<1: raise ValueError('fusion geometry must be positive')
        self.d_model=d_model; self.group_count=group_count
        if group_count>1:
            self.base_norm=nn.LayerNorm(d_model)
            self.delta_norm=nn.LayerNorm(d_model)
            self.group_score=nn.Sequential(nn.Linear(2*d_model+6,d_model),nn.GELU(),nn.Linear(d_model,1))
            self.group_bias=nn.Parameter(torch.zeros(group_count))
            self.none_score=nn.Linear(d_model+1,1)
            self.projection=nn.Linear(group_count*d_model,d_model,bias=False)
            nn.init.zeros_(self.group_score[-1].weight); nn.init.zeros_(self.group_score[-1].bias)
            nn.init.zeros_(self.none_score.weight); nn.init.zeros_(self.none_score.bias)
            with torch.no_grad(): self.projection.weight.copy_(torch.eye(d_model).repeat(1,group_count))

    def forward(self,base:Tensor,deltas:Tensor,routes:Tensor,rho:Tensor,available:Tensor)->GroupFusionOutput:
        shape=base.shape[:-1]; g=self.group_count; d=self.d_model
        if base.ndim!=4 or base.shape[-1]!=d or deltas.shape!=(*shape,g,d) or routes.shape!=(*shape,g,4) or rho.shape!=shape or available.shape!=(*shape,g):
            raise ValueError('fusion input shape mismatch')
        if not all(torch.isfinite(t).all() for t in (base,deltas,routes,rho)):
            raise ValueError('fusion input must be finite')
        eligible=available.bool() & (rho<1)[...,None]
        if g==1:
            weights=torch.stack((~eligible[...,0],eligible[...,0]),-1).to(base.dtype)
            contributions=deltas.masked_fill(~eligible[...,None],0)
            return GroupFusionOutput(base+contributions.squeeze(-2),weights,contributions)
        normalized=self.base_norm(base)
        rms=deltas.square().mean(-1).add(1e-12).sqrt()
        features=torch.cat((normalized[...,None,:].expand_as(deltas),self.delta_norm(deltas),routes,
                            rho[...,None,None].expand(*shape,g,1),torch.log1p(rms)[...,None]),-1)
        scores=self.group_score(features).squeeze(-1)+self.group_bias
        none=self.none_score(torch.cat((normalized,rho[...,None]),-1))
        weights=torch.softmax(torch.cat((none,scores.masked_fill(~eligible,-torch.inf)),-1),-1)
        weighted=weights[...,1:,None]*deltas
        # Keep each projected contribution for diagnostics; no bias may write at no evidence.
        blocks=self.projection.weight.reshape(d,g,d).permute(1,0,2)
        contributions=torch.einsum('...gi,goi->...go',weighted,blocks)
        latents=base+contributions.sum(-2)
        return GroupFusionOutput(latents,weights,contributions)
