"""Position-dependent positive overlap weights within the recovery module."""
import torch
from torch import nn


class OverlapFusion(nn.Module):
    def __init__(self,d_model,patch_length):
        super().__init__()
        self.norm=nn.LayerNorm(d_model)
        self.score=nn.Sequential(nn.Linear(d_model+3*patch_length+5,32),nn.GELU(),
                                nn.Linear(32,patch_length))
        nn.init.zeros_(self.score[-1].weight);nn.init.zeros_(self.score[-1].bias)

    def forward(self,inputs,routed):
        original=inputs.patch_mask.to(torch.bool)&inputs.patch_validity.to(torch.bool)
        features=torch.cat((self.norm(routed.latents),inputs.patch_values.masked_fill(~original,0),
            original.to(routed.latents.dtype),inputs.patch_validity.to(routed.latents.dtype),
            routed.weights,inputs.rho[...,None]),-1)
        return torch.exp(2*torch.tanh(self.score(features))).masked_fill(~inputs.patch_validity,0)


def normalized_overlap_weights(weights,validity,*,window_len,stride):
    """Normalize each coordinate across its overlapping patch copies."""
    b,c,n,p=weights.shape;denominator=weights.new_zeros(b,c,window_len)
    for i in range(n):
        start=i*stride;length=min(p,window_len-start)
        if length<=0:break
        denominator[:,:,start:start+length]+=weights[:,:,i,:length]*validity[:,:,i,:length]
    normalized=torch.zeros_like(weights)
    for i in range(n):
        start=i*stride;length=min(p,window_len-start)
        if length<=0:break
        normalized[:,:,i,:length]=weights[:,:,i,:length]/denominator[:,:,start:start+length].clamp_min(1e-8)
    return normalized.masked_fill(~validity,0)
