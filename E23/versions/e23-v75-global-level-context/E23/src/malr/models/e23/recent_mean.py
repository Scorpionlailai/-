"""Repair the recent aggregate level after pointwise expert reconstruction."""
from dataclasses import replace
import torch
from torch import nn
from malr.data.patching import patchify_preserving_values
from .coordinates import repair_scale

class RecentMeanProjector(nn.Module):
    def __init__(self,variables,width=64):
        super().__init__()
        if width!=64 or variables!=7:raise ValueError("recent mean requires original ETTh1 width64")
        self.width,self.variables=width,variables
        self.network=nn.Sequential(nn.Linear(3*width+6+variables,width),nn.GELU(),nn.Linear(width,1))
        nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)

    def forward(self,explicit,router,points,mean,std):
        sequence=explicit.normalized_sequence;B,T,C=sequence.shape
        if T!=96 or C!=self.variables or points.shape!=(B*C,self.width,T):raise ValueError("recent mean geometry mismatch")
        valid=explicit.sequence_validity;observed=explicit.sequence_observed & valid
        r=repair_scale(std);ratio=(std/r)[:,None]
        z=torch.where(valid,sequence*ratio,0)
        visible=torch.where(observed,explicit.observed_sequence*ratio,0)
        denom=valid[:,-32:].sum(1).clamp_min(1)
        coarse_mean=z[:,-32:].sum(1)/denom
        visible_mean=visible[:,-32:].sum(1)/observed[:,-32:].sum(1).clamp_min(1)
        fraction=observed[:,-32:].sum(1)/denom
        all_fraction=observed.sum(1)/valid.sum(1).clamp_min(1)
        phase=torch.linspace(-1,1,32,device=z.device,dtype=z.dtype)
        slope=(z[:,-32:]*phase[None,:,None]).sum(1)/phase.square().sum()
        index=torch.arange(T,device=z.device)[None,:,None]
        last=torch.where(observed,index,-1).amax(1)
        last_value=visible.gather(1,last.clamp_min(0)[:,None]).squeeze(1).masked_fill(last.lt(0),0)
        numeric=torch.stack((coarse_mean,visible_mean,fraction,all_fraction,slope,last_value),-1)
        x=points.reshape(B,C,self.width,T);v=valid.transpose(1,2)[:,:,None]
        recent=(x[:,:,:,-32:]*v[:,:,:,-32:]).sum(-1)/v[:,:,:,-32:].sum(-1).clamp_min(1)
        whole=(x*v).sum(-1)/v.sum(-1).clamp_min(1)
        weights=explicit.validity_patches.sum(-1).to(x.dtype)
        latent=(router.latents*weights[...,None]).sum(2)/weights.sum(2).clamp_min(1)[...,None]
        identity=torch.eye(C,device=x.device,dtype=x.dtype)[None].expand(B,-1,-1)
        feature=torch.cat((recent,whole,latent,numeric,identity),-1)
        delta=2*torch.tanh(self.network(feature).squeeze(-1)/2)
        scope=torch.arange(T,device=z.device)[None,:,None].ge(T-32)
        hidden=valid & ~observed & scope
        count=hidden.sum(1)
        share=(32/count.clamp_min(1)).clamp_max(4)
        global_delta=delta*r
        normalized=sequence+torch.where(hidden,(global_delta*share/std)[:,None],0)
        patches=patchify_preserving_values(normalized,explicit.sequence_observed,16,8,validity=valid)
        result=replace(explicit,normalized_sequence=normalized,standardized_sequence=normalized*std[:,None]+mean[:,None],repaired_patches=patches.values.masked_fill(~explicit.validity_patches,0))
        return result,torch.where(count.gt(0),global_delta,0)
