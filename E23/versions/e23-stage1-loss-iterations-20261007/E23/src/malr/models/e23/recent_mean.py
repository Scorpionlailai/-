"""Repair the recent aggregate level after pointwise expert reconstruction."""
from dataclasses import replace
import torch
from torch import nn
from malr.data.patching import patchify_preserving_values
from .coordinates import repair_scale

class RecentMeanProjector(nn.Module):
    def __init__(self,variables,width=64,*,mode="mean_v1"):
        super().__init__()
        if width!=64 or variables!=7:raise ValueError("recent mean requires original ETTh1 width64")
        if mode not in ("mean_v1", "global_level_mean_v2", "global_recent_bands_v3"):raise ValueError("unsupported recent mean mode")
        self.width,self.variables=width,variables
        self.network=nn.Sequential(nn.Linear(3*width+6+variables,width),nn.GELU(),nn.Linear(width,1))
        nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)
        self.global_level_network=None
        self.recent_band_network=None
        if mode in ("global_level_mean_v2", "global_recent_bands_v3"):
            with torch.random.fork_rng(devices=[]):
                self.global_level_network=nn.Sequential(nn.Linear(7*variables,width),nn.GELU(),nn.Linear(width,1))
                nn.init.zeros_(self.global_level_network[-1].weight)
                nn.init.zeros_(self.global_level_network[-1].bias)
        if mode == "global_recent_bands_v3":
            with torch.random.fork_rng(devices=[]):
                self.recent_band_network=nn.Sequential(nn.Linear(94,width),nn.GELU(),nn.Linear(width,2))
                nn.init.zeros_(self.recent_band_network[-1].weight)
                nn.init.zeros_(self.recent_band_network[-1].bias)

    def _recent_band_features(self,explicit,mean,std,global_context):
        observed=explicit.sequence_observed & explicit.sequence_validity
        safe=torch.where(observed,explicit.observed_sequence,0)
        values=torch.where(observed,safe*std[:,None]+mean[:,None],0)
        summaries=[]
        coarse=[]
        # Extra source summaries distinguish last8, last16, and preceding16.
        for start,stop in ((88,96),(80,96),(64,80)):
            count=observed[:,start:stop].sum(1)
            average=values[:,start:stop].sum(1)/count.clamp_min(1)
            fraction=count/explicit.sequence_validity[:,start:stop].sum(1).clamp_min(1)
            summaries.extend((average,fraction))
            valid=explicit.sequence_validity[:,start:stop]
            z=torch.where(valid,explicit.normalized_sequence[:,start:stop],0)
            current=torch.where(valid,z*std[:,None]+mean[:,None],0)
            coarse.append(current.sum(1)/valid.sum(1).clamp_min(1))
        # 49 global summaries/target identity + 42 visible summaries + 3 target coarse levels.
        B,C=mean.shape
        context=torch.cat(summaries,-1)[:,None].expand(B,C,42)
        return torch.cat((global_context,context,torch.stack(coarse,-1)),-1)

    def _global_level_features(self,explicit,mean,std):
        valid=explicit.sequence_validity
        observed=explicit.sequence_observed & valid
        safe=torch.where(observed,explicit.observed_sequence,0)
        global_values=torch.where(observed,safe*std[:,None]+mean[:,None],0)
        count=observed.sum(1);recent_count=observed[:,-32:].sum(1)
        whole_mean=global_values.sum(1)/count.clamp_min(1)
        recent_mean=global_values[:,-32:].sum(1)/recent_count.clamp_min(1)
        index=torch.arange(global_values.shape[1],device=global_values.device)[None,:,None]
        last=torch.where(observed,index,-1).amax(1)
        last_value=global_values.gather(1,last.clamp_min(0)[:,None]).squeeze(1).masked_fill(last.lt(0),0)
        whole_fraction=count/valid.sum(1).clamp_min(1)
        recent_fraction=recent_count/valid[:,-32:].sum(1).clamp_min(1)
        visible_std=torch.where(count.gt(0),std,0)
        summary=torch.stack((whole_mean,recent_mean,last_value,whole_fraction,recent_fraction,visible_std),-1)
        summary=torch.where(count.gt(0)[...,None],summary,0)
        B,C=mean.shape
        context=summary.reshape(B,6*C)[:,None].expand(B,C,6*C)
        identity=torch.eye(C,device=context.device,dtype=context.dtype)[None].expand(B,-1,-1)
        return torch.cat((context,identity),-1),count.sum(-1).gt(0)


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
        if self.global_level_network is not None:
            feature,has_source=self._global_level_features(explicit,mean,std)
            extra=2*torch.tanh(self.global_level_network(feature).squeeze(-1)/2)
            global_delta=global_delta+torch.where(has_source[:,None],extra,0)

        normalized=sequence+torch.where(hidden,(global_delta*share/std)[:,None],0)
        if self.recent_band_network is not None:
            features=self._recent_band_features(explicit,mean,std,feature)
            band_delta=2*torch.tanh(self.recent_band_network(features)/2)
            band_delta=torch.where(has_source[:,None,None],band_delta,0)
            extra=torch.zeros_like(sequence)
            for i,(start,stop) in enumerate(((64,80),(80,96))):
                scope=hidden[:,start:stop]
                n=scope.sum(1)
                band_share=(16/n.clamp_min(1)).clamp_max(4)
                addition=(band_delta[...,i]*band_share/std)[:,None]
                extra[:,start:stop]=torch.where(scope,addition,0)
            normalized=normalized+extra
        patches=patchify_preserving_values(normalized,explicit.sequence_observed,16,8,validity=valid)
        result=replace(explicit,normalized_sequence=normalized,standardized_sequence=normalized*std[:,None]+mean[:,None],repaired_patches=patches.values.masked_fill(~explicit.validity_patches,0))
        return result,torch.where(count.gt(0),global_delta,0)
