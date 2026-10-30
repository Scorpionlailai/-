"""Combine calibration paths after one shared coarse repair."""
from dataclasses import replace
from math import isfinite

import torch
from torch import nn

from malr.data.patching import patchify_preserving_values
from .recent_mean import RecentMeanProjector


class RecentMeanEnsembleProjector(nn.Module):
    def __init__(self,variables,*,widths,weights):
        super().__init__()
        if len(widths)!=len(weights) or len(widths)<2:
            raise ValueError('calibration widths and weights require at least two members')
        if any(type(w) is not int or w not in (64,128,256) for w in widths):
            raise ValueError('unsupported calibration member width')
        if any(type(w) not in (int,float) or not isfinite(w) or w<=0 for w in weights) or abs(sum(weights)-1)>1e-8:
            raise ValueError('calibration weights must be positive finite and sum to one')
        with torch.random.fork_rng(devices=[]):
            self.members=nn.ModuleList([RecentMeanProjector(variables,mode='global_dense_context_gate_v6',global_dense_width=w) for w in widths])
        self.register_buffer('weights',torch.tensor(weights,dtype=torch.float64))

    def forward(self,explicit,router,points,mean,std):
        outputs=[member(explicit,router,points,mean,std) for member in self.members]
        weights=self.weights.to(dtype=explicit.normalized_sequence.dtype)
        normalized=sum(w*result.normalized_sequence for w,(result,_) in zip(weights,outputs))
        valid=explicit.sequence_validity;observed=explicit.sequence_observed & valid
        active=valid & ~observed & observed.flatten(1).any(-1)[:,None,None]
        normalized=torch.where(active,normalized,explicit.normalized_sequence)
        normalized=torch.where(observed,explicit.observed_sequence,normalized)
        patches=patchify_preserving_values(normalized,explicit.sequence_observed,16,8,validity=valid)
        result=replace(explicit,normalized_sequence=normalized,
            standardized_sequence=normalized*std[:,None]+mean[:,None],
            repaired_patches=patches.values.masked_fill(~explicit.validity_patches,0))
        delta=sum(w*d for w,(_,d) in zip(weights,outputs))
        delta=torch.where(observed.flatten(1).any(-1)[:,None],delta,0)
        return result,delta
