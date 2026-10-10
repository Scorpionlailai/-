"""Distinct expert groups and layers over one shared historical candidate lookup."""
from dataclasses import dataclass, replace
from typing import Mapping

import torch
from torch import Tensor, nn
from .contracts import ExpertInputs, ExpertEvidence, RouterOutput
from .exposure import sample_specialist_exposure
from .router import EXPERTS
from .stacked_group import TVHGroup
from .stacked_group_kernels import prepare_grouped_variable_sources, grouped_tv_h
from .stacked_fusion import StackedGroupFusion


@dataclass(frozen=True)
class StackLayerDiagnostics:
    layer_index: int
    latents: Tensor
    routes: Tensor
    preview_gates: Tensor
    group_weights: Tensor
    group_deltas: Tensor
    allowed: Tensor
    update_rms: Tensor


@dataclass(frozen=True)
class StackedRepairState:
    router: RouterOutput
    evidence: Mapping[str, ExpertEvidence]
    layers: tuple[StackLayerDiagnostics, ...]
    specialist_selected: Tensor | None


class StackedTVHLayer(nn.Module):
    def __init__(self,variable_count,patch_length,d_model,group_count,**kwargs):
        super().__init__()
        self.groups=nn.ModuleList([TVHGroup(variable_count,patch_length,d_model,**kwargs) for _ in range(group_count)])
        self.fusion=StackedGroupFusion(d_model,group_count)


class StackedTVHController(nn.Module):
    def __init__(self,variable_count:int,patch_length:int,d_model:int,*,layers:int,groups:int,
                 activation_checkpointing:bool=False,variable_topk:int=4,variable_offsets:tuple[int,...]=(-1,0,1),
                 temporal_local_radius:int=2,temporal_boundaries:tuple[int,...]=(2,4,8,16)):
        super().__init__()
        if type(layers) is not int or type(groups) is not int or min(layers,groups)<1:
            raise ValueError('stack layer and group counts must be positive integers')
        self.group_count=groups; self.activation_checkpointing=activation_checkpointing
        self.layers=nn.ModuleList([StackedTVHLayer(variable_count,patch_length,d_model,groups,
            variable_topk=variable_topk,variable_offsets=variable_offsets,
            temporal_local_radius=temporal_local_radius,temporal_boundaries=temporal_boundaries) for _ in range(layers)])

    def forward(self,inputs:ExpertInputs,*,historical,epoch:int=0,explore_variable:bool=False,
                specialist_exposure:float=0,specialist_seed:int=0,disabled_experts:frozenset[str]=frozenset(),
                disabled_groups:frozenset[tuple[int,int]]=frozenset())->StackedRepairState:
        if not set(disabled_experts).issubset(EXPERTS): raise ValueError('unknown disabled expert')
        if any(type(l) is not int or type(g) is not int or l<0 or l>=len(self.layers) or g<0 or g>=self.group_count for l,g in disabled_groups):
            raise ValueError('disabled group index out of range')
        h_cache=historical.prepare_candidates(inputs)
        caches=[prepare_grouped_variable_sources(inputs,[g.variable for g in layer.groups],epoch=epoch,explore=explore_variable) for layer in self.layers]
        visible=(inputs.patch_mask.bool() & inputs.patch_validity.bool()).any(-1)
        t_available=(visible.sum(-1,keepdim=True)-visible.long()).gt(0) & inputs.damaged.bool()
        availability=torch.stack([torch.stack([torch.stack((h_cache.available,cache.available,t_available),-1) for cache in layer_caches],-2) for layer_caches in caches],-3)
        for index,name in enumerate(EXPERTS):
            if name in disabled_experts: availability[...,index]=False
        for l,g in disabled_groups: availability[...,l,g,:]=False
        selected=None; allowed=availability
        if specialist_exposure:
            union=availability.any(dim=(-3,-2))
            kind_allowed,selected=sample_specialist_exposure(inputs.damaged,union,fraction=specialist_exposure,seed=specialist_seed)
            allowed=availability & kind_allowed[...,None,None,:]
        source_context_allowed=None if selected is None else ~selected
        state=inputs.base_latents; diagnostics=[]
        for index,layer in enumerate(self.layers):
            current=replace(inputs,base_latents=state)
            routed=grouped_tv_h(current,layer.groups,h_cache,caches[index],allowed[...,index,:,:],source_context_allowed)
            fused=layer.fusion(state,routed.latents-state[...,None,:],routed.weights,inputs.rho,routed.available[...,1:].any(-1))
            diagnostics.append(StackLayerDiagnostics(index+1,fused.latents,routed.weights,routed.preview_gates,
                fused.group_weights,fused.projected_deltas,allowed[...,index,:,:],(fused.latents-state).square().mean(-1).sqrt()))
            state=fused.latents
        # Public summaries describe last-layer routes; detailed group states remain in layers.
        group_weights=fused.group_weights[...,1:]
        weights=(group_weights[...,None]*routed.weights).sum(-2)
        weights=torch.cat((weights[...,:1]+fused.group_weights[...,:1],weights[...,1:]),-1)
        evidence={name:ExpertEvidence((item.message*group_weights[...,None]).sum(-2),
            (item.quality*group_weights[...,None]).sum(-2),item.available.any(-1),{'group_weights':group_weights.detach()}) for name,item in routed.evidence.items()}
        deltas={name:(delta*group_weights[...,None]).sum(-2) for name,delta in routed.deltas.items()}
        return StackedRepairState(RouterOutput(state,weights,deltas,routed.available.any(-2)),evidence,tuple(diagnostics),selected)
