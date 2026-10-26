"""Runnable R0/A/B adapters sharing original frontend, loss and history lifecycle."""
from types import SimpleNamespace
import torch
from torch import nn
from dataclasses import replace
from bootstrap import V125Adapter,strip_postprocessing
from candidates import SourceAdapters
from reader import UnifiedReader
from history import HistoricalCandidateProvider
from integrations.v125_ablation.model.contracts import ExpertInputs
from integrations.v125_ablation.model.point_evidence import point_patches
from integrations.v125_ablation.model.coordinates import repair_scale
from malr.data.patching import patchify_with_state
from malr.models.explicit_repair import build_explicit_repair

class UnifiedEncoder(nn.Module):
    def __init__(self,old):
        super().__init__()
        for name in ('base_encoder','front_context','decoder','temporal','variable'):
            setattr(self,name,getattr(old,name))
        for name in ('window_length','patch_length','stride','variable_count','d_model','patch_count','repair_coordinates'):
            setattr(self,name,getattr(old,name))
        for name in ('point_refiner','recent_mean_projector','point_router','point_candidate_readout','overlap_confidence','router','iterative'):
            setattr(self,name,None)
        oldh=old.historical
        with torch.random.fork_rng(devices=[]):
            self.historical=HistoricalCandidateProvider(16,64,64,self.variable_count,topk=oldh.topk,
                similarity_threshold=oldh.similarity_threshold,retrieval_strategy=oldh.retrieval_strategy)
        self.historical.key_encoder.load_state_dict(oldh.key_encoder.state_dict())
        self.historical.retrieval.load_state_dict(oldh.retrieval.state_dict())
        self.historical.key_policy=oldh.key_policy
        self.adapters=SourceAdapters();self.reader=UnifiedReader()
        # These heads performed the deleted per-expert pooling or dynamic reranking.
        for module,names in [(self.temporal,('message_projector','quality_projector')),
                             (self.variable,('message_projector','quality_projector','context_query','context_key'))]:
            for name in names:setattr(module,name,None)
        # Hard argmax selection supplies no gradient to these projections in B.
        self.temporal.query_projection.requires_grad_(False)
        self.temporal.key_projection.requires_grad_(False)

    def forward(self,values,observed_mask,query_mean,query_std,patch_starts,*,valid_mask=None,
                epoch=0,explore_variable=False,specialist_exposure=0.,specialist_seed=0,
                disabled_experts=frozenset(),**kwargs):
        if values.shape[1:]!=(96,self.variable_count):raise ValueError('expected [B,96,C]')
        patches=patchify_with_state(values,observed_mask,16,8,validity=valid_mask)
        seen=patches.effective_mask;valid=patches.validity
        patch_values=torch.where(seen,patches.values,0)
        base=self.base_encoder(torch.cat((patch_values,seen.to(values.dtype)),-1))
        context,points,visible=self.front_context(values,observed_mask,query_mean,query_std,valid,
            valid_mask=valid_mask,return_points=True)
        base=base+context
        inputs=ExpertInputs(base,patch_values,seen,valid,patches.rho,patch_starts,query_mean,query_std,patches.rho.lt(1))
        bundles,selected=self.adapters(inputs,self.temporal,self.variable,self.historical,
            epoch=epoch,explore=explore_variable,exposure=specialist_exposure,seed=specialist_seed,disabled=disabled_experts)
        delta,trace=self.reader(base,point_patches(points,valid),bundles,valid&~seen)
        base_prediction=self.decoder(base)
        scaled=delta*repair_scale(query_std)[:,:,None,None]/query_std[:,:,None,None]
        candidate=base_prediction+scaled
        explicit=build_explicit_repair(candidate,patch_values,seen,valid,query_mean,query_std,
            window_len=96,patch_len=16,stride=8)
        explicit=replace(explicit,repaired_patches=explicit.repaired_patches.masked_fill(~explicit.validity_patches,0))
        return SimpleNamespace(explicit=explicit,base_latents=base,router=None,evidence=bundles,
            specialist_selected=selected,point_candidate_attention=trace,base_prediction=base_prediction,
            evidence_correction=scaled)


def make_model(data,route,seed=2023):
    if route=='A-conf':
        model=make_model(data,'R0-conf',seed)
        model.core.iterative.parallel=True
        model.core.point_router=None
        model.route=route
        return model
    if route not in ('R0-equal','R0-conf','A','B'):raise ValueError('unknown route')
    model=strip_postprocessing(V125Adapter(data,variant='A0',seed=seed))
    if route!='R0-conf':model.core.overlap_confidence=None
    if route=='A':
        model.core.iterative.parallel=True
        model.core.point_router=None
    elif route=='B':
        model.core=UnifiedEncoder(model.core)
        model.ema_encoder=model.core.historical.retrieval.ema_encoder
        model.frozen_encoder=model.core.historical.retrieval.frozen_encoder
    model.route=route
    return model
