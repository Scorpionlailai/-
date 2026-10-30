"""R0 latest selected recovery; R1 summaries; R2 no segment organization."""
from types import SimpleNamespace
from dataclasses import replace
import torch
from torch import nn
import bootstrap
from legacy_model import make_model
from interfaces import SummaryReadout,DirectReadout,DirectConfidence
from source_content import temporal_content,variable_content,historical_content
from single import make_head
from integrations.v125_ablation.model.contracts import ExpertInputs
from integrations.v125_ablation.model.exposure import sample_specialist_exposure
from integrations.v125_ablation.model.coordinates import repair_scale
from malr.data.patching import patchify_with_state,patchify_preserving_values
from malr.models.explicit_repair import build_explicit_repair

class Calibration(nn.Module):
    def __init__(self,full):
        super().__init__();self.global_dense_refiner=full.global_dense_refiner
        self.global_dense_refiner.temporal.layers=nn.ModuleList([self.global_dense_refiner.temporal.layers[0]])
        self.global_dense_refiner.temporal.num_layers=1
    def forward(self,explicit,router,points,mean,std):
        y=self.global_dense_refiner(explicit.normalized_sequence,explicit,mean,std)
        p=patchify_preserving_values(y,explicit.sequence_observed,16,8,validity=explicit.sequence_validity)
        return replace(explicit,normalized_sequence=y,standardized_sequence=y*std[:,None]+mean[:,None],repaired_patches=p.values.masked_fill(~explicit.validity_patches,0)),torch.zeros_like(mean)

def make_calibrator(variables,variant,seed):return Calibration(make_head(variables,64,seed))

class DirectCore(nn.Module):
    def __init__(self,old):
        super().__init__()
        for n in ['base_encoder','front_context','decoder','temporal','variable','historical']:setattr(self,n,getattr(old,n))
        for n in ['window_length','patch_length','stride','variable_count','d_model','patch_count','repair_coordinates']:setattr(self,n,getattr(old,n))
        self.candidate_condition=nn.Parameter(old.iterative.round_embedding.weight[0].detach().clone())
        self.point_candidate_readout=DirectReadout(old.point_candidate_readout)
        self.overlap_confidence=DirectConfidence(64,coordinates=old.overlap_confidence.coordinates)
        self.overlap_confidence.load_state_dict(old.overlap_confidence.state_dict(),strict=True)
        # Only the source-level quality projections are segment-only. Candidate
        # message encoders and within-source weighting operators remain intact.
        for source in [self.temporal,self.variable,self.historical]:source.quality_projector=None
        for n in ['router','iterative','point_router','point_refiner','recent_mean_projector']:setattr(self,n,None)

    def forward(self,values,observed_mask,query_mean,query_std,patch_starts,*,valid_mask=None,epoch=0,explore_variable=False,specialist_exposure=0.,specialist_seed=0,disabled_experts=frozenset(),**kwargs):
        patches=patchify_with_state(values,observed_mask,16,8,validity=valid_mask)
        valid=patches.validity;seen=patches.effective_mask;safe=patches.values.masked_fill(~seen,0)
        base=self.base_encoder(torch.cat((safe,seen.to(values.dtype)),-1))
        context,points,visible=self.front_context(values,observed_mask,query_mean,query_std,valid,valid_mask=valid_mask,return_points=True)
        base=base+context
        inputs=ExpertInputs(base,safe,seen,valid,patches.rho,patch_starts,query_mean,query_std,patches.rho.lt(1))
        hc=self.historical.prepare_candidates(inputs);vc=self.variable.prepare_sources(inputs,epoch=epoch,explore=explore_variable);tc=self.temporal.prepare_sources(inputs)
        visible_sources=tc.observed.any(-1);tav=(visible_sources.sum(-1,keepdim=True)-visible_sources.long()).gt(0)&inputs.damaged
        available=torch.stack((hc.available,vc.available,tav),-1)
        for i,n in enumerate(('historical','variable','temporal')):
            if n in disabled_experts:available[...,i]=False
        allowed,selected=sample_specialist_exposure(inputs.damaged,available,fraction=specialist_exposure,seed=specialist_seed) if specialist_exposure else (available,None)
        source_allowed=None if selected is None else ~selected
        evidence={
          'temporal':temporal_content(self.temporal,inputs,tc,round_embedding=self.candidate_condition,source_context_allowed=source_allowed),
          'variable':variable_content(self.variable,inputs,vc,round_embedding=self.candidate_condition,source_context_allowed=source_allowed),
          'historical':historical_content(self.historical,inputs,hc,round_embedding=self.candidate_condition)}
        for i,n in enumerate(('historical','variable','temporal')):evidence[n]=replace(evidence[n],available=evidence[n].available&allowed[...,i])
        legality=torch.stack((torch.ones_like(inputs.damaged,dtype=torch.bool),*(evidence[n].available&inputs.damaged for n in ('historical','variable','temporal'))),-1)
        interface=SimpleNamespace(latents=base,available=legality) # No source weights/deltas.
        base_prediction=self.decoder(base)
        correction,attention=self.point_candidate_readout(inputs,interface,evidence,self.decoder,points,visible,base_prediction)
        scale=repair_scale(query_std)[:,:,None,None]/query_std[:,:,None,None]
        candidate=base_prediction+correction*scale
        confidence=self.overlap_confidence(inputs,interface,self.decoder,points,visible,candidate)
        explicit=build_explicit_repair(candidate,safe,seen,valid,query_mean,query_std,window_len=96,patch_len=16,stride=8,candidate_weights=confidence)
        explicit=replace(explicit,repaired_patches=explicit.repaired_patches.masked_fill(~explicit.validity_patches,0))
        return SimpleNamespace(explicit=explicit,router=interface,evidence=evidence,base_latents=base,specialist_selected=selected,iterations=None,point_candidate_attention=attention,base_prediction=base_prediction)

def make_repair(data,variant,seed=2023):
    if variant not in ['R0','R1','R2']:raise ValueError(variant)
    model=make_model(data,'A-conf',seed)
    if variant=='R1':model.core.point_candidate_readout.__class__=SummaryReadout
    if variant=='R2':model.core=DirectCore(model.core)
    return model

def retrieval_state(model):return model.core.historical.retrieval.state_dict()
def restore_retrieval(model,state):model.core.historical.retrieval.load_state_dict(state)
def library_hash(model):return model.memory.snapshot_hash()
