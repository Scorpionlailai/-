"""R3 source removal or matched-budget selection ablations. E0 is reused."""
from types import SimpleNamespace
from dataclasses import replace
import torch
import bootstrap
import variants
from aligned_h_model import make_repair as r3_factory
from r2_variants import DirectCore
from source_content import temporal_content,variable_content,historical_content
from selection_controls import empty,controlled_evidence
from integrations.v125_ablation.model.contracts import ExpertInputs
from integrations.v125_ablation.model.exposure import sample_specialist_exposure
from integrations.v125_ablation.model.coordinates import repair_scale
from malr.data.patching import patchify_with_state
from malr.models.explicit_repair import build_explicit_repair

class AblationCore(DirectCore):
 def forward(self,values,observed_mask,query_mean,query_std,patch_starts,*,valid_mask=None,epoch=0,explore_variable=False,specialist_exposure=0.,specialist_seed=0,disabled_experts=frozenset(),**kwargs):
  patches=patchify_with_state(values,observed_mask,16,8,validity=valid_mask)
  valid=patches.validity;seen=patches.effective_mask;safe=patches.values.masked_fill(~seen,0)
  base=self.base_encoder(torch.cat((safe,seen.to(values.dtype)),-1))
  context,points,visible=self.front_context(values,observed_mask,query_mean,query_std,valid,valid_mask=valid_mask,return_points=True)
  base=base+context
  inputs=ExpertInputs(base,safe,seen,valid,patches.rho,patch_starts,query_mean,query_std,patches.rho.lt(1))
  is_control=self.ablation_variant in ['E4','E5']
  if is_control:
   with torch.no_grad():
    hc=self.historical.prepare_candidates(inputs);vc=self.variable.prepare_sources(inputs,epoch=epoch,explore=explore_variable)
   tc=None
   # Same-weights reference R3 counts control the candidate budget, not a teacher prediction.
   from selection_controls import temporal_reference_count
   avail=torch.stack((hc.available,vc.available,temporal_reference_count(inputs)>0),-1)
  else:
   hc=None if 'historical' in disabled_experts else self.historical.prepare_candidates(inputs)
   vc=None if 'variable' in disabled_experts else self.variable.prepare_sources(inputs,epoch=epoch,explore=explore_variable)
   tc=None if 'temporal' in disabled_experts else self.temporal.prepare_sources(inputs)
   tav=torch.zeros_like(inputs.damaged) if tc is None else ((tc.observed.any(-1).sum(-1,keepdim=True)-tc.observed.any(-1).long()).gt(0)&inputs.damaged)
   avail=torch.stack((torch.zeros_like(tav) if hc is None else hc.available,torch.zeros_like(tav) if vc is None else vc.available,tav),-1)
  allowed,selected=sample_specialist_exposure(inputs.damaged,avail,fraction=specialist_exposure,seed=specialist_seed) if specialist_exposure else (avail,None)
  source_allowed=None if selected is None else ~selected
  trace=None
  if is_control:
   evidence,trace,new_allowed,new_selected=controlled_evidence(self,inputs,self.ablation_variant,hc,vc,source_allowed,epoch if explore_variable else 0,specialist_exposure,specialist_seed)
   if self.ablation_variant=='E5':allowed,selected=new_allowed,new_selected
  else:
   evidence={
    'historical':empty(inputs,5,4) if hc is None else historical_content(self.historical,inputs,hc,round_embedding=self.candidate_condition),
    'variable':empty(inputs,4,12) if vc is None else variable_content(self.variable,inputs,vc,round_embedding=self.candidate_condition,source_context_allowed=source_allowed),
    'temporal':empty(inputs,5,10) if tc is None else temporal_content(self.temporal,inputs,tc,round_embedding=self.candidate_condition,source_context_allowed=source_allowed)}
  for i,n in enumerate(['historical','variable','temporal']):evidence[n]=replace(evidence[n],available=evidence[n].available&allowed[...,i])
  legality=torch.stack((torch.ones_like(inputs.damaged,dtype=torch.bool),*(evidence[n].available&inputs.damaged for n in ['historical','variable','temporal'])),-1)
  interface=SimpleNamespace(latents=base,available=legality)
  prediction=self.decoder(base)
  correction,attention=self.point_candidate_readout(inputs,interface,evidence,self.decoder,points,visible,prediction)
  scale=repair_scale(query_std)[:,:,None,None]/query_std[:,:,None,None]
  candidate=prediction+correction*scale
  confidence=self.overlap_confidence(inputs,interface,self.decoder,points,visible,candidate)
  explicit=build_explicit_repair(candidate,safe,seen,valid,query_mean,query_std,window_len=96,patch_len=16,stride=8,candidate_weights=confidence)
  explicit=replace(explicit,repaired_patches=explicit.repaired_patches.masked_fill(~explicit.validity_patches,0))
  return SimpleNamespace(explicit=explicit,router=interface,evidence=evidence,base_latents=base,specialist_selected=selected,iterations=None,point_candidate_attention=attention,base_prediction=prediction,selection_trace=trace)

DROP={'E1':'temporal','E2':'variable','E3':'historical'}
def make_repair(data,variant,seed=2023):
 if variant not in ['E0','E1','E2','E3','E4','E5','E0_CHECK']:raise ValueError(variant)
 model=r3_factory(data,'H_GLOBAL',seed)
 if variant=='E0':return model
 model.core.__class__=AblationCore;model.core.ablation_variant=variant
 model.disabled=frozenset([DROP[variant]]) if variant in DROP else frozenset()
 # Disabled source parameters remain registered to preserve shared initialization;
 # they are frozen and bypassed, not counted as effective inference parameters.
 for source in model.disabled:
  for p in getattr(model.core,source).parameters():p.requires_grad_(False)
  for p in model.core.point_candidate_readout.quality[source].parameters():p.requires_grad_(False)
  if source=='historical':
   model.ema_encoder.requires_grad_(False);model.frozen_encoder.requires_grad_(False)
 return model

def install():variants.make_repair=make_repair


def effective_parameters(full):
 """Inference-active scalar count; retained unused registration is excluded."""
 disabled=full.frozen.repair.disabled
 total=0
 for name,param in full.named_parameters():
  prefixes=['frozen.repair.ema_encoder.','frozen.repair.core.historical.key_encoder.']
  for source in disabled:
   prefixes.extend(['frozen.repair.core.'+source+'.','frozen.repair.core.point_candidate_readout.quality.'+source+'.'])
  if 'historical' in disabled:prefixes.append('frozen.repair.frozen_encoder.')
  if any(name.startswith(prefix) for prefix in prefixes):continue
  count=param.numel()
  if name=='frozen.repair.core.point_candidate_readout.kind.weight':count-=len(disabled)*param.shape[-1]
  # Deleted-source available/alternative columns are constant zero throughout this arm.
  if name=='frozen.repair.core.point_candidate_readout.query.0.weight':count-=len(disabled)*param.shape[0]
  if name=='frozen.repair.core.overlap_confidence.score.0.weight':count-=3*len(disabled)*param.shape[0]
  total+=count
 return total
