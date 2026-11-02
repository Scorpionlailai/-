"""Current R2 baseline (FULL), R2 source summaries (SUM), context capacity (CTX)."""
from dataclasses import replace
from types import SimpleNamespace
import torch
from torch import nn
import bootstrap
from r2_variants import make_repair as r2_make_repair, make_calibrator
from interfaces import DirectReadout,summarize_candidates
from integrations.v125_ablation.model.router import EXPERTS
from integrations.v125_ablation.model.point_evidence import point_patches
from integrations.v125_ablation.model.coordinates import numeric_features,repair_scale
from malr.data.patching import patchify_with_state
from malr.models.explicit_repair import build_explicit_repair
from malr.prototypes.retrieval import MaskAwareRetrievalEncoder,masked_pair_loss
from integrations.v125_ablation.adapter import V125Adapter
from integrations.physionet_baselines.losses import hidden_error,mean_or_zero,chunked_contrastive_loss

class R2SummaryReadout(DirectReadout):
    def forward(self,inputs,router,evidence,decoder,points,visible_features,candidates):
        compact={}
        for i,name in enumerate(EXPERTS):
            item=evidence[name];cand=item.candidates
            valid=cand.valid&item.available[...,None]&router.available[...,i+1,None]
            compact[name]=replace(item,candidates=summarize_candidates(cand,valid))
        return super().forward(inputs,router,compact,decoder,points,visible_features,candidates)

class ContextCore(nn.Module):
    def __init__(self,template,seed):
        super().__init__()
        for name in ['base_encoder','front_context','decoder','overlap_confidence']:
            setattr(self,name,getattr(template,name))
        for name in ['patch_count','d_model','variable_count','repair_coordinates']:
            setattr(self,name,getattr(template,name))
        self.context_norm=template.point_candidate_readout.context_norm
        self.latent_norm=template.point_candidate_readout.latent_norm
        # Analytic effective-inference budget matching, not validation tuning.
        target=sum(p.numel() for p in template.parameters())-sum(p.numel() for p in template.historical.key_encoder.parameters())+sum(p.numel() for p in template.historical.retrieval.frozen_encoder.parameters())
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.context_encoder=MaskAwareRetrievalEncoder(16,64)
            fixed=sum(p.numel() for p in self.parameters())
            # MLP 199 -> hidden -> 64 -> 1 has 264*hidden+129 parameters.
            self.capacity_hidden=max(1,round((target-fixed-129)/264))
            self.context_mlp=nn.Sequential(nn.Linear(199,self.capacity_hidden),nn.GELU(),nn.Linear(self.capacity_hidden,64),nn.GELU(),nn.Linear(64,1))
            nn.init.zeros_(self.context_mlp[-1].weight);nn.init.zeros_(self.context_mlp[-1].bias)
        self.target_core_parameters=target
    def forward(self,values,observed,mean,std,positions,*,valid_mask=None,**kwargs):
        p=patchify_with_state(values,observed,16,8,validity=valid_mask);seen=p.effective_mask;valid=p.validity;safe=p.values.masked_fill(~seen,0)
        context,points,visible=self.front_context(values,observed,mean,std,valid,valid_mask=valid_mask,return_points=True)
        z0=self.base_encoder(torch.cat((safe,seen.to(values.dtype)),-1))+context
        base=self.decoder(z0);embedding=self.context_encoder(safe.flatten(0,2),seen.flatten(0,2)).reshape_as(z0)
        F=point_patches(points,valid);obs=point_patches(visible,valid)
        shape=base.shape
        phase=torch.linspace(-1,1,16,device=values.device,dtype=values.dtype).expand(*shape)[...,None]
        x=torch.cat((self.context_norm(F),self.latent_norm(z0)[...,None,:].expand(*shape,64),embedding[...,None,:].expand(*shape,64),obs,phase,numeric_features(base,mean,std,self.repair_coordinates)[...,None]),-1)
        delta=self.context_mlp(x).squeeze(-1).masked_fill(~(valid&~seen),0)
        candidate=base+delta*repair_scale(std)[:,:,None,None]/std[:,:,None,None]
        available=torch.zeros(*z0.shape[:-1],4,device=z0.device,dtype=torch.bool);available[...,0]=True
        interface=SimpleNamespace(latents=z0,available=available)
        inputs=SimpleNamespace(base_latents=z0,patch_validity=valid,patch_mask=seen,query_mean=mean,query_std=std)
        weights=self.overlap_confidence(inputs,interface,self.decoder,points,visible,candidate)
        explicit=build_explicit_repair(candidate,safe,seen,valid,mean,std,window_len=96,patch_len=16,stride=8,candidate_weights=weights)
        explicit=replace(explicit,repaired_patches=explicit.repaired_patches.masked_fill(~explicit.validity_patches,0))
        return SimpleNamespace(explicit=explicit,router=interface,base_latents=z0,base_prediction=base,evidence={},point_candidate_attention=None)

class ContextAdapter(nn.Module):
    context=V125Adapter.context
    def __init__(self,data,template,seed):
        super().__init__();self.data=data;self.seed=seed;self.epoch=0;self.batch_index=0;self.core=ContextCore(template,seed)
    def output(self,values,observed,starts,*,training_coarse=False):
        norm,positions=self.context(values,observed,starts);valid=torch.from_numpy(self.data.valid_for(starts)).to(values.device)
        return self.core(norm.student,observed,norm.mean,norm.std,positions,valid_mask=valid),norm,positions
    def forward(self,values,observed,starts):
        result,_,_=self.output(values,observed,starts)
        return torch.where(observed,values,result.explicit.standardized_sequence)
    def loss(self,truth,observed,starts):
        import numpy as np
        result,norm,_=self.output(truth,observed,starts,training_coarse=True);valid=torch.from_numpy(self.data.valid_for(starts)).to(truth.device)
        error=hidden_error(result.explicit.standardized_sequence,truth,observed,valid)
        tail=torch.where(~observed&valid,result.explicit.standardized_sequence-truth,0.)[:,-32:]
        query=patchify_with_state(norm.student,observed,16,8)
        clean=patchify_with_state(torch.where(valid,norm.target,0.),valid,16,8)
        eligible=query.rho.lt(1.)&clean.effective_mask.any(-1)
        if eligible.any():
            encoder=self.core.context_encoder;q=encoder(query.values[eligible],query.effective_mask[eligible]);t=encoder(clean.values[eligible],clean.effective_mask[eligible])
            auxiliary=masked_pair_loss(q,t)+.5*chunked_contrastive_loss(q,t)/np.log(max(len(q),2))
        else:auxiliary=error.sum()*0
        return mean_or_zero(error.square())+tail.mean(1).square().mean()+.1*auxiliary
    def after_step(self):pass
    def publish_epoch(self,epoch):pass

def make_repair(data,variant,seed=2023):
    if variant not in ('FULL','SUM','CTX'):raise ValueError(variant)
    model=r2_make_repair(data,'R2',seed)
    if variant=='SUM':model.core.point_candidate_readout.__class__=R2SummaryReadout
    if variant=='CTX':return ContextAdapter(data,model.core,seed)
    return model

def retrieval_state(model):return None if isinstance(model,ContextAdapter) else model.core.historical.retrieval.state_dict()
def restore_retrieval(model,state):
    if isinstance(model,ContextAdapter):
        if state is not None:raise ValueError('CTX cannot restore memory')
    else:model.core.historical.retrieval.load_state_dict(state)
def library_hash(model):return None if isinstance(model,ContextAdapter) else model.memory.snapshot_hash()
