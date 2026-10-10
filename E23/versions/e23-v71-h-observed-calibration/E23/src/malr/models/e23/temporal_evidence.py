"""Repeated attention from recent repair queries to original visible time evidence."""
from dataclasses import replace
import torch
from torch import nn
from torch.nn import functional as F
from malr.data.patching import patchify_preserving_values
from .coordinates import repair_scale

class _VisibleTimeBlock(nn.Module):
    def __init__(self,width=64,heads=8):
        super().__init__();self.width,self.heads=width,heads
        self.norm=nn.LayerNorm(width);self.q=nn.Linear(width,width,bias=False)
        self.k=nn.Linear(width,width,bias=False);self.v=nn.Linear(width,width,bias=False)
        self.projection=nn.Linear(width,width)
        self.ffn_norm=nn.LayerNorm(width);self.ffn=nn.Sequential(nn.Linear(width,2*width),nn.GELU(),nn.Linear(2*width,width))
        self.relative_bias=nn.Parameter(torch.zeros(heads,191));self.null_bias=nn.Parameter(torch.zeros(heads))
        self.register_buffer('distance_scales',torch.tensor([8.,12.,16.,24.,32.,48.,64.,96.]),persistent=False)

    def forward(self,query,source,source_observed):
        BC,Q,D=query.shape;T=source.shape[1];H=self.heads;dh=D//H
        q=self.q(self.norm(query)).reshape(BC,Q,H,dh).transpose(1,2)
        k=self.k(source).reshape(BC,T,H,dh).transpose(1,2)
        v=self.v(source).reshape(BC,T,H,dh).transpose(1,2)
        difference=torch.arange(T,device=query.device)[None,:]-torch.arange(T-Q,T,device=query.device)[:,None]
        logits=q@k.transpose(-1,-2)/(dh**.5)
        logits=logits+self.relative_bias[:,difference+95][None]-difference.abs()[None,None]/self.distance_scales[None,:,None,None]
        logits=logits.masked_fill(~source_observed[:,None,None],float('-inf'))
        null=self.null_bias[None,:,None,None].expand(BC,H,Q,1)
        weights=torch.softmax(torch.cat((null,logits),-1),-1)
        update=(weights[...,1:]@v).transpose(1,2).reshape(BC,Q,D)
        query=query+.1*self.projection(update)
        query=query+.1*self.ffn(self.ffn_norm(query))
        return query,weights

class TemporalEvidenceRefiner(nn.Module):
    def __init__(self,variables,width=64):
        super().__init__()
        if variables!=7 or width!=64:raise ValueError('temporal evidence requires original ETTh1 width64')
        self.variables,self.width=variables,width
        self.query=nn.Sequential(nn.Linear(2*width+6+variables,width),nn.GELU())
        self.source=nn.Sequential(nn.Linear(width+5+variables,width),nn.GELU(),nn.LayerNorm(width))
        self.blocks=nn.ModuleList([_VisibleTimeBlock(width,8),_VisibleTimeBlock(width,8)])
        self.output=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,1))
        nn.init.zeros_(self.output[-1].weight);nn.init.zeros_(self.output[-1].bias)

    def _points(self,latents,validity,coverage):
        B,C,N,D=latents.shape;P=16
        values=latents.permute(0,1,3,2)[...,None].expand(B,C,D,N,P)*validity[:,:,None]
        columns=values.permute(0,1,2,4,3).reshape(B*C,D*P,N)
        folded=F.fold(columns,(1,104),(1,P),stride=(1,8)).squeeze(2)[...,:96]
        return folded/coverage.transpose(1,2).reshape(B*C,1,96).clamp_min(1)

    def forward(self,explicit,router,points,visible_features,mean,std):
        seq=explicit.normalized_sequence;B,T,C=seq.shape
        if (T,C)!=(96,self.variables) or points.shape!=(B*C,self.width,T) or visible_features.shape!=(B*C,5,T):raise ValueError('temporal evidence shape mismatch')
        observed=explicit.sequence_observed & explicit.sequence_validity
        identity=torch.eye(C,device=seq.device,dtype=seq.dtype)[None,:,None].expand(B,C,T,C).reshape(B*C,T,C)
        context=points.transpose(1,2);features=visible_features.transpose(1,2)
        ratio=(std/repair_scale(std)).reshape(B*C,1,1)
        coarse=seq.transpose(1,2).reshape(B*C,T,1)*ratio
        latent=self._points(router.latents,explicit.validity_patches,explicit.coverage).transpose(1,2)
        query=self.query(torch.cat((context,latent,coarse,features,identity),-1)[:,-32:])
        source=self.source(torch.cat((context,features,identity),-1))
        source_observed=observed.transpose(1,2).reshape(B*C,T)
        attention=[]
        for block in self.blocks:
            query,w=block(query,source,source_observed);attention.append(w)
        delta=2*torch.tanh(self.output(query).squeeze(-1)/2)
        correction=F.pad(delta,(64,0)).reshape(B,C,T).transpose(1,2)*repair_scale(std)[:,None]/std[:,None]
        scope=torch.arange(T,device=seq.device)[None,:,None].ge(64)
        eligible=explicit.sequence_missing & scope
        repaired=torch.where(eligible,seq+correction,seq)
        patches=patchify_preserving_values(repaired,explicit.sequence_observed,16,8,validity=explicit.sequence_validity)
        result=replace(explicit,normalized_sequence=repaired,standardized_sequence=repaired*std[:,None]+mean[:,None],repaired_patches=patches.values.masked_fill(~explicit.validity_patches,0))
        return result,dict(weights=torch.stack(attention,1),source_observed=source_observed,queries=32,heads=8,layers=2)
