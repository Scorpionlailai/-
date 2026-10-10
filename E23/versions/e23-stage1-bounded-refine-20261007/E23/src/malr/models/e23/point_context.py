"""Direct numeric repair from unpooled visible context and routed H/V/T features."""
from dataclasses import replace
import torch
from torch import nn
from torch.nn import functional as F
from malr.data.patching import patchify_preserving_values
from .coordinates import repair_scale


class PointContextReadout(nn.Module):
    def __init__(self, variables: int, width: int = 64):
        super().__init__()
        if variables != 7 or width != 64:
            raise ValueError('point context requires ETTh1 width64')
        self.variables, self.width = variables, width
        self.network = nn.Sequential(nn.Linear(2*width+6+variables, width),
                                     nn.GELU(), nn.Linear(width, 1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def _points(self, latents, validity, coverage):
        B,C,N,D=latents.shape
        values=latents.permute(0,1,3,2)[...,None].expand(B,C,D,N,16)*validity[:,:,None]
        columns=values.permute(0,1,2,4,3).reshape(B*C,D*16,N)
        folded=F.fold(columns,(1,104),(1,16),stride=(1,8)).squeeze(2)[...,:96]
        return folded/coverage.transpose(1,2).reshape(B*C,1,96).clamp_min(1)

    def forward(self, explicit, router, points, visible_features, mean, std):
        seq=explicit.normalized_sequence;B,T,C=seq.shape
        if (T,C)!=(96,self.variables) or points.shape!=(B*C,self.width,T) or visible_features.shape!=(B*C,5,T):
            raise ValueError('point context shape mismatch')
        identity=torch.eye(C,device=seq.device,dtype=seq.dtype)[None,:,None].expand(B,C,T,C).reshape(B*C,T,C)
        coarse=seq.transpose(1,2).reshape(B*C,T,1)*(std/repair_scale(std)).reshape(B*C,1,1)
        latent=self._points(router.latents,explicit.validity_patches,explicit.coverage).transpose(1,2)
        features=torch.cat((points.transpose(1,2),latent,coarse,visible_features.transpose(1,2),identity),-1)[:,-32:]
        delta=2*torch.tanh(self.network(features).squeeze(-1)/2)
        correction=F.pad(delta,(64,0)).reshape(B,C,T).transpose(1,2)*repair_scale(std)[:,None]/std[:,None]
        eligible=explicit.sequence_missing & torch.arange(T,device=seq.device)[None,:,None].ge(64)
        repaired=torch.where(eligible,seq+correction,seq)
        patches=patchify_preserving_values(repaired,explicit.sequence_observed,16,8,validity=explicit.sequence_validity)
        return replace(explicit,normalized_sequence=repaired,
                       standardized_sequence=repaired*std[:,None]+mean[:,None],
                       repaired_patches=patches.values.masked_fill(~explicit.validity_patches,0))
