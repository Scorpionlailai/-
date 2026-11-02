"""Read unpooled expert candidates separately at each original missing point."""
from math import sqrt

import torch
from torch import nn

from .point_evidence import evidence_features
from .router import EXPERTS
from .coordinates import numeric_features
from .variable_readout import VariableReadoutConditioner


class PointCandidateReadout(nn.Module):
    def __init__(self, width=64, heads=8, *, coordinates="global_v1", variable_readout="none",variable_count=0):
        super().__init__()
        self.coordinates = coordinates
        if width != 64 or heads != 8:
            raise ValueError("point candidate readout requires width64/eight heads")
        self.width, self.heads = width, heads
        self.context_norm = nn.LayerNorm(width)
        self.latent_norm = nn.LayerNorm(width)
        self.query = nn.Sequential(nn.Linear(143, width), nn.GELU(), nn.Linear(width, width))
        self.message_norm = nn.LayerNorm(width)
        self.quality = nn.ModuleDict({name:nn.Linear(4 if name == 'variable' else 5, width)
                                     for name in EXPERTS})
        self.kind = nn.Embedding(3, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.output = nn.Sequential(nn.LayerNorm(2*width), nn.Linear(2*width, width),
                                    nn.GELU(), nn.Linear(width, 1))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)
        self.variable_readout = VariableReadoutConditioner(variable_count,width) if variable_readout != "none" else None

    def forward(self, inputs, router, evidence, decoder, points, visible_features, candidates):
        context, visible, _, prior, available, phase, _ = evidence_features(
            inputs, router, decoder, points, visible_features,self.coordinates)
        B, C, N, P, _ = context.shape
        anchor = self.latent_norm(router.latents)[..., None, :].expand(B,C,N,P,self.width)
        numerical = numeric_features(candidates,inputs.query_mean,inputs.query_std,self.coordinates)[...,None]
        query = self.query(torch.cat((self.context_norm(context), anchor, visible,
                                      phase, numerical, prior, available.to(context.dtype)), -1))
        messages, masses, allowed, counts = [], [], [], []
        for i, name in enumerate(EXPERTS):
            item = evidence[name]
            cand = item.candidates
            if cand is None:
                raise ValueError(f"{name} candidates required for point readout")
            valid = cand.valid & item.available[..., None] & router.available[..., i+1, None]
            # Mask operands before normalization/projection, including unused NaN/Inf.
            raw = torch.where(valid[..., None], cand.messages, 0)
            quality = torch.where(valid[..., None], cand.quality, 0)
            if getattr(self, 'neutral_quality', False):
                quality = torch.zeros_like(quality)
            encoded = self.message_norm(raw)+self.quality[name](quality)+self.kind.weight[i]
            messages.append(encoded)
            weights = torch.where(valid, cand.weights, 0)
            masses.append(prior[..., i+1, None]*weights[..., None, :])
            allowed.append(valid)
            counts.append(cand.messages.shape[-2])
        encoded = torch.cat(messages, -2)
        valid = torch.cat(allowed, -1)
        K = encoded.shape[-2]
        H, D = self.heads, self.width//self.heads
        q = query.reshape(B,C,N,P,H,D).transpose(-3,-2)
        k = self.key(encoded).reshape(B,C,N,K,H,D).transpose(-3,-2)
        v = self.value(encoded).reshape(B,C,N,K,H,D).transpose(-3,-2)
        # Average only valid missing-point queries. Keep the original per-point
        # query for the pointwise output network. All candidates in this sealed
        # reader have patch-level legality; an assertion guards future changes.
        if getattr(self, 'patch_shared', False):
            missing = inputs.patch_validity & ~inputs.patch_mask
            denominator = missing.sum(-1, keepdim=True).clamp_min(1)
            shared_q = (query * missing[...,None]).sum(-2, keepdim=True) / denominator[...,None]
            q = shared_q.expand_as(query).reshape(B,C,N,P,H,D).transpose(-3,-2)
            shared_prior = (prior * missing[...,None]).sum(-2, keepdim=True) / denominator[...,None]
            # Empty eligible patches still have a legal null token.
            shared_prior = torch.where(missing.any(-1)[...,None,None], shared_prior, prior[..., :1, :])
            masses = []
            for i, name in enumerate(EXPERTS):
                cand = evidence[name].candidates
                mask = allowed[i]
                masses.append(shared_prior[..., i+1, None].expand(B,C,N,P,1) * torch.where(mask,cand.weights,0)[...,None,:])
            prior = shared_prior.expand_as(prior)
        logits = torch.matmul(q,k.transpose(-1,-2))/sqrt(D)
        mass = torch.cat(masses, -1)
        logits = logits+mass.clamp_min(1e-8).log()[..., None, :, :]
        logits = logits.masked_fill(~valid[..., None, None, :], -torch.inf)
        # Always-valid null token has zero value. Fully unavailable experts stay absent.
        null = prior[..., :1].clamp_min(1e-8).log()[..., None, :, :].expand(B,C,N,H,P,1)
        attention = torch.softmax(torch.cat((null,logits), -1), -1)
        readout = torch.matmul(attention[..., 1:],v).transpose(-3,-2).reshape(B,C,N,P,self.width)
        features = torch.cat((query,readout), -1)
        if self.variable_readout is None:
            correction = self.output(features).squeeze(-1)
        else:
            hidden = self.output[2](self.output[1](self.output[0](features)))
            correction = self.output[-1](self.variable_readout(hidden)).squeeze(-1)
        eligible = inputs.patch_validity & ~inputs.patch_mask & valid.any(-1)[..., None]
        correction = torch.where(eligible,correction,0)
        return correction, dict(weights=attention, candidate_counts=tuple(counts), valid=valid)
