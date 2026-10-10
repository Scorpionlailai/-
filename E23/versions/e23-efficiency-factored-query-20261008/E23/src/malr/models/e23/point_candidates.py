"""Read unpooled expert candidates separately at each original missing point."""
from math import sqrt

import torch
from torch import nn
from torch.nn import functional as F

from .point_evidence import evidence_features, point_patches
from .router import EXPERTS
from .coordinates import numeric_features
from .variable_readout import VariableReadoutConditioner


class PointCandidateReadout(nn.Module):
    def __init__(self, width=64, heads=8, *, coordinates="global_v1", variable_readout="none",variable_count=0, factor_query=False):
        super().__init__()
        self.coordinates = coordinates
        if width != 64 or heads != 8:
            raise ValueError("point candidate readout requires width64/eight heads")
        self.width, self.heads = width, heads
        self.factor_query = factor_query
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
        numerical = numeric_features(candidates,inputs.query_mean,inputs.query_std,self.coordinates)[...,None]
        if self.factor_query:
            # Split the SAME first linear map by inputs. Context is shared by
            # overlapping copies; patch latents are shared by P query points.
            # No parameter replacement or reduced feature width is involved.
            first = self.query[0]
            weight = first.weight
            point_part = F.linear(self.context_norm(points.transpose(1, 2)),
                                  weight[:, :self.width])
            point_part = point_patches(point_part.transpose(1, 2), inputs.patch_validity)
            # Old context is zero BEFORE LayerNorm at invalid padded copies.
            zero_part = F.linear(self.context_norm.bias, weight[:, :self.width])
            point_part = torch.where(inputs.patch_validity[..., None], point_part, zero_part)
            patch_part = F.linear(self.latent_norm(router.latents),
                                  weight[:, self.width:2*self.width])[..., None, :]
            local = torch.cat((visible, phase, numerical, prior, available.to(context.dtype)), -1)
            hidden = point_part + patch_part + F.linear(local, weight[:, 2*self.width:], first.bias)
            query = self.query[2](self.query[1](hidden))
        else:
            anchor = self.latent_norm(router.latents)[..., None, :].expand(B,C,N,P,self.width)
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
