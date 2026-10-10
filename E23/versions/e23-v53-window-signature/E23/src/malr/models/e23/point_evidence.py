"""Pointwise H/V/T reweighting and confidence for overlapping patch copies."""
import torch
from torch import nn
from torch.nn import functional as F
from .router import EXPERTS


def point_patches(points, validity):
    B, C, N, P = validity.shape
    end = (N-1)*8+P
    cols = F.pad(points, (0, end-96)).unfold(-1, P, 8)
    D = points.shape[1]
    return cols.permute(0, 2, 3, 1).reshape(B, C, N, P, D).masked_fill(~validity[..., None], 0)


def evidence_features(inputs, router, decoder, points, visible_features):
    valid = inputs.patch_validity
    context = point_patches(points, valid)
    visible = point_patches(visible_features, valid)
    deltas = torch.stack([router.deltas[name] for name in EXPERTS], -2)
    numeric_deltas = F.linear(deltas, decoder.weight, None).transpose(-1, -2)
    numeric_deltas = numeric_deltas.masked_fill(~router.available[..., None, 1:], 0)
    base = decoder(inputs.base_latents)
    alternatives = torch.cat((base[..., None], base[..., None]+numeric_deltas), -1)
    alternatives = alternatives*inputs.query_std[:, :, None, None, None]+inputs.query_mean[:, :, None, None, None]
    available = router.available[..., None, :].expand_as(alternatives)
    alternatives = alternatives.masked_fill(~available, 0)
    weights = (router.weights[..., None, :].expand_as(alternatives)
               if router.point_weights is None else router.point_weights)
    phase = torch.linspace(-1, 1, valid.shape[-1], device=context.device, dtype=context.dtype)
    phase = phase.expand(*valid.shape)[..., None]
    return context, visible, alternatives, weights, available, phase, numeric_deltas


class PointEvidenceRouter(nn.Module):
    """Keep patch routing as prior; choose a convex mixture separately per point."""
    def __init__(self, width=64):
        super().__init__()
        self.context_norm = nn.LayerNorm(width)
        self.quality = nn.ModuleDict({name:nn.Linear(width, 4) for name in EXPERTS})
        self.score = nn.Sequential(nn.Linear(94, width), nn.GELU(), nn.Linear(width, 4))
        nn.init.zeros_(self.score[-1].weight);nn.init.zeros_(self.score[-1].bias)

    def forward(self, inputs, router, evidence, decoder, points, visible_features):
        context, visible, alternatives, prior, available, phase, increments = evidence_features(
            inputs, router, decoder, points, visible_features)
        qualities = torch.cat([self.quality[name](evidence[name].quality).masked_fill(
            ~router.available[..., i+1, None], 0) for i, name in enumerate(EXPERTS)], -1)
        qualities = qualities[..., None, :].expand(*context.shape[:-1], 12)
        features = torch.cat((self.context_norm(context), qualities, alternatives, prior,
                              available.to(context.dtype), phase, visible), -1)
        eligible = inputs.patch_validity & ~inputs.patch_mask
        logits = (2*torch.tanh(self.score(features)/2)).masked_fill(~eligible[..., None], 0)
        # expm1 gives an exactly zero correction at initialization, avoiding
        # log/softmax roundoff that would perturb the reference's initial output.
        shift = prior*torch.expm1(logits)
        mass = shift.sum(-1, keepdim=True)
        difference = (shift-prior*mass)/(1+mass)
        correction = (difference[..., 1:]*increments).sum(-1)
        return correction, prior+difference


class OverlapConfidence(nn.Module):
    """Predict positive copy weights using visible context and evidence agreement."""
    def __init__(self, width=64):
        super().__init__()
        self.context_norm = nn.LayerNorm(width)
        self.score = nn.Sequential(nn.Linear(84, 32), nn.GELU(), nn.Linear(32, 1))
        nn.init.zeros_(self.score[-1].weight);nn.init.zeros_(self.score[-1].bias)

    def forward(self, inputs, router, decoder, points, visible_features, candidates):
        context, visible, alternatives, weights, available, phase, _ = evidence_features(
            inputs, router, decoder, points, visible_features)
        mean = (weights*alternatives).sum(-1, keepdim=True)
        disagreement = (weights*(alternatives-mean).square()).sum(-1, keepdim=True).clamp_min(1e-6).sqrt()
        numerical = (candidates*inputs.query_std[:, :, None, None]+inputs.query_mean[:, :, None, None])[..., None]
        features = torch.cat((self.context_norm(context), visible, phase, alternatives, weights,
                              available.to(context.dtype), numerical, disagreement), -1)
        eligible = inputs.patch_validity & ~inputs.patch_mask
        logits = (2*torch.tanh(self.score(features).squeeze(-1)/2)).masked_fill(~eligible, 0)
        return logits.exp().masked_fill(~inputs.patch_validity, 0)
