"""Reconcile routed H/V/T evidence across time and variables before decoding."""

from __future__ import annotations

from math import log

import torch
from torch import Tensor, nn

from .contracts import RouterOutput


class EvidenceAwareFusionRefiner(nn.Module):
    """One identity-initialized, reliability-gated factorized context step."""

    def __init__(self, d_model: int, patch_count: int, *, heads: int = 4) -> None:
        super().__init__()
        if d_model < 1 or patch_count < 1 or heads < 1 or d_model % heads:
            raise ValueError("post-fusion refiner dimensions are invalid")
        self.d_model = d_model
        self.patch_count = patch_count
        self.input_norm = nn.LayerNorm(d_model)
        self.variable_norm = nn.LayerNorm(d_model)
        self.feed_forward_norm = nn.LayerNorm(d_model)
        self.feature_projection = nn.Linear(4, d_model, bias=False)
        self.position_embedding = nn.Parameter(torch.zeros(patch_count, d_model))
        self.temporal_attention = nn.MultiheadAttention(
            d_model, heads, dropout=0.0, batch_first=True,
        )
        self.variable_attention = nn.MultiheadAttention(
            d_model, heads, dropout=0.0, batch_first=True,
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Linear(2 * d_model, d_model),
        )
        self.output_projection = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        self.confidence_gate = nn.Linear(4, 1)
        nn.init.zeros_(self.confidence_gate.weight)
        nn.init.constant_(self.confidence_gate.bias, -2.0)

    def forward(self, routed: RouterOutput, visible_fraction: Tensor) -> Tensor:
        latents = routed.latents
        if latents.ndim != 4 or latents.shape[-1] != self.d_model:
            raise ValueError("refiner latents must have shape [B,C,N,D]")
        batch, variables, patches, width = latents.shape
        if patches != self.patch_count or visible_fraction.shape != (batch, variables, patches):
            raise ValueError("refiner patch geometry mismatch")
        if routed.weights.shape != (batch, variables, patches, 4):
            raise ValueError("refiner route weights must have four options")
        if not torch.isfinite(visible_fraction).all() or not torch.all(
            (visible_fraction >= 0) & (visible_fraction <= 1)
        ):
            raise ValueError("refiner visible fraction is invalid")
        expert_deltas = torch.stack(
            [routed.deltas[name] for name in ("historical", "variable", "temporal")],
            dim=-2,
        )
        if expert_deltas.shape != (batch, variables, patches, 3, width):
            raise ValueError("refiner expert delta geometry mismatch")
        expert_weights = routed.weights[..., 1:]
        weight_sum = expert_weights.sum(dim=-1).clamp_min(1e-8)
        mean_delta = (expert_weights[..., None] * expert_deltas).sum(dim=-2)
        mean_delta = mean_delta / weight_sum[..., None]
        variance = (expert_weights[..., None]
                    * (expert_deltas - mean_delta[..., None, :]).square()).sum(dim=(-2, -1))
        disagreement = torch.log1p(variance / (weight_sum * width))
        entropy = -(routed.weights * routed.weights.clamp_min(1e-8).log()).sum(dim=-1)
        entropy = entropy / log(4)
        features = torch.stack((visible_fraction, routed.weights[..., 0],
                                entropy, disagreement), dim=-1)

        enriched = (self.input_norm(latents) + self.feature_projection(features)
                    + self.position_embedding[None, None])
        temporal_tokens = enriched.reshape(batch * variables, patches, width)
        temporal_context, _ = self.temporal_attention(
            temporal_tokens, temporal_tokens, temporal_tokens, need_weights=False,
        )
        hidden = latents + temporal_context.reshape_as(latents)
        variable_tokens = (self.variable_norm(hidden) + self.feature_projection(features))
        variable_tokens = variable_tokens.permute(0, 2, 1, 3).reshape(
            batch * patches, variables, width,
        )
        variable_context, _ = self.variable_attention(
            variable_tokens, variable_tokens, variable_tokens, need_weights=False,
        )
        hidden = hidden + variable_context.reshape(
            batch, patches, variables, width,
        ).permute(0, 2, 1, 3)
        hidden = hidden + self.feed_forward(self.feed_forward_norm(hidden))
        delta = self.output_projection(hidden)
        confidence = torch.sigmoid(self.confidence_gate(features))
        return latents + (1.0 - visible_fraction)[..., None] * confidence * delta


__all__ = ["EvidenceAwareFusionRefiner"]
