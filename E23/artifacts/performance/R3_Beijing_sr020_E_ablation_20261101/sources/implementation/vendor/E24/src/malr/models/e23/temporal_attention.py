"""Visibility-aware multihead attention over current-window temporal evidence."""

from __future__ import annotations

from math import sqrt

import torch
from torch import Tensor, nn

from .contracts import ExpertEvidence, ExpertInputs


class VisibleTemporalAttentionExpert(nn.Module):
    """Attend from a damaged patch to observed patches of the same variable."""

    def __init__(
        self, patch_length: int, d_model: int, *, heads: int = 4,
        boundaries: tuple[int, ...] = (2, 4, 8, 16),
    ) -> None:
        super().__init__()
        if patch_length < 1 or d_model < 1 or heads < 1 or d_model % heads:
            raise ValueError("T attention needs positive dimensions and divisible head width")
        if boundaries != (2, 4, 8, 16):
            raise ValueError("T attention requires the locked distance boundaries")
        self.patch_length = patch_length
        self.d_model = d_model
        self.heads = heads
        self.head_width = d_model // heads
        self.boundaries = boundaries
        self.source_encoder = nn.Sequential(
            nn.Linear(2 * patch_length, d_model), nn.GELU(), nn.Linear(d_model, d_model),
        )
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.relative_bias = nn.Embedding(2 * (len(boundaries) + 1), heads)
        nn.init.zeros_(self.relative_bias.weight)
        self.output_projection = nn.Linear(d_model, d_model)
        self.message_projector = nn.Sequential(
            nn.Linear(2 * d_model + 4, d_model), nn.GELU(), nn.Linear(d_model, d_model),
        )
        self.quality_projector = nn.Sequential(nn.Linear(5, d_model), nn.Tanh())

    def forward(self, inputs: ExpertInputs) -> ExpertEvidence:
        values, mask, validity = inputs.patch_values, inputs.patch_mask, inputs.patch_validity
        base = inputs.base_latents
        if values.ndim != 4:
            raise ValueError("T patch values must have shape [B,C,N,P]")
        batch, variables, patches, length = values.shape
        if (length != self.patch_length or mask.shape != values.shape
                or validity.shape != values.shape
                or base.shape != (batch, variables, patches, self.d_model)
                or inputs.damaged.shape != (batch, variables, patches)):
            raise ValueError("T attention input shape mismatch")
        observed = mask.to(torch.bool) & validity.to(torch.bool)
        visible = values.masked_fill(~observed, 0)
        source = self.source_encoder(torch.cat((visible, observed.to(values.dtype)), dim=-1))
        query = self.query_projection(base).reshape(
            batch, variables, patches, self.heads, self.head_width,
        )
        key = self.key_projection(source).reshape(
            batch, variables, patches, self.heads, self.head_width,
        )
        value = self.value_projection(source).reshape(
            batch, variables, patches, self.heads, self.head_width,
        )
        similarity = torch.einsum("bcnhd,bcmhd->bchnm", query, key) / sqrt(self.head_width)

        positions = torch.arange(patches, device=values.device)
        signed_distance = positions[None, :] - positions[:, None]
        distance = signed_distance.abs()
        relative_distance = distance.to(values.dtype) / max(patches - 1, 1)
        bucket = torch.bucketize(
            distance, torch.tensor(self.boundaries, device=values.device), right=False,
        )
        bucket = bucket + (signed_distance > 0).long() * (len(self.boundaries) + 1)
        relative_bias = self.relative_bias(bucket).permute(2, 0, 1)

        valid_count = validity.to(torch.bool).sum(dim=-1).clamp_min(1)
        source_fraction = observed.sum(dim=-1).to(values.dtype) / valid_count
        common = observed[..., :, None, :] & observed[..., None, :, :]
        overlap_count = common.sum(dim=-1)
        overlap_error = ((visible[..., :, None, :] - visible[..., None, :, :]).abs()
                         * common).sum(dim=-1) / overlap_count.clamp_min(1)
        consistency = torch.exp(-overlap_error) * overlap_count.gt(0)
        eligible = (
            observed.any(dim=-1)[:, :, None, :]
            & inputs.damaged.to(torch.bool)[..., None]
            & distance.gt(0)[None, None]
        )
        logits = (similarity + relative_bias[None, None]
                  + 0.5 * source_fraction[:, :, None, None, :]
                  + 0.25 * consistency[:, :, None]
                  - 0.05 * relative_distance[None, None, None])
        logits = logits.masked_fill(~eligible[:, :, None], -1e9)
        weights = torch.softmax(logits, dim=-1) * eligible[:, :, None]
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        attended = torch.einsum("bchnm,bcmhd->bcnhd", weights, value)
        context = self.output_projection(attended.reshape(batch, variables, patches, self.d_model))

        mean_weights = weights.mean(dim=2)
        source_fraction_at_target = source_fraction[:, :, None, :].expand(
            batch, variables, patches, patches,
        )
        distance_at_target = relative_distance[None, None].expand_as(source_fraction_at_target)
        target_fraction = source_fraction[..., None].expand_as(source_fraction_at_target)
        local = distance.le(self.boundaries[0]).to(values.dtype)[None, None].expand_as(
            source_fraction_at_target,
        )
        evidence_features = torch.stack((source_fraction_at_target, distance_at_target,
                                         consistency, target_fraction, local), dim=-1)
        summary = (mean_weights[..., None] * evidence_features).sum(dim=-2)
        message = self.message_projector(torch.cat((base, context, summary[..., :4]), dim=-1))
        quality = self.quality_projector(summary)
        available = eligible.any(dim=-1)
        zeros = torch.zeros_like(base)
        message = torch.where(available[..., None], message, zeros)
        quality = torch.where(available[..., None], quality, zeros)
        source_indices = positions.view(1, 1, 1, patches).expand(
            batch, variables, patches, patches,
        )
        return ExpertEvidence(
            message, quality, available,
            {"source_indices": source_indices.detach(),
             "candidate_valid": eligible.detach(),
             "candidate_count": eligible.sum(dim=-1).detach(),
             "weights": mean_weights.detach()},
        )


__all__ = ["VisibleTemporalAttentionExpert"]
