"""One-pass source encoding and full-lookback, banded temporal evidence."""

from __future__ import annotations

from math import sqrt

import torch
from torch import Tensor, nn

from .contracts import CandidateEvidence, ExpertEvidence, ExpertInputs


class MultiScaleTemporalExpert(nn.Module):
    """Return one T message from two local and eight global directional slots."""

    def __init__(
        self, patch_length: int, d_model: int, *, local_radius: int = 2,
        boundaries: tuple[int, ...] = (2, 4, 8, 16),
    ) -> None:
        super().__init__()
        if patch_length < 1 or d_model < 1 or local_radius < 1:
            raise ValueError("patch length, width and local radius must be positive")
        if boundaries[0] != local_radius or tuple(sorted(set(boundaries))) != boundaries:
            raise ValueError("boundaries must start at local radius and increase")
        self.patch_length = patch_length
        self.d_model = d_model
        self.local_radius = local_radius
        self.boundaries = boundaries
        self.source_encoder = nn.Sequential(
            nn.Linear(2 * patch_length, d_model), nn.GELU(), nn.Linear(d_model, d_model),
        )
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.message_projector = nn.Sequential(
            nn.Linear(2 * d_model + 4, d_model), nn.GELU(), nn.Linear(d_model, d_model),
        )
        self.quality_projector = nn.Sequential(nn.Linear(5, d_model), nn.Tanh())

    def forward(self, inputs: ExpertInputs) -> ExpertEvidence:
        values, mask, validity = inputs.patch_values, inputs.patch_mask, inputs.patch_validity
        base = inputs.base_latents
        if values.ndim != 4:
            raise ValueError("patch values must have shape [B,C,N,P]")
        batch, variables, patches, length = values.shape
        if length != self.patch_length or mask.shape != values.shape or validity.shape != values.shape:
            raise ValueError("patch value/mask shape mismatch")
        if base.shape != (batch, variables, patches, self.d_model):
            raise ValueError("base latent shape mismatch")
        if inputs.damaged.shape != (batch, variables, patches):
            raise ValueError("damaged shape mismatch")

        observed = mask.to(torch.bool) & validity.to(torch.bool)
        visible = values * observed
        # The encoder is called exactly once per forward on [B,C,N,2P].
        source_encoded = self.source_encoder(torch.cat((visible, observed.to(values.dtype)), dim=-1))
        query = self.query_projection(base)
        key = self.key_projection(source_encoded)
        similarity = torch.einsum("bcnd,bcmd->bcnm", query, key) / sqrt(self.d_model)

        positions = torch.arange(patches, device=values.device)
        separation = positions[:, None] - positions[None, :]
        source_eligible = observed.any(dim=-1)
        eligible = source_eligible[:, :, None, :] & inputs.damaged[..., None].to(torch.bool)
        # Score all ten directional bands in one reduction. The slot order is
        # identical to the scalar loop: five past bands, then five future bands.
        signed = torch.tensor((1, -1), device=values.device)[:, None, None] * separation
        lower = torch.tensor((0, *self.boundaries), device=values.device)
        upper = torch.tensor((*self.boundaries, patches), device=values.device)
        band = ((signed[:, None] > lower[None, :, None, None])
                & (signed[:, None] <= upper[None, :, None, None]))
        band = band.reshape(-1, patches, patches)
        local_flags = torch.tensor(
            (True, *(False for _ in self.boundaries)) * 2,
            device=values.device, dtype=torch.bool,
        )
        allowed = eligible[:, :, None] & band[None, None]
        local_rank = -signed[:, None].expand(-1, len(lower), -1, -1).reshape(
            -1, patches, patches,
        ).to(similarity.dtype)
        rank = torch.where(
            local_flags[None, None, :, None, None],
            local_rank[None, None], similarity[:, :, None],
        ).masked_fill(~allowed, -torch.inf)
        source_indices = rank.argmax(dim=-1).permute(0, 1, 3, 2)
        candidate_valid = allowed.any(dim=-1).permute(0, 1, 3, 2)
        available = candidate_valid.any(dim=-1)
        count = candidate_valid.sum(dim=-1)

        b_index = torch.arange(batch, device=values.device)[:, None, None, None]
        c_index = torch.arange(variables, device=values.device)[None, :, None, None]
        source = source_encoded[b_index, c_index, source_indices]
        source_values = visible[b_index, c_index, source_indices]
        source_mask = observed[b_index, c_index, source_indices]
        target_mask = observed[..., None, :]
        target_values = visible[..., None, :]
        common = source_mask & target_mask
        error = ((source_values - target_values).abs() * common).sum(-1)
        error = error / common.sum(-1).clamp_min(1)
        consistency = torch.exp(-error) * common.any(dim=-1)
        fraction = source_mask.float().mean(dim=-1)
        target_fraction = observed.float().mean(dim=-1)[..., None].expand_as(fraction)
        distance = (source_indices - positions[None, None, :, None]).abs()
        relative_distance = distance.to(values.dtype) / max(patches - 1, 1)
        is_local = local_flags.to(values.dtype)
        is_local = is_local.view(1, 1, 1, -1).expand_as(fraction)

        candidates = torch.cat((
            base[..., None, :].expand_as(source), source,
            fraction[..., None], relative_distance[..., None],
            consistency[..., None], is_local[..., None],
        ), dim=-1)
        candidate_messages = self.message_projector(candidates)
        selected_score = similarity.gather(dim=-1, index=source_indices)
        selected_score = selected_score + 0.5 * fraction - 0.05 * relative_distance
        selected_score = selected_score.masked_fill(~candidate_valid, -1e9)
        weights = torch.softmax(selected_score, dim=-1) * candidate_valid
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        message = (weights[..., None] * candidate_messages).sum(dim=-2)
        quality_features = torch.stack((
            fraction, relative_distance, consistency, target_fraction, is_local,
        ), dim=-1)
        quality = self.quality_projector((weights[..., None] * quality_features).sum(dim=-2))
        zeros = torch.zeros_like(base)
        message = torch.where(available[..., None], message, zeros)
        quality = torch.where(available[..., None], quality, zeros)
        return ExpertEvidence(
            message, quality, available,
            {"source_indices": source_indices.detach(),
             "candidate_valid": candidate_valid.detach(),
             "candidate_count": count.detach(),
             "weights": weights.detach()},
            CandidateEvidence(candidate_messages, quality_features, weights, candidate_valid),
        )


__all__ = ["MultiScaleTemporalExpert"]
