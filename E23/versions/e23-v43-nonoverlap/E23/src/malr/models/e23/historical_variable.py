"""Cross-variable evidence at H prototype starts in the train-only archive."""

from __future__ import annotations

from math import sqrt

import torch
from torch import Tensor, nn

from .contracts import ExpertEvidence, ExpertInputs


class HistoricalVariableExpert(nn.Module):
    def __init__(
        self, variable_count: int, patch_length: int, d_model: int,
        *, topk: int, archive: Tensor, bounded_values: bool = False,
    ) -> None:
        super().__init__()
        if min(variable_count, patch_length, d_model, topk) < 1:
            raise ValueError("historical variable geometry must be positive")
        if archive.ndim != 2 or archive.shape[1] != variable_count or archive.shape[0] < patch_length:
            raise ValueError("historical archive must have shape [train_time, variables]")
        if not torch.isfinite(archive).all():
            raise ValueError("historical archive contains nonfinite values")
        self.variable_count = variable_count
        self.patch_length = patch_length
        self.d_model = d_model
        self.topk = min(topk, variable_count - 1)
        self.bounded_values = bounded_values
        self.register_buffer("archive", archive.detach().clone(), persistent=False)
        self.source_embedding = nn.Embedding(variable_count, d_model)
        self.target_embedding = nn.Embedding(variable_count, d_model)
        self.message_projector = nn.Sequential(
            nn.Linear(2 * patch_length + d_model + 2, d_model),
            nn.GELU(), nn.Linear(d_model, d_model),
        )
        self.quality_projector = nn.Sequential(nn.Linear(3, d_model), nn.Tanh())

    def _selected_sources(self, *, epoch: int, explore: bool) -> tuple[Tensor, Tensor]:
        relation = self.source_embedding.weight @ self.target_embedding.weight.T
        relation = relation / sqrt(self.d_model)
        ranked = torch.argsort(
            relation.T.masked_fill(
                torch.eye(self.variable_count, dtype=torch.bool, device=relation.device),
                -torch.inf,
            ), dim=-1, descending=True, stable=True,
        )
        source = ranked[:, :self.topk].clone()
        if explore and self.variable_count - 1 > self.topk:
            other = ranked[:, self.topk:self.variable_count - 1]
            target = torch.arange(self.variable_count, device=relation.device)
            source[:, -1] = other.gather(1, ((epoch + target) % other.shape[1])[:, None]).squeeze(1)
        return source, relation.T.gather(1, source)

    def forward(
        self, inputs: ExpertInputs, historical: ExpertEvidence,
        *, epoch: int, explore: bool,
    ) -> ExpertEvidence:
        base = inputs.base_latents
        if base.ndim != 4 or base.shape[1] != self.variable_count or base.shape[-1] != self.d_model:
            raise ValueError("historical variable base shape mismatch")
        if epoch < 0:
            raise ValueError("epoch must be nonnegative")
        batch, variables, patches, _ = base.shape
        starts = historical.provenance["candidate_starts"]
        h_valid = historical.provenance["candidate_valid"].to(torch.bool)
        h_raw = historical.provenance["candidate_raw"]
        h_weights = historical.provenance["weights"]
        if starts.shape != h_valid.shape or h_raw.shape != (*starts.shape, self.patch_length):
            raise ValueError("H provenance geometry mismatch")
        if h_weights.shape != starts.shape or starts.shape[:3] != (batch, variables, patches):
            raise ValueError("H candidate weights shape mismatch")
        kh = starts.shape[-1]
        zeros = torch.zeros_like(base)
        empty = torch.zeros((batch, variables, patches), dtype=torch.bool, device=base.device)
        if self.topk == 0:
            return ExpertEvidence(zeros, zeros, empty, {
                "candidate_count": torch.zeros_like(empty, dtype=torch.long),
            })
        query_starts = inputs.patch_starts
        if query_starts.shape == (batch, patches):
            query_starts = query_starts[:, None, :].expand(batch, variables, patches)
        if query_starts.shape != (batch, variables, patches):
            raise ValueError("query patch starts shape mismatch")
        anchor_valid = (
            h_valid & historical.available[..., None] & inputs.damaged[..., None]
            & (starts >= 0)
            & (starts + self.patch_length <= query_starts[..., :1, None])
            & (starts + self.patch_length <= self.archive.shape[0])
        )
        positions = starts.clamp(0, self.archive.shape[0] - self.patch_length)[..., None]
        positions = positions + torch.arange(self.patch_length, device=base.device)
        target_ix = torch.arange(variables, device=base.device)[None, :, None, None, None]
        aligned_target = self.archive[positions, target_ix]
        if anchor_valid.any() and not torch.allclose(
            aligned_target[anchor_valid], h_raw[anchor_valid], atol=1e-4, rtol=1e-4,
        ):
            raise ValueError("H prototype target differs from aligned training archive")
        sources, prior = self._selected_sources(epoch=epoch, explore=explore)
        kv = self.topk
        source_ix = sources[None, :, None, None, None, :]
        source_raw = self.archive[positions[..., None], source_ix]
        source_raw = source_raw.permute(0, 1, 2, 3, 5, 4)
        source_mean = inputs.query_mean[:, sources][:, :, None, None, :, None]
        source_std = inputs.query_std[:, sources][:, :, None, None, :, None]
        source_values = (source_raw - source_mean) / source_std
        target_values = (h_raw - inputs.query_mean[:, :, None, None, None]) / inputs.query_std[:, :, None, None, None]
        if self.bounded_values:
            source_values = 8 * torch.tanh(source_values / 8)
            target_values = 8 * torch.tanh(target_values / 8)
        candidate_valid = anchor_valid[..., None].expand(batch, variables, patches, kh, kv)
        prior_full = prior[None, :, None, None, :].expand_as(candidate_valid)
        h_weight_full = h_weights[..., None].expand_as(candidate_valid)
        features = torch.cat((
            source_values,
            target_values[..., None, :].expand_as(source_values),
            base[..., None, None, :].expand(batch, variables, patches, kh, kv, self.d_model),
            prior_full[..., None], h_weight_full[..., None],
        ), dim=-1)
        score = (prior_full + h_weight_full.clamp_min(1e-8).log()).masked_fill(~candidate_valid, -1e9)
        weights = torch.softmax(score.flatten(-2), dim=-1).reshape_as(score)
        weights = weights * candidate_valid
        weights = weights / weights.sum(dim=(-1, -2), keepdim=True).clamp_min(1e-8)
        message = (self.message_projector(features) * weights[..., None]).sum(dim=(-3, -2))
        correlation = torch.exp(-(source_values - target_values[..., None, :]).abs().mean(-1))
        quality_raw = torch.stack((prior_full, h_weight_full, correlation), dim=-1)
        quality = self.quality_projector((quality_raw * weights[..., None]).sum(dim=(-3, -2)))
        available = candidate_valid.any(dim=(-1, -2))
        return ExpertEvidence(
            torch.where(available[..., None], message, zeros),
            torch.where(available[..., None], quality, zeros), available,
            {"source_variables": sources.detach(),
             "candidate_starts": starts.detach(),
             "candidate_valid": candidate_valid.detach(),
             "candidate_count": candidate_valid.sum(dim=(-1, -2)).detach(),
             "source_values": source_values.detach().masked_fill(~candidate_valid[..., None], 0),
             "weights": weights.detach()},
        )


class HistoricalVariableAdapter(nn.Module):
    """Zero-gated historical correction after the selected flat H/V/T router."""

    def __init__(
        self, variable_count: int, patch_length: int, d_model: int,
        *, topk: int, archive: Tensor, bounded_values: bool = False,
    ) -> None:
        super().__init__()
        self.expert = HistoricalVariableExpert(
            variable_count, patch_length, d_model, topk=topk, archive=archive,
            bounded_values=bounded_values,
        )
        self.residual = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model),
        )
        self.gates = nn.Parameter(torch.zeros(variable_count))

    def forward(
        self, routed_latents: Tensor, inputs: ExpertInputs, historical: ExpertEvidence,
        *, epoch: int, explore: bool,
    ) -> tuple[Tensor, ExpertEvidence]:
        evidence = self.expert(inputs, historical, epoch=epoch, explore=explore)
        if routed_latents.shape != evidence.message.shape:
            raise ValueError("historical adapter routed latent shape mismatch")
        proposed = self.residual(torch.cat((routed_latents, evidence.message), dim=-1))
        scale = (0.1 * torch.tanh(self.gates))[None, :, None, None]
        correction = proposed * scale * evidence.available[..., None]
        return correction, evidence


__all__ = ["HistoricalVariableExpert", "HistoricalVariableAdapter"]
