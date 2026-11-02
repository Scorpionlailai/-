"""Independent full-lookback temporal context for E22.

The E21 historical and variable branches remain the source of the H/V
messages.  This module is deliberately a separate T operator: it uses the
current lookback window, keeps the nearest reliable left/right anchors as a
local context, and lets every other eligible patch contribute through a
global context.  Its output is a bounded residual applied only at missing
coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .candidates_v2 import TemporalCandidateBatch


def _masked_softmax(scores: Tensor, selected: Tensor) -> Tensor:
    """Softmax over selected candidates, returning exact zero on empty sets."""

    masked = scores.masked_fill(~selected, -torch.inf)
    maximum = masked.max(dim=-1, keepdim=True).values
    maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    numerators = torch.where(
        selected,
        torch.exp(masked - maximum),
        torch.zeros_like(scores),
    )
    return numerators / numerators.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(scores.dtype).tiny
    )


def _nearest_one(absolute_lag: Tensor, selected: Tensor) -> Tensor:
    """Select one nearest item in each target group, if that group is nonempty."""

    sortable = absolute_lag.masked_fill(~selected, torch.inf)
    index = sortable.argmin(dim=-1, keepdim=True)
    result = torch.zeros_like(selected).scatter(-1, index, True)
    return result & selected.any(dim=-1, keepdim=True)


@dataclass(frozen=True)
class DynamicTemporalContextOutput:
    """Temporal residual and the routing diagnostics needed by E22."""

    residual_patches: Tensor
    local_weights: Tensor
    global_weights: Tensor
    gate_weights: Tensor
    local_selected: Tensor
    global_selected: Tensor
    local_available: Tensor
    global_available: Tensor
    local_context: Tensor
    global_context: Tensor


class DynamicTemporalContextRepair(nn.Module):
    """Full-lookback temporal repair with local/global/none routing.

    The operator never uses generated repairs as source candidates.  Source
    values are masked before encoding, and source reliability (``rho``) is a
    score bias rather than a hard ranking signal.  A zero-initialized final
    residual layer makes the newly introduced T path an exact no-op at the
    start of fine-tuning.
    """

    def __init__(
        self,
        d_model: int,
        patch_len: int,
        feature_dim: int,
        *,
        hidden_dim: int | None = None,
        residual_limit: float = 2.0,
    ) -> None:
        super().__init__()
        if d_model <= 0 or patch_len <= 0 or feature_dim <= 0:
            raise ValueError("d_model, patch_len, and feature_dim must be positive")
        if not torch.isfinite(torch.tensor(float(residual_limit))) or residual_limit <= 0:
            raise ValueError("residual_limit must be finite and positive")
        hidden = int(hidden_dim or max(32, d_model * 2))
        self.d_model = int(d_model)
        self.patch_len = int(patch_len)
        self.feature_dim = int(feature_dim)
        self.residual_limit = float(residual_limit)

        self.source_encoder = nn.Sequential(
            nn.Linear(2 * patch_len, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        self.query_encoder = nn.Sequential(
            nn.Linear(d_model + patch_len + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        self.score_head = nn.Sequential(
            nn.Linear(2 * d_model + feature_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.context_gate = nn.Sequential(
            nn.Linear(3 * d_model + 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3),
        )
        self.residual_head = nn.Sequential(
            nn.Linear(2 * d_model + patch_len, hidden),
            nn.GELU(),
            nn.Linear(hidden, patch_len),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)
        nn.init.zeros_(self.score_head[-1].weight)
        nn.init.zeros_(self.score_head[-1].bias)
        nn.init.zeros_(self.context_gate[-1].weight)
        nn.init.zeros_(self.context_gate[-1].bias)

    def _gather_sources(self, patches: Tensor, source_patch: Tensor) -> Tensor:
        batch, channels, patch_count, patch_len = patches.shape
        source_index = source_patch.to(device=patches.device, dtype=torch.long)
        source_index = source_index.unsqueeze(-1).expand(*source_index.shape, patch_len)
        expanded = patches.unsqueeze(2).expand(
            batch, channels, patch_count, patch_count, patch_len
        )
        return torch.gather(expanded, 3, source_index)

    def forward(
        self,
        hv_latents: Tensor,
        hv_patches: Tensor,
        patch_values: Tensor,
        patch_mask: Tensor,
        patch_validity: Tensor,
        rho: Tensor,
        candidates: TemporalCandidateBatch,
        damaged: Tensor,
    ) -> DynamicTemporalContextOutput:
        """Produce a bounded residual for the missing coordinates."""

        if hv_latents.ndim != 4:
            raise ValueError("hv_latents must have shape [B,C,N,D]")
        batch, channels, patch_count, d_model = hv_latents.shape
        expected_patch_shape = (batch, channels, patch_count, self.patch_len)
        for name, tensor in (
            ("hv_patches", hv_patches),
            ("patch_values", patch_values),
            ("patch_mask", patch_mask),
            ("patch_validity", patch_validity),
        ):
            if tensor.shape != expected_patch_shape:
                raise ValueError(f"{name} must have shape {expected_patch_shape}")
        if d_model != self.d_model:
            raise ValueError("hv_latents last dimension does not match d_model")
        if rho.shape != (batch, channels, patch_count):
            raise ValueError("rho must have shape [B,C,N]")
        if damaged.shape != rho.shape:
            raise ValueError("damaged must have shape [B,C,N]")
        if candidates.eligible.shape[:3] != (batch, channels, patch_count):
            raise ValueError("candidate batch shape does not match the inputs")
        if candidates.features.shape[-1] != self.feature_dim:
            raise ValueError("candidate feature dimension does not match feature_dim")

        observed = patch_mask.to(torch.bool) & patch_validity.to(torch.bool)
        target_ratio = observed.to(hv_latents.dtype).mean(dim=-1, keepdim=True)
        query_input = torch.cat(
            (hv_latents, hv_patches * observed.to(hv_patches.dtype), target_ratio),
            dim=-1,
        )
        query = self.query_encoder(query_input)

        source_values = self._gather_sources(patch_values, candidates.source_patch)
        source_mask = self._gather_sources(
            patch_mask.to(torch.bool), candidates.source_patch
        )
        source_validity = self._gather_sources(
            patch_validity.to(torch.bool), candidates.source_patch
        )
        source_observed = source_mask & source_validity
        source_input = torch.cat(
            (
                source_values * source_observed.to(source_values.dtype),
                source_observed.to(source_values.dtype),
            ),
            dim=-1,
        )
        source_encoded = self.source_encoder(source_input)

        candidate_features = candidates.features.to(
            device=hv_latents.device, dtype=hv_latents.dtype
        )
        query_for_score = query.unsqueeze(-2).expand_as(source_encoded)
        score_input = torch.cat(
            (query_for_score, source_encoded, candidate_features), dim=-1
        )
        scores = self.score_head(score_input).squeeze(-1)
        source_rho = self._gather_sources(rho.unsqueeze(-1), candidates.source_patch).squeeze(-1)
        scores = scores + source_rho.clamp_min(torch.finfo(scores.dtype).tiny).log()

        candidate_eligible = candidates.eligible.to(
            device=hv_latents.device, dtype=torch.bool
        )
        candidate_eligible = candidate_eligible & source_validity.any(dim=-1)
        candidate_eligible = candidate_eligible & damaged.to(
            device=hv_latents.device, dtype=torch.bool
        ).unsqueeze(-1)
        signed_lag = candidates.signed_patch_lag.to(device=hv_latents.device)
        is_local = candidates.is_local.to(device=hv_latents.device, dtype=torch.bool)
        local_left = candidate_eligible & is_local & signed_lag.lt(0)
        local_right = candidate_eligible & is_local & signed_lag.gt(0)
        absolute_lag = signed_lag.abs().to(dtype=hv_latents.dtype)
        local_selected = _nearest_one(absolute_lag, local_left) | _nearest_one(
            absolute_lag, local_right
        )
        global_selected = candidate_eligible & ~local_selected

        local_weights = _masked_softmax(scores, local_selected)
        global_weights = _masked_softmax(scores, global_selected)
        local_context = (local_weights.unsqueeze(-1) * source_encoded).sum(dim=-2)
        global_context = (global_weights.unsqueeze(-1) * source_encoded).sum(dim=-2)
        local_available = local_selected.any(dim=-1)
        global_available = global_selected.any(dim=-1)

        gate_input = torch.cat(
            (
                query,
                local_context,
                global_context,
                local_available.to(query.dtype).unsqueeze(-1),
                global_available.to(query.dtype).unsqueeze(-1),
            ),
            dim=-1,
        )
        gate_logits = self.context_gate(gate_input)
        gate_available = torch.stack(
            (
                torch.ones_like(local_available),
                local_available,
                global_available,
            ),
            dim=-1,
        )
        gate_weights = _masked_softmax(gate_logits, gate_available)
        observed_patches = hv_patches * observed.to(hv_patches.dtype)
        local_raw = self.residual_head(
            torch.cat((query, local_context, observed_patches), dim=-1)
        )
        global_raw = self.residual_head(
            torch.cat((query, global_context, observed_patches), dim=-1)
        )
        residual = self.residual_limit * (
            gate_weights[..., 1:2] * torch.tanh(local_raw)
            + gate_weights[..., 2:3] * torch.tanh(global_raw)
        )
        missing = (
            ~patch_mask.to(torch.bool)
            & patch_validity.to(torch.bool)
            & damaged.to(device=hv_latents.device, dtype=torch.bool).unsqueeze(-1)
        )
        residual = residual.masked_fill(~missing, 0.0)
        return DynamicTemporalContextOutput(
            residual_patches=residual,
            local_weights=local_weights,
            global_weights=global_weights,
            gate_weights=gate_weights,
            local_selected=local_selected,
            global_selected=global_selected,
            local_available=local_available,
            global_available=global_available,
            local_context=local_context,
            global_context=global_context,
        )


__all__ = ["DynamicTemporalContextOutput", "DynamicTemporalContextRepair"]
