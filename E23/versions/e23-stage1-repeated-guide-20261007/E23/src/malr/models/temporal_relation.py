"""Dynamic local and multiscale temporal relation messages."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .candidates_v2 import TemporalCandidateBatch
from .relation_outputs import RelationBranchOutput, TemporalAggregationPlan


@dataclass(frozen=True)
class TemporalSelection:
    """Hard temporal edge masks and the interpolation regime for each target."""

    local_selected: Tensor
    global_selected: Tensor
    local_mode: Tensor


def _best_in_group(score: Tensor, group: Tensor) -> Tensor:
    """Select a stable best item from each final-axis group, if it is nonempty."""

    if score.shape[-1] == 0:
        return torch.zeros_like(group)
    safe_score = score.masked_fill(~group, -torch.inf)
    index = safe_score.argmax(dim=-1, keepdim=True)
    selected = torch.zeros_like(group).scatter(-1, index, True)
    return selected & group.any(dim=-1, keepdim=True)


def _quota_global(
    score: Tensor,
    eligible: Tensor,
    direction: Tensor,
    band: Tensor,
    band_count: int,
) -> Tensor:
    """Reserve at most one long-range edge per direction and distance band."""

    selected = torch.zeros_like(eligible)
    for direction_value in (-1, 1):
        for band_value in range(1, band_count):
            group = eligible & direction.eq(direction_value) & band.eq(band_value)
            selected |= _best_in_group(score, group)
    return selected


def _nearest(
    absolute_offset: Tensor, group: Tensor, count: int
) -> tuple[Tensor, Tensor]:
    """Return the ``count`` nearest eligible candidates and whether enough exist."""

    if count <= 0:
        raise ValueError("count must be positive")
    if absolute_offset.shape[-1] == 0:
        return torch.zeros_like(group), torch.zeros_like(group.any(dim=-1))
    sortable = absolute_offset.masked_fill(~group, torch.inf)
    order = torch.argsort(sortable, dim=-1, stable=True)
    ranked_valid = group.gather(-1, order)
    rank = torch.arange(group.shape[-1], device=group.device).view(
        *((1,) * (group.ndim - 1)), -1
    )
    chosen = ranked_valid & rank.lt(count)
    selected = torch.zeros_like(group).scatter(-1, order, chosen)
    return selected, group.sum(dim=-1).ge(count)


def select_temporal_candidates(
    candidates: TemporalCandidateBatch,
    predicted_gain: Tensor,
    candidate_gain_threshold: float = 0.0,
    *,
    global_score: Tensor | None = None,
    distance_band_count: int | None = None,
) -> TemporalSelection:
    """Select local anchors and quota-preserving long-range candidates.

    Local selection is geometric rather than score-ranked.  Positive-gain
    anchors on both sides yield bilateral interpolation; otherwise two anchors
    on one side yield a local extrapolation/backcast.
    """

    if predicted_gain.shape != candidates.eligible.shape:
        raise ValueError("predicted_gain must have shape [B,C,N,K]")
    if not predicted_gain.is_floating_point() or not torch.isfinite(predicted_gain).all():
        raise ValueError("predicted_gain must be finite and floating-point")
    if not isfinite(candidate_gain_threshold):
        raise ValueError("candidate_gain_threshold must be finite")

    if global_score is not None:
        if global_score.shape != predicted_gain.shape or not torch.isfinite(global_score).all():
            raise ValueError("global_score must be finite with shape [B,C,N,K]")
    eligible = candidates.eligible.to(device=predicted_gain.device, dtype=torch.bool)
    direction = candidates.direction.to(device=predicted_gain.device)
    band = candidates.distance_band.to(device=predicted_gain.device)
    is_local = candidates.is_local.to(device=predicted_gain.device, dtype=torch.bool)
    offset = candidates.signed_time_offset.to(
        device=predicted_gain.device, dtype=predicted_gain.dtype
    )
    admissible = eligible & predicted_gain.gt(candidate_gain_threshold)
    local = admissible & is_local
    left = local & direction.lt(0)
    right = local & direction.gt(0)
    absolute_offset = offset.abs()
    nearest_left, has_left = _nearest(absolute_offset, left, 1)
    nearest_right, has_right = _nearest(absolute_offset, right, 1)
    left_two, has_left_two = _nearest(absolute_offset, left, 2)
    right_two, has_right_two = _nearest(absolute_offset, right, 2)

    bilateral = has_left & has_right
    left_extrapolation = ~has_right & has_left_two
    right_backcast = ~has_left & has_right_two
    local_selected = (
        (bilateral.unsqueeze(-1) & (nearest_left | nearest_right))
        | (left_extrapolation.unsqueeze(-1) & left_two)
        | (right_backcast.unsqueeze(-1) & right_two)
    )
    local_mode = torch.where(
        bilateral,
        torch.ones_like(has_left, dtype=torch.long),
        torch.where(
            left_extrapolation,
            torch.full_like(has_left, 2, dtype=torch.long),
            torch.where(
                right_backcast,
                torch.full_like(has_left, 3, dtype=torch.long),
                torch.zeros_like(has_left, dtype=torch.long),
            ),
        ),
    )
    band_count = (
        int(distance_band_count)
        if distance_band_count is not None
        else int(band.max().item()) + 1 if band.numel() else 1
    )
    if band_count <= 1:
        raise ValueError("distance_band_count must be greater than one")
    if band.numel() and (band.lt(0).any() or band.ge(band_count).any()):
        raise ValueError("candidate distance bands must lie in distance_band_count")
    global_selected = _quota_global(
        predicted_gain if global_score is None else global_score,
        admissible & ~is_local,
        direction,
        band,
        band_count,
    )
    return TemporalSelection(local_selected, global_selected, local_mode)


def _masked_softmax(score: Tensor, selected: Tensor) -> Tensor:
    """Finite softmax that returns exact zeros for an empty selected set."""

    masked = score.masked_fill(~selected, -torch.inf)
    maximum = masked.max(dim=-1, keepdim=True).values
    maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    numerators = torch.where(selected, torch.exp(masked - maximum), torch.zeros_like(score))
    return numerators / numerators.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(score.dtype).tiny
    )


def _selected_mean(values: Tensor, selected: Tensor) -> Tensor:
    weights = selected.to(dtype=values.dtype)
    return (weights * values).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0)


class DynamicTemporalRelation(nn.Module):
    """Build target-space messages from local anchors and multiscale edges."""

    def __init__(
        self,
        d_model: int,
        feature_dim: int,
        distance_band_count: int,
        candidate_gain_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model <= 0 or feature_dim <= 0 or distance_band_count <= 1:
            raise ValueError("d_model, feature_dim, and distance_band_count must be positive")
        if not isfinite(candidate_gain_threshold):
            raise ValueError("candidate_gain_threshold must be finite")
        self.d_model = d_model
        self.feature_dim = feature_dim
        self.distance_band_count = distance_band_count
        self.candidate_gain_threshold = float(candidate_gain_threshold)
        hidden = max(d_model, feature_dim)
        self.candidate_projector = nn.Sequential(
            nn.Linear(d_model + feature_dim, hidden), nn.GELU(), nn.Linear(hidden, d_model)
        )
        edge_width = 2 * d_model + feature_dim + 2
        self.gain_head = nn.Sequential(
            nn.Linear(edge_width, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        self.uncertainty_head = nn.Sequential(
            nn.Linear(edge_width, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear(3 * d_model + 4, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        # A positive initial gain lets valid edges exercise both message paths;
        # learned edge evidence remains responsible for later admission.
        nn.init.zeros_(self.gain_head[-1].weight)
        nn.init.constant_(self.gain_head[-1].bias, 1.0)

    def _validate(
        self, nodes: Tensor, candidates: TemporalCandidateBatch, rho: Tensor
    ) -> None:
        if nodes.ndim != 4 or nodes.shape[-1] != self.d_model:
            raise ValueError("nodes must have shape [B,C,N,D]")
        if rho.shape != nodes.shape[:-1]:
            raise ValueError("rho must have shape [B,C,N]")
        expected = (*nodes.shape[:-1], candidates.eligible.shape[-1])
        if candidates.eligible.shape != expected:
            raise ValueError("candidate tensors must have shape [B,C,N,K]")
        if candidates.features.shape != (*expected, self.feature_dim):
            raise ValueError("features must have shape [B,C,N,K,feature_dim]")
        metadata = (
            candidates.source_patch,
            candidates.signed_patch_lag,
            candidates.signed_time_offset,
            candidates.direction,
            candidates.distance_band,
            candidates.is_local,
        )
        if any(value.shape != expected for value in metadata):
            raise ValueError("candidate metadata must have shape [B,C,N,K]")
        if candidates.candidate_count.shape != nodes.shape[:-1]:
            raise ValueError("candidate_count must have shape [B,C,N]")
        if not nodes.is_floating_point() or not rho.is_floating_point():
            raise TypeError("nodes and rho must be floating-point tensors")
        if not torch.isfinite(nodes).all() or not torch.isfinite(rho).all():
            raise ValueError("nodes and rho must be finite")
        if not torch.isfinite(candidates.features).all():
            raise ValueError("candidate features must be finite")

    def project_candidate_messages(
        self, nodes: Tensor, candidates: TemporalCandidateBatch
    ) -> Tensor:
        """Project temporal source candidates into their target representation."""

        source_index = candidates.source_patch.to(device=nodes.device, dtype=torch.long)
        source = nodes.unsqueeze(2).expand(-1, -1, nodes.shape[2], -1, -1).gather(
            3, source_index.unsqueeze(-1).expand(-1, -1, -1, -1, self.d_model)
        )
        features = candidates.features.to(device=nodes.device, dtype=nodes.dtype)
        return self.candidate_projector(torch.cat((source, features), dim=-1))

    def _fuse_message(
        self,
        nodes: Tensor,
        local_message: Tensor,
        global_message: Tensor,
        plan: TemporalAggregationPlan,
    ) -> tuple[Tensor, Tensor]:
        gate_input = torch.cat(
            (
                nodes,
                local_message,
                global_message,
                plan.local_gain.unsqueeze(-1),
                plan.global_gain.unsqueeze(-1),
                plan.local_uncertainty.unsqueeze(-1),
                plan.global_uncertainty.unsqueeze(-1),
            ),
            dim=-1,
        )
        local_gate = torch.sigmoid(self.fusion_gate(gate_input).squeeze(-1))
        both = plan.local_available & plan.global_available
        message = torch.where(
            both.unsqueeze(-1),
            local_gate.unsqueeze(-1) * local_message
            + (1.0 - local_gate).unsqueeze(-1) * global_message,
            torch.where(
                plan.local_available.unsqueeze(-1),
                local_message,
                torch.where(
                    plan.global_available.unsqueeze(-1),
                    global_message,
                    torch.zeros_like(local_message),
                ),
            ),
        )
        return message, local_gate

    def replay_message(
        self,
        nodes: Tensor,
        candidate_messages: Tensor,
        plan: TemporalAggregationPlan,
    ) -> Tensor:
        """Replay local/global aggregation with a plan and trainable fusion gate."""

        local_message = (plan.local_coefficients.unsqueeze(-1) * candidate_messages).sum(dim=-2)
        local_message = torch.where(
            plan.local_available.unsqueeze(-1), local_message, torch.zeros_like(local_message)
        )
        global_message = (plan.global_coefficients.unsqueeze(-1) * candidate_messages).sum(dim=-2)
        global_message = torch.where(
            plan.global_available.unsqueeze(-1), global_message, torch.zeros_like(global_message)
        )
        return self._fuse_message(nodes, local_message, global_message, plan)[0]

    def forward_with_plan(
        self, nodes: Tensor, candidates: TemporalCandidateBatch, rho: Tensor
    ) -> tuple[RelationBranchOutput, TemporalAggregationPlan]:
        self._validate(nodes, candidates, rho)
        width = candidates.eligible.shape[-1]
        source_index = candidates.source_patch.to(device=nodes.device, dtype=torch.long)
        source_rho = rho.unsqueeze(2).expand(-1, -1, nodes.shape[2], -1).gather(
            3, source_index
        )
        target_rho = rho.unsqueeze(-1).expand_as(source_rho)
        features = candidates.features.to(device=nodes.device, dtype=nodes.dtype)
        candidate_messages = self.project_candidate_messages(nodes, candidates)
        edge_input = torch.cat(
            (
                nodes.unsqueeze(-2).expand_as(candidate_messages),
                candidate_messages,
                features,
                target_rho.unsqueeze(-1),
                source_rho.unsqueeze(-1),
            ),
            dim=-1,
        )
        candidate_gain = self.gain_head(edge_input).squeeze(-1)
        candidate_uncertainty = F.softplus(
            self.uncertainty_head(edge_input).squeeze(-1)
        ) + torch.finfo(nodes.dtype).eps
        selection = select_temporal_candidates(
            candidates,
            candidate_gain,
            self.candidate_gain_threshold,
            global_score=candidate_gain - candidate_uncertainty,
            distance_band_count=self.distance_band_count,
        )
        local_available = selection.local_mode.ne(0)
        global_available = selection.global_selected.any(dim=-1)

        offset = candidates.signed_time_offset.to(device=nodes.device, dtype=nodes.dtype)
        local_weight = selection.local_selected.to(dtype=nodes.dtype) * source_rho
        local_weight = local_weight / offset.abs().clamp_min(1.0)
        local_weight = local_weight / local_weight.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(nodes.dtype).tiny
        )
        bilateral_message = (local_weight.unsqueeze(-1) * candidate_messages).sum(dim=-2)
        local_distance = offset.abs().masked_fill(~selection.local_selected, torch.inf)
        local_order = torch.argsort(local_distance, dim=-1, stable=True)
        near_index = local_order[..., :1]
        far_index = local_order[..., 1:2] if width > 1 else near_index
        near = candidate_messages.gather(
            -2, near_index.unsqueeze(-1).expand(-1, -1, -1, -1, self.d_model)
        ).squeeze(-2)
        far = candidate_messages.gather(
            -2, far_index.unsqueeze(-1).expand(-1, -1, -1, -1, self.d_model)
        ).squeeze(-2)
        near_time = offset.gather(-1, near_index).squeeze(-1)
        far_time = offset.gather(-1, far_index).squeeze(-1)
        alpha = near_time.abs() / (near_time - far_time).abs().clamp_min(1.0)
        alpha = alpha.clamp(0.0, 2.0)
        one_sided_message = near + alpha.unsqueeze(-1) * (near - far)
        local_message = torch.where(
            selection.local_mode.eq(1).unsqueeze(-1), bilateral_message, one_sided_message
        )
        local_message = torch.where(
            local_available.unsqueeze(-1), local_message, torch.zeros_like(local_message)
        )
        one_sided_coefficients = torch.zeros_like(local_weight)
        one_sided_coefficients = one_sided_coefficients.scatter_add(
            -1, near_index, (1.0 + alpha).unsqueeze(-1)
        )
        one_sided_coefficients = one_sided_coefficients.scatter_add(
            -1, far_index, (-alpha).unsqueeze(-1)
        )
        local_coefficients = torch.where(
            selection.local_mode.eq(1).unsqueeze(-1), local_weight, one_sided_coefficients
        )
        local_coefficients = torch.where(
            local_available.unsqueeze(-1), local_coefficients, torch.zeros_like(local_coefficients)
        )
        local_gain = _selected_mean(candidate_gain, selection.local_selected)
        local_uncertainty = _selected_mean(candidate_uncertainty, selection.local_selected)

        global_score = candidate_gain - candidate_uncertainty
        global_coefficients = _masked_softmax(global_score, selection.global_selected)
        global_message = (global_coefficients.unsqueeze(-1) * candidate_messages).sum(dim=-2)
        global_message = torch.where(
            global_available.unsqueeze(-1), global_message, torch.zeros_like(global_message)
        )
        global_gain = (global_coefficients * candidate_gain).sum(dim=-1)
        global_uncertainty = (global_coefficients * candidate_uncertainty).sum(dim=-1)
        available = local_available | global_available
        plan = TemporalAggregationPlan(
            local_coefficients=local_coefficients,
            global_coefficients=global_coefficients,
            local_selected=selection.local_selected,
            global_selected=selection.global_selected,
            local_mode=selection.local_mode,
            local_available=local_available,
            global_available=global_available,
            local_gain=local_gain,
            global_gain=global_gain,
            local_uncertainty=local_uncertainty,
            global_uncertainty=global_uncertainty,
            available=available,
        )
        message, local_gate = self._fuse_message(nodes, local_message, global_message, plan)
        both = local_available & global_available
        gain = torch.where(
            both,
            local_gate * local_gain + (1.0 - local_gate) * global_gain,
            torch.where(local_available, local_gain, torch.where(global_available, global_gain, torch.zeros_like(local_gain))),
        )
        uncertainty = torch.where(
            both,
            local_gate * local_uncertainty + (1.0 - local_gate) * global_uncertainty,
            torch.where(local_available, local_uncertainty, torch.where(global_available, global_uncertainty, torch.ones_like(local_uncertainty))),
        )
        selected = selection.local_selected | selection.global_selected
        output = RelationBranchOutput(
            message=message,
            predicted_gain=gain,
            uncertainty=uncertainty,
            available=available,
            candidate_messages=candidate_messages,
            candidate_predicted_gain=candidate_gain,
            candidate_uncertainty=candidate_uncertainty,
            candidate_eligible=candidates.eligible.to(device=nodes.device, dtype=torch.bool),
            candidate_selected=selected,
            diagnostics={
                "local_selected": selection.local_selected,
                "global_selected": selection.global_selected,
                "local_mode": selection.local_mode,
                "local_available": local_available,
                "global_available": global_available,
            },
        )
        return output, plan

    def forward(
        self, nodes: Tensor, candidates: TemporalCandidateBatch, rho: Tensor
    ) -> RelationBranchOutput:
        output, _ = self.forward_with_plan(nodes, candidates, rho)
        return output


__all__ = [
    "DynamicTemporalRelation",
    "TemporalSelection",
    "select_temporal_candidates",
]
