from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from malr.data.patching import PATCH_OBSERVED
from malr.prototypes.bank import GroupedPrototypeBank, PrototypeBank
from malr.prototypes.retrieval import MaskAwareRetrievalEncoder

from .filtering import EdgeConfidenceFilter


@dataclass(frozen=True)
class RelationSelection:
    message: Tensor
    weights: Tensor
    selected: Tensor
    candidate_count: Tensor
    pool_count: Tensor | None = None
    filter_probabilities: Tensor | None = None
    hard_gate: Tensor | None = None
    retained_count: Tensor | None = None
    keep_loss: Tensor | None = None
    group_similarity: Tensor | None = None
    visible_count: Tensor | None = None
    selected_values: Tensor | None = None
    selected_weights: Tensor | None = None


def stable_masked_topk(
    scores: Tensor,
    eligible: Tensor,
    k: int,
    *,
    largest: bool = True,
    normalize: str = "softmax",
) -> tuple[Tensor, Tensor]:
    """Direct hard Top-K with deterministic ascending-index tie-breaking."""

    if scores.shape != eligible.shape or scores.ndim < 1:
        raise ValueError("scores and eligible must have the same non-scalar shape")
    if k <= 0:
        raise ValueError("k must be positive")
    if normalize not in {"softmax", "none"}:
        raise ValueError("normalize must be 'softmax' or 'none'")
    candidate_count = scores.shape[-1]
    width = min(k, candidate_count)
    if width == 0:
        return torch.zeros_like(scores), torch.empty(
            (*scores.shape[:-1], 0), dtype=torch.long, device=scores.device
        )

    eligible = eligible.to(device=scores.device, dtype=torch.bool)
    fill = -torch.inf if largest else torch.inf
    sortable = torch.where(eligible, scores, torch.full_like(scores, fill))
    order = torch.argsort(sortable, dim=-1, descending=largest, stable=True)
    chosen_raw = order[..., :width]
    selected_eligible = eligible.gather(-1, chosen_raw)
    selected_scores = scores.gather(-1, chosen_raw)
    chosen = torch.where(
        selected_eligible, chosen_raw, torch.full_like(chosen_raw, -1)
    )

    if normalize == "softmax":
        safe_scores = torch.where(
            selected_eligible, selected_scores, torch.full_like(selected_scores, -torch.inf)
        )
        maximum = torch.where(
            selected_eligible,
            safe_scores,
            torch.full_like(safe_scores, -torch.inf),
        ).max(dim=-1, keepdim=True).values
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        numerator = torch.where(
            selected_eligible,
            torch.exp(safe_scores - maximum),
            torch.zeros_like(selected_scores),
        )
        selected_weights = numerator / numerator.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(scores.dtype).tiny
        )
    else:
        selected_weights = selected_eligible.to(dtype=scores.dtype)

    dense_weights = torch.zeros_like(scores)
    safe_indices = chosen.clamp_min(0)
    dense_weights.scatter_add_(-1, safe_indices, selected_weights)
    return dense_weights, chosen


def _gather_temporal_values(values: Tensor, selected: Tensor) -> Tensor:
    """Gather temporal source values into selected edge slots."""

    patch_count = values.shape[2]
    safe = selected.clamp_min(0)
    gathered = values.unsqueeze(-3).expand(
        *values.shape[:2], patch_count, patch_count, values.shape[-1]
    ).gather(3, safe.unsqueeze(-1).expand(*safe.shape, values.shape[-1]))
    return gathered * selected.ge(0).unsqueeze(-1).to(values.dtype)


def _gather_variable_values(values: Tensor, selected: Tensor) -> Tensor:
    """Gather variable source values into [B,C,N,K,D] edge slots."""

    batch_size, variable_count, patch_count, d_model = values.shape
    selected_by_patch = selected.permute(0, 2, 1, 3).contiguous()
    safe = selected_by_patch.clamp_min(0)
    values_by_patch = values.permute(0, 2, 1, 3).contiguous()
    gathered = values_by_patch.unsqueeze(2).expand(
        batch_size, patch_count, variable_count, variable_count, d_model
    ).gather(
        3,
        safe.unsqueeze(-1).expand(
            batch_size, patch_count, variable_count, safe.shape[-1], d_model
        ),
    ).permute(0, 2, 1, 3, 4).contiguous()
    return gathered * selected.ge(0).unsqueeze(-1).to(values.dtype)


class TemporalRelation(nn.Module):
    def __init__(
        self,
        d_model: int,
        topk: int = 6,
        bias: bool = True,
        *,
        fine_radius: int = 0,
        coarse_stride: int = 0,
        coarse_radius: int = 0,
    ) -> None:
        super().__init__()
        if fine_radius < 0 or coarse_stride < 0 or coarse_radius < 0:
            raise ValueError("temporal scope parameters must be non-negative")
        if (coarse_stride == 0) != (coarse_radius == 0):
            raise ValueError("coarse_stride and coarse_radius must be set together")
        self.topk = topk
        self.fine_radius = int(fine_radius)
        self.coarse_stride = int(coarse_stride)
        self.coarse_radius = int(coarse_radius)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)

    def _scope(self, patch_count: int, device: torch.device) -> Tensor:
        if self.fine_radius == 0 and self.coarse_stride == 0:
            scope = torch.ones((patch_count, patch_count), dtype=torch.bool, device=device)
            scope.fill_diagonal_(False)
            return scope
        offsets = set(range(1, self.fine_radius + 1))
        offsets.update(
            self.coarse_stride * index
            for index in range(1, self.coarse_radius + 1)
        )
        distance = torch.arange(patch_count, device=device)
        distance = (distance.view(1, -1) - distance.view(-1, 1)).abs()
        scope = torch.zeros_like(distance, dtype=torch.bool)
        for offset in offsets:
            scope |= distance == int(offset)
        scope.fill_diagonal_(False)
        return scope

    def select(
        self,
        query: Tensor,
        sources: Tensor,
        rho: Tensor,
        target_eligible: Tensor | None = None,
        *,
        filter_query: Tensor | None = None,
        edge_filter: EdgeConfidenceFilter | None = None,
    ) -> RelationSelection:
        if query.shape != sources.shape or query.ndim != 4:
            raise ValueError("query and sources must match [batch, variables, patches, d]")
        if rho.shape != query.shape[:-1]:
            raise ValueError("rho must match relation node dimensions")
        patch_count = query.shape[2]
        keys = self.W_k(sources)
        values = self.W_v(sources)
        scores = torch.einsum("bcid,bcjd->bcij", query, keys) / sqrt(query.shape[-1])
        eligible = (rho > 0).unsqueeze(-2).expand_as(scores)
        identity = torch.eye(patch_count, dtype=torch.bool, device=query.device)
        eligible = eligible & ~identity.view(1, 1, patch_count, patch_count)
        eligible = eligible & self._scope(patch_count, query.device).view(
            1, 1, patch_count, patch_count
        )
        if target_eligible is not None:
            if target_eligible.shape != rho.shape:
                raise ValueError("target_eligible must match rho")
            eligible = eligible & target_eligible.unsqueeze(-1)
        if (filter_query is None) != (edge_filter is None):
            raise ValueError("filter_query and edge_filter must be supplied together")
        if edge_filter is not None:
            assert filter_query is not None
            if filter_query.shape != query.shape:
                raise ValueError("filter_query must match temporal query")
            source_keys = keys.unsqueeze(-3).expand(
                *keys.shape[:2], patch_count, patch_count, keys.shape[-1]
            )
            source_rho = rho.unsqueeze(-2).expand_as(scores)
            index = torch.arange(patch_count, device=query.device)
            signed_distance = index.view(1, -1) - index.view(-1, 1)
            signed_distance = signed_distance.to(query.dtype) / max(
                patch_count - 1, 1
            )
            auxiliary = torch.stack(
                (
                    signed_distance.view(1, 1, patch_count, patch_count).expand_as(
                        scores
                    ),
                    source_rho,
                ),
                dim=-1,
            )
            filtered = edge_filter(
                filter_query,
                source_keys,
                auxiliary,
                scores,
                eligible,
            )
            message = torch.einsum("bcij,bcjd->bcid", filtered.weights, values)
            selected_values = _gather_temporal_values(values, filtered.selected)
            selected_weights = filtered.weights.gather(
                -1, filtered.selected.clamp_min(0)
            )
            selected_weights = selected_weights * filtered.selected.ge(0).to(
                selected_weights.dtype
            )
            active_target = (
                target_eligible
                if target_eligible is not None
                else torch.ones_like(rho, dtype=torch.bool)
            )
            scope_count = self._scope(patch_count, query.device).sum(dim=-1)
            pool_count = active_target.to(torch.long) * scope_count.view(1, 1, -1)
            return RelationSelection(
                message=message,
                weights=filtered.weights,
                selected=filtered.selected,
                candidate_count=filtered.candidate_count,
                pool_count=pool_count,
                filter_probabilities=filtered.probabilities,
                hard_gate=filtered.hard_gate,
                retained_count=filtered.retained_count,
                keep_loss=filtered.keep_loss,
                selected_values=selected_values,
                selected_weights=selected_weights,
            )
        weights, selected = stable_masked_topk(
            scores, eligible, self.topk, largest=True, normalize="softmax"
        )
        message = torch.einsum("bcij,bcjd->bcid", weights, values)
        selected_values = _gather_temporal_values(values, selected)
        selected_weights = weights.gather(-1, selected.clamp_min(0))
        selected_weights = selected_weights * selected.ge(0).to(selected_weights.dtype)
        return RelationSelection(
            message,
            weights,
            selected,
            eligible.sum(dim=-1),
            selected_values=selected_values,
            selected_weights=selected_weights,
        )

    def forward(self, query: Tensor, sources: Tensor, rho: Tensor) -> Tensor:
        return self.select(query, sources, rho).message


class VariableRelation(nn.Module):
    def __init__(
        self,
        d_model: int,
        topk: int = 4,
        bias: bool = True,
        *,
        use_global_summary: bool = False,
    ) -> None:
        super().__init__()
        self.topk = topk
        self.use_global_summary = bool(use_global_summary)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)

    def select(
        self,
        query: Tensor,
        sources: Tensor,
        rho: Tensor,
        target_eligible: Tensor | None = None,
        *,
        filter_query: Tensor | None = None,
        edge_filter: EdgeConfidenceFilter | None = None,
        variable_embeddings: Tensor | None = None,
    ) -> RelationSelection:
        if query.shape != sources.shape or query.ndim != 4:
            raise ValueError("query and sources must match [batch, variables, patches, d]")
        if rho.shape != query.shape[:-1]:
            raise ValueError("rho must match relation node dimensions")
        variable_count = query.shape[1]
        keys = self.W_k(sources)
        values = self.W_v(sources)
        if self.use_global_summary:
            source_weight = rho.unsqueeze(-1)
            source_summary = (sources * source_weight).sum(dim=2) / source_weight.sum(
                dim=2
            ).clamp_min(1.0)
            query_weight = rho.unsqueeze(-1)
            query_summary = (query * query_weight).sum(dim=2) / query_weight.sum(
                dim=2
            ).clamp_min(1.0)
            global_keys = self.W_k(source_summary)
            scores = torch.einsum("bcd,bvd->bcv", query_summary, global_keys)
            scores = scores.unsqueeze(1).expand(-1, query.shape[2], -1, -1)
        else:
            scores = torch.einsum("bcnd,bvnd->bncv", query, keys) / sqrt(query.shape[-1])
        source_eligible = (rho > 0).permute(0, 2, 1)
        eligible = source_eligible.unsqueeze(-2).expand_as(scores)
        identity = torch.eye(variable_count, dtype=torch.bool, device=query.device)
        eligible = eligible & ~identity.view(1, 1, variable_count, variable_count)
        if target_eligible is not None:
            if target_eligible.shape != rho.shape:
                raise ValueError("target_eligible must match rho")
            eligible = eligible & target_eligible.permute(0, 2, 1).unsqueeze(-1)
        if (filter_query is None) != (edge_filter is None):
            raise ValueError("filter_query and edge_filter must be supplied together")
        if edge_filter is not None:
            assert filter_query is not None
            if filter_query.shape != query.shape:
                raise ValueError("filter_query must match variable query")
            if variable_embeddings is None or variable_embeddings.shape != (
                variable_count,
                query.shape[-1],
            ):
                raise ValueError("variable_embeddings must have shape [C,D]")
            batch_size, _, patch_count, d_model = query.shape
            query_by_patch = filter_query.permute(0, 2, 1, 3).contiguous()
            keys_by_patch = keys.permute(0, 2, 1, 3).contiguous()
            if self.use_global_summary:
                source_weight = rho.unsqueeze(-1)
                source_summary = (sources * source_weight).sum(dim=2) / source_weight.sum(
                    dim=2
                ).clamp_min(1.0)
                global_keys = self.W_k(source_summary)
                source_keys = global_keys.unsqueeze(1).unsqueeze(1).expand(
                    batch_size, patch_count, variable_count, variable_count, d_model
                )
            else:
                source_keys = keys_by_patch.unsqueeze(-3).expand(
                    batch_size,
                    patch_count,
                    variable_count,
                    variable_count,
                    d_model,
                )
            target_identity = variable_embeddings.view(
                1, 1, variable_count, 1, d_model
            ).expand_as(source_keys)
            source_identity = variable_embeddings.view(
                1, 1, 1, variable_count, d_model
            ).expand_as(source_keys)
            source_rho = rho.permute(0, 2, 1).unsqueeze(-2).expand_as(scores)
            auxiliary = torch.cat(
                (target_identity, source_identity, source_rho.unsqueeze(-1)),
                dim=-1,
            )
            filtered = edge_filter(
                query_by_patch,
                source_keys,
                auxiliary,
                scores,
                eligible,
            )
            message = torch.einsum("bncv,bvnd->bcnd", filtered.weights, values)
            active_target = (
                target_eligible
                if target_eligible is not None
                else torch.ones_like(rho, dtype=torch.bool)
            )
            pool_count = active_target.to(torch.long) * max(variable_count - 1, 0)
            selected = filtered.selected.permute(0, 2, 1, 3).contiguous()
            selected_values = _gather_variable_values(values, selected)
            selected_weights = filtered.weights.gather(
                -1, filtered.selected.clamp_min(0)
            ).permute(0, 2, 1, 3).contiguous()
            selected_weights = selected_weights * selected.ge(0).to(
                selected_weights.dtype
            )
            return RelationSelection(
                message=message,
                weights=filtered.weights.permute(0, 2, 1, 3).contiguous(),
                selected=filtered.selected.permute(0, 2, 1, 3).contiguous(),
                candidate_count=filtered.candidate_count.permute(
                    0, 2, 1
                ).contiguous(),
                pool_count=pool_count,
                filter_probabilities=filtered.probabilities.permute(
                    0, 2, 1, 3
                ).contiguous(),
                hard_gate=filtered.hard_gate.permute(0, 2, 1, 3).contiguous(),
                retained_count=filtered.retained_count.permute(
                    0, 2, 1
                ).contiguous(),
                keep_loss=filtered.keep_loss,
                selected_values=selected_values,
                selected_weights=selected_weights,
            )
        weights, selected = stable_masked_topk(
            scores, eligible, self.topk, largest=True, normalize="softmax"
        )
        message = torch.einsum("bncv,bvnd->bcnd", weights, values)
        selected = selected.permute(0, 2, 1, 3).contiguous()
        selected_values = _gather_variable_values(values, selected)
        selected_weights = weights.gather(-1, selected.permute(0, 2, 1, 3).contiguous().clamp_min(0))
        selected_weights = selected_weights.permute(0, 2, 1, 3).contiguous()
        selected_weights = selected_weights * selected.ge(0).to(selected_weights.dtype)
        return RelationSelection(
            message=message,
            weights=weights.permute(0, 2, 1, 3).contiguous(),
            selected=selected,
            candidate_count=eligible.sum(dim=-1).permute(0, 2, 1).contiguous(),
            selected_values=selected_values,
            selected_weights=selected_weights,
        )

    def forward(self, query: Tensor, sources: Tensor, rho: Tensor) -> Tensor:
        return self.select(query, sources, rho).message


class HistoricalRelation(nn.Module):
    """Variable-specific prototype retrieval with caller-supplied center encoding."""

    def __init__(
        self,
        bank: PrototypeBank,
        topk: int = 4,
        *,
        candidate_pool: int | None = None,
        patch_len: int = 16,
        patch_stride: int = 8,
        d_model: int,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if topk <= 0 or patch_len != 16 or patch_stride != 8 or d_model <= 0:
            raise ValueError("invalid stage-1 historical relation geometry")
        if candidate_pool is not None and candidate_pool <= 0:
            raise ValueError("candidate_pool must be positive")
        self.bank = bank
        self.topk = topk
        self.candidate_pool = topk if candidate_pool is None else candidate_pool
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self._d_model = d_model
        self.eps = eps
        maximum_prototypes = max(
            (len(records) for records in bank.records), default=0
        )
        padded_prototypes = max(maximum_prototypes, 1)
        variable_count = len(bank.records)
        prototype_factors = np.zeros(
            (variable_count, padded_prototypes, 7), dtype=np.float32
        )
        prototype_available = np.zeros(
            (variable_count, padded_prototypes, 7), dtype=np.bool_
        )
        prototype_centers = np.zeros(
            (variable_count, padded_prototypes, patch_len), dtype=np.float32
        )
        prototype_exists = np.zeros(
            (variable_count, padded_prototypes), dtype=np.bool_
        )
        for variable_index, records in enumerate(bank.records):
            for prototype_index, record in enumerate(records):
                prototype_factors[variable_index, prototype_index] = (
                    record.factors.values
                )
                prototype_available[variable_index, prototype_index] = (
                    record.factors.available
                )
                prototype_centers[variable_index, prototype_index] = (
                    record.raw_segment[record.center_slice]
                )
                prototype_exists[variable_index, prototype_index] = True
        self.register_buffer(
            "prototype_factors",
            torch.from_numpy(prototype_factors),
            persistent=False,
        )
        self.register_buffer(
            "prototype_available",
            torch.from_numpy(prototype_available),
            persistent=False,
        )
        self.register_buffer(
            "prototype_centers",
            torch.from_numpy(prototype_centers),
            persistent=False,
        )
        self.register_buffer(
            "prototype_exists",
            torch.from_numpy(prototype_exists),
            persistent=False,
        )
        self.register_buffer(
            "factor_mean",
            torch.from_numpy(np.asarray(bank.factor_mean, dtype=np.float32)),
            persistent=False,
        )
        self.register_buffer(
            "factor_std",
            torch.from_numpy(np.asarray(bank.factor_std, dtype=np.float32)),
            persistent=False,
        )

    @property
    def d_model(self) -> int:
        return self._d_model

    @staticmethod
    def _masked_side_factors(
        values: Tensor,
        valid: Tensor,
        indices: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        counts = valid.sum(dim=-1)
        selected = torch.where(valid, values, torch.zeros_like(values))
        safe_counts = counts.clamp_min(1).to(dtype=values.dtype)
        mean = selected.sum(dim=-1) / safe_counts
        mean_available = counts > 0

        index_values = indices.to(device=values.device, dtype=values.dtype)
        index_sum = torch.where(
            valid, index_values, torch.zeros_like(index_values)
        ).sum(dim=-1)
        index_mean = index_sum / safe_counts
        centered_index = index_values - index_mean.unsqueeze(-1)
        centered_values = values - mean.unsqueeze(-1)
        slope_denominator = torch.where(
            valid, centered_index.square(), torch.zeros_like(values)
        ).sum(dim=-1)
        slope_numerator = torch.where(
            valid,
            centered_index * centered_values,
            torch.zeros_like(values),
        ).sum(dim=-1)
        slope_available = counts >= 2
        slope = torch.where(
            slope_available,
            slope_numerator / slope_denominator.clamp_min(
                torch.finfo(values.dtype).eps
            ),
            torch.zeros_like(mean),
        )

        difference_valid = valid[..., :-1] & valid[..., 1:]
        differences = values[..., 1:] - values[..., :-1]
        difference_count = difference_valid.sum(dim=-1)
        safe_difference_count = difference_count.clamp_min(1).to(values.dtype)
        difference_mean = torch.where(
            difference_valid, differences, torch.zeros_like(differences)
        ).sum(dim=-1) / safe_difference_count
        difference_variance = torch.where(
            difference_valid,
            (differences - difference_mean.unsqueeze(-1)).square(),
            torch.zeros_like(differences),
        ).sum(dim=-1) / safe_difference_count
        volatility_available = difference_count > 0
        volatility = torch.where(
            volatility_available,
            torch.sqrt(difference_variance),
            torch.zeros_like(mean),
        )
        return (
            mean,
            mean_available,
            slope,
            slope_available,
            volatility,
            volatility_available,
        )

    def _batched_query_factors(
        self,
        context: Tensor,
        context_mask: Tensor,
        patch_count: int,
        train_std: Tensor,
    ) -> tuple[Tensor, Tensor]:
        context_by_variable = context.permute(0, 2, 1)
        mask_by_variable = context_mask.permute(0, 2, 1).to(torch.bool)
        required_length = (patch_count - 1) * self.patch_stride + 64
        right_padding = required_length - (context.shape[1] + 24)
        if right_padding < 0:
            raise ValueError("context is longer than the configured patch geometry")
        padded_context = F.pad(context_by_variable, (24, right_padding))
        padded_mask = F.pad(mask_by_variable, (24, right_padding), value=False)
        segments = padded_context.unfold(-1, 64, self.patch_stride)
        valid = padded_mask.unfold(-1, 64, self.patch_stride)
        if segments.shape[2] != patch_count:
            raise ValueError("context and patch state geometry are inconsistent")

        left_values, left_valid = segments[..., :24], valid[..., :24]
        right_values, right_valid = segments[..., 40:], valid[..., 40:]
        left = self._masked_side_factors(
            left_values, left_valid, torch.arange(24, device=context.device)
        )
        right = self._masked_side_factors(
            right_values,
            right_valid,
            torch.arange(40, 64, device=context.device),
        )

        left_positions = torch.arange(24, device=context.device)
        left_last_index = torch.where(
            left_valid,
            left_positions,
            torch.full_like(left_positions, -1),
        ).max(dim=-1).values
        right_positions = torch.arange(24, device=context.device)
        right_first_index = torch.where(
            right_valid,
            right_positions,
            torch.full_like(right_positions, 24),
        ).min(dim=-1).values
        boundary_available = (left_last_index >= 0) & (right_first_index < 24)
        safe_left_index = left_last_index.clamp_min(0).unsqueeze(-1)
        safe_right_index = right_first_index.clamp_max(23).unsqueeze(-1)
        left_boundary = left_values.gather(-1, safe_left_index).squeeze(-1)
        right_boundary = right_values.gather(-1, safe_right_index).squeeze(-1)
        boundary_gap = torch.where(
            boundary_available,
            (right_boundary - left_boundary)
            / train_std.to(context.dtype).view(1, -1, 1),
            torch.zeros_like(left_boundary),
        )

        factor_values = torch.stack(
            (left[0], left[2], left[4], right[0], right[2], right[4], boundary_gap),
            dim=-1,
        )
        factor_available = torch.stack(
            (
                left[1],
                left[3],
                left[5],
                right[1],
                right[3],
                right[5],
                boundary_available,
            ),
            dim=-1,
        )
        return factor_values, factor_available

    def select(
        self,
        context: Tensor,
        context_mask: Tensor,
        state: Tensor,
        query_mean: Tensor,
        query_std: Tensor,
        train_std: Tensor,
        *,
        prototype_projection: nn.Module,
        filter_query: Tensor | None = None,
        edge_filter: EdgeConfidenceFilter | None = None,
    ) -> RelationSelection:
        if context.ndim != 3 or context_mask.shape != context.shape:
            raise ValueError("context and mask must match [batch, time, variables]")
        batch_size, _, variable_count = context.shape
        if variable_count != len(self.bank.records):
            raise ValueError("prototype bank variable count does not match context")
        if state.ndim != 3 or state.shape[:2] != (batch_size, variable_count):
            raise ValueError("state must have shape [batch, variables, patches]")
        if query_mean.shape != (batch_size, variable_count) or query_std.shape != query_mean.shape:
            raise ValueError("query normalization statistics have invalid shape")
        if train_std.shape != (variable_count,):
            raise ValueError("train_std must contain one value per variable")
        if (filter_query is None) != (edge_filter is None):
            raise ValueError("filter_query and edge_filter must be supplied together")
        if filter_query is not None and filter_query.shape != (
            batch_size,
            variable_count,
            state.shape[2],
            self.d_model,
        ):
            raise ValueError("filter_query has invalid historical node geometry")

        patch_count = state.shape[2]
        message = context.new_zeros(
            batch_size, variable_count, patch_count, self.d_model
        )
        selected = torch.full(
            (batch_size, variable_count, patch_count, self.topk),
            -1,
            dtype=torch.long,
            device=context.device,
        )
        selected_weights = context.new_zeros(
            batch_size, variable_count, patch_count, self.topk
        )
        empty_count = torch.zeros(
            (batch_size, variable_count, patch_count),
            dtype=torch.long,
            device=context.device,
        )
        filtering = edge_filter is not None

        with torch.no_grad():
            if not filtering and not bool(self.prototype_exists.any()):
                return RelationSelection(
                    message, selected_weights, selected, empty_count
                )
            query_factors, query_available = self._batched_query_factors(
                context, context_mask, patch_count, train_std
            )
            factor_mean = self.factor_mean.to(dtype=context.dtype)
            factor_std = self.factor_std.to(dtype=context.dtype)
            standardized_query = (
                query_factors - factor_mean.view(1, variable_count, 1, 7)
            ) / factor_std.view(1, variable_count, 1, 7)
            standardized_prototypes = (
                self.prototype_factors.to(dtype=context.dtype)
                - factor_mean.unsqueeze(1)
            ) / factor_std.unsqueeze(1)
            shared = (
                query_available.unsqueeze(-2)
                & self.prototype_available.unsqueeze(0).unsqueeze(2)
            )
            shared_count = shared.sum(dim=-1)
            difference = (
                standardized_query.unsqueeze(-2)
                - standardized_prototypes.unsqueeze(0).unsqueeze(2)
            )
            distances = torch.sqrt(
                torch.where(
                    shared, difference.square(), torch.zeros_like(difference)
                ).sum(dim=-1)
                / shared_count.clamp_min(1).to(context.dtype)
            )
            eligible = (
                (shared_count > 0)
                & self.prototype_exists.unsqueeze(0).unsqueeze(2)
                & (state != PATCH_OBSERVED).unsqueeze(-1)
            )
            all_candidate_count = eligible.sum(dim=-1)
            retrieval_width = self.candidate_pool if filtering else self.topk
            _, chosen = stable_masked_topk(
                distances,
                eligible,
                retrieval_width,
                largest=False,
                normalize="none",
            )
            actual_width = chosen.shape[-1]
            safe_chosen = chosen.clamp_min(0)
            chosen_eligible = eligible.gather(-1, safe_chosen) & (chosen >= 0)
            chosen_distances = distances.gather(-1, safe_chosen)

            centers = self.prototype_centers.to(dtype=context.dtype)
            centers = centers.unsqueeze(0).unsqueeze(2).expand(
                batch_size, variable_count, patch_count, -1, self.patch_len
            )
            chosen_centers = centers.gather(
                3,
                safe_chosen.unsqueeze(-1).expand(
                    batch_size,
                    variable_count,
                    patch_count,
                    actual_width,
                    self.patch_len,
                ),
            )
            normalized_centers = (
                chosen_centers - query_mean[:, :, None, None, None]
            ) / query_std[:, :, None, None, None]

            if filtering:
                chosen_shared_count = shared_count.gather(-1, safe_chosen)
                available_count = self.prototype_available.sum(dim=-1)
                available_count = available_count.unsqueeze(0).unsqueeze(2).expand(
                    batch_size, variable_count, patch_count, -1
                )
                chosen_available_count = available_count.gather(-1, safe_chosen)
            else:
                inverse = torch.where(
                    chosen_eligible,
                    1.0 / (chosen_distances + self.eps),
                    torch.zeros_like(chosen_distances),
                )
                weights = inverse / inverse.sum(dim=-1, keepdim=True).clamp_min(
                    torch.finfo(context.dtype).tiny
                )
                selected[..., :actual_width] = chosen
                selected_weights[..., :actual_width] = weights

        prototype_latents = prototype_projection(
            normalized_centers.reshape(-1, self.patch_len)
        ).reshape(
            batch_size,
            variable_count,
            patch_count,
            actual_width,
            self.d_model,
        )
        if not filtering:
            message = (weights.unsqueeze(-1) * prototype_latents).sum(dim=-2)
            return RelationSelection(
                message,
                selected_weights,
                selected,
                all_candidate_count,
                selected_values=prototype_latents,
                selected_weights=selected_weights,
            )

        assert edge_filter is not None and filter_query is not None
        auxiliary = torch.stack(
            (
                chosen_distances,
                chosen_shared_count.to(context.dtype) / 7.0,
                chosen_available_count.to(context.dtype) / 7.0,
            ),
            dim=-1,
        )
        base_scores = -torch.log(chosen_distances + self.eps)
        filtered = edge_filter(
            filter_query,
            prototype_latents,
            auxiliary,
            base_scores,
            chosen_eligible,
        )
        message = (filtered.weights.unsqueeze(-1) * prototype_latents).sum(dim=-2)
        safe_selected = filtered.selected.clamp_min(0)
        mapped_ids = chosen.gather(-1, safe_selected)
        mapped_ids = torch.where(
            filtered.selected >= 0, mapped_ids, torch.full_like(mapped_ids, -1)
        )
        mapped_weights = filtered.weights.gather(-1, safe_selected)
        mapped_weights = torch.where(
            filtered.selected >= 0,
            mapped_weights,
            torch.zeros_like(mapped_weights),
        )
        return RelationSelection(
            message=message,
            weights=mapped_weights,
            selected=mapped_ids,
            candidate_count=filtered.candidate_count,
            pool_count=chosen_eligible.sum(dim=-1),
            filter_probabilities=filtered.probabilities,
            hard_gate=filtered.hard_gate,
            retained_count=filtered.retained_count,
            keep_loss=filtered.keep_loss,
            selected_values=prototype_latents.gather(
                -2,
                filtered.selected.clamp_min(0).unsqueeze(-1).expand(
                    *filtered.selected.shape, self.d_model
                ),
            ) * filtered.selected.ge(0).unsqueeze(-1).to(prototype_latents.dtype),
            selected_weights=filtered.weights,
        )

    def forward(
        self,
        context: Tensor,
        context_mask: Tensor,
        state: Tensor,
        query_mean: Tensor,
        query_std: Tensor,
        train_std: Tensor,
        *,
        prototype_projection: nn.Module,
    ) -> Tensor:
        return self.select(
            context,
            context_mask,
            state,
            query_mean,
            query_std,
            train_std,
            prototype_projection=prototype_projection,
        ).message


class GroupedHistoricalRelation(nn.Module):
    """Frozen two-level clean–mask retrieval over a grouped prototype bank."""

    def __init__(
        self,
        bank: GroupedPrototypeBank,
        *,
        retrieval_encoder: nn.Module,
        topk: int = 4,
        similarity_threshold: float = 0.7,
        min_visible_points: int = 4,
        overlap_exclusion: int = 16,
        retrieval_temperature: float = 1.0,
        patch_len: int = 16,
        patch_stride: int = 8,
        d_model: int,
    ) -> None:
        super().__init__()
        if topk <= 0 or patch_len <= 0 or patch_stride <= 0 or d_model <= 0:
            raise ValueError("grouped historical relation geometry is invalid")
        if not 0.0 < similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must lie in (0, 1]")
        if min_visible_points <= 0 or overlap_exclusion < 0:
            raise ValueError("historical availability constraints are invalid")
        if retrieval_temperature <= 0.0:
            raise ValueError("retrieval_temperature must be positive")
        if bank.patch_len != patch_len or bank.patch_stride != patch_stride:
            raise ValueError("grouped bank patch geometry does not match relation")
        if bank.retrieval_dim <= 0:
            raise ValueError("grouped bank retrieval dimension must be positive")
        self.bank = bank
        self.topk = int(topk)
        self.similarity_threshold = float(similarity_threshold)
        self.min_visible_points = int(min_visible_points)
        self.overlap_exclusion = int(overlap_exclusion)
        self.retrieval_temperature = float(retrieval_temperature)
        self.patch_len = int(patch_len)
        self.patch_stride = int(patch_stride)
        self._d_model = int(d_model)
        self.retrieval_encoder = retrieval_encoder.eval()
        for parameter in self.retrieval_encoder.parameters():
            parameter.requires_grad_(False)

        variable_count = bank.variable_count
        group_count = max((len(groups) for groups in bank.groups), default=0)
        member_count = max(
            (len(group.members) for groups in bank.groups for group in groups),
            default=0,
        )
        group_count = max(group_count, 1)
        member_count = max(member_count, 1)
        self.member_count = member_count
        group_centers = np.zeros(
            (variable_count, group_count, bank.retrieval_dim), dtype=np.float32
        )
        group_exists = np.zeros((variable_count, group_count), dtype=np.bool_)
        member_raw = np.zeros(
            (variable_count, group_count, member_count, patch_len), dtype=np.float32
        )
        member_keys = np.zeros(
            (variable_count, group_count, member_count, bank.retrieval_dim), dtype=np.float32
        )
        member_starts = np.full(
            (variable_count, group_count, member_count), -1, dtype=np.int64
        )
        member_exists = np.zeros(
            (variable_count, group_count, member_count), dtype=np.bool_
        )
        for variable_id, groups in enumerate(bank.groups):
            for group_index, group in enumerate(groups):
                group_centers[variable_id, group_index] = group.center
                group_exists[variable_id, group_index] = True
                for member_index, member in enumerate(group.members):
                    member_raw[variable_id, group_index, member_index] = member.raw_patch
                    member_keys[variable_id, group_index, member_index] = member.clean_key
                    member_starts[variable_id, group_index, member_index] = member.start
                    member_exists[variable_id, group_index, member_index] = True
        self.register_buffer("group_centers", torch.from_numpy(group_centers), persistent=False)
        self.register_buffer("group_exists", torch.from_numpy(group_exists), persistent=False)
        self.register_buffer("member_raw", torch.from_numpy(member_raw), persistent=False)
        self.register_buffer("member_keys", torch.from_numpy(member_keys), persistent=False)
        self.register_buffer("member_starts", torch.from_numpy(member_starts), persistent=False)
        self.register_buffer("member_exists", torch.from_numpy(member_exists), persistent=False)

    @property
    def d_model(self) -> int:
        return self._d_model

    def _query_patches(
        self, context: Tensor, context_mask: Tensor, patch_count: int
    ) -> tuple[Tensor, Tensor]:
        values = context.permute(0, 2, 1)
        masks = context_mask.permute(0, 2, 1).to(torch.bool)
        if values.shape[-1] < self.patch_len:
            raise ValueError("historical context is shorter than patch_len")
        replicated_values = values[..., -1:].expand(-1, -1, self.patch_stride)
        replicated_masks = masks[..., -1:].expand(-1, -1, self.patch_stride)
        replicated_validity = torch.zeros_like(replicated_masks)
        padded_values = torch.cat((values, replicated_values), dim=-1)
        padded_masks = torch.cat((masks, replicated_masks), dim=-1)
        padded_validity = torch.cat(
            (torch.ones_like(masks), replicated_validity), dim=-1
        )
        patches = padded_values.unfold(-1, self.patch_len, self.patch_stride)
        patch_masks = padded_masks.unfold(-1, self.patch_len, self.patch_stride)
        patch_validity = padded_validity.unfold(-1, self.patch_len, self.patch_stride)
        patch_masks = patch_masks & patch_validity
        if patches.shape[2] != patch_count:
            raise ValueError("historical context and state patch geometry disagree")
        return patches, patch_masks

    def select(
        self,
        context: Tensor,
        context_mask: Tensor,
        state: Tensor,
        query_mean: Tensor,
        query_std: Tensor,
        train_std: Tensor,
        *,
        prototype_projection: nn.Module,
        patch_starts: Tensor | None = None,
    ) -> RelationSelection:
        del train_std
        if context.ndim != 3 or context_mask.shape != context.shape:
            raise ValueError("context and context_mask must have shape [batch, time, variables]")
        batch_size, _, variable_count = context.shape
        if variable_count != self.bank.variable_count:
            raise ValueError("grouped bank variable count does not match context")
        if state.ndim != 3 or state.shape[:2] != (batch_size, variable_count):
            raise ValueError("state must have shape [batch, variables, patches]")
        patch_count = state.shape[2]
        if query_mean.shape != (batch_size, variable_count) or query_std.shape != query_mean.shape:
            raise ValueError("query normalization statistics have invalid shape")
        if patch_starts is None:
            patch_starts = torch.arange(
                patch_count, device=context.device, dtype=torch.long
            ).view(1, -1) * self.patch_stride
            patch_starts = patch_starts.expand(batch_size, -1)
        if patch_starts.shape != (batch_size, patch_count):
            raise ValueError("patch_starts must have shape [batch, patches]")

        values, masks = self._query_patches(context, context_mask, patch_count)
        visible_count = masks.sum(dim=-1)
        target_eligible = (state != PATCH_OBSERVED) & (
            visible_count >= self.min_visible_points
        )

        with torch.no_grad():
            query_keys = self.retrieval_encoder(
                values.reshape(-1, self.patch_len),
                masks.reshape(-1, self.patch_len),
            ).reshape(
                batch_size,
                variable_count,
                patch_count,
                self.bank.retrieval_dim,
            )
            group_scores = torch.einsum(
                "bcnd,cgd->bcng",
                query_keys,
                self.group_centers.to(dtype=query_keys.dtype),
            )
            group_eligible = target_eligible.unsqueeze(-1) & self.group_exists.view(
                1, variable_count, 1, -1
            )
            sortable_group_scores = torch.where(
                group_eligible,
                group_scores,
                torch.full_like(group_scores, -torch.inf),
            )
            best_group = sortable_group_scores.argmax(dim=-1)
            has_group = group_eligible.any(dim=-1)
            best_group_similarity = sortable_group_scores.max(dim=-1).values
            group_similarity = torch.where(
                has_group,
                best_group_similarity,
                torch.zeros_like(best_group_similarity),
            )

            variable_index = torch.arange(
                variable_count, device=context.device
            ).view(1, variable_count, 1).expand(
                batch_size, variable_count, patch_count
            )
            raw_candidates = self.member_raw[
                variable_index, best_group
            ].to(dtype=context.dtype)
            candidate_starts = self.member_starts[variable_index, best_group]
            member_exists = self.member_exists[variable_index, best_group]

            candidate_masks = masks.unsqueeze(-2).expand(
                batch_size,
                variable_count,
                patch_count,
                self.member_count,
                self.patch_len,
            )
            active_query = has_group & member_exists.any(dim=-1)
            active_candidates = raw_candidates[active_query]
            active_masks = candidate_masks[active_query]
            encoded_candidates = self.retrieval_encoder(
                (active_candidates * active_masks.to(context.dtype)).reshape(
                    -1, self.patch_len
                ),
                active_masks.reshape(-1, self.patch_len),
            ).reshape(-1, self.member_count, self.bank.retrieval_dim)
            candidate_keys = query_keys.new_zeros(
                batch_size,
                variable_count,
                patch_count,
                self.member_count,
                self.bank.retrieval_dim,
            )
            candidate_keys[active_query] = encoded_candidates
            similarities = (
                candidate_keys * query_keys.unsqueeze(-2)
            ).sum(dim=-1)
            no_overlap = (
                candidate_starts - patch_starts.unsqueeze(1).unsqueeze(-1)
            ).abs() >= self.overlap_exclusion
            eligible = (
                target_eligible.unsqueeze(-1)
                & has_group.unsqueeze(-1)
                & member_exists
                & no_overlap
                & (similarities >= self.similarity_threshold)
            )
            candidate_count = eligible.sum(dim=-1)
            dense_weights, chosen = stable_masked_topk(
                similarities / self.retrieval_temperature,
                eligible,
                self.topk,
                largest=True,
                normalize="softmax",
            )

        width = chosen.shape[-1]
        safe_chosen = chosen.clamp_min(0)
        chosen_raw = raw_candidates.gather(
            -2,
            safe_chosen.unsqueeze(-1).expand(
                batch_size,
                variable_count,
                patch_count,
                width,
                self.patch_len,
            ),
        )
        chosen_starts = candidate_starts.gather(-1, safe_chosen)
        chosen_weights = dense_weights.gather(-1, safe_chosen)
        chosen_valid = chosen >= 0
        chosen_starts = torch.where(
            chosen_valid, chosen_starts, torch.full_like(chosen_starts, -1)
        )
        chosen_weights = chosen_weights * chosen_valid.to(chosen_weights.dtype)
        chosen_raw = (
            chosen_raw - query_mean.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        ) / query_std.clamp_min(1e-6).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        chosen_raw = chosen_raw * chosen_valid.unsqueeze(-1).to(chosen_raw.dtype)

        selected_raw = context.new_zeros(
            batch_size, variable_count, patch_count, self.topk, self.patch_len
        )
        selected = torch.full(
            (batch_size, variable_count, patch_count, self.topk),
            -1,
            dtype=torch.long,
            device=context.device,
        )
        selected_weights = context.new_zeros(
            batch_size, variable_count, patch_count, self.topk
        )
        selected_raw[..., :width, :] = chosen_raw
        selected[..., :width] = chosen_starts
        selected_weights[..., :width] = chosen_weights
        projected = prototype_projection(
            selected_raw.reshape(-1, self.patch_len)
        ).reshape(
            batch_size,
            variable_count,
            patch_count,
            self.topk,
            self.d_model,
        )
        message = (selected_weights.unsqueeze(-1) * projected).sum(dim=-2)
        return RelationSelection(
            message,
            selected_weights,
            selected,
            candidate_count,
            group_similarity=group_similarity,
            visible_count=visible_count,
            selected_values=projected,
            selected_weights=selected_weights,
        )

    def forward(
        self,
        context: Tensor,
        context_mask: Tensor,
        state: Tensor,
        query_mean: Tensor,
        query_std: Tensor,
        train_std: Tensor,
        *,
        prototype_projection: nn.Module,
        patch_starts: Tensor | None = None,
    ) -> Tensor:
        return self.select(
            context,
            context_mask,
            state,
            query_mean,
            query_std,
            train_std,
            prototype_projection=prototype_projection,
            patch_starts=patch_starts,
        ).message
