from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from malr.data.patching import PATCH_OBSERVED
from malr.prototypes.bank import PrototypeBank


@dataclass(frozen=True)
class RelationSelection:
    message: Tensor
    weights: Tensor
    selected: Tensor
    candidate_count: Tensor


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


class TemporalRelation(nn.Module):
    def __init__(self, d_model: int, topk: int = 6, bias: bool = True) -> None:
        super().__init__()
        self.topk = topk
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)

    def select(
        self,
        query: Tensor,
        sources: Tensor,
        rho: Tensor,
        target_eligible: Tensor | None = None,
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
        if target_eligible is not None:
            if target_eligible.shape != rho.shape:
                raise ValueError("target_eligible must match rho")
            eligible = eligible & target_eligible.unsqueeze(-1)
        weights, selected = stable_masked_topk(
            scores, eligible, self.topk, largest=True, normalize="softmax"
        )
        message = torch.einsum("bcij,bcjd->bcid", weights, values)
        return RelationSelection(message, weights, selected, eligible.sum(dim=-1))

    def forward(self, query: Tensor, sources: Tensor, rho: Tensor) -> Tensor:
        return self.select(query, sources, rho).message


class VariableRelation(nn.Module):
    def __init__(self, d_model: int, topk: int = 4, bias: bool = True) -> None:
        super().__init__()
        self.topk = topk
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)

    def select(
        self,
        query: Tensor,
        sources: Tensor,
        rho: Tensor,
        target_eligible: Tensor | None = None,
    ) -> RelationSelection:
        if query.shape != sources.shape or query.ndim != 4:
            raise ValueError("query and sources must match [batch, variables, patches, d]")
        if rho.shape != query.shape[:-1]:
            raise ValueError("rho must match relation node dimensions")
        variable_count = query.shape[1]
        keys = self.W_k(sources)
        values = self.W_v(sources)
        scores = torch.einsum("bcnd,bvnd->bncv", query, keys) / sqrt(query.shape[-1])
        source_eligible = (rho > 0).permute(0, 2, 1)
        eligible = source_eligible.unsqueeze(-2).expand_as(scores)
        identity = torch.eye(variable_count, dtype=torch.bool, device=query.device)
        eligible = eligible & ~identity.view(1, 1, variable_count, variable_count)
        if target_eligible is not None:
            if target_eligible.shape != rho.shape:
                raise ValueError("target_eligible must match rho")
            eligible = eligible & target_eligible.permute(0, 2, 1).unsqueeze(-1)
        weights, selected = stable_masked_topk(
            scores, eligible, self.topk, largest=True, normalize="softmax"
        )
        message = torch.einsum("bncv,bvnd->bcnd", weights, values)
        return RelationSelection(
            message=message,
            weights=weights.permute(0, 2, 1, 3).contiguous(),
            selected=selected.permute(0, 2, 1, 3).contiguous(),
            candidate_count=eligible.sum(dim=-1).permute(0, 2, 1).contiguous(),
        )

    def forward(self, query: Tensor, sources: Tensor, rho: Tensor) -> Tensor:
        return self.select(query, sources, rho).message


class HistoricalRelation(nn.Module):
    """Frozen variable-specific prototype retrieval and center encoding."""

    def __init__(
        self,
        bank: PrototypeBank,
        teacher_projection: nn.Module,
        topk: int = 4,
        *,
        patch_len: int = 16,
        patch_stride: int = 8,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if topk <= 0 or patch_len != 16 or patch_stride != 8:
            raise ValueError("invalid stage-1 historical relation geometry")
        self.bank = bank
        self.teacher = teacher_projection
        self.topk = topk
        self.patch_len = patch_len
        self.patch_stride = patch_stride
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
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
        self.teacher.eval()

    @property
    def d_model(self) -> int:
        if hasattr(self.teacher, "out_features"):
            return int(self.teacher.out_features)
        raise ValueError("teacher projection must expose out_features")

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

        patch_count = state.shape[2]
        width = self.topk
        message = context.new_zeros(batch_size, variable_count, patch_count, self.d_model)
        selected = torch.full(
            (batch_size, variable_count, patch_count, width),
            -1,
            dtype=torch.long,
            device=context.device,
        )
        selected_weights = context.new_zeros(batch_size, variable_count, patch_count, width)
        candidate_count = torch.zeros(
            (batch_size, variable_count, patch_count),
            dtype=torch.long,
            device=context.device,
        )

        with torch.no_grad():
            if not bool(self.prototype_exists.any()):
                return RelationSelection(
                    message, selected_weights, selected, candidate_count
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
                torch.where(shared, difference.square(), torch.zeros_like(difference)).sum(
                    dim=-1
                )
                / shared_count.clamp_min(1).to(context.dtype)
            )
            eligible = (
                (shared_count > 0)
                & self.prototype_exists.unsqueeze(0).unsqueeze(2)
                & (state != PATCH_OBSERVED).unsqueeze(-1)
            )
            candidate_count = eligible.sum(dim=-1)
            _, chosen = stable_masked_topk(
                distances,
                eligible,
                self.topk,
                largest=False,
                normalize="none",
            )
            actual_width = chosen.shape[-1]
            safe_chosen = chosen.clamp_min(0)
            chosen_eligible = eligible.gather(-1, safe_chosen) & (chosen >= 0)
            chosen_distances = distances.gather(-1, safe_chosen)
            inverse = torch.where(
                chosen_eligible,
                1.0 / (chosen_distances + self.eps),
                torch.zeros_like(chosen_distances),
            )
            weights = inverse / inverse.sum(dim=-1, keepdim=True).clamp_min(
                torch.finfo(context.dtype).tiny
            )

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
            prototype_latents = self.teacher(
                normalized_centers.reshape(-1, self.patch_len)
            ).reshape(
                batch_size,
                variable_count,
                patch_count,
                actual_width,
                self.d_model,
            )
            message = (weights.unsqueeze(-1) * prototype_latents).sum(dim=-2)
            selected[..., :actual_width] = chosen
            selected_weights[..., :actual_width] = weights
        return RelationSelection(message, selected_weights, selected, candidate_count)

    def forward(
        self,
        context: Tensor,
        context_mask: Tensor,
        state: Tensor,
        query_mean: Tensor,
        query_std: Tensor,
        train_std: Tensor,
    ) -> Tensor:
        return self.select(
            context, context_mask, state, query_mean, query_std, train_std
        ).message
