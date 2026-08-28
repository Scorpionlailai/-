from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
import torch
from torch import Tensor, nn

from malr.data.patching import PATCH_OBSERVED
from malr.prototypes.bank import PrototypeBank
from malr.prototypes.factors import compute_context_factors


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
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
        self.teacher.eval()

    @property
    def d_model(self) -> int:
        if hasattr(self.teacher, "out_features"):
            return int(self.teacher.out_features)
        raise ValueError("teacher projection must expose out_features")

    def _query_segment(
        self,
        context: Tensor,
        context_mask: Tensor,
        batch_index: int,
        variable_index: int,
        patch_index: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        raw_start = patch_index * self.patch_stride - 24
        segment = np.zeros(64, dtype=np.float64)
        valid = np.zeros(64, dtype=np.bool_)
        source_start = max(raw_start, 0)
        source_stop = min(raw_start + 64, context.shape[1])
        if source_start < source_stop:
            target_start = source_start - raw_start
            target_stop = target_start + source_stop - source_start
            segment[target_start:target_stop] = (
                context[batch_index, source_start:source_stop, variable_index]
                .detach()
                .cpu()
                .numpy()
            )
            valid[target_start:target_stop] = (
                context_mask[batch_index, source_start:source_stop, variable_index]
                .detach()
                .cpu()
                .numpy()
            )
        valid[24:40] = False
        return segment, valid

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
            for batch_index in range(batch_size):
                for variable_index, records in enumerate(self.bank.records):
                    if not records:
                        continue
                    factor_mean = self.bank.factor_mean[variable_index]
                    factor_std = self.bank.factor_std[variable_index]
                    for patch_index in range(patch_count):
                        if int(state[batch_index, variable_index, patch_index]) == PATCH_OBSERVED:
                            continue
                        segment, valid = self._query_segment(
                            context,
                            context_mask,
                            batch_index,
                            variable_index,
                            patch_index,
                        )
                        query_factors = compute_context_factors(
                            segment,
                            valid,
                            center_slice=slice(24, 40),
                            train_std=float(train_std[variable_index]),
                        )
                        query_values = (query_factors.values - factor_mean) / factor_std
                        distances: list[tuple[float, int]] = []
                        for prototype_id, record in enumerate(records):
                            shared = query_factors.available & record.factors.available
                            if not shared.any():
                                continue
                            prototype_values = (
                                record.factors.values - factor_mean
                            ) / factor_std
                            distance = float(
                                np.sqrt(np.mean((query_values[shared] - prototype_values[shared]) ** 2))
                            )
                            distances.append((distance, prototype_id))
                        distances.sort(key=lambda item: (item[0], item[1]))
                        chosen = distances[: self.topk]
                        candidate_count[batch_index, variable_index, patch_index] = len(distances)
                        if not chosen:
                            continue
                        distance_tensor = context.new_tensor([item[0] for item in chosen])
                        inverse = 1.0 / (distance_tensor + self.eps)
                        weights = inverse / inverse.sum()
                        centers = context.new_tensor(
                            np.stack(
                                [
                                    records[item[1]].raw_segment[24:40]
                                    for item in chosen
                                ]
                            )
                        )
                        centers = (
                            centers - query_mean[batch_index, variable_index]
                        ) / query_std[batch_index, variable_index]
                        prototype_latents = self.teacher(centers)
                        count = len(chosen)
                        message[batch_index, variable_index, patch_index] = (
                            weights.unsqueeze(-1) * prototype_latents
                        ).sum(dim=0)
                        selected[batch_index, variable_index, patch_index, :count] = torch.tensor(
                            [item[1] for item in chosen],
                            dtype=torch.long,
                            device=context.device,
                        )
                        selected_weights[
                            batch_index, variable_index, patch_index, :count
                        ] = weights
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
