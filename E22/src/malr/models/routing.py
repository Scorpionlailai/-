"""Seven-way hard relation routing and training-time oracle labels."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


ROUTE_OPTIONS: tuple[str, ...] = (
    "none",
    "historical",
    "temporal",
    "variable",
    "historical_temporal",
    "historical_variable",
    "temporal_variable",
)

_ROUTE_RELATION_MASK = torch.tensor(
    [
        [False, False, False],
        [True, False, False],
        [False, True, False],
        [False, False, True],
        [True, True, False],
        [True, False, True],
        [False, True, True],
    ],
    dtype=torch.bool,
)


@dataclass(frozen=True)
class OracleRoute:
    """Detached route labels and the availability mask used to derive them."""

    index: Tensor
    one_hot: Tensor
    available: Tensor
    selected_error: Tensor
    relative_gain: Tensor


def route_relation_mask(*, device: torch.device | None = None) -> Tensor:
    """Return the fixed route-to-relation incidence matrix."""

    return _ROUTE_RELATION_MASK.to(device=device).clone()


def route_availability_mask(relation_available: Tensor) -> Tensor:
    """Expand historical/temporal/variable availability to seven routes."""

    if relation_available.ndim < 1 or relation_available.shape[-1] != 3:
        raise ValueError("relation_available must have shape [..., 3]")
    relation_available = relation_available.to(dtype=torch.bool)
    relation_mask = route_relation_mask(device=relation_available.device)
    route_available = relation_available.unsqueeze(-2) | ~relation_mask.view(
        *((1,) * (relation_available.ndim - 1)), 7, 3
    )
    return route_available.all(dim=-1)


def _validate_oracle_inputs(
    combination_error: Tensor,
    baseline_error: Tensor,
    relation_available: Tensor,
) -> None:
    if combination_error.ndim < 1 or combination_error.shape[-1] != len(ROUTE_OPTIONS):
        raise ValueError("combination_error must have shape [..., 7]")
    if baseline_error.shape != combination_error.shape[:-1]:
        raise ValueError("baseline_error must match combination_error without route dimension")
    if relation_available.shape != (*combination_error.shape[:-1], 3):
        raise ValueError("relation_available must have shape [..., 3]")
    if not combination_error.is_floating_point() or not baseline_error.is_floating_point():
        raise ValueError("route errors must be floating-point")
    if not torch.isfinite(combination_error).all() or not torch.isfinite(baseline_error).all():
        raise ValueError("route errors must contain only finite values")
    if torch.any(combination_error < 0.0) or torch.any(baseline_error < 0.0):
        raise ValueError("route errors must be non-negative")


def oracle_route_labels(
    combination_error: Tensor,
    baseline_error: Tensor,
    relation_available: Tensor,
    *,
    double_improvement_threshold: float = 0.05,
    eps: float = 1e-8,
) -> OracleRoute:
    """Build detached seven-way labels from batched counterfactual errors."""

    _validate_oracle_inputs(combination_error, baseline_error, relation_available)
    if not torch.isfinite(torch.as_tensor(double_improvement_threshold)) or double_improvement_threshold < 0.0:
        raise ValueError("double_improvement_threshold must be non-negative and finite")
    if not torch.isfinite(torch.as_tensor(eps)) or eps <= 0.0:
        raise ValueError("eps must be positive and finite")

    available = route_availability_mask(relation_available)
    single_available = available[..., 1:4]
    double_available = available[..., 4:7]
    positive_infinity = torch.full_like(combination_error[..., 0], torch.inf)

    single_errors = combination_error[..., 1:4].masked_fill(~single_available, torch.inf)
    best_single_error, best_single_offset = single_errors.min(dim=-1)
    best_single_index = best_single_offset + 1
    has_single = single_available.any(dim=-1)

    double_errors = combination_error[..., 4:7].masked_fill(~double_available, torch.inf)
    best_double_error, best_double_offset = double_errors.min(dim=-1)
    best_double_index = best_double_offset + 4
    has_double = double_available.any(dim=-1)

    double_gain_over_single = (best_single_error - best_double_error) / (
        best_single_error + float(eps)
    )
    choose_double = has_double & (
        ~has_single | (double_gain_over_single >= float(double_improvement_threshold))
    )
    choose_single = has_single & ~choose_double
    candidate_index = torch.where(
        choose_double,
        best_double_index,
        torch.where(choose_single, best_single_index, torch.zeros_like(best_single_index)),
    )
    candidate_error = torch.where(
        choose_double,
        best_double_error,
        torch.where(choose_single, best_single_error, baseline_error),
    )
    candidate_error = torch.where(
        torch.isfinite(candidate_error), candidate_error, positive_infinity
    )
    candidate_gain = (baseline_error - candidate_error) / (
        baseline_error + float(eps)
    )
    has_positive_gain = (candidate_gain > 0.0) & torch.isfinite(candidate_gain)
    index = torch.where(has_positive_gain, candidate_index, torch.zeros_like(candidate_index))
    selected_error = torch.where(index == 0, baseline_error, candidate_error)
    relative_gain = (baseline_error - selected_error) / (baseline_error + float(eps))
    one_hot = F.one_hot(index, num_classes=len(ROUTE_OPTIONS)).to(
        dtype=combination_error.dtype
    )
    return OracleRoute(
        index=index.detach(),
        one_hot=one_hot.detach(),
        available=available.detach(),
        selected_error=selected_error.detach(),
        relative_gain=relative_gain.detach(),
    )


def hard_route_select(logits: Tensor, available: Tensor) -> Tensor:
    """Select the highest-scoring available route with deterministic argmax."""

    if logits.ndim < 1 or logits.shape[-1] != len(ROUTE_OPTIONS):
        raise ValueError("logits must have shape [..., 7]")
    if available.shape != logits.shape:
        raise ValueError("available must match logits")
    if not logits.is_floating_point() or not torch.isfinite(logits).all():
        raise ValueError("logits must be finite floating-point values")
    available = available.to(device=logits.device, dtype=torch.bool)
    if not available.any(dim=-1).all():
        raise ValueError("each target must have at least one available route")
    masked_logits = logits.masked_fill(~available, -torch.inf)
    return masked_logits.argmax(dim=-1)


def route_cross_entropy(
    logits: Tensor,
    oracle_index: Tensor,
    available: Tensor | None = None,
) -> Tensor:
    """Compute hard-route cross-entropy, optionally masking unavailable routes."""

    if logits.ndim < 1 or logits.shape[-1] != len(ROUTE_OPTIONS):
        raise ValueError("logits must have shape [..., 7]")
    if oracle_index.shape != logits.shape[:-1]:
        raise ValueError("oracle_index must match logits without route dimension")
    if not oracle_index.dtype in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise ValueError("oracle_index must be an integer tensor")
    if torch.any((oracle_index < 0) | (oracle_index >= len(ROUTE_OPTIONS))):
        raise ValueError("oracle_index contains an unknown route")
    if not logits.is_floating_point() or not torch.isfinite(logits).all():
        raise ValueError("logits must be finite floating-point values")
    masked_logits = logits
    if available is not None:
        if available.shape != logits.shape:
            raise ValueError("available must match logits")
        available = available.to(device=logits.device, dtype=torch.bool)
        if not available.any(dim=-1).all():
            raise ValueError("each target must have at least one available route")
        oracle_available = available.gather(-1, oracle_index.to(logits.device).unsqueeze(-1)).squeeze(-1)
        if not oracle_available.all():
            raise ValueError("oracle_index selects an unavailable route")
        masked_logits = logits.masked_fill(~available, -torch.inf)
    if logits.numel() == 0:
        return logits.sum() * 0.0
    return F.cross_entropy(
        masked_logits.reshape(-1, len(ROUTE_OPTIONS)),
        oracle_index.to(device=logits.device).reshape(-1),
    )


__all__ = [
    "OracleRoute",
    "ROUTE_OPTIONS",
    "hard_route_select",
    "oracle_route_labels",
    "route_availability_mask",
    "route_cross_entropy",
    "route_relation_mask",
]
