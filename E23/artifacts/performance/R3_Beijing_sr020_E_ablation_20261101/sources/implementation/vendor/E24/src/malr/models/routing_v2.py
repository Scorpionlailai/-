"""Eight-way hard relation routing with stability-gated oracle labels."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import torch
from torch import Tensor, nn
from torch.nn import functional as F


V2_ROUTE_OPTIONS: tuple[str, ...] = (
    "none",
    "historical",
    "temporal",
    "variable",
    "historical_temporal",
    "historical_variable",
    "temporal_variable",
    "historical_temporal_variable",
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
        [True, True, True],
    ],
    dtype=torch.bool,
)
_ROUTE_COMPLEXITY = _ROUTE_RELATION_MASK.sum(dim=-1)
_STRICT_SUBSETS: tuple[tuple[int, ...], ...] = (
    (),
    (0,),
    (0,),
    (0,),
    (0, 1, 2),
    (0, 1, 3),
    (0, 2, 3),
    (0, 1, 2, 3, 4, 5, 6),
)


@dataclass(frozen=True)
class OracleRoute:
    """Detached V2 route labels and the availability used to derive them."""

    index: Tensor
    one_hot: Tensor
    available: Tensor
    selected_error: Tensor
    relative_gain: Tensor


def route_relation_mask(*, device: torch.device | None = None) -> Tensor:
    """Return the fixed eight-route relation-incidence matrix."""

    return _ROUTE_RELATION_MASK.to(device=device).clone()


def route_availability_mask(relation_available: Tensor) -> Tensor:
    """Expand historical/temporal/variable availability to eight routes."""

    if relation_available.ndim < 1 or relation_available.shape[-1] != 3:
        raise ValueError("relation_available must have shape [..., 3]")
    relation_available = relation_available.to(dtype=torch.bool)
    relation_mask = route_relation_mask(device=relation_available.device)
    route_available = relation_available.unsqueeze(-2) | ~relation_mask.view(
        *((1,) * (relation_available.ndim - 1)), len(V2_ROUTE_OPTIONS), 3
    )
    return route_available.all(dim=-1)


def _validate_error_inputs(
    combination_error: Tensor,
    baseline_error: Tensor,
    relation_available: Tensor,
) -> None:
    if combination_error.ndim < 1 or combination_error.shape[-1] != len(V2_ROUTE_OPTIONS):
        raise ValueError("combination_error must have shape [..., 8]")
    if baseline_error.shape != combination_error.shape[:-1]:
        raise ValueError("baseline_error must match combination_error without route dimension")
    if relation_available.shape != (*combination_error.shape[:-1], 3):
        raise ValueError("relation_available must have shape [..., 3]")
    if not combination_error.is_floating_point() or not baseline_error.is_floating_point():
        raise TypeError("route errors must be floating-point")
    if not torch.isfinite(baseline_error).all():
        raise ValueError("baseline_error must contain only finite values")
    finite_errors = torch.isfinite(combination_error)
    if torch.any((combination_error < 0.0) & finite_errors) or torch.any(
        baseline_error < 0.0
    ):
        raise ValueError("route errors must be non-negative")


def _validate_margin(complexity_margin: Tensor) -> None:
    if not isinstance(complexity_margin, Tensor):
        raise TypeError("complexity_margin must be a Tensor")
    if complexity_margin.shape != (3,):
        raise ValueError("complexity_margin must have shape [3]")
    if not complexity_margin.is_floating_point():
        raise TypeError("complexity_margin must be floating-point")
    if not torch.isfinite(complexity_margin).all() or torch.any(complexity_margin < 0.0):
        raise ValueError("complexity_margin must be finite and non-negative")


def oracle_route_labels(
    combination_error: Tensor,
    baseline_error: Tensor,
    relation_available: Tensor,
    *,
    complexity_margin: Tensor,
    eps: float = 1e-8,
    allow_combinations: bool = True,
) -> OracleRoute:
    """Admit routes only when they improve over their best strict subset.

    ``allow_combinations=False`` keeps the legacy eight-way tensor protocol
    while restricting supervision to ``none`` and the three single relations.
    """

    _validate_error_inputs(combination_error, baseline_error, relation_available)
    _validate_margin(complexity_margin)
    if not isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be positive and finite")

    available = route_availability_mask(relation_available)
    if not isinstance(allow_combinations, bool):
        raise TypeError("allow_combinations must be a bool")
    if not allow_combinations:
        available = available.clone()
        available[..., 4:] = False
    errors = combination_error.clone()
    errors[..., 0] = baseline_error
    finite = torch.isfinite(errors)
    candidate = torch.zeros_like(available)
    candidate[..., 0] = True
    margin = complexity_margin.to(device=errors.device, dtype=errors.dtype)

    for route_index in range(1, len(V2_ROUTE_OPTIONS)):
        strict_subset = _STRICT_SUBSETS[route_index]
        parent_error = errors[..., strict_subset].masked_fill(
            ~(available[..., strict_subset] & finite[..., strict_subset]), torch.inf
        ).min(dim=-1).values
        has_parent = torch.isfinite(parent_error)
        improvement = (parent_error - errors[..., route_index]) / (parent_error + eps)
        complexity = int(_ROUTE_COMPLEXITY[route_index].item())
        candidate[..., route_index] = (
            available[..., route_index]
            & finite[..., route_index]
            & has_parent
            & improvement.ge(margin[complexity - 1])
        )

    selected_index = torch.zeros_like(errors[..., 0], dtype=torch.long)
    selected_error = errors[..., 0]
    selected_complexity = torch.zeros_like(selected_index)
    for route_index in range(1, len(V2_ROUTE_OPTIONS)):
        route_error = errors[..., route_index]
        complexity = int(_ROUTE_COMPLEXITY[route_index].item())
        prefer_route = candidate[..., route_index] & (
            route_error.lt(selected_error)
            | (
                route_error.eq(selected_error)
                & selected_complexity.gt(complexity)
            )
        )
        selected_index = torch.where(
            prefer_route,
            torch.full_like(selected_index, route_index),
            selected_index,
        )
        selected_error = torch.where(prefer_route, route_error, selected_error)
        selected_complexity = torch.where(
            prefer_route,
            torch.full_like(selected_complexity, complexity),
            selected_complexity,
        )

    relative_gain = (
        (baseline_error - selected_error) / (baseline_error + eps)
    ).nan_to_num(nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
    one_hot = F.one_hot(selected_index, num_classes=len(V2_ROUTE_OPTIONS)).to(
        dtype=combination_error.dtype
    )
    return OracleRoute(
        index=selected_index.detach(),
        one_hot=one_hot.detach(),
        available=available.detach(),
        selected_error=selected_error.detach(),
        relative_gain=relative_gain.detach(),
    )


def direct_single_route_select(predicted_gain: Tensor, available: Tensor) -> Tensor:
    """Select the best positive H/T/V gain, otherwise return the none route.

    The returned route indices retain the fixed V2 numbering: ``0`` is none,
    followed by historical, temporal, and variable at indices ``1``--``3``.
    Combination route indices are intentionally never produced.
    """

    if predicted_gain.ndim < 1 or predicted_gain.shape[-1] != 3:
        raise ValueError("predicted_gain must have shape [..., 3]")
    if available.shape != predicted_gain.shape:
        raise ValueError("available must match predicted_gain")
    if not predicted_gain.is_floating_point() or not torch.isfinite(predicted_gain).all():
        raise ValueError("predicted_gain must be finite floating-point values")
    available = available.to(device=predicted_gain.device, dtype=torch.bool)
    masked_gain = predicted_gain.masked_fill(~available, -torch.inf)
    best_gain, best_branch = masked_gain.max(dim=-1)
    return torch.where(
        best_gain.gt(0.0),
        best_branch + 1,
        torch.zeros_like(best_branch),
    )


class RouteStabilityTracker(nn.Module):
    """Train-only EMA variation of strict-subset improvements by complexity."""

    def __init__(
        self,
        base_margin: float = 0.10,
        min_targets: int = 128,
        decay: float = 0.9,
    ) -> None:
        super().__init__()
        if not isfinite(base_margin) or base_margin < 0.0:
            raise ValueError("base_margin must be finite and non-negative")
        if type(min_targets) is not int or min_targets <= 0:
            raise ValueError("min_targets must be a positive integer")
        if not isfinite(decay) or not 0.0 <= decay < 1.0:
            raise ValueError("decay must be finite and lie in [0, 1)")
        self.base_margin = float(base_margin)
        self.min_targets = min_targets
        self.decay = float(decay)
        self.register_buffer("mean", torch.zeros(3))
        self.register_buffer("second_moment", torch.zeros(3))
        self.register_buffer("count", torch.zeros(3, dtype=torch.long))

    def margins(self) -> Tensor:
        """Return a base margin until each complexity has enough observations."""

        std = (self.second_moment - self.mean.square()).clamp_min(0.0).sqrt()
        learned = torch.maximum(std, torch.full_like(std, self.base_margin))
        return torch.where(
            self.count.ge(self.min_targets), learned, torch.full_like(std, self.base_margin)
        )

    def _validate_update_inputs(self, combination_error: Tensor, available: Tensor) -> None:
        if combination_error.ndim < 1 or combination_error.shape[-1] != len(V2_ROUTE_OPTIONS):
            raise ValueError("combination_error must have shape [..., 8]")
        if available.shape != combination_error.shape:
            raise ValueError("available must match combination_error")
        if not combination_error.is_floating_point():
            raise TypeError("combination_error must be floating-point")
        if available.dtype != torch.bool:
            raise TypeError("available must be bool")
        finite_errors = torch.isfinite(combination_error)
        if torch.any((combination_error < 0.0) & finite_errors):
            raise ValueError("combination_error must be non-negative when finite")

    @torch.no_grad()
    def update(self, combination_error: Tensor, available: Tensor) -> None:
        """Aggregate finite available strict-subset improvements during training."""

        self._validate_update_inputs(combination_error, available)
        if not self.training:
            return
        errors = combination_error.detach().to(device=self.mean.device, dtype=self.mean.dtype)
        route_available = available.detach().to(device=self.mean.device, dtype=torch.bool)
        finite = torch.isfinite(errors)
        improvements: list[Tensor] = []
        complexities: list[Tensor] = []
        for route_index in range(1, len(V2_ROUTE_OPTIONS)):
            strict_subset = _STRICT_SUBSETS[route_index]
            parent_error = errors[..., strict_subset].masked_fill(
                ~(route_available[..., strict_subset] & finite[..., strict_subset]), torch.inf
            ).min(dim=-1).values
            take = (
                route_available[..., route_index]
                & finite[..., route_index]
                & torch.isfinite(parent_error)
            )
            if not take.any():
                continue
            improvement = (parent_error - errors[..., route_index]) / (parent_error + 1e-8)
            # A near-zero parent error is a valid observation, but its raw
            # relative ratio is not a useful stability statistic.  Bound only
            # the tracker observation; route admission still compares errors
            # directly and remains exact.
            improvement = torch.nan_to_num(
                improvement, nan=0.0, posinf=1.0, neginf=-1.0
            ).clamp(-1.0, 1.0)
            improvements.append(improvement[take])
            complexity = int(_ROUTE_COMPLEXITY[route_index].item()) - 1
            complexities.append(
                torch.full(
                    (int(take.sum().item()),), complexity, device=self.mean.device, dtype=torch.long
                )
            )
        if not improvements:
            return

        values = torch.cat(improvements)
        level = torch.cat(complexities)
        observations = torch.zeros_like(self.mean)
        improvement_sum = torch.zeros_like(self.mean)
        second_sum = torch.zeros_like(self.mean)
        observations.scatter_add_(0, level, torch.ones_like(values))
        improvement_sum.scatter_add_(0, level, values)
        second_sum.scatter_add_(0, level, values.square())
        present = observations.gt(0)
        batch_mean = improvement_sum / observations.clamp_min(1.0)
        batch_second = second_sum / observations.clamp_min(1.0)
        initialized = self.count.gt(0)
        updated_mean = torch.where(
            initialized,
            self.decay * self.mean + (1.0 - self.decay) * batch_mean,
            batch_mean,
        )
        updated_second = torch.where(
            initialized,
            self.decay * self.second_moment + (1.0 - self.decay) * batch_second,
            batch_second,
        )
        self.mean[present] = updated_mean[present]
        self.second_moment[present] = updated_second[present]
        self.count[present] += observations[present].to(dtype=self.count.dtype)


def hard_route_select(logits: Tensor, available: Tensor) -> Tensor:
    """Select the highest-scoring available V2 route with deterministic argmax."""

    if logits.ndim < 1 or logits.shape[-1] != len(V2_ROUTE_OPTIONS):
        raise ValueError("logits must have shape [..., 8]")
    if available.shape != logits.shape:
        raise ValueError("available must match logits")
    if not logits.is_floating_point() or not torch.isfinite(logits).all():
        raise ValueError("logits must be finite floating-point values")
    available = available.to(device=logits.device, dtype=torch.bool)
    if not available.any(dim=-1).all():
        raise ValueError("each target must have at least one available route")
    return logits.masked_fill(~available, -torch.inf).argmax(dim=-1)


def route_cross_entropy(
    logits: Tensor,
    oracle_index: Tensor,
    available: Tensor | None = None,
) -> Tensor:
    """Compute V2 hard-route cross-entropy with optional availability masking."""

    if logits.ndim < 1 or logits.shape[-1] != len(V2_ROUTE_OPTIONS):
        raise ValueError("logits must have shape [..., 8]")
    if oracle_index.shape != logits.shape[:-1]:
        raise ValueError("oracle_index must match logits without route dimension")
    if oracle_index.is_floating_point() or oracle_index.dtype == torch.bool:
        raise TypeError("oracle_index must have an integer dtype")
    if torch.any((oracle_index < 0) | (oracle_index >= len(V2_ROUTE_OPTIONS))):
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
        selected_available = available.gather(
            -1, oracle_index.to(logits.device).unsqueeze(-1)
        ).squeeze(-1)
        if not selected_available.all():
            raise ValueError("oracle_index selects an unavailable route")
        masked_logits = logits.masked_fill(~available, -torch.inf)
    if logits.numel() == 0:
        return logits.sum() * 0.0
    return F.cross_entropy(
        masked_logits.reshape(-1, len(V2_ROUTE_OPTIONS)),
        oracle_index.to(device=logits.device).reshape(-1),
    )


__all__ = [
    "OracleRoute",
    "RouteStabilityTracker",
    "V2_ROUTE_OPTIONS",
    "direct_single_route_select",
    "hard_route_select",
    "oracle_route_labels",
    "route_availability_mask",
    "route_cross_entropy",
    "route_relation_mask",
]
