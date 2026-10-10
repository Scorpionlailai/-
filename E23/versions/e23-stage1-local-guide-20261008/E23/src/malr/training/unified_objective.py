"""Fixed first-round objective weights and residual warm-up schedule."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


DEFAULT_IMPUTATION_HUBER_WEIGHT = 0.10
DEFAULT_EDGE_GAIN_WEIGHT = 0.05
DEFAULT_ROUTE_WEIGHT = 0.05
DEFAULT_MEMORY_WEIGHT = 0.01


def residual_injection_weight(epoch_index: int, warmup_epochs: int = 10) -> float:
    """Linearly ramp the final residual from zero to one over ``warmup_epochs``."""

    if type(epoch_index) is not int or epoch_index < 0:
        raise ValueError("epoch_index must be a non-negative integer")
    if type(warmup_epochs) is not int or warmup_epochs <= 0:
        raise ValueError("warmup_epochs must be a positive integer")
    return min(1.0, float(epoch_index) / float(warmup_epochs))


@dataclass(frozen=True)
class UnifiedStage1Loss:
    total: Tensor
    forecast: Tensor
    imputation_huber: Tensor
    edge_gain: Tensor
    route: Tensor
    memory: Tensor
    weighted_imputation_huber: Tensor
    weighted_edge_gain: Tensor
    weighted_route: Tensor
    weighted_memory: Tensor


def _validate_component(name: str, value: Tensor) -> None:
    if not isinstance(value, Tensor) or value.numel() != 1:
        raise ValueError(f"{name} must be a scalar tensor")
    if not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite and floating-point")


def unified_stage1_loss(
    forecast: Tensor,
    imputation_huber: Tensor,
    edge_gain: Tensor,
    route: Tensor,
    memory: Tensor,
    *,
    lambda_imputation_huber: float = DEFAULT_IMPUTATION_HUBER_WEIGHT,
    lambda_edge: float = DEFAULT_EDGE_GAIN_WEIGHT,
    lambda_route: float = DEFAULT_ROUTE_WEIGHT,
    lambda_memory: float = DEFAULT_MEMORY_WEIGHT,
) -> UnifiedStage1Loss:
    """Combine the fixed first-round objective without legacy auxiliary terms."""

    components = {
        "forecast": forecast,
        "imputation_huber": imputation_huber,
        "edge_gain": edge_gain,
        "route": route,
        "memory": memory,
    }
    for name, value in components.items():
        _validate_component(name, value)
    weights = {
        "lambda_imputation_huber": lambda_imputation_huber,
        "lambda_edge": lambda_edge,
        "lambda_route": lambda_route,
        "lambda_memory": lambda_memory,
    }
    for name, value in weights.items():
        if not torch.isfinite(torch.as_tensor(value)) or value < 0.0:
            raise ValueError("loss weights must be finite and non-negative")

    weighted_imputation_huber = float(lambda_imputation_huber) * imputation_huber
    weighted_edge_gain = float(lambda_edge) * edge_gain
    weighted_route = float(lambda_route) * route
    weighted_memory = float(lambda_memory) * memory
    total = (
        forecast
        + weighted_imputation_huber
        + weighted_edge_gain
        + weighted_route
        + weighted_memory
    )
    return UnifiedStage1Loss(
        total=total,
        forecast=forecast,
        imputation_huber=imputation_huber,
        edge_gain=edge_gain,
        route=route,
        memory=memory,
        weighted_imputation_huber=weighted_imputation_huber,
        weighted_edge_gain=weighted_edge_gain,
        weighted_route=weighted_route,
        weighted_memory=weighted_memory,
    )


__all__ = [
    "DEFAULT_EDGE_GAIN_WEIGHT",
    "DEFAULT_IMPUTATION_HUBER_WEIGHT",
    "DEFAULT_MEMORY_WEIGHT",
    "DEFAULT_ROUTE_WEIGHT",
    "UnifiedStage1Loss",
    "residual_injection_weight",
    "unified_stage1_loss",
]
