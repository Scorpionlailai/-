from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class TrainVariableStats:
    mean: Tensor
    std: Tensor

    def __post_init__(self) -> None:
        if self.mean.ndim != 1 or self.std.ndim != 1:
            raise ValueError("train variable statistics must be one-dimensional")
        if self.mean.shape != self.std.shape:
            raise ValueError("train mean and std shapes must match")
        if torch.any(self.std <= 0):
            raise ValueError("train variable std must be positive")


@dataclass(frozen=True)
class MaskedNormalization:
    student: Tensor
    target: Tensor
    mean: Tensor
    std: Tensor
    effective_mask: Tensor
    used_fallback: Tensor


def fit_train_variable_stats(values: Tensor, eps: float = 1e-5) -> TrainVariableStats:
    if values.ndim != 2:
        raise ValueError("train values must have shape [time, variables]")
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("train values cannot be empty")
    if eps < 0:
        raise ValueError("eps cannot be negative")
    if not torch.isfinite(values).all():
        raise ValueError("train values must be finite")
    mean = values.mean(dim=0).detach()
    variance = ((values - mean) ** 2).mean(dim=0)
    std = torch.sqrt(variance + eps).detach()
    std = torch.where(std > 0, std, torch.ones_like(std))
    return TrainVariableStats(mean=mean, std=std)


def standardize_values(values: Tensor, stats: TrainVariableStats) -> Tensor:
    """Apply train-fitted, per-variable StandardScaler statistics."""

    if values.ndim != 2:
        raise ValueError("values must have shape [time, variables]")
    if values.shape[1] != stats.mean.numel():
        raise ValueError("standardization statistics do not match variable count")
    if not torch.isfinite(values).all():
        raise ValueError("values must be finite")
    mean = stats.mean.to(device=values.device, dtype=values.dtype)
    std = stats.std.to(device=values.device, dtype=values.dtype)
    return (values - mean) / std


def _broadcast_validity(validity: Tensor | None, reference: Tensor) -> Tensor:
    if validity is None:
        return torch.ones_like(reference, dtype=torch.bool)
    result = validity.to(device=reference.device, dtype=torch.bool)
    if result.shape == reference.shape[:2]:
        result = result.unsqueeze(-1).expand_as(reference)
    elif result.shape == (*reference.shape[:2], 1):
        result = result.expand_as(reference)
    if result.shape != reference.shape:
        raise ValueError("validity must broadcast to [batch, time, variables]")
    return result


def masked_window_normalize(
    clean: Tensor,
    observed_mask: Tensor,
    fallback: TrainVariableStats,
    eps: float = 1e-5,
    validity: Tensor | None = None,
) -> MaskedNormalization:
    if clean.ndim != 3:
        raise ValueError("clean input must have shape [batch, time, variables]")
    if observed_mask.shape != clean.shape:
        raise ValueError("observed mask must match clean input")
    if clean.shape[-1] != fallback.mean.numel():
        raise ValueError("fallback statistics do not match variable count")
    if eps <= 0:
        raise ValueError("eps must be positive")

    valid = _broadcast_validity(validity, clean)
    effective = observed_mask.to(dtype=torch.bool, device=clean.device) & valid
    counts = effective.sum(dim=1)
    safe_counts = counts.clamp_min(1).to(dtype=clean.dtype)
    observed_values = torch.where(effective, clean, torch.zeros_like(clean))
    observed_mean = observed_values.sum(dim=1) / safe_counts
    used_fallback = counts == 0

    fallback_mean = fallback.mean.to(device=clean.device, dtype=clean.dtype)
    fallback_std = fallback.std.to(device=clean.device, dtype=clean.dtype)
    mean = torch.where(used_fallback, fallback_mean.unsqueeze(0), observed_mean)
    centered = torch.where(
        effective,
        clean - mean.unsqueeze(1),
        torch.zeros_like(clean),
    )
    observed_variance = centered.square().sum(dim=1) / safe_counts
    observed_std = torch.sqrt(observed_variance + eps)
    std = torch.where(used_fallback, fallback_std.unsqueeze(0), observed_std)

    mean = mean.detach()
    std = std.detach()
    normalized = (clean - mean.unsqueeze(1)) / std.unsqueeze(1)
    student = torch.where(effective, normalized, torch.zeros_like(normalized))
    target = normalized.detach()
    return MaskedNormalization(
        student=student,
        target=target,
        mean=mean,
        std=std,
        effective_mask=effective,
        used_fallback=used_fallback,
    )


def denormalize_forecast(forecast: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    if forecast.ndim != 3:
        raise ValueError("forecast must have shape [batch, horizon, variables]")
    if mean.ndim != 2 or std.ndim != 2 or mean.shape != std.shape:
        raise ValueError("mean and std must have matching [batch, variables] shapes")
    if forecast.shape[0] != mean.shape[0] or forecast.shape[2] != mean.shape[1]:
        raise ValueError("forecast does not match normalization statistics")
    return forecast * std.unsqueeze(1) + mean.unsqueeze(1)
