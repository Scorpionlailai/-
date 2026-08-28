from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from malr.data.patching import PATCH_OBSERVED


def forecast_mse(prediction: Tensor, target: Tensor) -> Tensor:
    if prediction.shape != target.shape:
        raise ValueError("forecast prediction and target shapes must match")
    return torch.mean((prediction - target) ** 2)


def latent_recovery_mse(
    output: Tensor,
    target: Tensor,
    state: Tensor,
    patch_valid: Tensor,
) -> Tensor:
    if output.shape != target.shape or output.ndim != 4:
        raise ValueError("latent output and target must match [batch, variables, patches, d]")
    if state.shape != output.shape[:-1] or patch_valid.shape != state.shape:
        raise ValueError("state and patch_valid must match latent nodes")
    damaged = (state != PATCH_OBSERVED) & patch_valid.to(dtype=torch.bool)
    if not damaged.any():
        return output.sum() * 0.0
    squared_error = (output - target).square().mean(dim=-1)
    return squared_error[damaged].mean()


@dataclass(frozen=True)
class Stage1Loss:
    total: Tensor
    forecast_mse: Tensor
    recovery_mse: Tensor
    damaged_patch_count: int

    @classmethod
    def create(
        cls,
        prediction: Tensor,
        target: Tensor,
        recovery_mse: Tensor,
        *,
        damaged_patch_count: int,
        lambda_rec: float,
    ) -> "Stage1Loss":
        if lambda_rec < 0:
            raise ValueError("lambda_rec cannot be negative")
        prediction_loss = forecast_mse(prediction, target)
        total = prediction_loss + float(lambda_rec) * recovery_mse
        return cls(total, prediction_loss, recovery_mse, damaged_patch_count)

    def detached_metrics(self) -> dict[str, float | int]:
        return {
            "forecast_mse": float(self.forecast_mse.detach()),
            "recovery_mse": float(self.recovery_mse.detach()),
            "damaged_patch_count": self.damaged_patch_count,
        }
