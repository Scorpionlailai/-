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


def _missing_region(
    reconstruction: Tensor,
    target: Tensor,
    observed_mask: Tensor,
    coverage: Tensor | None,
) -> Tensor:
    if reconstruction.shape != target.shape or reconstruction.ndim != 3:
        raise ValueError(
            "reconstruction and target must match [batch, time, variables]"
        )
    if observed_mask.shape != reconstruction.shape:
        raise ValueError("observed mask must match reconstruction")
    missing = ~observed_mask.to(device=reconstruction.device, dtype=torch.bool)
    if coverage is not None:
        if coverage.shape != reconstruction.shape:
            raise ValueError("coverage must match reconstruction")
        missing = missing & coverage.to(
            device=reconstruction.device, dtype=torch.bool
        )
    return missing


def imputation_mse(
    reconstruction: Tensor,
    target: Tensor,
    observed_mask: Tensor,
    coverage: Tensor | None = None,
) -> Tensor:
    missing = _missing_region(reconstruction, target, observed_mask, coverage)
    if not missing.any():
        return reconstruction.sum() * 0.0
    return (reconstruction - target).square()[missing].mean()


def imputation_mae(
    reconstruction: Tensor,
    target: Tensor,
    observed_mask: Tensor,
    coverage: Tensor | None = None,
) -> Tensor:
    missing = _missing_region(reconstruction, target, observed_mask, coverage)
    if not missing.any():
        return reconstruction.sum() * 0.0
    return (reconstruction - target).abs()[missing].mean()


def imputation_huber(
    reconstruction: Tensor,
    target: Tensor,
    observed_mask: Tensor,
    coverage: Tensor | None = None,
    *,
    delta: float = 1.0,
) -> Tensor:
    if delta <= 0.0:
        raise ValueError("Huber delta must be positive")
    missing = _missing_region(reconstruction, target, observed_mask, coverage)
    if not missing.any():
        return reconstruction.sum() * 0.0
    absolute_error = (reconstruction - target).abs()[missing]
    quadratic = 0.5 * absolute_error.square()
    linear = float(delta) * (absolute_error - 0.5 * float(delta))
    return torch.where(absolute_error <= float(delta), quadratic, linear).mean()


@dataclass(frozen=True)
class Stage1Loss:
    total: Tensor
    forecast_mse: Tensor
    recovery_mse: Tensor
    filter_keep: Tensor
    damaged_patch_count: int
    imputation_mse: Tensor
    imputation_mae: Tensor
    weighted_imputation_mse: Tensor
    weighted_imputation_mae: Tensor
    imputation_huber: Tensor
    weighted_imputation_huber: Tensor
    edge_utility: Tensor
    weighted_edge_utility: Tensor

    @classmethod
    def create(
        cls,
        prediction: Tensor,
        target: Tensor,
        recovery_mse: Tensor,
        filter_keep: Tensor | None = None,
        *,
        damaged_patch_count: int,
        lambda_rec: float,
        lambda_keep: float = 0.0,
        imputation_mse: Tensor | None = None,
        imputation_mae: Tensor | None = None,
        lambda_imputation_mse: float = 0.0,
        lambda_imputation_mae: float = 0.0,
        imputation_huber: Tensor | None = None,
        lambda_imputation_huber: float = 0.0,
        edge_utility: Tensor | None = None,
        lambda_edge_utility: float = 0.0,
    ) -> "Stage1Loss":
        if (
            lambda_rec < 0
            or lambda_keep < 0
            or lambda_imputation_mse < 0
            or lambda_imputation_mae < 0
            or lambda_imputation_huber < 0
            or lambda_edge_utility < 0
        ):
            raise ValueError("loss weights cannot be negative")
        prediction_loss = forecast_mse(prediction, target)
        if filter_keep is None:
            filter_keep = prediction_loss * 0.0
        if imputation_mse is None:
            imputation_mse = prediction_loss * 0.0
        if imputation_mae is None:
            imputation_mae = prediction_loss * 0.0
        if imputation_huber is None:
            imputation_huber = prediction_loss * 0.0
        if edge_utility is None:
            edge_utility = prediction_loss * 0.0
        weighted_imputation_mse = float(lambda_imputation_mse) * imputation_mse
        weighted_imputation_mae = float(lambda_imputation_mae) * imputation_mae
        weighted_imputation_huber = (
            float(lambda_imputation_huber) * imputation_huber
        )
        weighted_edge_utility = float(lambda_edge_utility) * edge_utility
        total = (
            prediction_loss
            + float(lambda_rec) * recovery_mse
            + float(lambda_keep) * filter_keep
            + weighted_imputation_mse
            + weighted_imputation_mae
            + weighted_imputation_huber
            + weighted_edge_utility
        )
        return cls(
            total,
            prediction_loss,
            recovery_mse,
            filter_keep,
            damaged_patch_count,
            imputation_mse,
            imputation_mae,
            weighted_imputation_mse,
            weighted_imputation_mae,
            imputation_huber,
            weighted_imputation_huber,
            edge_utility,
            weighted_edge_utility,
        )

    def detached_metrics(self) -> dict[str, float | int]:
        return {
            "forecast_mse": float(self.forecast_mse.detach()),
            "recovery_mse": float(self.recovery_mse.detach()),
            "filter_keep": float(self.filter_keep.detach()),
            "imputation_mse": float(self.imputation_mse.detach()),
            "imputation_mae": float(self.imputation_mae.detach()),
            "weighted_imputation_mse": float(
                self.weighted_imputation_mse.detach()
            ),
            "weighted_imputation_mae": float(
                self.weighted_imputation_mae.detach()
            ),
            "imputation_huber": float(self.imputation_huber.detach()),
            "weighted_imputation_huber": float(
                self.weighted_imputation_huber.detach()
            ),
            "edge_utility": float(self.edge_utility.detach()),
            "weighted_edge_utility": float(self.weighted_edge_utility.detach()),
            "damaged_patch_count": self.damaged_patch_count,
        }
