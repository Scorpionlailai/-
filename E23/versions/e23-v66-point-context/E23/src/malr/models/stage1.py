from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

import torch
from torch import Tensor, nn

from malr.data.normalization import denormalize_forecast
from malr.data.patching import PATCH_OBSERVED
from malr.training.losses import (
    Stage1Loss,
    imputation_huber,
    imputation_mae,
    imputation_mse,
    latent_recovery_mse,
)

from .patchtst import PatchTSTCore
from .recovery import MissingAwareRecovery, RecoveryOutput


class ModelVariant(str, Enum):
    CLEAN = "clean"
    MISSING = "missing"
    MISSING_PRETRAINED = "missing_pretrained"
    OURS = "ours"


@dataclass(frozen=True)
class Stage1Forward:
    forecast: Tensor
    normalized_forecast: Tensor
    latents: Tensor
    recovery: RecoveryOutput | None
    teacher_target: Tensor | None
    reconstruction: Tensor | None = None
    reconstruction_coverage: Tensor | None = None


@dataclass(frozen=True)
class Stage1TrainingResult:
    output: Stage1Forward
    loss: Stage1Loss
    teacher_target: Tensor
    prototype_latents: Tensor
    diagnostics: dict[str, Tensor]


class Stage1Model(nn.Module):
    """Single composition root for all four stage-1 forecasting variants."""

    def __init__(
        self,
        core: PatchTSTCore,
        variant: ModelVariant | str,
        *,
        recovery: MissingAwareRecovery | None = None,
    ) -> None:
        super().__init__()
        self.core = core
        self.variant = ModelVariant(variant)
        self.recovery = recovery
        if self.variant is ModelVariant.OURS and recovery is None:
            raise ValueError("ours variant requires MissingAwareRecovery")
        if self.variant is not ModelVariant.OURS and recovery is not None:
            raise ValueError("only ours variant may own a recovery front-end")
        self._clean_initialized = self.variant is not ModelVariant.MISSING_PRETRAINED

    @property
    def teacher_projection(self) -> nn.Module:
        if self.variant is not ModelVariant.OURS:
            raise AttributeError("this model has no recovery teacher projection")
        return self.core.W_P

    def teacher_target(
        self,
        clean_raw_patches: Tensor,
        query_mean: Tensor,
        query_std: Tensor,
    ) -> Tensor:
        if self.variant is not ModelVariant.OURS:
            raise ValueError("teacher targets are only defined for ours")
        if clean_raw_patches.ndim != 4:
            raise ValueError("clean_raw_patches must have four dimensions")
        if query_mean.shape != clean_raw_patches.shape[:2] or query_std.shape != query_mean.shape:
            raise ValueError("query normalization statistics do not match clean patches")
        normalized = (
            clean_raw_patches - query_mean.unsqueeze(-1).unsqueeze(-1)
        ) / query_std.unsqueeze(-1).unsqueeze(-1)
        with torch.no_grad():
            return self.core.project_clean_patches(normalized).detach()

    def forward(
        self,
        patch_values: Tensor,
        patch_mask: Tensor,
        rho: Tensor,
        state: Tensor,
        query_mean: Tensor,
        query_std: Tensor,
        *,
        raw_context: Tensor | None = None,
        context_mask: Tensor | None = None,
        train_std: Tensor | None = None,
        clean_raw_patches: Tensor | None = None,
        patch_starts: Tensor | None = None,
        patch_valid: Tensor | None = None,
        edge_utility_enabled: bool = False,
    ) -> Stage1Forward:
        if self.variant is ModelVariant.MISSING_PRETRAINED and not self._clean_initialized:
            raise RuntimeError(f"{self.variant.value} requires clean-checkpoint initialization")
        if clean_raw_patches is not None and not self.training:
            raise ValueError("clean teacher input is training-only")

        recovery_output: RecoveryOutput | None = None
        base_latents = self.core.project_clean_patches(patch_values)
        teacher_target = None
        if clean_raw_patches is not None:
            teacher_target = self.teacher_target(clean_raw_patches, query_mean, query_std)
        if self.variant is ModelVariant.OURS:
            assert self.recovery is not None
            recovery_output = self.recovery(
                patch_values,
                patch_mask,
                rho,
                state,
                raw_context=raw_context,
                context_mask=context_mask,
                query_mean=query_mean,
                query_std=query_std,
                train_std=train_std,
                base_latents=base_latents,
                prototype_projection=self.core.W_P,
                patch_starts=patch_starts,
                clean_target_latent=teacher_target,
                patch_valid=patch_valid,
                edge_utility_enabled=edge_utility_enabled,
            )
            latents = recovery_output.latents
        else:
            latents = base_latents

        normalized_forecast = self.core.forecast_from_latents(latents)
        forecast = denormalize_forecast(
            normalized_forecast.permute(0, 2, 1), query_mean, query_std
        ).permute(0, 2, 1).contiguous()
        return Stage1Forward(
            forecast=forecast,
            normalized_forecast=normalized_forecast,
            latents=latents,
            recovery=recovery_output,
            teacher_target=teacher_target,
        )

    def training_objective(
        self,
        patch_values: Tensor,
        patch_mask: Tensor,
        rho: Tensor,
        state: Tensor,
        query_mean: Tensor,
        query_std: Tensor,
        forecast_target: Tensor,
        clean_raw_patches: Tensor | None,
        patch_valid: Tensor,
        *,
        raw_context: Tensor | None = None,
        context_mask: Tensor | None = None,
        train_std: Tensor | None = None,
        lambda_rec: float = 1.0,
        lambda_keep: float = 0.0,
        use_recovery_loss: bool = True,
        imputation_target: Tensor | None = None,
        lambda_imputation_mse: float = 0.0,
        lambda_imputation_mae: float = 0.0,
        lambda_imputation_huber: float = 0.0,
        imputation_huber_delta: float = 1.0,
        lambda_edge_utility: float = 0.0,
        patch_starts: Tensor | None = None,
    ) -> Stage1TrainingResult:
        if self.variant is ModelVariant.OURS and use_recovery_loss and clean_raw_patches is None:
            raise ValueError("ours recovery loss requires clean raw patches")
        imputation_enabled = (
            float(lambda_imputation_mse) > 0.0
            or float(lambda_imputation_mae) > 0.0
            or float(lambda_imputation_huber) > 0.0
        )
        if imputation_enabled and self.variant is not ModelVariant.OURS:
            raise ValueError("imputation supervision is only defined for ours")
        if imputation_enabled and imputation_target is None:
            raise ValueError("imputation supervision requires an imputation target")
        if imputation_enabled and context_mask is None:
            raise ValueError("imputation supervision requires a context mask")
        edge_utility_enabled = float(lambda_edge_utility) > 0.0
        if edge_utility_enabled and clean_raw_patches is None:
            raise ValueError("edge utility supervision requires clean raw patches")
        output = self(
            patch_values,
            patch_mask,
            rho,
            state,
            query_mean,
            query_std,
            raw_context=raw_context,
            context_mask=context_mask,
            train_std=train_std,
            patch_starts=patch_starts,
            clean_raw_patches=(
                clean_raw_patches
                if self.variant is ModelVariant.OURS
                and (use_recovery_loss or edge_utility_enabled)
                else None
            ),
            patch_valid=patch_valid,
            edge_utility_enabled=edge_utility_enabled,
        )
        if output.teacher_target is not None and use_recovery_loss:
            recovery_loss = latent_recovery_mse(
                output.latents, output.teacher_target, state, patch_valid
            )
            teacher_target = output.teacher_target
        else:
            recovery_loss = output.latents.sum() * 0.0
            teacher_target = torch.empty(0, device=output.latents.device)
        if imputation_enabled:
            assert self.recovery is not None
            assert imputation_target is not None
            assert context_mask is not None
            reconstruction, reconstruction_coverage = self.recovery.reconstruct(
                output.latents
            )
            reconstruction_loss_mse = imputation_mse(
                reconstruction,
                imputation_target,
                context_mask,
                reconstruction_coverage,
            )
            reconstruction_loss_mae = imputation_mae(
                reconstruction,
                imputation_target,
                context_mask,
                reconstruction_coverage,
            )
            reconstruction_loss_huber = imputation_huber(
                reconstruction,
                imputation_target,
                context_mask,
                reconstruction_coverage,
                delta=imputation_huber_delta,
            )
            output = replace(
                output,
                reconstruction=reconstruction,
                reconstruction_coverage=reconstruction_coverage,
            )
        else:
            reconstruction_loss_mse = output.latents.sum() * 0.0
            reconstruction_loss_mae = output.latents.sum() * 0.0
            reconstruction_loss_huber = output.latents.sum() * 0.0
        damaged_count = int(
            ((state != PATCH_OBSERVED) & patch_valid.to(dtype=torch.bool)).sum().item()
        )
        loss = Stage1Loss.create(
            output.forecast,
            forecast_target,
            recovery_loss,
            (
                output.recovery.filter_keep_loss
                if output.recovery is not None
                else output.latents.sum() * 0.0
            ),
            damaged_patch_count=damaged_count,
            lambda_rec=lambda_rec if use_recovery_loss else 0.0,
            lambda_keep=lambda_keep,
            imputation_mse=reconstruction_loss_mse,
            imputation_mae=reconstruction_loss_mae,
            lambda_imputation_mse=(
                float(lambda_imputation_mse) if imputation_enabled else 0.0
            ),
            lambda_imputation_mae=(
                float(lambda_imputation_mae) if imputation_enabled else 0.0
            ),
            imputation_huber=reconstruction_loss_huber,
            lambda_imputation_huber=(
                float(lambda_imputation_huber) if imputation_enabled else 0.0
            ),
            edge_utility=(
                output.recovery.edge_utility_loss
                if output.recovery is not None
                else output.latents.sum() * 0.0
            ),
            lambda_edge_utility=(
                float(lambda_edge_utility) if edge_utility_enabled else 0.0
            ),
        )
        if output.recovery is None:
            prototype_latents = torch.empty(0, device=output.latents.device)
            diagnostics: dict[str, Tensor] = {}
        else:
            prototype_latents = output.recovery.messages.historical.detach()
            diagnostics = output.recovery.diagnostics
        return Stage1TrainingResult(
            output=output,
            loss=loss,
            teacher_target=teacher_target.detach(),
            prototype_latents=prototype_latents,
            diagnostics=diagnostics,
        )


def _extract_core_state(checkpoint: Mapping[str, Any]) -> Mapping[str, Tensor]:
    state: Mapping[str, Any] = checkpoint
    if "model_state" in state:
        state = state["model_state"]
    if not state:
        raise ValueError("clean checkpoint state is empty")
    if all(str(key).startswith("core.") for key in state):
        return {str(key)[5:]: value for key, value in state.items()}
    return state  # type: ignore[return-value]


def initialize_from_clean_checkpoint(
    model: Stage1Model,
    checkpoint: Mapping[str, Any],
) -> None:
    core_state = _extract_core_state(checkpoint)
    incompatible = model.core.load_state_dict(core_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError("clean checkpoint is incompatible with PatchTST core")
    if model.variant is ModelVariant.OURS:
        assert model.recovery is not None
    model._clean_initialized = True
