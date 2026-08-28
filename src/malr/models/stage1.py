from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
from torch import Tensor, nn

from malr.data.normalization import denormalize_forecast
from malr.data.patching import PATCH_OBSERVED
from malr.training.losses import Stage1Loss, latent_recovery_mse

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
        self._clean_initialized = self.variant in {
            ModelVariant.CLEAN,
            ModelVariant.MISSING,
        }

    @property
    def teacher_projection(self) -> nn.Module:
        if (
            self.variant is not ModelVariant.OURS
            or self.recovery is None
            or self.recovery.historical is None
        ):
            raise AttributeError("this model has no clean teacher projection")
        return self.recovery.historical.teacher

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
            return self.teacher_projection(normalized).detach()

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
    ) -> Stage1Forward:
        if self.variant in {ModelVariant.MISSING_PRETRAINED, ModelVariant.OURS} and not self._clean_initialized:
            raise RuntimeError(f"{self.variant.value} requires clean-checkpoint initialization")
        if clean_raw_patches is not None and not self.training:
            raise ValueError("clean teacher input is training-only")

        recovery_output: RecoveryOutput | None = None
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
            )
            latents = recovery_output.latents
        else:
            latents = self.core.project_clean_patches(patch_values)

        normalized_forecast = self.core.forecast_from_latents(latents)
        forecast = denormalize_forecast(
            normalized_forecast.permute(0, 2, 1), query_mean, query_std
        ).permute(0, 2, 1).contiguous()
        target = None
        if clean_raw_patches is not None:
            target = self.teacher_target(clean_raw_patches, query_mean, query_std)
        return Stage1Forward(
            forecast=forecast,
            normalized_forecast=normalized_forecast,
            latents=latents,
            recovery=recovery_output,
            teacher_target=target,
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
        use_recovery_loss: bool = True,
    ) -> Stage1TrainingResult:
        if self.variant is ModelVariant.OURS and use_recovery_loss and clean_raw_patches is None:
            raise ValueError("ours recovery loss requires clean raw patches")
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
            clean_raw_patches=(
                clean_raw_patches
                if self.variant is ModelVariant.OURS and use_recovery_loss
                else None
            ),
        )
        if output.teacher_target is not None and use_recovery_loss:
            recovery_loss = latent_recovery_mse(
                output.latents, output.teacher_target, state, patch_valid
            )
            teacher_target = output.teacher_target
        else:
            recovery_loss = output.latents.sum() * 0.0
            teacher_target = torch.empty(0, device=output.latents.device)
        damaged_count = int(
            ((state != PATCH_OBSERVED) & patch_valid.to(dtype=torch.bool)).sum().item()
        )
        loss = Stage1Loss.create(
            output.forecast,
            forecast_target,
            recovery_loss,
            damaged_patch_count=damaged_count,
            lambda_rec=lambda_rec if use_recovery_loss else 0.0,
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
        model.recovery.W_x.load_state_dict(model.core.W_P.state_dict(), strict=True)
        model.teacher_projection.load_state_dict(model.core.W_P.state_dict(), strict=True)
        model.teacher_projection.eval()
        for parameter in model.teacher_projection.parameters():
            parameter.requires_grad_(False)
    model._clean_initialized = True
