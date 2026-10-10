"""Forecast-loss-driven fusion of none, historical, and variable route latents."""

from __future__ import annotations

from math import isfinite

import torch
from torch import Tensor, nn


_FUSION_ROUTE_INDICES: tuple[int, int, int] = (0, 1, 3)


class ForecastHVFusion(nn.Module):
    """Learn a soft forecast mixture while retaining a detached teacher prior."""

    def __init__(
        self,
        d_model: int,
        patch_len: int,
        hidden_dim: int | None = None,
        teacher_prior_scale: float = 0.25,
    ) -> None:
        super().__init__()
        if type(d_model) is not int or d_model <= 0:
            raise ValueError("d_model must be a positive integer")
        if type(patch_len) is not int or patch_len <= 0:
            raise ValueError("patch_len must be a positive integer")
        hidden = d_model if hidden_dim is None else hidden_dim
        if type(hidden) is not int or hidden <= 0:
            raise ValueError("hidden_dim must be a positive integer")
        if not isfinite(teacher_prior_scale) or teacher_prior_scale < 0.0:
            raise ValueError("teacher_prior_scale must be finite and non-negative")

        self.d_model = d_model
        self.patch_len = patch_len
        self.teacher_prior_scale = float(teacher_prior_scale)
        self.logit_mlp = nn.Sequential(
            nn.Linear(3 * d_model + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3),
        )

    def _validate_forward_inputs(
        self,
        route_latents: Tensor,
        route_available: Tensor,
        teacher_route_logits: Tensor,
        missing_indicator: Tensor,
    ) -> None:
        inputs = {
            "route_latents": route_latents,
            "route_available": route_available,
            "teacher_route_logits": teacher_route_logits,
            "missing_indicator": missing_indicator,
        }
        if any(not isinstance(value, Tensor) for value in inputs.values()):
            raise TypeError("forecast fusion inputs must be tensors")
        if (
            route_latents.ndim != 5
            or route_latents.shape[-2] != 8
            or route_latents.shape[-1] != self.d_model
        ):
            raise ValueError("route_latents must have shape [B,C,N,8,d_model]")
        leading_shape = route_latents.shape[:3]
        if route_available.shape != (*leading_shape, 8):
            raise ValueError("route_available must have shape [B,C,N,8]")
        if teacher_route_logits.shape != (*leading_shape, 8):
            raise ValueError("teacher_route_logits must have shape [B,C,N,8]")
        if missing_indicator.shape != (*leading_shape, self.patch_len):
            raise ValueError(
                "missing_indicator must have shape [B,C,N,patch_len]"
            )
        if not route_latents.is_floating_point():
            raise TypeError("route_latents must be floating-point")
        if route_available.dtype != torch.bool:
            raise TypeError("route_available must be bool")
        if not teacher_route_logits.is_floating_point():
            raise TypeError("teacher_route_logits must be floating-point")
        if missing_indicator.dtype != torch.bool:
            raise TypeError("missing_indicator must be bool")
        if not torch.isfinite(route_latents).all():
            raise ValueError("route_latents must contain only finite values")
        if not torch.isfinite(teacher_route_logits).all():
            raise ValueError(
                "teacher_route_logits must contain only finite values"
            )
        if any(value.device != route_latents.device for value in inputs.values()):
            raise ValueError("forecast fusion inputs must share a device")

    def forward(
        self,
        route_latents: Tensor,
        route_available: Tensor,
        teacher_route_logits: Tensor,
        missing_indicator: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return a forecast latent and four-route none/H/T/V weights."""

        self._validate_forward_inputs(
            route_latents,
            route_available,
            teacher_route_logits,
            missing_indicator,
        )
        selected_latents = route_latents[..., _FUSION_ROUTE_INDICES, :]
        selected_available = route_available[..., _FUSION_ROUTE_INDICES].clone()
        selected_available[..., 0] = True

        visible_latents = selected_latents.masked_fill(
            ~selected_available.unsqueeze(-1), 0.0
        )
        missing_fraction = missing_indicator.to(dtype=route_latents.dtype).mean(
            dim=-1, keepdim=True
        )
        features = torch.cat(
            (
                visible_latents.flatten(start_dim=-2),
                missing_fraction,
            ),
            dim=-1,
        )
        learned_logits = self.logit_mlp(features)
        teacher_prior = teacher_route_logits[
            ..., _FUSION_ROUTE_INDICES
        ].detach().to(dtype=learned_logits.dtype)
        fusion_logits = learned_logits + self.teacher_prior_scale * teacher_prior
        fusion_logits = fusion_logits.masked_fill(~selected_available, -torch.inf)
        selected_weights = torch.softmax(fusion_logits, dim=-1)

        temporal_weight = torch.zeros_like(selected_weights[..., 0])
        weights = torch.stack(
            (
                selected_weights[..., 0],
                selected_weights[..., 1],
                temporal_weight,
                selected_weights[..., 2],
            ),
            dim=-1,
        ).to(dtype=route_latents.dtype)
        fused = (
            selected_latents * selected_weights.to(route_latents.dtype).unsqueeze(-1)
        ).sum(dim=-2)
        if not torch.isfinite(weights).all() or not torch.isfinite(fused).all():
            raise ValueError("forecast fusion produced non-finite values")
        return fused, weights


__all__ = ["ForecastHVFusion"]
