from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class GlobalRefinementOutput:
    latents: Tensor
    delta: Tensor
    applied_delta: Tensor


class GlobalLatentRefiner(nn.Module):
    """Jointly reconcile all variable-patch nodes before PatchTST encoding."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_multiplier: int = 2,
        dropout: float = 0.0,
        output_init: str = "zero",
    ) -> None:
        super().__init__()
        if type(d_model) is not int or d_model <= 0:
            raise ValueError("refiner d_model must be a positive integer")
        if type(n_heads) is not int or n_heads <= 0:
            raise ValueError("refiner n_heads must be a positive integer")
        if d_model % n_heads:
            raise ValueError("refiner d_model must be divisible by n_heads")
        if type(ff_multiplier) is not int or ff_multiplier <= 0:
            raise ValueError("refiner ff_multiplier must be a positive integer")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("refiner dropout must be in [0, 1)")
        if output_init != "zero":
            raise ValueError("refiner output_init must be zero")

        hidden_dim = ff_multiplier * d_model
        self.input_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.feed_forward_norm = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )
        self.residual_dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, latents: Tensor, prior: Tensor, rho: Tensor) -> GlobalRefinementOutput:
        if latents.ndim != 4 or prior.shape != latents.shape:
            raise ValueError("refiner latents and prior must match [batch, variables, patches, d]")
        if rho.shape != latents.shape[:-1]:
            raise ValueError("refiner rho must match latent node dimensions")
        if not torch.isfinite(rho).all() or not torch.all((rho >= 0.0) & (rho <= 1.0)):
            raise ValueError("refiner rho must be finite and in [0, 1]")

        batch, variables, patches, d_model = latents.shape
        joint = (latents + prior).reshape(batch, variables * patches, d_model)
        normalized = self.input_norm(joint)
        attention, _ = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        hidden = joint + self.residual_dropout(attention)
        feed_forward = self.feed_forward(self.feed_forward_norm(hidden))
        hidden = hidden + self.residual_dropout(feed_forward)
        delta = self.output_projection(hidden).reshape_as(latents)
        applied_delta = (1.0 - rho).unsqueeze(-1).to(latents.dtype) * delta
        refined = latents + applied_delta
        return GlobalRefinementOutput(refined, delta, applied_delta)
