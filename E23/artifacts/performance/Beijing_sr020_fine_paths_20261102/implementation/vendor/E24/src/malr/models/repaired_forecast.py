from __future__ import annotations

import torch
from torch import Tensor, nn

from malr.models.patchtst import PatchTSTCore


class PatchVariableMixer(nn.Module):
    """Exchange information across variables at each forecast patch."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        if d_model < 1 or d_model % 4:
            raise ValueError("variable mixer width must be divisible by four")
        self.d_model = d_model
        self.normalization = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(d_model, 4, dropout=0.0, batch_first=True)
        self.residual_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, latents: Tensor) -> Tensor:
        if latents.ndim != 4 or latents.shape[-1] != self.d_model:
            raise ValueError("variable mixer expects [batch, variables, patches, width]")
        batch, variables, patches, width = latents.shape
        tokens = latents.permute(0, 2, 1, 3).reshape(batch * patches, variables, width)
        normalized = self.normalization(tokens)
        mixed, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        residual = mixed.reshape(batch, patches, variables, width).permute(0, 2, 1, 3)
        return latents + self.residual_scale * residual


class MaskAwareForecastModel(nn.Module):
    """Independent Stage-2 predictor over repaired values plus missingness."""

    def __init__(self, core: PatchTSTCore, *, variable_mixer: bool = False) -> None:
        super().__init__()
        self.core = core
        self.mask_projection = nn.Linear(core.patch_len, core.d_model, bias=False)
        nn.init.zeros_(self.mask_projection.weight)
        self.variable_mixer = PatchVariableMixer(core.d_model) if variable_mixer else None

    def forward(
        self,
        patch_values: Tensor,
        patch_mask: Tensor,
        query_mean: Tensor,
        query_std: Tensor,
        *,
        patch_validity: Tensor | None = None,
    ) -> Tensor:
        if patch_mask.shape != patch_values.shape:
            raise ValueError("forecast mask must match repaired patch values")
        if patch_validity is None:
            validity = torch.ones_like(patch_mask, dtype=torch.bool)
        else:
            validity = patch_validity.to(
                device=patch_values.device, dtype=torch.bool
            )
            if validity.shape != patch_values.shape:
                raise ValueError("forecast patch validity must match values")
        missing = (~patch_mask.to(dtype=torch.bool)) & validity
        latents = self.core.project_clean_patches(patch_values)
        latents = latents + self.mask_projection(missing.to(patch_values.dtype))
        if self.variable_mixer is not None:
            latents = self.variable_mixer(latents)
        normalized = self.core.forecast_from_latents(latents)
        if query_mean.shape != query_std.shape or query_mean.shape != (
            patch_values.shape[0],
            self.core.c_in,
        ):
            raise ValueError("forecast query statistics have invalid shape")
        return normalized * query_std.unsqueeze(-1) + query_mean.unsqueeze(-1)


__all__ = ["MaskAwareForecastModel"]
