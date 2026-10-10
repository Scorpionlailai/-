from __future__ import annotations

import torch
from torch import Tensor, nn

from malr.models.patchtst import PatchTSTCore


class MaskAwareForecastModel(nn.Module):
    """Independent Stage-2 predictor over repaired values plus missingness."""

    def __init__(self, core: PatchTSTCore) -> None:
        super().__init__()
        self.core = core
        self.mask_projection = nn.Linear(core.patch_len, core.d_model, bias=False)
        nn.init.zeros_(self.mask_projection.weight)

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
        normalized = self.core.forecast_from_latents(latents)
        if query_mean.shape != query_std.shape or query_mean.shape != (
            patch_values.shape[0],
            self.core.c_in,
        ):
            raise ValueError("forecast query statistics have invalid shape")
        return normalized * query_std.unsqueeze(-1) + query_mean.unsqueeze(-1)


__all__ = ["MaskAwareForecastModel"]
