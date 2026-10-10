from __future__ import annotations

import torch
from torch import Tensor, nn


class LatentReconstructionHead(nn.Module):
    """Map each recovered patch latent to one standardized value patch."""

    def __init__(self, d_model: int, patch_len: int) -> None:
        super().__init__()
        if d_model <= 0 or patch_len <= 0:
            raise ValueError("reconstruction dimensions must be positive")
        self.d_model = d_model
        self.patch_len = patch_len
        self.projection = nn.Linear(d_model, patch_len)

    def forward(self, latents: Tensor) -> Tensor:
        if latents.ndim != 4:
            raise ValueError(
                "latents must have shape [batch, variables, patches, d_model]"
            )
        if latents.shape[-1] != self.d_model:
            raise ValueError(
                f"expected latent width {self.d_model}, got {latents.shape[-1]}"
            )
        return self.projection(latents)


def aggregate_patch_predictions(
    patches: Tensor,
    *,
    window_len: int,
    patch_len: int,
    stride: int,
    validity: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Place patches on the time axis and average overlapping valid values.

    ``patches`` and ``validity`` use ``[batch, variables, patches, patch_len]``;
    the returned window and coverage use ``[batch, time, variables]``.
    """

    if patches.ndim != 4:
        raise ValueError("patches must have shape [batch, variables, patches, patch_len]")
    if window_len <= 0 or patch_len <= 0 or stride <= 0:
        raise ValueError("window_len, patch_len, and stride must be positive")
    if patches.shape[-1] != patch_len:
        raise ValueError("patch length does not match reconstruction patches")
    if not torch.isfinite(patches).all():
        raise ValueError("reconstruction patches must be finite")

    if validity is None:
        valid = torch.ones_like(patches, dtype=torch.bool)
    else:
        if validity.shape != patches.shape:
            raise ValueError("validity must match reconstruction patches")
        valid = validity.to(device=patches.device, dtype=torch.bool)

    batch, variables = patches.shape[:2]
    sums = torch.zeros(
        batch,
        window_len,
        variables,
        dtype=patches.dtype,
        device=patches.device,
    )
    coverage = torch.zeros(
        batch,
        window_len,
        variables,
        dtype=torch.long,
        device=patches.device,
    )
    for patch_index in range(patches.shape[2]):
        start = patch_index * stride
        if start >= window_len:
            continue
        length = min(patch_len, window_len - start)
        values = patches[:, :, patch_index, :length].permute(0, 2, 1)
        mask = valid[:, :, patch_index, :length].permute(0, 2, 1)
        sums[:, start : start + length] += torch.where(
            mask, values, torch.zeros_like(values)
        )
        coverage[:, start : start + length] += mask.to(dtype=torch.long)

    if torch.any(coverage == 0):
        raise ValueError("reconstruction has no valid coverage for a time point")
    window = sums / coverage.to(dtype=patches.dtype)
    if not torch.isfinite(window).all():
        raise ValueError("aggregated reconstruction is not finite")
    return window, coverage
