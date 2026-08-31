from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


PATCH_OBSERVED = 0
PATCH_PARTIAL = 1
PATCH_MISSING = 2


@dataclass(frozen=True)
class PatchBatch:
    values: Tensor
    effective_mask: Tensor
    validity: Tensor
    rho: Tensor
    state: Tensor


def _validity_like(validity: Tensor | None, reference: Tensor) -> Tensor:
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


def patchify_with_state(
    values: Tensor,
    observed_mask: Tensor,
    length: int,
    stride: int,
    validity: Tensor | None = None,
) -> PatchBatch:
    if values.ndim != 3:
        raise ValueError("values must have shape [batch, time, variables]")
    if observed_mask.shape != values.shape:
        raise ValueError("observed mask must match values")
    if length <= 0 or stride <= 0 or stride > length:
        raise ValueError("patch length/stride are invalid")
    if values.shape[1] < length:
        raise ValueError("input sequence is shorter than one patch")

    valid = _validity_like(validity, values)
    observed = observed_mask.to(device=values.device, dtype=torch.bool)
    effective_values = torch.where(
        observed & valid, values, torch.zeros_like(values)
    )
    replicated_values = effective_values[:, -1:, :].expand(-1, stride, -1)
    replicated_observed = observed[:, -1:, :].expand(-1, stride, -1)
    padded_values = torch.cat((effective_values, replicated_values), dim=1)
    padded_observed = torch.cat((observed, replicated_observed), dim=1)
    padded_validity = torch.cat(
        (
            valid,
            torch.zeros(
                values.shape[0],
                stride,
                values.shape[2],
                dtype=torch.bool,
                device=values.device,
            ),
        ),
        dim=1,
    )
    effective = padded_observed & padded_validity

    value_patches = padded_values.unfold(1, length, stride).permute(0, 2, 1, 3)
    mask_patches = effective.unfold(1, length, stride).permute(0, 2, 1, 3)
    validity_patches = padded_validity.unfold(1, length, stride).permute(
        0, 2, 1, 3
    )
    valid_count = validity_patches.sum(dim=-1)
    if torch.any(valid_count == 0):
        raise ValueError("patching produced a patch with no valid time points")
    observed_count = mask_patches.sum(dim=-1)
    rho = observed_count.to(values.dtype) / valid_count.to(values.dtype)
    state = torch.full_like(valid_count, PATCH_MISSING, dtype=torch.long)
    state = torch.where(rho > 0, torch.full_like(state, PATCH_PARTIAL), state)
    state = torch.where(rho == 1, torch.full_like(state, PATCH_OBSERVED), state)
    return PatchBatch(
        values=value_patches.contiguous(),
        effective_mask=mask_patches.contiguous(),
        validity=validity_patches.contiguous(),
        rho=rho.contiguous(),
        state=state.contiguous(),
    )
