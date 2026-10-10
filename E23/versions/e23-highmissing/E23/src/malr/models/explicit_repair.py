"""Differentiable sequence-level repair and provenance-preserving repatching."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from malr.data.patching import patchify_preserving_values


@dataclass(frozen=True)
class ExplicitRepairBatch:
    normalized_sequence: Tensor
    standardized_sequence: Tensor
    observed_sequence: Tensor
    sequence_observed: Tensor
    sequence_validity: Tensor
    sequence_missing: Tensor
    repaired_patches: Tensor
    observation_patches: Tensor
    validity_patches: Tensor
    coverage: Tensor


def _aggregate_allow_empty(
    patches: Tensor,
    validity: Tensor,
    *,
    window_len: int,
    patch_len: int,
    stride: int,
) -> tuple[Tensor, Tensor]:
    if patches.ndim != 4:
        raise ValueError("patches must have shape [batch, variables, patches, patch_len]")
    if validity.shape != patches.shape:
        raise ValueError("validity must match patches")
    if patches.shape[-1] != patch_len:
        raise ValueError("patch length does not match patches")
    if window_len <= 0 or patch_len <= 0 or stride <= 0:
        raise ValueError("window_len, patch_len, and stride must be positive")
    batch, variables = patches.shape[:2]
    sums = torch.zeros(
        batch, window_len, variables, dtype=patches.dtype, device=patches.device
    )
    coverage = torch.zeros(
        batch, window_len, variables, dtype=torch.long, device=patches.device
    )
    valid = validity.to(device=patches.device, dtype=torch.bool)
    if stride == patch_len:
        if patches.shape[2] * patch_len < window_len:
            raise ValueError("nonoverlap candidates do not cover the window")
        values = patches.reshape(batch, variables, -1)[..., :window_len].transpose(1, 2)
        mask = valid.reshape(batch, variables, -1)[..., :window_len].transpose(1, 2)
        return torch.where(mask, values, torch.zeros_like(values)), mask.to(torch.long)
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
        coverage[:, start : start + length] += mask.to(torch.long)
    safe = coverage.to(patches.dtype).clamp_min(1.0)
    return sums / safe, coverage


def _check_stats(stats: Tensor, batch: int, variables: int, name: str) -> Tensor:
    if stats.ndim == 1 and stats.shape == (variables,):
        stats = stats.unsqueeze(0).expand(batch, -1)
    if stats.shape != (batch, variables):
        raise ValueError(f"{name} must have shape [B,C] or [C]")
    if not torch.isfinite(stats).all():
        raise ValueError(f"{name} must be finite")
    return stats


def build_explicit_repair(
    candidate_patches: Tensor,
    observed_patch_values: Tensor,
    observed_patch_mask: Tensor,
    patch_validity: Tensor,
    query_mean: Tensor,
    query_std: Tensor,
    *,
    window_len: int,
    patch_len: int,
    stride: int,
    strict_provenance: bool = True,
) -> ExplicitRepairBatch:
    """Aggregate candidate patches once per time coordinate and clamp observations."""

    if candidate_patches.ndim != 4:
        raise ValueError("candidate_patches must have shape [B,C,N,P]")
    if observed_patch_values.shape != candidate_patches.shape:
        raise ValueError("observed_patch_values must match candidate_patches")
    if observed_patch_mask.shape != candidate_patches.shape:
        raise ValueError("observed_patch_mask must match candidate_patches")
    if patch_validity.shape != candidate_patches.shape:
        raise ValueError("patch_validity must match candidate_patches")
    if not torch.isfinite(candidate_patches).all():
        raise ValueError("candidate_patches must be finite")
    batch, variables = candidate_patches.shape[:2]
    observed_mask = observed_patch_mask.to(
        device=candidate_patches.device, dtype=torch.bool
    )
    validity = patch_validity.to(device=candidate_patches.device, dtype=torch.bool)

    candidate_sequence, coverage = _aggregate_allow_empty(
        candidate_patches,
        validity,
        window_len=window_len,
        patch_len=patch_len,
        stride=stride,
    )
    valid_count = coverage
    observed_count_values, observed_coverage = _aggregate_allow_empty(
        observed_patch_values,
        validity & observed_mask,
        window_len=window_len,
        patch_len=patch_len,
        stride=stride,
    )
    observed_count = observed_coverage
    inconsistent = (observed_count > 0) & (observed_count < valid_count)
    if strict_provenance and inconsistent.any():
        raise ValueError("overlapping provenance masks disagree")

    sequence_validity = valid_count.gt(0)
    sequence_observed = observed_count.eq(valid_count) & sequence_validity
    sequence_missing = sequence_validity & ~sequence_observed
    observed_sequence = observed_count_values
    normalized_sequence = torch.where(
        sequence_observed,
        observed_sequence,
        candidate_sequence,
    ).masked_fill(~sequence_validity, 0.0)

    mean = _check_stats(query_mean.to(candidate_patches), batch, variables, "query_mean")
    std = _check_stats(query_std.to(candidate_patches), batch, variables, "query_std")
    if torch.any(std <= 0):
        raise ValueError("query_std must be positive")
    standardized_sequence = normalized_sequence * std.unsqueeze(1) + mean.unsqueeze(1)

    sequence_values_for_patching = normalized_sequence
    repatched = patchify_preserving_values(
        sequence_values_for_patching,
        sequence_observed,
        patch_len,
        stride,
        validity=sequence_validity,
    )
    return ExplicitRepairBatch(
        normalized_sequence=normalized_sequence,
        standardized_sequence=standardized_sequence,
        observed_sequence=observed_sequence,
        sequence_observed=sequence_observed,
        sequence_validity=sequence_validity,
        sequence_missing=sequence_missing,
        repaired_patches=repatched.values,
        observation_patches=repatched.effective_mask,
        validity_patches=repatched.validity,
        coverage=coverage,
    )


__all__ = ["ExplicitRepairBatch", "build_explicit_repair"]
