"""Untruncated temporal and cross-variable candidate pools for relation V2."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


DEFAULT_DISTANCE_BOUNDARIES: tuple[int, ...] = (2, 4, 8, 16)

TEMPORAL_FEATURE_NAMES: tuple[str, ...] = (
    "signed_time_offset",
    "direction",
    "normalized_absolute_time_offset",
    "log1p_patch_distance",
    "source_visible_ratio",
    "target_visible_ratio",
    "mask_overlap",
    "source_position",
    "target_position",
    "is_local",
    "normalized_distance_band",
    "shape_evidence",
)

VARIABLE_FEATURE_NAMES: tuple[str, ...] = (
    "source_visible_ratio",
    "target_visible_ratio",
    "mask_overlap",
    "source_variable",
    "target_variable",
    "signed_lag",
    "direction",
    "log1p_patch_distance",
    "is_synchronous",
    "normalized_lag_band",
    "shape_evidence",
)


@dataclass(frozen=True)
class TemporalCandidateBatch:
    """All non-self, same-variable patches for every target patch."""

    source_patch: Tensor
    signed_patch_lag: Tensor
    signed_time_offset: Tensor
    direction: Tensor
    distance_band: Tensor
    eligible: Tensor
    is_local: Tensor
    features: Tensor
    candidate_count: Tensor


@dataclass(frozen=True)
class VariableCandidateBatch:
    """The full ordered source-variable by source-patch candidate pool."""

    source_variable: Tensor
    source_patch: Tensor
    lag: Tensor
    lag_band: Tensor
    eligible: Tensor
    is_synchronous: Tensor
    features: Tensor
    candidate_count: Tensor


def distance_band(abs_patch_distance: Tensor, boundaries: tuple[int, ...]) -> Tensor:
    """Map absolute patch distances to inclusive, logarithmic distance bands."""

    if not boundaries or any(boundary <= 0 for boundary in boundaries):
        raise ValueError("distance boundaries must be non-empty and positive")
    if tuple(sorted(set(boundaries))) != boundaries:
        raise ValueError("distance boundaries must be strictly increasing")
    limits = torch.tensor(boundaries, device=abs_patch_distance.device)
    return torch.bucketize(abs_patch_distance, limits, right=False)


def _validate_patch_inputs(
    values: Tensor,
    mask: Tensor,
    rho: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    if values.ndim != 4:
        raise ValueError("values must have shape [batch, variables, patches, patch_len]")
    if mask.shape != values.shape:
        raise ValueError("mask must match values")
    if rho.shape != values.shape[:-1]:
        raise ValueError("rho must have shape [batch, variables, patches]")
    if not torch.isfinite(values).all():
        raise ValueError("values must contain only finite values")
    rho = rho.to(device=values.device, dtype=values.dtype)
    if not torch.isfinite(rho).all() or not torch.all((rho >= 0.0) & (rho <= 1.0)):
        raise ValueError("rho must be finite and lie in [0, 1]")
    return values, mask.to(device=values.device, dtype=torch.bool), rho


def _validate_target_eligible(
    target_eligible: Tensor | None,
    rho: Tensor,
) -> Tensor:
    if target_eligible is None:
        return torch.ones_like(rho, dtype=torch.bool)
    if target_eligible.shape != rho.shape:
        raise ValueError("target_eligible must match [batch, variables, patches]")
    return target_eligible.to(device=rho.device, dtype=torch.bool)


def _batched_shape_evidence(
    target_values: Tensor,
    target_mask: Tensor,
    source_values: Tensor,
    source_mask: Tensor,
) -> Tensor:
    """Compute masked Pearson evidence as a feature, never a selector."""

    shared = target_mask & source_mask
    weights = shared.to(dtype=target_values.dtype)
    count = weights.sum(dim=-1)
    safe_count = count.clamp_min(1.0)
    target_mean = (target_values * weights).sum(dim=-1) / safe_count
    source_mean = (source_values * weights).sum(dim=-1) / safe_count
    target_centered = (target_values - target_mean.unsqueeze(-1)) * weights
    source_centered = (source_values - source_mean.unsqueeze(-1)) * weights
    numerator = (target_centered * source_centered).sum(dim=-1)
    denominator = target_centered.norm(dim=-1) * source_centered.norm(dim=-1)
    eps = torch.finfo(target_values.dtype).eps
    evidence = numerator / denominator.clamp_min(eps)
    return torch.where((count >= 2.0) & (denominator > eps), evidence, 0.0)


def _temporal_source_indices(patch_count: int, device: torch.device) -> Tensor:
    patch_ids = torch.arange(patch_count, device=device)
    sources = patch_ids.unsqueeze(0).expand(patch_count, -1)
    targets = patch_ids.unsqueeze(1)
    return sources[sources.ne(targets)].reshape(patch_count, patch_count - 1)


def build_temporal_candidates(
    values: Tensor,
    mask: Tensor,
    rho: Tensor,
    patch_starts: Tensor,
    distance_boundaries: tuple[int, ...],
    target_eligible: Tensor | None = None,
) -> TemporalCandidateBatch:
    """Build all same-variable, non-self candidates with real time offsets."""

    values, mask, rho = _validate_patch_inputs(values, mask, rho)
    target_eligible = _validate_target_eligible(target_eligible, rho)
    batch_size, variable_count, patch_count, patch_len = values.shape
    if patch_count < 2:
        raise ValueError("temporal candidates require at least two patches")
    if patch_starts.shape != (batch_size, patch_count):
        raise ValueError("patch_starts must have shape [batch, patches]")
    patch_starts = patch_starts.to(device=values.device)
    if not torch.isfinite(patch_starts).all():
        raise ValueError("patch_starts must contain only finite values")

    source_by_target = _temporal_source_indices(patch_count, values.device)
    width = source_by_target.shape[-1]
    source_patch = source_by_target.view(1, 1, patch_count, width).expand(
        batch_size, variable_count, -1, -1
    )
    target_patch = torch.arange(patch_count, device=values.device).view(
        1, 1, patch_count, 1
    )
    signed_patch_lag = source_patch - target_patch
    abs_patch_distance = signed_patch_lag.abs()
    bands = distance_band(abs_patch_distance, distance_boundaries)

    source_starts = torch.gather(
        patch_starts.unsqueeze(1).expand(-1, patch_count, -1),
        2,
        source_by_target.unsqueeze(0).expand(batch_size, -1, -1),
    )
    signed_time_offset = (
        source_starts - patch_starts.unsqueeze(-1)
    ).unsqueeze(1).expand(-1, variable_count, -1, -1)
    direction = signed_time_offset.sign()
    source_rho = torch.gather(
        rho.unsqueeze(2).expand(-1, -1, patch_count, -1), 3, source_patch
    )
    eligible = target_eligible.unsqueeze(-1) & source_rho.gt(0.0)

    source_index = source_patch.unsqueeze(-1).expand(-1, -1, -1, -1, patch_len)
    source_values = torch.gather(
        values.unsqueeze(2).expand(-1, -1, patch_count, -1, -1), 3, source_index
    )
    source_mask = torch.gather(
        mask.unsqueeze(2).expand(-1, -1, patch_count, -1, -1), 3, source_index
    )
    target_values = values.unsqueeze(3)
    target_mask = mask.unsqueeze(3)
    overlap = (target_mask & source_mask).to(values.dtype).mean(dim=-1)
    shape_evidence = _batched_shape_evidence(
        target_values, target_mask, source_values, source_mask
    )

    position_denominator = float(patch_count - 1)
    time_scale = (
        patch_starts.max(dim=-1).values - patch_starts.min(dim=-1).values
    ).abs().clamp_min(1).to(dtype=values.dtype).view(batch_size, 1, 1, 1)
    normalized_abs_offset = signed_time_offset.abs().to(values.dtype) / time_scale
    is_local = bands.eq(0) & eligible
    features = torch.stack(
        (
            signed_time_offset.to(values.dtype),
            direction.to(values.dtype),
            normalized_abs_offset,
            torch.log1p(abs_patch_distance.to(values.dtype)),
            source_rho,
            rho.unsqueeze(-1).expand_as(source_rho),
            overlap,
            source_patch.to(values.dtype) / position_denominator,
            target_patch.to(values.dtype).expand_as(source_rho) / position_denominator,
            is_local.to(values.dtype),
            bands.to(values.dtype) / float(len(distance_boundaries)),
            shape_evidence,
        ),
        dim=-1,
    ).masked_fill(~eligible.unsqueeze(-1), 0.0)

    return TemporalCandidateBatch(
        source_patch=source_patch,
        signed_patch_lag=signed_patch_lag,
        signed_time_offset=signed_time_offset,
        direction=direction,
        distance_band=bands,
        eligible=eligible,
        is_local=is_local,
        features=features,
        candidate_count=eligible.sum(dim=-1),
    )


def build_variable_candidates(
    values: Tensor,
    mask: Tensor,
    rho: Tensor,
    target_eligible: Tensor | None = None,
    distance_boundaries: tuple[int, ...] = DEFAULT_DISTANCE_BOUNDARIES,
) -> VariableCandidateBatch:
    """Build every in-window source patch for every ordered variable pair."""

    values, mask, rho = _validate_patch_inputs(values, mask, rho)
    target_eligible = _validate_target_eligible(target_eligible, rho)
    batch_size, variable_count, patch_count, patch_len = values.shape
    if variable_count < 2:
        raise ValueError("variable candidates require at least two variables")

    variable_ids = torch.arange(variable_count, device=values.device)
    all_sources = variable_ids.unsqueeze(0).expand(variable_count, -1)
    source_variables_by_target = all_sources[
        all_sources.ne(variable_ids.unsqueeze(1))
    ].reshape(variable_count, variable_count - 1)
    source_count = source_variables_by_target.shape[-1]
    source_patches = torch.arange(patch_count, device=values.device)
    width = source_count * patch_count

    source_variable = source_variables_by_target.view(
        1, variable_count, 1, source_count, 1
    ).expand(batch_size, -1, patch_count, -1, patch_count).flatten(-2)
    source_patch = source_patches.view(1, 1, 1, 1, patch_count).expand(
        batch_size, variable_count, patch_count, source_count, -1
    ).flatten(-2)
    target_patch = torch.arange(patch_count, device=values.device).view(
        1, 1, patch_count, 1
    )
    lag = source_patch - target_patch
    lag_band = distance_band(lag.abs(), distance_boundaries)
    is_synchronous = lag.eq(0)

    source_values_by_variable = values[:, source_variables_by_target]
    source_mask_by_variable = mask[:, source_variables_by_target]
    source_rho_by_variable = rho[:, source_variables_by_target]
    source_values = source_values_by_variable.unsqueeze(2).expand(
        -1, -1, patch_count, -1, -1, -1
    ).flatten(3, 4)
    source_mask = source_mask_by_variable.unsqueeze(2).expand(
        -1, -1, patch_count, -1, -1, -1
    ).flatten(3, 4)
    source_rho = source_rho_by_variable.unsqueeze(2).expand(
        -1, -1, patch_count, -1, -1
    ).flatten(3, 4)
    eligible = target_eligible.unsqueeze(-1) & source_rho.gt(0.0)

    target_values = values.unsqueeze(3)
    target_mask = mask.unsqueeze(3)
    overlap = (target_mask & source_mask).to(values.dtype).mean(dim=-1)
    shape_evidence = _batched_shape_evidence(
        target_values, target_mask, source_values, source_mask
    )

    variable_denominator = float(variable_count - 1)
    features = torch.stack(
        (
            source_rho,
            rho.unsqueeze(-1).expand_as(source_rho),
            overlap,
            source_variable.to(values.dtype) / variable_denominator,
            variable_ids.view(1, variable_count, 1, 1).to(values.dtype).expand_as(source_rho)
            / variable_denominator,
            lag.to(values.dtype) / float(max(patch_count - 1, 1)),
            lag.sign().to(values.dtype),
            torch.log1p(lag.abs().to(values.dtype)),
            is_synchronous.to(values.dtype),
            lag_band.to(values.dtype) / float(len(distance_boundaries)),
            shape_evidence,
        ),
        dim=-1,
    ).masked_fill(~eligible.unsqueeze(-1), 0.0)

    return VariableCandidateBatch(
        source_variable=source_variable,
        source_patch=source_patch,
        lag=lag,
        lag_band=lag_band,
        eligible=eligible,
        is_synchronous=is_synchronous,
        features=features,
        candidate_count=eligible.sum(dim=-1),
    )


__all__ = [
    "DEFAULT_DISTANCE_BOUNDARIES",
    "TEMPORAL_FEATURE_NAMES",
    "VARIABLE_FEATURE_NAMES",
    "TemporalCandidateBatch",
    "VariableCandidateBatch",
    "build_temporal_candidates",
    "build_variable_candidates",
    "distance_band",
]
