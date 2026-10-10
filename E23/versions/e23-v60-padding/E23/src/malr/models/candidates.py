"""Deterministic candidate-pool construction for the next relation protocol.

The builders in this module are intentionally independent from the legacy
relation selectors.  They construct masked-data candidate pools and expose a
separate downstream capacity; edge scoring and routing are implemented later.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import torch
from torch import Tensor


TEMPORAL_FEATURE_NAMES: tuple[str, ...] = (
    "signed_distance",
    "direction",
    "absolute_distance",
    "source_visible_ratio",
    "target_visible_ratio",
    "mask_overlap",
    "target_position",
    "source_position",
    "is_local",
    "shape_evidence",
)

VARIABLE_FEATURE_NAMES: tuple[str, ...] = (
    "global_prior",
    "local_relation",
    "source_visible_ratio",
    "target_visible_ratio",
    "mask_overlap",
    "source_variable",
    "target_variable",
    "lag_signed",
    "lag_direction",
    "lag_absolute",
    "is_synchronous",
)

VARIABLE_LAGS: tuple[int, ...] = (-3, -2, -1, 0, 1, 2, 3)


def temporal_candidate_capacity(patch_count: int) -> int:
    """Return the downstream time-edge capacity for ``patch_count`` patches."""

    if type(patch_count) is not int or patch_count <= 0:
        raise ValueError("patch_count must be a positive integer")
    return min(24, max(6, ceil(0.3 * (patch_count - 1))))


def variable_candidate_capacity(variable_count: int) -> int:
    """Return the downstream variable-edge capacity for ``variable_count`` vars."""

    if type(variable_count) is not int or variable_count <= 0:
        raise ValueError("variable_count must be a positive integer")
    return min(8, max(4, ceil(0.5 * (variable_count - 1))))


@dataclass(frozen=True)
class TemporalCandidateBatch:
    """Merged local/global temporal candidates for every target patch."""

    source_patch: Tensor
    eligible: Tensor
    is_local: Tensor
    features: Tensor
    candidate_count: Tensor
    capacity: int


@dataclass(frozen=True)
class VariableCandidateBatch:
    """Synchronous and lagged variable candidates for every target patch."""

    source_variable: Tensor
    source_patch: Tensor
    lag: Tensor
    eligible: Tensor
    is_synchronous: Tensor
    features: Tensor
    candidate_count: Tensor
    capacity: int


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
    """Return masked Pearson evidence without device-to-host scalar reads."""

    shared = target_mask & source_mask
    weights = shared.to(dtype=target_values.dtype)
    count = weights.sum(dim=-1)
    denominator_count = count.clamp_min(1.0)
    target_mean = (target_values * weights).sum(dim=-1) / denominator_count
    source_mean = (source_values * weights).sum(dim=-1) / denominator_count
    target_centered = (target_values - target_mean.unsqueeze(-1)) * weights
    source_centered = (source_values - source_mean.unsqueeze(-1)) * weights
    numerator = (target_centered * source_centered).sum(dim=-1)
    denominator = target_centered.norm(dim=-1) * source_centered.norm(dim=-1)
    eps = torch.finfo(target_values.dtype).eps
    evidence = numerator / denominator.clamp_min(eps)
    return torch.where((count >= 2.0) & (denominator > eps), evidence, 0.0)


def _compact_permutation(available: Tensor) -> Tensor:
    """Move available candidates left while preserving their protocol order."""

    return torch.argsort((~available).to(torch.int8), dim=-1, stable=True)


def build_temporal_candidates(
    values: Tensor,
    mask: Tensor,
    rho: Tensor,
    target_eligible: Tensor | None = None,
) -> TemporalCandidateBatch:
    """Build a deterministic local-first temporal candidate pool.

    ``values`` are expected to be the masked values supplied to the model.  A
    source with ``rho == 0`` is represented as unavailable and is never marked
    eligible.  The pool is padded to ``N-1`` slots; ``capacity`` is the later
    edge-selection budget and does not truncate this pool.
    """

    values, mask, rho = _validate_patch_inputs(values, mask, rho)
    target_eligible = _validate_target_eligible(target_eligible, rho)
    batch_size, variable_count, patch_count, patch_len = values.shape
    width = max(patch_count - 1, 0)
    feature_dim = len(TEMPORAL_FEATURE_NAMES)
    distance_denominator = float(max(patch_count - 1, 1))

    patch_ids = torch.arange(patch_count, device=values.device)
    target_ids = patch_ids.unsqueeze(1)
    source_ids = patch_ids.unsqueeze(0).expand(patch_count, -1)
    distance = source_ids - target_ids
    different_patch = distance.ne(0)
    local = different_patch & distance.abs().le(4)
    group = torch.where(local, 0, torch.where(different_patch, 1, 2))
    protocol_key = group * max(patch_count, 1) + source_ids
    protocol_order = protocol_key.argsort(dim=-1)[..., :width]
    base_local = local.gather(-1, protocol_order)

    expanded_order = protocol_order.view(1, 1, patch_count, width).expand(
        batch_size, variable_count, -1, -1
    )
    source_rho_in_protocol_order = torch.gather(
        rho.unsqueeze(2).expand(-1, -1, patch_count, -1),
        3,
        expanded_order,
    )
    available = target_eligible.unsqueeze(-1) & source_rho_in_protocol_order.gt(0.0)
    candidate_count = available.sum(dim=-1)
    permutation = _compact_permutation(available)
    compacted_source = expanded_order.gather(-1, permutation)
    compacted_local = base_local.view(1, 1, patch_count, width).expand_as(
        expanded_order
    ).gather(-1, permutation)
    slots = torch.arange(width, device=values.device).view(1, 1, 1, width)
    eligible = slots < candidate_count.unsqueeze(-1)
    source_patch = torch.where(eligible, compacted_source, -1)
    is_local = compacted_local & eligible

    safe_source = source_patch.clamp_min(0)
    gather_index = safe_source.unsqueeze(-1).expand(-1, -1, -1, -1, patch_len)
    source_values = torch.gather(
        values.unsqueeze(2).expand(-1, -1, patch_count, -1, -1),
        3,
        gather_index,
    )
    source_mask = torch.gather(
        mask.unsqueeze(2).expand(-1, -1, patch_count, -1, -1),
        3,
        gather_index,
    )
    source_rho = torch.gather(
        rho.unsqueeze(2).expand(-1, -1, patch_count, -1),
        3,
        safe_source,
    )
    target_values = values.unsqueeze(3)
    target_mask = mask.unsqueeze(3)
    overlap = (target_mask & source_mask).to(values.dtype).mean(dim=-1)
    shape_evidence = _batched_shape_evidence(
        target_values,
        target_mask,
        source_values,
        source_mask,
    )
    target_patch = patch_ids.view(1, 1, patch_count, 1).expand_as(safe_source)
    signed_distance = (safe_source - target_patch).to(values.dtype)
    features = torch.stack(
        (
            signed_distance / distance_denominator,
            signed_distance.sign(),
            signed_distance.abs() / distance_denominator,
            source_rho,
            rho.unsqueeze(-1).expand_as(source_rho),
            overlap,
            target_patch.to(values.dtype) / distance_denominator,
            safe_source.to(values.dtype) / distance_denominator,
            is_local.to(values.dtype),
            shape_evidence,
        ),
        dim=-1,
    ).masked_fill(~eligible.unsqueeze(-1), 0.0)

    return TemporalCandidateBatch(
        source_patch=source_patch,
        eligible=eligible,
        is_local=is_local,
        features=features,
        candidate_count=candidate_count,
        capacity=temporal_candidate_capacity(patch_count),
    )


def build_variable_candidates(
    values: Tensor,
    mask: Tensor,
    rho: Tensor,
    global_prior: Tensor,
    target_eligible: Tensor | None = None,
) -> VariableCandidateBatch:
    """Build sync plus one strongest non-zero-lag candidate per variable pair."""

    values, mask, rho = _validate_patch_inputs(values, mask, rho)
    target_eligible = _validate_target_eligible(target_eligible, rho)
    batch_size, variable_count, patch_count, _ = values.shape
    if global_prior.ndim != 3 or global_prior.shape != (variable_count, variable_count, 7):
        raise ValueError("global_prior must have shape [variables, variables, 7]")
    global_prior = global_prior.to(device=values.device, dtype=values.dtype)
    if not torch.isfinite(global_prior).all():
        raise ValueError("global_prior must contain only finite values")

    width = max(2 * (variable_count - 1), 0)
    feature_dim = len(VARIABLE_FEATURE_NAMES)
    variable_denominator = float(max(variable_count - 1, 1))

    target_variables = torch.arange(variable_count, device=values.device)
    all_sources = target_variables.unsqueeze(0).expand(variable_count, -1)
    source_variables = all_sources[
        all_sources.ne(target_variables.unsqueeze(1))
    ].reshape(variable_count, variable_count - 1)
    source_count = source_variables.shape[1]

    source_values_by_variable = values[:, source_variables]
    source_mask_by_variable = mask[:, source_variables]
    source_rho_by_variable = rho[:, source_variables]
    source_values_sync = source_values_by_variable.permute(0, 1, 3, 2, 4)
    source_mask_sync = source_mask_by_variable.permute(0, 1, 3, 2, 4)
    source_rho_sync = source_rho_by_variable.permute(0, 1, 3, 2)
    target_values = values.unsqueeze(3)
    target_mask = mask.unsqueeze(3)
    sync_relation = _batched_shape_evidence(
        target_values,
        target_mask,
        source_values_sync,
        source_mask_sync,
    )
    sync_overlap = (target_mask & source_mask_sync).to(values.dtype).mean(dim=-1)
    sync_available = target_eligible.unsqueeze(-1) & source_rho_sync.gt(0.0)

    preferred_lags = torch.tensor(
        (-1, 1, -2, 2, -3, 3), dtype=torch.long, device=values.device
    )
    target_patches = torch.arange(patch_count, device=values.device)
    lagged_patch = target_patches.unsqueeze(1) + preferred_lags.unsqueeze(0)
    lag_in_bounds = (lagged_patch >= 0) & (lagged_patch < patch_count)
    safe_lagged_patch = lagged_patch.clamp(0, max(patch_count - 1, 0))
    lag_index = safe_lagged_patch.view(1, 1, patch_count, 1, -1, 1).expand(
        batch_size, variable_count, patch_count, source_count, -1, values.shape[-1]
    )
    source_values_lagged = torch.gather(
        source_values_by_variable.unsqueeze(2).expand(
            -1, -1, patch_count, -1, -1, -1
        ),
        4,
        lag_index,
    )
    source_mask_lagged = torch.gather(
        source_mask_by_variable.unsqueeze(2).expand(
            -1, -1, patch_count, -1, -1, -1
        ),
        4,
        lag_index,
    )
    rho_lag_index = safe_lagged_patch.view(1, 1, patch_count, 1, -1).expand(
        batch_size, variable_count, patch_count, source_count, -1
    )
    source_rho_lagged = torch.gather(
        source_rho_by_variable.unsqueeze(2).expand(
            -1, -1, patch_count, -1, -1
        ),
        4,
        rho_lag_index,
    )
    lagged_relation = _batched_shape_evidence(
        target_values.unsqueeze(4),
        target_mask.unsqueeze(4),
        source_values_lagged,
        source_mask_lagged,
    )
    lagged_overlap = (
        target_mask.unsqueeze(4) & source_mask_lagged
    ).to(values.dtype).mean(dim=-1)
    lagged_available = (
        target_eligible.unsqueeze(-1).unsqueeze(-1)
        & lag_in_bounds.view(1, 1, patch_count, 1, -1)
        & source_rho_lagged.gt(0.0)
    )
    lagged_scores = lagged_relation.abs().masked_fill(~lagged_available, -torch.inf)
    best_choice = lagged_scores.argmax(dim=-1, keepdim=True)
    best_available = lagged_available.any(dim=-1)
    best_relation = lagged_relation.gather(-1, best_choice).squeeze(-1)
    best_overlap = lagged_overlap.gather(-1, best_choice).squeeze(-1)
    best_source_rho = source_rho_lagged.gather(-1, best_choice).squeeze(-1)
    best_patch = safe_lagged_patch.view(1, 1, patch_count, 1, -1).expand(
        batch_size, variable_count, patch_count, source_count, -1
    ).gather(-1, best_choice).squeeze(-1)
    best_lag = preferred_lags.view(1, 1, 1, 1, -1).expand(
        batch_size, variable_count, patch_count, source_count, -1
    ).gather(-1, best_choice).squeeze(-1)

    candidate_available = torch.stack((sync_available, best_available), dim=-1)
    candidate_relation = torch.stack((sync_relation, best_relation), dim=-1)
    candidate_overlap = torch.stack((sync_overlap, best_overlap), dim=-1)
    candidate_source_rho = torch.stack((source_rho_sync, best_source_rho), dim=-1)
    sync_patch = target_patches.view(1, 1, patch_count, 1).expand_as(best_patch)
    candidate_source_patch = torch.stack((sync_patch, best_patch), dim=-1)
    candidate_lag = torch.stack((torch.zeros_like(best_lag), best_lag), dim=-1)
    candidate_sync = torch.zeros_like(candidate_available)
    candidate_sync[..., 0] = True
    candidate_source_variable = source_variables.view(
        1, variable_count, 1, source_count, 1
    ).expand_as(candidate_lag)
    candidate_target_variable = target_variables.view(
        1, variable_count, 1, 1, 1
    ).expand_as(candidate_lag)
    candidate_prior = global_prior[
        candidate_source_variable,
        candidate_target_variable,
        candidate_lag + 3,
    ]
    candidate_features = torch.stack(
        (
            candidate_prior,
            candidate_relation,
            candidate_source_rho,
            rho.unsqueeze(-1).unsqueeze(-1).expand_as(candidate_source_rho),
            candidate_overlap,
            candidate_source_variable.to(values.dtype) / variable_denominator,
            candidate_target_variable.to(values.dtype) / variable_denominator,
            candidate_lag.to(values.dtype) / 3.0,
            candidate_lag.sign().to(values.dtype),
            candidate_lag.abs().to(values.dtype) / 3.0,
            candidate_sync.to(values.dtype),
        ),
        dim=-1,
    )

    candidate_available = candidate_available.flatten(-2)
    candidate_source_variable = candidate_source_variable.flatten(-2)
    candidate_source_patch = candidate_source_patch.flatten(-2)
    candidate_lag = candidate_lag.flatten(-2)
    candidate_sync = candidate_sync.flatten(-2)
    candidate_features = candidate_features.flatten(-3, -2)
    candidate_count = candidate_available.sum(dim=-1)
    permutation = _compact_permutation(candidate_available)
    eligible = torch.arange(width, device=values.device).view(1, 1, 1, width) < candidate_count.unsqueeze(-1)
    source_variable = torch.where(
        eligible, candidate_source_variable.gather(-1, permutation), -1
    )
    source_patch = torch.where(
        eligible, candidate_source_patch.gather(-1, permutation), -1
    )
    lag = torch.where(eligible, candidate_lag.gather(-1, permutation), 0)
    is_synchronous = candidate_sync.gather(-1, permutation) & eligible
    features = candidate_features.gather(
        -2, permutation.unsqueeze(-1).expand(-1, -1, -1, -1, feature_dim)
    ).masked_fill(~eligible.unsqueeze(-1), 0.0)

    return VariableCandidateBatch(
        source_variable=source_variable,
        source_patch=source_patch,
        lag=lag,
        eligible=eligible,
        is_synchronous=is_synchronous,
        features=features,
        candidate_count=candidate_count,
        capacity=variable_candidate_capacity(variable_count),
    )


__all__ = [
    "TEMPORAL_FEATURE_NAMES",
    "VARIABLE_FEATURE_NAMES",
    "VARIABLE_LAGS",
    "TemporalCandidateBatch",
    "VariableCandidateBatch",
    "build_temporal_candidates",
    "build_variable_candidates",
    "temporal_candidate_capacity",
    "variable_candidate_capacity",
]
