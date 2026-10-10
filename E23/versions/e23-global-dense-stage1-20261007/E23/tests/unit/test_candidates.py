from __future__ import annotations

import torch
import pytest

from malr.models.candidates import (
    TEMPORAL_FEATURE_NAMES,
    VARIABLE_FEATURE_NAMES,
    build_temporal_candidates,
    build_variable_candidates,
    temporal_candidate_capacity,
    variable_candidate_capacity,
)


def _local_scalar_read_count(function: object) -> int:
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as profile:
        function()  # type: ignore[operator]
    return sum(
        event.count
        for event in profile.key_averages()
        if event.key == "aten::_local_scalar_dense"
    )


def test_temporal_candidates_merge_local_and_global_without_missing_sources() -> None:
    batch_size, variable_count, patch_count, patch_len = 1, 1, 12, 4
    values = torch.arange(
        batch_size * variable_count * patch_count * patch_len,
        dtype=torch.float32,
    ).reshape(batch_size, variable_count, patch_count, patch_len)
    mask = torch.ones_like(values, dtype=torch.bool)
    mask[0, 0, 10] = False
    rho = mask.to(torch.float32).mean(dim=-1)

    result = build_temporal_candidates(values, mask, rho)
    active = result.eligible[0, 0, 5]
    source_patch = result.source_patch[0, 0, 5][active]
    local = result.is_local[0, 0, 5][active]

    assert source_patch.tolist() == [1, 2, 3, 4, 6, 7, 8, 9, 0, 11]
    assert local.tolist() == [True] * 8 + [False, False]
    assert result.candidate_count[0, 0, 5].item() == 10
    assert result.capacity == temporal_candidate_capacity(patch_count) == 6
    assert result.features.shape[-1] == len(TEMPORAL_FEATURE_NAMES)
    signed_index = TEMPORAL_FEATURE_NAMES.index("signed_distance")
    assert result.features[0, 0, 5, 0, signed_index].item() == pytest.approx(-4.0 / 11.0)
    assert torch.isfinite(result.features).all()


def test_temporal_target_availability_masks_the_whole_candidate_row() -> None:
    values = torch.zeros(1, 1, 6, 4)
    mask = torch.ones_like(values, dtype=torch.bool)
    rho = torch.ones(1, 1, 6)
    target_eligible = torch.ones_like(rho, dtype=torch.bool)
    target_eligible[0, 0, 3] = False

    result = build_temporal_candidates(values, mask, rho, target_eligible)

    assert not result.eligible[0, 0, 3].any()
    assert result.candidate_count[0, 0, 3].item() == 0


def test_variable_candidates_keep_sync_and_strongest_nonzero_lag() -> None:
    values = torch.zeros(1, 3, 5, 4)
    mask = torch.ones_like(values, dtype=torch.bool)
    target = torch.tensor([1.0, 2.0, 3.0, 4.0])
    values[0, 0, 2] = target
    values[0, 1, 2] = target
    values[0, 1, 4] = target
    values[0, 1, 1] = torch.tensor([0.0, 2.0, 3.0, 4.0])
    values[0, 1, 3] = torch.tensor([1.0, 2.0, 3.0, 5.0])
    mask[0, 2] = False
    rho = mask.to(torch.float32).mean(dim=-1)
    global_prior = torch.zeros(3, 3, 7)
    global_prior[1, 0, 3] = 0.25  # lag 0
    global_prior[1, 0, 5] = 0.80  # lag +2

    result = build_variable_candidates(values, mask, rho, global_prior)
    active = result.eligible[0, 0, 2]

    assert result.source_variable[0, 0, 2][active].tolist() == [1, 1]
    assert result.source_patch[0, 0, 2][active].tolist() == [2, 4]
    assert result.lag[0, 0, 2][active].tolist() == [0, 2]
    assert result.is_synchronous[0, 0, 2][active].tolist() == [True, False]
    assert result.candidate_count[0, 0, 2].item() == 2
    assert result.features.shape[-1] == len(VARIABLE_FEATURE_NAMES)
    prior_index = VARIABLE_FEATURE_NAMES.index("global_prior")
    assert result.features[0, 0, 2, 1, prior_index].item() == pytest.approx(0.80)


def test_variable_candidates_exclude_missing_sources_and_respect_patch_boundaries() -> None:
    values = torch.zeros(1, 3, 5, 4)
    mask = torch.ones_like(values, dtype=torch.bool)
    mask[0, 2] = False
    rho = mask.to(torch.float32).mean(dim=-1)
    global_prior = torch.zeros(3, 3, 7)

    result = build_variable_candidates(values, mask, rho, global_prior)

    active = result.eligible[0, 0, 0]
    assert torch.all(result.source_variable[0, 0, 0][active] != 2)
    assert torch.all((result.source_patch[0, 0, 0][active] >= 0) & (result.source_patch[0, 0, 0][active] < 5))
    assert result.capacity == variable_candidate_capacity(3) == 4


def test_temporal_candidates_do_not_read_one_host_scalar_per_candidate() -> None:
    values = torch.randn(2, 3, 6, 4)
    mask = torch.ones_like(values, dtype=torch.bool)
    rho = torch.ones(2, 3, 6)

    scalar_reads = _local_scalar_read_count(
        lambda: build_temporal_candidates(values, mask, rho)
    )

    assert scalar_reads <= 8


def test_variable_candidates_do_not_read_one_host_scalar_per_candidate() -> None:
    values = torch.randn(2, 3, 6, 4)
    mask = torch.ones_like(values, dtype=torch.bool)
    rho = torch.ones(2, 3, 6)
    global_prior = torch.zeros(3, 3, 7)

    scalar_reads = _local_scalar_read_count(
        lambda: build_variable_candidates(values, mask, rho, global_prior)
    )

    assert scalar_reads <= 8
