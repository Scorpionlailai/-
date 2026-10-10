from __future__ import annotations

import pytest
import torch

from malr.models.candidates_v2 import (
    TEMPORAL_FEATURE_NAMES,
    VARIABLE_FEATURE_NAMES,
    build_temporal_candidates,
    build_variable_candidates,
    distance_band,
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


def test_distance_band_uses_inclusive_logarithmic_boundaries() -> None:
    distances = torch.tensor([1, 2, 3, 4, 5, 8, 9, 16, 17])

    result = distance_band(distances, (2, 4, 8, 16))

    assert result.tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4]


@pytest.mark.parametrize("boundaries", [(), (0, 2), (2, 2), (4, 2)])
def test_distance_band_rejects_invalid_boundaries(
    boundaries: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError):
        distance_band(torch.tensor([1]), boundaries)


def test_temporal_candidates_label_true_offsets_and_log_bands() -> None:
    values = torch.randn(1, 1, 12, 16)
    mask = torch.ones_like(values, dtype=torch.bool)
    rho = torch.ones(1, 1, 12)
    starts = (torch.arange(12) * 8).unsqueeze(0)

    result = build_temporal_candidates(values, mask, rho, starts, (2, 4, 8, 16))

    row = result.eligible[0, 0, 5]
    by_source = {
        int(source): (int(lag), int(offset), int(band))
        for source, lag, offset, band in zip(
            result.source_patch[0, 0, 5][row],
            result.signed_patch_lag[0, 0, 5][row],
            result.signed_time_offset[0, 0, 5][row],
            result.distance_band[0, 0, 5][row],
            strict=True,
        )
    }
    assert by_source[4] == (-1, -8, 0)
    assert by_source[0] == (-5, -40, 2)
    assert by_source[11] == (6, 48, 2)
    assert result.candidate_count[0, 0, 5].item() == 11
    assert result.features.shape[-1] == len(TEMPORAL_FEATURE_NAMES)


def test_temporal_candidates_keep_all_nonself_sources_without_builder_topk() -> None:
    values = torch.randn(1, 1, 10, 4)
    mask = torch.ones_like(values, dtype=torch.bool)
    rho = torch.ones(1, 1, 10)
    starts = (torch.arange(10) * 8).unsqueeze(0)

    result = build_temporal_candidates(values, mask, rho, starts, (2, 4, 8, 16))

    active = result.eligible[0, 0, 4]
    assert active.sum().item() == 9
    assert set(result.source_patch[0, 0, 4][active].tolist()) == set(range(10)) - {4}


def test_fully_missing_target_keeps_all_valid_variable_lags() -> None:
    values = torch.randn(1, 3, 5, 4)
    mask = torch.ones_like(values, dtype=torch.bool)
    mask[0, 0, 2] = False
    rho = mask.float().mean(-1)

    result = build_variable_candidates(values, mask, rho)

    active = result.eligible[0, 0, 2]
    assert active.sum().item() == 2 * 5
    assert set(result.lag[0, 0, 2][active].tolist()) == {-2, -1, 0, 1, 2}
    assert result.features.shape[-1] == len(VARIABLE_FEATURE_NAMES)


def test_variable_candidates_are_ordered_by_source_variable_then_patch() -> None:
    values = torch.randn(1, 3, 3, 4)
    mask = torch.ones_like(values, dtype=torch.bool)
    rho = torch.ones(1, 3, 3)

    result = build_variable_candidates(values, mask, rho)

    active = result.eligible[0, 1, 1]
    assert result.source_variable[0, 1, 1][active].tolist() == [0, 0, 0, 2, 2, 2]
    assert result.source_patch[0, 1, 1][active].tolist() == [0, 1, 2, 0, 1, 2]
    assert result.lag[0, 1, 1][active].tolist() == [-1, 0, 1, -1, 0, 1]
    assert result.is_synchronous[0, 1, 1][active].tolist() == [False, True, False] * 2


def test_zero_rho_sources_are_unavailable_without_reordering_valid_candidates() -> None:
    values = torch.randn(1, 3, 3, 4)
    mask = torch.ones_like(values, dtype=torch.bool)
    rho = torch.ones(1, 3, 3)
    rho[0, 2, 1] = 0

    result = build_variable_candidates(values, mask, rho)

    active = result.eligible[0, 0, 1]
    pairs = list(
        zip(
            result.source_variable[0, 0, 1][active].tolist(),
            result.source_patch[0, 0, 1][active].tolist(),
            strict=True,
        )
    )
    assert pairs == [(1, 0), (1, 1), (1, 2), (2, 0), (2, 2)]


def test_candidate_builders_avoid_per_candidate_host_scalar_reads() -> None:
    values = torch.randn(2, 3, 6, 4)
    mask = torch.ones_like(values, dtype=torch.bool)
    rho = torch.ones(2, 3, 6)
    starts = (torch.arange(6) * 8).unsqueeze(0).expand(2, -1)

    temporal_reads = _local_scalar_read_count(
        lambda: build_temporal_candidates(values, mask, rho, starts, (2, 4, 8, 16))
    )
    variable_reads = _local_scalar_read_count(
        lambda: build_variable_candidates(values, mask, rho)
    )

    assert temporal_reads <= 8
    assert variable_reads <= 8
