from pathlib import Path

import numpy as np
import pytest

from malr.data.masks import (
    EpochMaskManager,
    MaskExpectation,
    generate_variable_missing_mask,
    load_fixed_mask,
    save_fixed_mask,
)


def test_variable_missing_has_unique_starts_and_valid_bounds() -> None:
    mask, metadata = generate_variable_missing_mask(
        100,
        3,
        start_rate=0.06,
        block_len=5,
        seed=2021,
    )

    assert mask.shape == (100, 3)
    assert mask.dtype == np.bool_
    assert all(
        len(starts) == len(set(starts)) == round(0.06 * 100)
        for starts in metadata.starts
    )
    assert all(0 <= start <= 95 for starts in metadata.starts for start in starts)
    assert metadata.final_rate == pytest.approx((~mask).sum() / mask.size)


def test_final_rate_counts_unique_missing_valid_elements() -> None:
    validity = np.ones((40, 2), dtype=bool)
    validity[-5:] = False

    mask, metadata = generate_variable_missing_mask(
        40,
        2,
        start_rate=0.25,
        block_len=8,
        seed=9,
        validity=validity,
    )

    expected = np.logical_and(~mask, validity).sum() / validity.sum()
    assert metadata.final_rate == pytest.approx(expected)
    assert metadata.final_rate <= 1.0


def test_epoch_masks_change_but_are_reproducible() -> None:
    first = EpochMaskManager(200, 7, 0.06, 5, base_seed=2021)
    second = EpochMaskManager(200, 7, 0.06, 5, base_seed=2021)

    np.testing.assert_array_equal(
        first.mask_for_epoch(0),
        second.mask_for_epoch(0),
    )
    assert not np.array_equal(
        first.mask_for_epoch(0),
        first.mask_for_epoch(1),
    )


def test_sample_masks_are_reproducible_by_epoch_and_sample_id() -> None:
    first = EpochMaskManager(200, 7, 0.06, 5, base_seed=2021)
    second = EpochMaskManager(200, 7, 0.06, 5, base_seed=2021)

    np.testing.assert_array_equal(
        first.mask_for_sample(3, 17, length=96),
        second.mask_for_sample(3, 17, length=96),
    )
    assert first.mask_for_sample(3, 17, length=96).shape == (96, 7)
    assert not np.array_equal(
        first.mask_for_sample(3, 17, length=96),
        first.mask_for_sample(3, 18, length=96),
    )
    assert not np.array_equal(
        first.mask_for_sample(3, 17, length=96),
        first.mask_for_sample(4, 17, length=96),
    )


def test_missing_and_ours_receive_identical_dynamic_masks() -> None:
    missing = EpochMaskManager(8640, 7, 0.10, 5, base_seed=2021)
    ours = EpochMaskManager(8640, 7, 0.10, 5, base_seed=2021)

    for epoch in (0, 1, 17, 99):
        np.testing.assert_array_equal(
            missing.mask_for_epoch(epoch),
            ours.mask_for_epoch(epoch),
        )


def test_fixed_mask_round_trip_validates_metadata(tmp_path: Path) -> None:
    mask, metadata = generate_variable_missing_mask(
        120,
        4,
        start_rate=0.06,
        block_len=5,
        seed=2021,
        dataset="ETTh1",
        split="validation",
        support_start=8544,
    )
    path = tmp_path / "etth1_validation.npz"

    checksum = save_fixed_mask(path, mask, metadata)
    loaded, loaded_metadata = load_fixed_mask(
        path,
        MaskExpectation(
            dataset="ETTh1",
            split="validation",
            seed=2021,
            start_rate=0.06,
            block_len=5,
            support_start=8544,
            support_end=8664,
            shape=(120, 4),
        ),
    )

    np.testing.assert_array_equal(loaded, mask)
    assert loaded_metadata == metadata
    assert len(checksum) == 64


def test_fixed_mask_rejects_wrong_split_or_support(tmp_path: Path) -> None:
    mask, metadata = generate_variable_missing_mask(
        80,
        2,
        start_rate=0.06,
        block_len=5,
        seed=2021,
        dataset="Weather",
        split="test",
        support_start=42061,
    )
    path = tmp_path / "weather_test.npz"
    save_fixed_mask(path, mask, metadata)

    with pytest.raises(ValueError, match="metadata mismatch"):
        load_fixed_mask(
            path,
            MaskExpectation(
                dataset="Weather",
                split="validation",
                seed=2021,
                start_rate=0.06,
                block_len=5,
                support_start=42061,
                support_end=42141,
                shape=(80, 2),
            ),
        )


def test_generation_rejects_impossible_start_count() -> None:
    with pytest.raises(ValueError, match="more unique starts"):
        generate_variable_missing_mask(
            6,
            1,
            start_rate=1.0,
            block_len=5,
            seed=1,
        )
