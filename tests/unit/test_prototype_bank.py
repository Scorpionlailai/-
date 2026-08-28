from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from malr.prototypes.bank import (
    build_prototype_bank,
    load_bank,
    save_bank,
)
from malr.prototypes.matrix_profile import candidate_matrix_profile


def test_matrix_profile_excludes_self_and_trivial_matches():
    candidates = np.stack(
        [np.arange(64), np.arange(64), np.arange(63, -1, -1)]
    ).astype(np.float64)
    starts = np.array([0, 16, 64], dtype=np.int64)

    result = candidate_matrix_profile(candidates, starts, exclusion=32)

    assert result.neighbor_indices[0] == 2
    assert result.neighbor_indices[1] == 2
    assert result.neighbor_indices[2] in (0, 1)


def test_motif_groups_keep_one_real_medoid_and_apply_cap():
    motif = np.r_[np.arange(32), np.arange(31, -1, -1)]
    series = np.tile(motif, 20).astype(np.float64)
    bank = build_prototype_bank(
        series[:, None],
        train_end=len(series),
        max_per_variable=2,
    )

    assert 0 < len(bank.records[0]) <= 2
    assert all(record.occurrence_count >= 2 for record in bank.records[0])
    assert all(
        np.array_equal(record.raw_segment, series[record.start : record.start + 64])
        for record in bank.records[0]
    )


def test_short_series_produces_empty_bank():
    series = np.arange(40, dtype=np.float64)[:, None]
    bank = build_prototype_bank(series, train_end=40)
    assert bank.records == ((),)


def test_bank_persistence_rejects_wrong_identity(tmp_path: Path):
    series = np.tile(np.r_[np.arange(32), np.arange(31, -1, -1)], 8)[
        :, None
    ].astype(np.float64)
    bank = build_prototype_bank(series, train_end=len(series), dataset="fixture")
    path = tmp_path / "prototype_bank.npz"
    save_bank(bank, path)

    restored = load_bank(
        path,
        source_train=series,
        dataset="fixture",
        config_sha256=bank.config_sha256,
    )
    assert restored.source_checksum == bank.source_checksum
    assert len(restored.records[0]) == len(bank.records[0])

    with pytest.raises(ValueError, match="dataset"):
        load_bank(
            path,
            source_train=series,
            dataset="other",
            config_sha256=bank.config_sha256,
        )

    changed = series.copy()
    changed[0, 0] += 1.0
    with pytest.raises(ValueError, match="checksum"):
        load_bank(
            path,
            source_train=changed,
            dataset="fixture",
            config_sha256=bank.config_sha256,
        )

    with pytest.raises(ValueError, match="config"):
        load_bank(
            path,
            source_train=series,
            dataset="fixture",
            config_sha256="wrong",
        )


def test_bank_records_are_immutable():
    series = np.tile(np.arange(64), 4)[:, None].astype(np.float64)
    bank = build_prototype_bank(series, train_end=len(series))
    if bank.records[0]:
        with pytest.raises(ValueError):
            bank.records[0][0].raw_segment[0] = -1.0
        with pytest.raises(Exception):
            replace(bank.records[0][0], start=-1).variable_id = 2
