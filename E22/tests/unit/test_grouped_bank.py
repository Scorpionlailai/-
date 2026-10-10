from pathlib import Path

import numpy as np
import pytest
from malr.prototypes.retrieval import MaskAwareRetrievalEncoder, retrieval_state_tuple

from malr.prototypes.bank import (
    GroupedPrototypeBank,
    build_grouped_prototype_bank,
    load_grouped_bank,
    save_grouped_bank,
)


def _fixture_bank() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts = np.arange(6, dtype=np.int64)
    patches = np.stack(
        [np.full(4, float(index), dtype=np.float64) for index in range(6)],
        axis=0,
    )
    keys = np.array(
        [
            [1.0, 0.0],
            [0.999, 0.0447],
            [0.998, 0.0632],
            [0.0, 1.0],
            [0.0447, 0.999],
            [0.0632, 0.998],
        ],
        dtype=np.float64,
    )
    return starts, patches, keys


def test_grouped_bank_uses_threshold_and_fifo_capacity() -> None:
    starts, patches, keys = _fixture_bank()
    bank = build_grouped_prototype_bank(
        patches[None],
        keys[None],
        starts,
        dataset="fixture",
        train_end=10,
        source_checksum="source",
        config_sha256="config",
        clean_checkpoint_sha256="clean",
        group_similarity_threshold=0.8,
        group_capacity=2,
        max_groups_per_variable=2,
    )

    assert isinstance(bank, GroupedPrototypeBank)
    assert len(bank.groups[0]) == 2
    assert [len(group.members) for group in bank.groups[0]] == [2, 2]
    assert [member.start for member in bank.groups[0][0].members] == [1, 2]
    assert [member.start for member in bank.groups[0][1].members] == [4, 5]
    assert [group.received_count for group in bank.groups[0]] == [3, 3]
    assert bank.diagnostics["evicted_members"] == 2


def test_grouped_bank_round_trip_and_identity_checks(tmp_path: Path) -> None:
    starts, patches, keys = _fixture_bank()
    encoder = MaskAwareRetrievalEncoder(4, 2, hidden_dim=5)
    bank = build_grouped_prototype_bank(
        patches[None],
        keys[None],
        starts,
        dataset="fixture",
        train_end=10,
        source_checksum="source",
        config_sha256="config",
        clean_checkpoint_sha256="clean",
        retrieval_state=retrieval_state_tuple(encoder),
    )
    path = tmp_path / "grouped.npz"
    save_grouped_bank(bank, path)

    restored = load_grouped_bank(
        path,
        dataset="fixture",
        train_end=10,
        source_checksum="source",
        config_sha256="config",
        clean_checkpoint_sha256="clean",
    )
    assert restored.format_version == 2
    assert restored.groups[0][0].members[0].start == bank.groups[0][0].members[0].start
    assert np.array_equal(
        restored.groups[0][0].members[0].raw_patch,
        bank.groups[0][0].members[0].raw_patch,
    )
    assert restored.retrieval_state[0][0] == "network.0.weight"
    assert np.array_equal(restored.retrieval_state[0][1], bank.retrieval_state[0][1])

    with pytest.raises(ValueError, match="clean checkpoint"):
        load_grouped_bank(
            path,
            dataset="fixture",
            train_end=10,
            source_checksum="source",
            config_sha256="config",
            clean_checkpoint_sha256="other",
        )


def test_grouped_bank_rejects_duplicate_absolute_patch() -> None:
    starts, patches, keys = _fixture_bank()
    with pytest.raises(ValueError, match="unique"):
        build_grouped_prototype_bank(
            patches[None, :2],
            keys[None, :2],
            np.array([0, 0], dtype=np.int64),
            dataset="fixture",
            train_end=10,
            source_checksum="source",
            config_sha256="config",
            clean_checkpoint_sha256="clean",
        )
