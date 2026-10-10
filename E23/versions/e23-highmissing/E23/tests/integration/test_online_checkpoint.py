from __future__ import annotations

from pathlib import Path

import pytest
import torch

from malr.training.checkpoints import save_checkpoint
from malr.training.online_checkpoint import (
    ONLINE_STATE_KEYS,
    load_online_checkpoint,
    save_online_checkpoint,
)

from test_online_stage1 import _fixture
from test_online_stage1_v2 import _v2_fixture


def _save_args() -> dict[str, object]:
    return {
        "model_state": {"weight": torch.ones(2)},
        "memory_state": {"format_version": 1, "snapshot": {"epoch": 3}},
        "ema_state": {"format_version": 1, "frozen_epoch": 3},
        "relation_prior_state": {"values": torch.zeros(2, 2, 7)},
        "metadata": {"dataset": "ETTh1", "variant": "ours"},
        "epoch": 3,
        "config_hash": "c" * 64,
        "data_hash": "d" * 64,
        "mask_hash": "m" * 64,
        "memory_hash": "b" * 64,
    }


def test_online_checkpoint_round_trip_preserves_all_paired_state(tmp_path: Path) -> None:
    path = tmp_path / "online.pt"
    save_online_checkpoint(path, **_save_args())

    restored = load_online_checkpoint(
        path,
        expected_metadata={"dataset": "ETTh1"},
        expected_hashes={"config_hash": "c" * 64, "memory_hash": "b" * 64},
    )

    assert set(restored["online_state"]) == set(ONLINE_STATE_KEYS)
    assert restored["online_state"]["epoch"] == 3
    assert restored["online_state"]["hashes"]["mask_hash"] == "m" * 64
    torch.testing.assert_close(restored["model_state"]["weight"], torch.ones(2))


def test_online_checkpoint_requires_complete_state_and_rejects_legacy_file(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.pt"
    save_checkpoint(legacy, {"weight": torch.ones(1)}, {"dataset": "ETTh1"})

    with pytest.raises(ValueError, match="online_state"):
        load_online_checkpoint(legacy)

    incomplete = tmp_path / "incomplete.pt"
    save_online_checkpoint(incomplete, **_save_args())
    payload = torch.load(incomplete, weights_only=False)
    payload["online_state"].pop("ema_state")
    torch.save(payload, incomplete)

    with pytest.raises(ValueError, match="incomplete"):
        load_online_checkpoint(incomplete)


def test_online_checkpoint_validates_expected_hashes(tmp_path: Path) -> None:
    path = tmp_path / "online.pt"
    save_online_checkpoint(path, **_save_args())

    with pytest.raises(ValueError, match="hash mismatch"):
        load_online_checkpoint(path, expected_hashes={"memory_hash": "x" * 64})


def test_v2_online_checkpoint_preserves_both_statistics(tmp_path: Path) -> None:
    """A V2 restore must retain variable and route learned statistics."""

    model, _ = _v2_fixture()
    model.variable_statistics.count[0, 1, 2] = 7
    model.route_statistics.count[2] = 9
    state = model.online_state()
    path = tmp_path / "relation-v2.pt"
    save_online_checkpoint(
        path,
        model.state_dict(),
        state["memory_state"],  # type: ignore[arg-type]
        state["ema_state"],  # type: ignore[arg-type]
        state["relation_prior_state"],  # type: ignore[arg-type]
        {"dataset": "ETTh1", "variant": "ours"},
        epoch=0,
        config_hash="c" * 64,
        data_hash="d" * 64,
        mask_hash="m" * 64,
        memory_hash=str(state["memory_hash"]),
    )

    restored, _ = _v2_fixture()
    restored.load_online_state(load_online_checkpoint(path)["online_state"])

    assert restored.variable_statistics.count[0, 1, 2].item() == 7
    assert restored.route_statistics.count[2].item() == 9


def test_legacy_model_rejects_a_v2_relation_state() -> None:
    """A legacy model must not silently interpret a V2 relation checkpoint."""

    legacy, _ = _fixture()
    v2, _ = _v2_fixture()
    state = v2.online_state()

    with pytest.raises(ValueError, match="format 1"):
        legacy.load_online_state(state)
