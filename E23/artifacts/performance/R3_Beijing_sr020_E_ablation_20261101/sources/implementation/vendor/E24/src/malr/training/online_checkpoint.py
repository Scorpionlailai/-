"""Paired checkpoint wrapper for online model and memory state."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from malr.training.checkpoints import load_checkpoint, save_checkpoint


ONLINE_STATE_KEYS: tuple[str, ...] = (
    "epoch",
    "memory_state",
    "ema_state",
    "relation_prior_state",
    "hashes",
)
_HASH_KEYS: tuple[str, ...] = (
    "config_hash",
    "data_hash",
    "mask_hash",
    "memory_hash",
)


def _validate_state_mapping(name: str, value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _validate_epoch(epoch: object) -> int:
    if type(epoch) is not int or epoch < 0:
        raise ValueError("online checkpoint epoch must be a non-negative integer")
    return epoch


def _validate_hashes(hashes: Mapping[str, object]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key in _HASH_KEYS:
        value = hashes.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"online checkpoint hash '{key}' must be a non-empty string")
        normalized[key] = value
    return normalized


def save_online_checkpoint(
    path: Path,
    model_state: Mapping[str, Any],
    memory_state: Mapping[str, Any],
    ema_state: Mapping[str, Any],
    relation_prior_state: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    epoch: int,
    config_hash: str,
    data_hash: str,
    mask_hash: str,
    memory_hash: str,
) -> None:
    """Atomically save model and all online states in one checkpoint payload."""

    model_state = _validate_state_mapping("model_state", model_state)
    memory_state = _validate_state_mapping("memory_state", memory_state)
    ema_state = _validate_state_mapping("ema_state", ema_state)
    relation_prior_state = _validate_state_mapping(
        "relation_prior_state", relation_prior_state
    )
    metadata = _validate_state_mapping("metadata", metadata)
    epoch = _validate_epoch(epoch)
    hashes = _validate_hashes(
        {
            "config_hash": config_hash,
            "data_hash": data_hash,
            "mask_hash": mask_hash,
            "memory_hash": memory_hash,
        }
    )
    save_checkpoint(
        path,
        model_state,
        metadata,
        online_state={
            "epoch": epoch,
            "memory_state": dict(memory_state),
            "ema_state": dict(ema_state),
            "relation_prior_state": dict(relation_prior_state),
            "hashes": hashes,
        },
    )


def load_online_checkpoint(
    path: Path,
    *,
    expected_metadata: Mapping[str, Any] | None = None,
    expected_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Load and validate a complete online checkpoint payload."""

    payload = load_checkpoint(path, expected_metadata=expected_metadata)
    online_state = payload.get("online_state")
    if not isinstance(online_state, Mapping):
        raise ValueError("online_state is incomplete")
    missing = [key for key in ONLINE_STATE_KEYS if key not in online_state]
    if missing:
        raise ValueError(f"online_state is incomplete: missing {missing}")
    _validate_epoch(online_state["epoch"])
    _validate_state_mapping("memory_state", online_state["memory_state"])
    _validate_state_mapping("ema_state", online_state["ema_state"])
    _validate_state_mapping("relation_prior_state", online_state["relation_prior_state"])
    hashes_value = online_state["hashes"]
    if not isinstance(hashes_value, Mapping):
        raise ValueError("online_state hashes are incomplete")
    hashes = _validate_hashes(hashes_value)
    if expected_hashes is not None:
        for key, expected in expected_hashes.items():
            if key not in hashes or hashes[key] != expected:
                raise ValueError("online checkpoint hash mismatch")
    return payload


__all__ = [
    "ONLINE_STATE_KEYS",
    "load_online_checkpoint",
    "save_online_checkpoint",
]
