from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: Path,
    model_state: Mapping[str, Any],
    metadata: Mapping[str, Any],
    **extra: Any,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "model_state": dict(model_state),
        "metadata": dict(metadata),
        **extra,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(
    path: Path,
    *,
    expected_metadata: Mapping[str, Any] | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format")
    if "model_state" not in payload or "metadata" not in payload:
        raise ValueError("checkpoint is incomplete")
    if expected_metadata is not None:
        actual = payload["metadata"]
        mismatches = {
            key: (actual.get(key), expected)
            for key, expected in expected_metadata.items()
            if actual.get(key) != expected
        }
        if mismatches:
            raise ValueError(f"checkpoint metadata mismatch: {mismatches}")
    return payload
