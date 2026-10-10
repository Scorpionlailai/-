"""Stage-2 weight-only continuation; deliberately not an optimizer/RNG resume."""

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from malr.models.online_stage1_v2 import RelationV2OnlineStage1Model
from .online_checkpoint import load_online_checkpoint


def _same_memory(left: object, right: object, *, field: str = "") -> bool:
    """Only unit-vector clean keys may differ by float64 reload roundoff."""
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return left.keys() == right.keys() and all(
            _same_memory(left[key], right[key], field=key) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            _same_memory(a, b, field=field) for a, b in zip(left, right)
        )
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        if left.shape != right.shape or left.dtype != right.dtype:
            return False
        if field == "clean_key":
            return bool(np.isfinite(left).all() and np.isfinite(right).all()
                        and np.allclose(left, right, rtol=0, atol=1e-14))
        return bool(np.array_equal(left, right))
    return type(left) is type(right) and left == right


def verify_frozen_online_state(before: Mapping, after: Mapping) -> dict[str, object]:
    for key in ("epoch", "ema_hash", "relation_prior_hash"):
        if after[key] != before[key]:
            raise ValueError(f"warm-start changed frozen Stage-1 online state: {key}")
    # OnlinePrototypeMember normalizes keys on each checkpoint load. Hashes
    # can differ by ~1e-17 without any changed sample, grouping or raw value.
    # Check ALL memory fields; never apply a blanket approximate comparison.
    if not _same_memory(before["memory_state"], after["memory_state"]):
        raise ValueError("warm-start changed frozen Stage-1 online state: memory")
    return {
        "memory_bitwise_hash_equal": before["memory_hash"] == after["memory_hash"],
        "memory_semantically_equal": True,
        "clean_key_atol": 1e-14,
        "other_memory_fields_exact": True,
        "ema_and_relation_prior_exact": True,
    }


def load_forecast_warm_start(
    model: RelationV2OnlineStage1Model,
    checkpoint: Path,
    *,
    source_epoch: int,
    dataset: str,
    expected_hashes: Mapping[str, str],
    stage1_checkpoint_sha256: str,
    teacher_mode: str | None,
    teacher_sha256: str | None,
) -> dict[str, object]:
    """Load full Stage-2 weights into an initialized Stage-1 handoff skeleton.

    Validate the frozen teacher and online library against that Stage-1 handoff.
    Old checkpoints have no training_epoch: callers must supply the verified
    forecast epoch, never infer it from the frozen library's online epoch.
    """
    if type(source_epoch) is not int or source_epoch <= 0:
        raise ValueError("warm-start source epoch must be a positive integer")
    payload = load_online_checkpoint(checkpoint, expected_hashes=expected_hashes)
    metadata = payload["metadata"]
    expected = {
        "dataset": dataset,
        "training_stage": "forecast",
        "stage1_checkpoint_sha256": stage1_checkpoint_sha256,
        "stage2_route_teacher": teacher_mode,
        "stage2_route_teacher_sha256": teacher_sha256,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"warm-start checkpoint lineage mismatch: {key}")
    if "training_epoch" in metadata and metadata["training_epoch"] != source_epoch:
        raise ValueError("warm-start source epoch disagrees with checkpoint training_epoch")
    before = model.online_state()
    model.load_state_dict(payload["model_state"], strict=True)
    model.load_online_state(payload["online_state"])
    after = model.online_state()
    verification = verify_frozen_online_state(before, after)
    if teacher_sha256 is not None:
        if model.route_teacher is None or model.route_teacher.state_sha256() != teacher_sha256:
            raise ValueError("warm-start changed frozen Stage-1 route teacher")
    return verification
