from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from typing import Iterable

import numpy as np
import torch
from torch import Tensor

from malr.data.patching import patchify_with_state


@dataclass(frozen=True)
class RepairedSplit:
    """Window-level handoff from repair pretraining to forecasting.

    ``values`` stay in the query-normalized value space used by PatchTST.  The
    original observation mask is stored independently, so repaired values are
    never erased merely because their coordinates were originally missing.
    """

    values: np.ndarray
    observed_mask: np.ndarray
    query_mean: np.ndarray
    query_std: np.ndarray
    targets: np.ndarray
    input_starts: np.ndarray
    confidence: np.ndarray | None = None

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        mask = np.asarray(self.observed_mask)
        mean = np.asarray(self.query_mean)
        std = np.asarray(self.query_std)
        targets = np.asarray(self.targets)
        starts = np.asarray(self.input_starts)
        if values.ndim != 3 or values.shape[0] == 0:
            raise ValueError("repaired values must have shape [windows, time, variables]")
        windows, _, variables = values.shape
        if mask.shape != values.shape or mask.dtype != np.bool_:
            raise ValueError("repaired observation mask must be boolean and match values")
        if mean.shape != (windows, variables) or std.shape != mean.shape:
            raise ValueError("repaired query statistics have invalid shape")
        if targets.ndim != 3 or targets.shape[:2] != (windows, variables):
            raise ValueError("repaired targets must have shape [windows, variables, horizon]")
        if starts.shape != (windows,) or not np.issubdtype(starts.dtype, np.integer):
            raise ValueError("repaired input starts must contain one integer per window")
        if self.confidence is not None:
            confidence = np.asarray(self.confidence)
            if confidence.shape != values.shape:
                raise ValueError("repaired confidence must match values")
            if np.any((confidence < 0.0) | (confidence > 1.0)):
                raise ValueError("repaired confidence must lie in [0, 1]")
        numeric = [values, mean, std, targets]
        if self.confidence is not None:
            numeric.append(np.asarray(self.confidence))
        if any(not np.isfinite(value).all() for value in numeric):
            raise ValueError("repaired split contains non-finite values")
        if np.any(std <= 0.0):
            raise ValueError("repaired query standard deviations must be positive")

    @property
    def window_count(self) -> int:
        return int(self.values.shape[0])

    @property
    def missing_rate(self) -> float:
        return float(1.0 - np.asarray(self.observed_mask, dtype=np.float64).mean())


@dataclass(frozen=True)
class RepairedForecastBatch:
    patch_values: Tensor
    patch_mask: Tensor
    patch_validity: Tensor
    query_mean: Tensor
    query_std: Tensor
    forecast_target: Tensor
    confidence: Tensor | None = None

    def to(self, device: str | torch.device) -> "RepairedForecastBatch":
        return RepairedForecastBatch(
            patch_values=self.patch_values.to(device),
            patch_mask=self.patch_mask.to(device),
            patch_validity=self.patch_validity.to(device),
            query_mean=self.query_mean.to(device),
            query_std=self.query_std.to(device),
            forecast_target=self.forecast_target.to(device),
            confidence=None if self.confidence is None else self.confidence.to(device),
        )


def save_repaired_split(path: Path, split: RepairedSplit) -> None:
    """Atomically persist one repaired split without pickle payloads."""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    payload = {
        "format_version": np.asarray(1, dtype=np.int64),
        "values": np.asarray(split.values, dtype=np.float32),
        "observed_mask": np.asarray(split.observed_mask, dtype=np.bool_),
        "query_mean": np.asarray(split.query_mean, dtype=np.float32),
        "query_std": np.asarray(split.query_std, dtype=np.float32),
        "targets": np.asarray(split.targets, dtype=np.float32),
        "input_starts": np.asarray(split.input_starts, dtype=np.int64),
    }
    if split.confidence is not None:
        payload["confidence"] = np.asarray(split.confidence, dtype=np.float32)
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            np.savez_compressed(temporary, **payload)
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def load_repaired_split(path: Path) -> RepairedSplit:
    try:
        with np.load(path, allow_pickle=False) as payload:
            if int(payload["format_version"]) != 1:
                raise ValueError("unsupported repaired split format")
            confidence = (
                np.asarray(payload["confidence"], dtype=np.float32)
                if "confidence" in payload.files
                else None
            )
            return RepairedSplit(
                values=np.asarray(payload["values"], dtype=np.float32),
                observed_mask=np.asarray(payload["observed_mask"], dtype=np.bool_),
                query_mean=np.asarray(payload["query_mean"], dtype=np.float32),
                query_std=np.asarray(payload["query_std"], dtype=np.float32),
                targets=np.asarray(payload["targets"], dtype=np.float32),
                input_starts=np.asarray(payload["input_starts"], dtype=np.int64),
                confidence=confidence,
            )
    except (OSError, KeyError, ValueError) as exc:
        raise ValueError(f"invalid repaired split artifact: {path}") from exc


def iter_repaired_batches(
    split: RepairedSplit,
    *,
    batch_size: int,
    patch_len: int,
    patch_stride: int,
    shuffle_seed: int | None = None,
    drop_last: bool = False,
) -> Iterable[RepairedForecastBatch]:
    """Patch repaired values while retaining their original missingness mask."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    indices = np.arange(split.window_count)
    if shuffle_seed is not None:
        np.random.default_rng(shuffle_seed).shuffle(indices)
    usable = len(indices) - len(indices) % batch_size if drop_last else len(indices)
    for offset in range(0, usable, batch_size):
        selected = indices[offset : offset + batch_size]
        values = torch.from_numpy(np.ascontiguousarray(split.values[selected]))
        observed = torch.from_numpy(
            np.ascontiguousarray(split.observed_mask[selected])
        )
        # Patching values with an all-observed mask is intentional: the values
        # at originally missing coordinates are the Stage-1 reconstruction and
        # must reach the forecast model.  Missingness metadata is patched in a
        # separate pass.
        value_patches = patchify_with_state(
            values,
            torch.ones_like(observed),
            length=patch_len,
            stride=patch_stride,
        )
        mask_patches = patchify_with_state(
            values,
            observed,
            length=patch_len,
            stride=patch_stride,
        )
        confidence = None
        if split.confidence is not None:
            confidence_values = torch.from_numpy(
                np.ascontiguousarray(split.confidence[selected])
            )
            confidence = patchify_with_state(
                confidence_values,
                torch.ones_like(observed),
                length=patch_len,
                stride=patch_stride,
            ).values
        yield RepairedForecastBatch(
            patch_values=value_patches.values,
            patch_mask=mask_patches.effective_mask,
            patch_validity=value_patches.validity,
            query_mean=torch.from_numpy(
                np.ascontiguousarray(split.query_mean[selected])
            ),
            query_std=torch.from_numpy(
                np.ascontiguousarray(split.query_std[selected])
            ),
            forecast_target=torch.from_numpy(
                np.ascontiguousarray(split.targets[selected])
            ),
            confidence=confidence,
        )


__all__ = [
    "RepairedForecastBatch",
    "RepairedSplit",
    "iter_repaired_batches",
    "load_repaired_split",
    "save_repaired_split",
]
