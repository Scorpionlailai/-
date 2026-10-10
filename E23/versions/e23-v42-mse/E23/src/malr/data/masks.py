from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile

import numpy as np


@dataclass(frozen=True)
class MaskMetadata:
    dataset: str
    split: str
    seed: int
    start_rate: float
    block_len: int
    support_start: int
    support_end: int
    shape: tuple[int, int]
    final_rate: float
    starts: tuple[tuple[int, ...], ...]

    def canonical_json(self) -> str:
        return json.dumps(
            asdict(self),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, payload: str) -> "MaskMetadata":
        raw = json.loads(payload)
        raw["shape"] = tuple(int(value) for value in raw["shape"])
        raw["starts"] = tuple(
            tuple(int(value) for value in starts) for starts in raw["starts"]
        )
        return cls(**raw)


@dataclass(frozen=True)
class MaskExpectation:
    dataset: str
    split: str
    seed: int
    start_rate: float
    block_len: int
    support_start: int
    support_end: int
    shape: tuple[int, int]

    def mismatches(self, metadata: MaskMetadata) -> dict[str, tuple[object, object]]:
        mismatches: dict[str, tuple[object, object]] = {}
        for field_name, expected in asdict(self).items():
            actual = getattr(metadata, field_name)
            if field_name == "shape":
                expected = tuple(expected)
            if actual != expected:
                mismatches[field_name] = (expected, actual)
        return mismatches


def _validity_matrix(
    validity: np.ndarray | None,
    length: int,
    n_vars: int,
) -> np.ndarray:
    if validity is None:
        return np.ones((length, n_vars), dtype=bool)
    result = np.asarray(validity, dtype=bool)
    if result.shape == (length,):
        result = np.broadcast_to(result[:, None], (length, n_vars))
    if result.shape != (length, n_vars):
        raise ValueError(
            f"validity must have shape {(length,)} or {(length, n_vars)}"
        )
    if result.sum() == 0:
        raise ValueError("validity must contain at least one valid element")
    return np.ascontiguousarray(result)


def generate_variable_missing_mask(
    length: int,
    n_vars: int,
    start_rate: float,
    block_len: int,
    seed: int,
    validity: np.ndarray | None = None,
    *,
    dataset: str = "",
    split: str = "",
    support_start: int = 0,
) -> tuple[np.ndarray, MaskMetadata]:
    if length <= 0 or n_vars <= 0:
        raise ValueError("length and n_vars must be positive")
    if block_len <= 0 or block_len > length:
        raise ValueError("block_len must be in [1, length]")
    if not 0.0 <= start_rate <= 1.0:
        raise ValueError("start_rate must be in [0, 1]")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    valid = _validity_matrix(validity, length, n_vars)
    candidate_count = length - block_len + 1
    n_start = round(start_rate * length)
    if n_start > candidate_count:
        raise ValueError(
            "start_rate requests more unique starts than the valid start set"
        )

    rng = np.random.default_rng(seed)
    observed = np.ones((length, n_vars), dtype=bool)
    starts_by_variable: list[tuple[int, ...]] = []
    for variable in range(n_vars):
        sampled = rng.choice(candidate_count, size=n_start, replace=False)
        starts = tuple(sorted(int(value) for value in sampled.tolist()))
        starts_by_variable.append(starts)
        for start in starts:
            observed[start : start + block_len, variable] = False

    missing_valid = np.logical_and(~observed, valid).sum(dtype=np.int64)
    final_rate = float(missing_valid / valid.sum(dtype=np.int64))
    metadata = MaskMetadata(
        dataset=dataset,
        split=split,
        seed=seed,
        start_rate=float(start_rate),
        block_len=block_len,
        support_start=support_start,
        support_end=support_start + length,
        shape=(length, n_vars),
        final_rate=final_rate,
        starts=tuple(starts_by_variable),
    )
    return observed, metadata


def _mask_checksum(observed_mask: np.ndarray, metadata_json: str) -> str:
    digest = sha256()
    digest.update(np.ascontiguousarray(observed_mask, dtype=np.uint8).tobytes())
    digest.update(metadata_json.encode("utf-8"))
    return digest.hexdigest()


def _read_mask_file(path: Path) -> tuple[np.ndarray, MaskMetadata, str]:
    try:
        with np.load(path, allow_pickle=False) as payload:
            observed = np.asarray(payload["observed_mask"], dtype=bool)
            metadata_json = str(payload["metadata_json"].item())
            stored_checksum = str(payload["checksum"].item())
    except (KeyError, OSError, ValueError) as exc:
        raise ValueError(f"invalid fixed mask file: {path}") from exc
    metadata = MaskMetadata.from_json(metadata_json)
    actual_checksum = _mask_checksum(observed, metadata_json)
    if actual_checksum != stored_checksum:
        raise ValueError("fixed mask checksum mismatch")
    if observed.shape != metadata.shape:
        raise ValueError("fixed mask shape does not match metadata")
    return observed, metadata, stored_checksum


def save_fixed_mask(
    path: Path,
    observed_mask: np.ndarray,
    metadata: MaskMetadata,
) -> str:
    observed = np.ascontiguousarray(observed_mask, dtype=bool)
    if observed.shape != metadata.shape:
        raise ValueError("observed mask shape does not match metadata")
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_json = metadata.canonical_json()
    checksum = _mask_checksum(observed, metadata_json)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            np.savez_compressed(
                temporary,
                observed_mask=observed,
                metadata_json=np.asarray(metadata_json),
                checksum=np.asarray(checksum),
            )
        loaded, loaded_metadata, loaded_checksum = _read_mask_file(temporary_path)
        if not np.array_equal(loaded, observed) or loaded_metadata != metadata:
            raise ValueError("temporary fixed mask failed round-trip validation")
        if loaded_checksum != checksum:
            raise ValueError("temporary fixed mask checksum changed")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return checksum


def load_fixed_mask(
    path: Path,
    expected: MaskExpectation,
) -> tuple[np.ndarray, MaskMetadata]:
    observed, metadata, _ = _read_mask_file(path)
    mismatches = expected.mismatches(metadata)
    if mismatches:
        detail = ", ".join(
            f"{name}: expected {values[0]!r}, got {values[1]!r}"
            for name, values in sorted(mismatches.items())
        )
        raise ValueError(f"fixed mask metadata mismatch: {detail}")
    return observed, metadata


@dataclass(frozen=True)
class EpochMaskManager:
    length: int
    n_vars: int
    start_rate: float
    block_len: int
    base_seed: int
    validity: np.ndarray | None = None

    def mask_for_epoch(self, epoch: int) -> np.ndarray:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        epoch_seed = int(
            np.random.SeedSequence([self.base_seed, epoch]).generate_state(1)[0]
        )
        mask, _ = generate_variable_missing_mask(
            self.length,
            self.n_vars,
            self.start_rate,
            self.block_len,
            epoch_seed,
            self.validity,
            dataset="train",
            split="train",
            support_start=0,
        )
        return mask

    def mask_for_sample(
        self,
        epoch: int,
        sample_id: int,
        *,
        length: int | None = None,
    ) -> np.ndarray:
        """Generate a deterministic local mask keyed by epoch and sample id."""

        if epoch < 0 or sample_id < 0:
            raise ValueError("epoch and sample_id must be non-negative")
        local_length = self.length if length is None else int(length)
        if local_length <= 0:
            raise ValueError("mask length must be positive")
        seed = int(
            np.random.SeedSequence(
                [self.base_seed, int(epoch), int(sample_id)]
            ).generate_state(1)[0]
        )
        mask, _ = generate_variable_missing_mask(
            local_length,
            self.n_vars,
            self.start_rate,
            self.block_len,
            seed,
            self.validity,
            dataset="train",
            split="train",
            support_start=0,
        )
        return mask
