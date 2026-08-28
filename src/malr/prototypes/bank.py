from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .factors import FACTOR_COUNT, FactorVector, compute_context_factors
from .matrix_profile import candidate_matrix_profile, discover_motif_groups


DEFAULT_CENTER = (24, 40)


def source_checksum(values: NDArray[np.float64]) -> str:
    contiguous = np.ascontiguousarray(values, dtype=np.float64)
    digest = sha256()
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _readonly(array: ArrayLike, dtype: np.dtype) -> NDArray:
    result = np.array(array, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class PrototypeRecord:
    variable_id: int
    start: int
    raw_segment: NDArray[np.float64]
    center_start: int
    center_stop: int
    factors: FactorVector
    occurrence_count: int
    mean_profile_distance: float

    def __post_init__(self) -> None:
        segment = _readonly(self.raw_segment, np.dtype(np.float64))
        if segment.shape != (64,):
            raise ValueError("prototype raw_segment must contain 64 points")
        if not 0 <= self.center_start < self.center_stop <= segment.size:
            raise ValueError("invalid prototype center slice")
        if self.occurrence_count < 2:
            raise ValueError("prototype must represent at least two occurrences")
        object.__setattr__(self, "raw_segment", segment)

    @property
    def center_slice(self) -> slice:
        return slice(self.center_start, self.center_stop)


@dataclass(frozen=True)
class PrototypeBank:
    dataset: str
    train_end: int
    source_checksum: str
    config_sha256: str
    records: tuple[tuple[PrototypeRecord, ...], ...]
    factor_mean: NDArray[np.float64]
    factor_std: NDArray[np.float64]

    def __post_init__(self) -> None:
        means = _readonly(self.factor_mean, np.dtype(np.float64))
        standard_deviations = _readonly(self.factor_std, np.dtype(np.float64))
        expected = (len(self.records), FACTOR_COUNT)
        if means.shape != expected or standard_deviations.shape != expected:
            raise ValueError("factor statistics must have shape [variables, 7]")
        if not np.isfinite(means).all() or not np.isfinite(standard_deviations).all():
            raise ValueError("factor statistics must be finite")
        if np.any(standard_deviations <= 0):
            raise ValueError("factor standard deviations must be positive")
        object.__setattr__(self, "factor_mean", means)
        object.__setattr__(self, "factor_std", standard_deviations)


def _settings_hash(settings: dict[str, object]) -> str:
    payload = json.dumps(settings, sort_keys=True, separators=(",", ":")).encode()
    return sha256(payload).hexdigest()


def _factor_statistics(factors: list[FactorVector]) -> tuple[NDArray, NDArray]:
    if not factors:
        return np.zeros(FACTOR_COUNT), np.ones(FACTOR_COUNT)
    values = np.stack([factor.values for factor in factors])
    available = np.stack([factor.available for factor in factors])
    means = np.zeros(FACTOR_COUNT, dtype=np.float64)
    standard_deviations = np.ones(FACTOR_COUNT, dtype=np.float64)
    for index in range(FACTOR_COUNT):
        selected = values[available[:, index], index]
        if selected.size:
            means[index] = selected.mean()
            deviation = selected.std(ddof=0)
            standard_deviations[index] = deviation if deviation > 1e-12 else 1.0
    return means, standard_deviations


def build_prototype_bank(
    source_train: ArrayLike,
    *,
    train_end: int,
    allowed_train_end: int | None = None,
    dataset: str = "unknown",
    config_sha256: str | None = None,
    window: int = 64,
    scan_stride: int = 32,
    exclusion: int = 32,
    motif_percentile: float = 0.20,
    min_occurrences: int = 2,
    max_per_variable: int = 256,
    center_slice: slice = slice(*DEFAULT_CENTER),
) -> PrototypeBank:
    values = np.asarray(source_train, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("source_train must have shape [train time, variables]")
    if values.shape[0] != train_end:
        raise ValueError("source_train must contain exactly the declared train interval")
    if allowed_train_end is not None and train_end != allowed_train_end:
        raise ValueError("prototype construction exceeds the declared train boundary")
    if not np.isfinite(values).all():
        raise ValueError("source_train must be finite")
    if window != 64 or scan_stride <= 0 or exclusion <= 0:
        raise ValueError("stage-1 requires a 64-point window and positive strides")
    if max_per_variable <= 0 or max_per_variable > 256:
        raise ValueError("max_per_variable must be between 1 and 256")
    center_start, center_stop, center_step = center_slice.indices(window)
    if center_step != 1 or (center_start, center_stop) != DEFAULT_CENTER:
        raise ValueError("stage-1 center slice must equal [24, 40)")

    settings = {
        "window": window,
        "scan_stride": scan_stride,
        "exclusion": exclusion,
        "motif_percentile": motif_percentile,
        "min_occurrences": min_occurrences,
        "max_per_variable": max_per_variable,
        "center": [center_start, center_stop],
    }
    identity = config_sha256 or _settings_hash(settings)
    starts = np.arange(0, max(train_end - window + 1, 0), scan_stride, dtype=np.int64)
    records_by_variable: list[tuple[PrototypeRecord, ...]] = []
    means_by_variable: list[NDArray] = []
    stds_by_variable: list[NDArray] = []

    for variable_id in range(values.shape[1]):
        train_std = float(values[:, variable_id].std(ddof=0))
        if train_std <= 1e-12:
            train_std = 1.0
        candidates = np.stack(
            [values[start : start + window, variable_id] for start in starts],
            axis=0,
        ) if starts.size else np.empty((0, window), dtype=np.float64)
        factors = [
            compute_context_factors(
                candidate,
                np.ones(window, dtype=bool),
                center_slice=slice(center_start, center_stop),
                train_std=train_std,
            )
            for candidate in candidates
        ]
        factor_mean, factor_std = _factor_statistics(factors)
        means_by_variable.append(factor_mean)
        stds_by_variable.append(factor_std)

        profile = candidate_matrix_profile(candidates, starts, exclusion=exclusion)
        groups = discover_motif_groups(
            profile,
            percentile=motif_percentile,
            min_occurrences=min_occurrences,
        )
        ranked_groups = sorted(
            groups,
            key=lambda group: (
                -len(group.occurrence_indices),
                group.mean_profile_distance,
                int(starts[group.medoid_index]),
            ),
        )[:max_per_variable]
        variable_records = tuple(
            PrototypeRecord(
                variable_id=variable_id,
                start=int(starts[group.medoid_index]),
                raw_segment=candidates[group.medoid_index],
                center_start=center_start,
                center_stop=center_stop,
                factors=factors[group.medoid_index],
                occurrence_count=len(group.occurrence_indices),
                mean_profile_distance=group.mean_profile_distance,
            )
            for group in ranked_groups
        )
        records_by_variable.append(variable_records)

    return PrototypeBank(
        dataset=dataset,
        train_end=train_end,
        source_checksum=source_checksum(values),
        config_sha256=identity,
        records=tuple(records_by_variable),
        factor_mean=np.stack(means_by_variable),
        factor_std=np.stack(stds_by_variable),
    )


def save_bank(bank: PrototypeBank, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata: list[dict[str, object]] = []
    segments: list[NDArray] = []
    factor_values: list[NDArray] = []
    factor_available: list[NDArray] = []
    for variable_records in bank.records:
        for record in variable_records:
            metadata.append(
                {
                    "variable_id": record.variable_id,
                    "start": record.start,
                    "center_start": record.center_start,
                    "center_stop": record.center_stop,
                    "occurrence_count": record.occurrence_count,
                    "mean_profile_distance": record.mean_profile_distance,
                }
            )
            segments.append(record.raw_segment)
            factor_values.append(record.factors.values)
            factor_available.append(record.factors.available)
    header = {
        "version": 1,
        "dataset": bank.dataset,
        "train_end": bank.train_end,
        "source_checksum": bank.source_checksum,
        "config_sha256": bank.config_sha256,
        "variable_count": len(bank.records),
        "records": metadata,
    }
    np.savez_compressed(
        path,
        header=np.array(json.dumps(header, sort_keys=True)),
        segments=np.stack(segments) if segments else np.empty((0, 64)),
        factor_values=(
            np.stack(factor_values) if factor_values else np.empty((0, FACTOR_COUNT))
        ),
        factor_available=(
            np.stack(factor_available)
            if factor_available
            else np.empty((0, FACTOR_COUNT), dtype=bool)
        ),
        factor_mean=bank.factor_mean,
        factor_std=bank.factor_std,
    )


def load_bank(
    path: Path,
    *,
    source_train: ArrayLike,
    dataset: str,
    config_sha256: str,
) -> PrototypeBank:
    values = np.asarray(source_train, dtype=np.float64)
    with np.load(path, allow_pickle=False) as archive:
        header = json.loads(str(archive["header"].item()))
        if header.get("version") != 1:
            raise ValueError("unsupported prototype bank version")
        if header["dataset"] != dataset:
            raise ValueError("prototype bank dataset identity mismatch")
        if header["config_sha256"] != config_sha256:
            raise ValueError("prototype bank config identity mismatch")
        if values.shape[0] != int(header["train_end"]):
            raise ValueError("prototype bank train boundary mismatch")
        if source_checksum(values) != header["source_checksum"]:
            raise ValueError("prototype bank source checksum mismatch")

        records: list[list[PrototypeRecord]] = [
            [] for _ in range(int(header["variable_count"]))
        ]
        for index, item in enumerate(header["records"]):
            record = PrototypeRecord(
                variable_id=int(item["variable_id"]),
                start=int(item["start"]),
                raw_segment=archive["segments"][index],
                center_start=int(item["center_start"]),
                center_stop=int(item["center_stop"]),
                factors=FactorVector(
                    archive["factor_values"][index],
                    archive["factor_available"][index],
                ),
                occurrence_count=int(item["occurrence_count"]),
                mean_profile_distance=float(item["mean_profile_distance"]),
            )
            records[record.variable_id].append(record)
        return PrototypeBank(
            dataset=header["dataset"],
            train_end=int(header["train_end"]),
            source_checksum=header["source_checksum"],
            config_sha256=header["config_sha256"],
            records=tuple(tuple(variable) for variable in records),
            factor_mean=archive["factor_mean"],
            factor_std=archive["factor_std"],
        )
