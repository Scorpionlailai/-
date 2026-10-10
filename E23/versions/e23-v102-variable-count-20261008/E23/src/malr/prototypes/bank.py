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


def _unit_vector(value: ArrayLike, *, name: str) -> NDArray[np.float64]:
    vector = np.asarray(value, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a finite non-empty vector")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError(f"{name} must have non-zero norm")
    return _readonly(vector / norm, np.dtype(np.float64))


@dataclass(frozen=True)
class GroupedPrototype:
    """A concrete clean patch retained by the grouped memory."""

    variable_id: int
    start: int
    raw_patch: NDArray[np.float64]
    clean_key: NDArray[np.float64]

    def __post_init__(self) -> None:
        raw_patch = np.asarray(self.raw_patch, dtype=np.float64)
        if raw_patch.ndim != 1 or raw_patch.size == 0:
            raise ValueError("grouped prototype raw_patch must be a non-empty vector")
        if not np.isfinite(raw_patch).all():
            raise ValueError("grouped prototype raw_patch must be finite")
        if self.variable_id < 0 or self.start < 0:
            raise ValueError("grouped prototype identifiers must be non-negative")
        object.__setattr__(self, "raw_patch", _readonly(raw_patch, np.dtype(np.float64)))
        object.__setattr__(self, "clean_key", _unit_vector(self.clean_key, name="clean_key"))


@dataclass(frozen=True)
class GroupedPrototypeGroup:
    """A bounded FIFO member list with an unbounded cumulative center."""

    variable_id: int
    group_id: int
    center_sum: NDArray[np.float64]
    members: tuple[GroupedPrototype, ...]
    received_count: int

    def __post_init__(self) -> None:
        center_sum = np.asarray(self.center_sum, dtype=np.float64)
        if center_sum.ndim != 1 or center_sum.size == 0:
            raise ValueError("group center_sum must be a non-empty vector")
        if not np.isfinite(center_sum).all() or np.linalg.norm(center_sum) <= 1e-12:
            raise ValueError("group center_sum must be finite and non-zero")
        if self.variable_id < 0 or self.group_id < 0:
            raise ValueError("group identifiers must be non-negative")
        if self.received_count < len(self.members) or self.received_count <= 0:
            raise ValueError("group received_count is inconsistent with members")
        if any(member.variable_id != self.variable_id for member in self.members):
            raise ValueError("group member variable does not match group")
        starts = [member.start for member in self.members]
        if len(starts) != len(set(starts)):
            raise ValueError("group members must have unique absolute starts")
        object.__setattr__(self, "center_sum", _readonly(center_sum, np.dtype(np.float64)))
        object.__setattr__(self, "members", tuple(self.members))

    @property
    def center(self) -> NDArray[np.float64]:
        return _unit_vector(self.center_sum, name="group center")


@dataclass(frozen=True)
class GroupedPrototypeBank:
    """Version-2 grouped clean-mask memory used by the historical relation."""

    dataset: str
    train_end: int
    source_checksum: str
    config_sha256: str
    clean_checkpoint_sha256: str
    patch_len: int
    patch_stride: int
    retrieval_dim: int
    group_similarity_threshold: float
    group_capacity: int
    max_groups_per_variable: int
    train_mean: NDArray[np.float64]
    train_std: NDArray[np.float64]
    groups: tuple[tuple[GroupedPrototypeGroup, ...], ...]
    diagnostics: dict[str, int | float]
    mask_fill: float = 0.0
    retrieval_state: tuple[tuple[str, NDArray[np.float32]], ...] = ()
    format_version: int = 2

    def __post_init__(self) -> None:
        if self.format_version != 2:
            raise ValueError("grouped prototype bank format version must equal 2")
        if self.train_end <= 0 or self.patch_len <= 0 or self.patch_stride <= 0:
            raise ValueError("grouped bank geometry must be positive")
        if not 0.0 < self.group_similarity_threshold <= 1.0:
            raise ValueError("group similarity threshold must lie in (0, 1]")
        if self.group_capacity <= 0 or self.max_groups_per_variable <= 0:
            raise ValueError("grouped bank capacities must be positive")
        means = np.asarray(self.train_mean, dtype=np.float64)
        standard_deviations = np.asarray(self.train_std, dtype=np.float64)
        if means.ndim != 1 or standard_deviations.shape != means.shape:
            raise ValueError("grouped bank normalization statistics must be vectors")
        if not np.isfinite(means).all() or not np.isfinite(standard_deviations).all():
            raise ValueError("grouped bank normalization statistics must be finite")
        if np.any(standard_deviations <= 0.0):
            raise ValueError("grouped bank train_std must be positive")
        if len(means) != len(self.groups):
            raise ValueError("grouped bank statistics and groups disagree on variables")
        for variable_id, variable_groups in enumerate(self.groups):
            if len(variable_groups) > self.max_groups_per_variable:
                raise ValueError("grouped bank exceeds max_groups_per_variable")
            for group in variable_groups:
                if group.variable_id != variable_id:
                    raise ValueError("grouped bank group variable id is inconsistent")
                if len(group.members) > self.group_capacity:
                    raise ValueError("grouped bank group exceeds group_capacity")
                if group.center_sum.size != self.retrieval_dim:
                    raise ValueError("group center dimension does not match retrieval_dim")
                if any(member.clean_key.size != self.retrieval_dim for member in group.members):
                    raise ValueError("group member key dimension does not match retrieval_dim")
        object.__setattr__(self, "train_mean", _readonly(means, np.dtype(np.float64)))
        object.__setattr__(
            self,
            "train_std",
            _readonly(standard_deviations, np.dtype(np.float64)),
        )
        object.__setattr__(self, "groups", tuple(tuple(groups) for groups in self.groups))
        normalized_state: list[tuple[str, NDArray[np.float32]]] = []
        for name, value in self.retrieval_state:
            array = np.asarray(value, dtype=np.float32)
            if not np.isfinite(array).all():
                raise ValueError("grouped bank retrieval state must be finite")
            normalized_state.append((str(name), _readonly(array, np.dtype(np.float32))))
        object.__setattr__(self, "retrieval_state", tuple(normalized_state))

    @property
    def variable_count(self) -> int:
        return len(self.groups)

    @property
    def member_count(self) -> int:
        return sum(len(group.members) for groups in self.groups for group in groups)


def build_grouped_prototype_bank(
    clean_patches: ArrayLike,
    clean_keys: ArrayLike,
    starts: ArrayLike,
    *,
    dataset: str,
    train_end: int,
    source_checksum: str,
    config_sha256: str,
    clean_checkpoint_sha256: str,
    patch_stride: int = 8,
    group_similarity_threshold: float = 0.8,
    group_capacity: int = 16,
    max_groups_per_variable: int = 128,
    train_mean: ArrayLike | None = None,
    train_std: ArrayLike | None = None,
    mask_fill: float = 0.0,
    retrieval_state: tuple[tuple[str, NDArray[np.float32]], ...] = (),
) -> GroupedPrototypeBank:
    patches = np.asarray(clean_patches, dtype=np.float64)
    keys = np.asarray(clean_keys, dtype=np.float64)
    absolute_starts = np.asarray(starts, dtype=np.int64)
    if patches.ndim != 3 or keys.ndim != 3:
        raise ValueError("clean_patches and clean_keys must have shape [variables, starts, features]")
    if patches.shape[:2] != keys.shape[:2] or patches.shape[1] != absolute_starts.size:
        raise ValueError("clean patches, keys, and starts have incompatible shapes")
    if patches.shape[1] == 0 or patches.shape[0] == 0:
        raise ValueError("grouped prototype bank requires at least one patch")
    if np.any(absolute_starts < 0) or len(np.unique(absolute_starts)) != absolute_starts.size:
        raise ValueError("absolute patch starts must be unique and non-negative")
    if train_end < int(absolute_starts.max()) + patches.shape[-1]:
        raise ValueError("prototype patch exceeds the declared train boundary")
    if not np.isfinite(patches).all() or not np.isfinite(keys).all():
        raise ValueError("clean patches and keys must be finite")
    if not 0.0 < group_similarity_threshold <= 1.0:
        raise ValueError("group_similarity_threshold must lie in (0, 1]")
    if group_capacity <= 0 or max_groups_per_variable <= 0 or patch_stride <= 0:
        raise ValueError("grouped bank capacities and patch_stride must be positive")
    order = np.argsort(absolute_starts, kind="stable")
    variable_count, _, patch_len = patches.shape
    retrieval_dim = keys.shape[-1]
    means = (
        np.zeros(variable_count, dtype=np.float64)
        if train_mean is None
        else np.asarray(train_mean, dtype=np.float64)
    )
    stds = (
        np.ones(variable_count, dtype=np.float64)
        if train_std is None
        else np.asarray(train_std, dtype=np.float64)
    )
    if means.shape != (variable_count,) or stds.shape != (variable_count,):
        raise ValueError("train_mean and train_std must have one value per variable")

    diagnostics: dict[str, int | float] = {
        "created_groups": 0,
        "joined_groups": 0,
        "evicted_members": 0,
        "evicted_groups": 0,
        "scanned_patches": int(variable_count * absolute_starts.size),
        "mean_group_similarity": 0.0,
    }
    similarity_values: list[float] = []
    groups_by_variable: list[tuple[GroupedPrototypeGroup, ...]] = []
    next_group_id = 0
    for variable_id in range(variable_count):
        variable_groups: list[GroupedPrototypeGroup] = []
        for source_index in order.tolist():
            key = _unit_vector(keys[variable_id, source_index], name="clean_key")
            best_index = -1
            best_similarity = -1.0
            for group_index, group in enumerate(variable_groups):
                similarity = float(np.dot(key, group.center))
                if similarity > best_similarity:
                    best_index = group_index
                    best_similarity = similarity
            if best_index >= 0:
                similarity_values.append(best_similarity)
            if best_index >= 0 and best_similarity >= group_similarity_threshold:
                group = variable_groups[best_index]
                member = GroupedPrototype(
                    variable_id=variable_id,
                    start=int(absolute_starts[source_index]),
                    raw_patch=patches[variable_id, source_index],
                    clean_key=key,
                )
                members = list(group.members)
                if len(members) >= group_capacity:
                    members.pop(0)
                    diagnostics["evicted_members"] = int(diagnostics["evicted_members"]) + 1
                members.append(member)
                variable_groups[best_index] = GroupedPrototypeGroup(
                    variable_id=group.variable_id,
                    group_id=group.group_id,
                    center_sum=group.center_sum + key,
                    members=tuple(members),
                    received_count=group.received_count + 1,
                )
                diagnostics["joined_groups"] = int(diagnostics["joined_groups"]) + 1
                continue

            if len(variable_groups) >= max_groups_per_variable:
                variable_groups.pop(0)
                diagnostics["evicted_groups"] = int(diagnostics["evicted_groups"]) + 1
            member = GroupedPrototype(
                variable_id=variable_id,
                start=int(absolute_starts[source_index]),
                raw_patch=patches[variable_id, source_index],
                clean_key=key,
            )
            variable_groups.append(
                GroupedPrototypeGroup(
                    variable_id=variable_id,
                    group_id=next_group_id,
                    center_sum=key,
                    members=(member,),
                    received_count=1,
                )
            )
            next_group_id += 1
            diagnostics["created_groups"] = int(diagnostics["created_groups"]) + 1
        groups_by_variable.append(tuple(variable_groups))
    if similarity_values:
        diagnostics["mean_group_similarity"] = float(np.mean(similarity_values))
    return GroupedPrototypeBank(
        dataset=dataset,
        train_end=int(train_end),
        source_checksum=source_checksum,
        config_sha256=config_sha256,
        clean_checkpoint_sha256=clean_checkpoint_sha256,
        patch_len=int(patch_len),
        patch_stride=int(patch_stride),
        retrieval_dim=int(retrieval_dim),
        group_similarity_threshold=float(group_similarity_threshold),
        group_capacity=int(group_capacity),
        max_groups_per_variable=int(max_groups_per_variable),
        train_mean=means,
        train_std=stds,
        groups=tuple(groups_by_variable),
        diagnostics=diagnostics,
        mask_fill=float(mask_fill),
        retrieval_state=retrieval_state,
    )


def save_grouped_bank(bank: GroupedPrototypeBank, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_patches: list[NDArray[np.float64]] = []
    clean_keys: list[NDArray[np.float64]] = []
    groups_metadata: list[dict[str, object]] = []
    for variable_groups in bank.groups:
        for group in variable_groups:
            member_indices: list[int] = []
            for member in group.members:
                member_indices.append(len(raw_patches))
                raw_patches.append(member.raw_patch)
                clean_keys.append(member.clean_key)
            groups_metadata.append(
                {
                    "variable_id": group.variable_id,
                    "group_id": group.group_id,
                    "center_sum": group.center_sum.tolist(),
                    "received_count": group.received_count,
                    "member_indices": member_indices,
                    "member_starts": [member.start for member in group.members],
                }
            )
    header = {
        "version": 2,
        "dataset": bank.dataset,
        "train_end": bank.train_end,
        "source_checksum": bank.source_checksum,
        "config_sha256": bank.config_sha256,
        "clean_checkpoint_sha256": bank.clean_checkpoint_sha256,
        "patch_len": bank.patch_len,
        "patch_stride": bank.patch_stride,
        "retrieval_dim": bank.retrieval_dim,
        "group_similarity_threshold": bank.group_similarity_threshold,
        "group_capacity": bank.group_capacity,
        "max_groups_per_variable": bank.max_groups_per_variable,
        "mask_fill": bank.mask_fill,
        "variable_count": bank.variable_count,
        "diagnostics": bank.diagnostics,
        "groups": groups_metadata,
        "retrieval_state_names": [name for name, _ in bank.retrieval_state],
    }
    retrieval_arrays = {
        f"retrieval_state_{index}": value
        for index, (_, value) in enumerate(bank.retrieval_state)
    }
    np.savez_compressed(
        path,
        header=np.asarray(json.dumps(header, sort_keys=True)),
        raw_patches=np.stack(raw_patches),
        clean_keys=np.stack(clean_keys),
        train_mean=bank.train_mean,
        train_std=bank.train_std,
        **retrieval_arrays,
    )


def load_grouped_bank(
    path: Path,
    *,
    dataset: str,
    train_end: int,
    source_checksum: str,
    config_sha256: str,
    clean_checkpoint_sha256: str,
) -> GroupedPrototypeBank:
    try:
        with np.load(path, allow_pickle=False) as archive:
            header = json.loads(str(archive["header"].item()))
            if header.get("version") != 2:
                raise ValueError("unsupported grouped prototype bank version")
            checks = {
                "dataset": (dataset, header.get("dataset")),
                "train boundary": (int(train_end), int(header.get("train_end", -1))),
                "source checksum": (source_checksum, header.get("source_checksum")),
                "config hash": (config_sha256, header.get("config_sha256")),
                "clean checkpoint": (
                    clean_checkpoint_sha256,
                    header.get("clean_checkpoint_sha256"),
                ),
            }
            for label, (expected, actual) in checks.items():
                if expected != actual:
                    raise ValueError(f"grouped bank {label} mismatch")
            raw_patches = np.asarray(archive["raw_patches"], dtype=np.float64)
            clean_keys = np.asarray(archive["clean_keys"], dtype=np.float64)
            groups_by_variable: list[list[GroupedPrototypeGroup]] = [
                [] for _ in range(int(header["variable_count"]))
            ]
            for group_item in header["groups"]:
                variable_id = int(group_item["variable_id"])
                indices = [int(index) for index in group_item["member_indices"]]
                starts = [int(start) for start in group_item["member_starts"]]
                if len(indices) != len(starts):
                    raise ValueError("grouped bank member metadata is inconsistent")
                members = tuple(
                    GroupedPrototype(
                        variable_id=variable_id,
                        start=start,
                        raw_patch=raw_patches[index],
                        clean_key=clean_keys[index],
                    )
                    for index, start in zip(indices, starts)
                )
                group = GroupedPrototypeGroup(
                    variable_id=variable_id,
                    group_id=int(group_item["group_id"]),
                    center_sum=np.asarray(group_item["center_sum"], dtype=np.float64),
                    members=members,
                    received_count=int(group_item["received_count"]),
                )
                groups_by_variable[variable_id].append(group)
            retrieval_state = tuple(
                (
                    str(name),
                    np.asarray(archive[f"retrieval_state_{index}"], dtype=np.float32),
                )
                for index, name in enumerate(header.get("retrieval_state_names", []))
            )
            return GroupedPrototypeBank(
                dataset=str(header["dataset"]),
                train_end=int(header["train_end"]),
                source_checksum=str(header["source_checksum"]),
                config_sha256=str(header["config_sha256"]),
                clean_checkpoint_sha256=str(header["clean_checkpoint_sha256"]),
                patch_len=int(header["patch_len"]),
                patch_stride=int(header["patch_stride"]),
                retrieval_dim=int(header["retrieval_dim"]),
                group_similarity_threshold=float(header["group_similarity_threshold"]),
                group_capacity=int(header["group_capacity"]),
                max_groups_per_variable=int(header["max_groups_per_variable"]),
                train_mean=archive["train_mean"],
                train_std=archive["train_std"],
                groups=tuple(tuple(groups) for groups in groups_by_variable),
                diagnostics={
                    str(key): value
                    for key, value in dict(header.get("diagnostics", {})).items()
                },
                mask_fill=float(header.get("mask_fill", 0.0)),
                retrieval_state=retrieval_state,
            )
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        if isinstance(exc, ValueError) and (
            str(exc).startswith("grouped bank")
            or "unsupported grouped prototype bank version" in str(exc)
        ):
            raise
        raise ValueError(f"invalid grouped prototype bank: {path}") from exc


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
