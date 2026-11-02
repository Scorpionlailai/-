from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
from types import MappingProxyType
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray


PrototypeIdentity: TypeAlias = tuple[int, int]


def _readonly(value: ArrayLike) -> NDArray[np.float64]:
    array = np.array(value, dtype=np.float64, copy=True)
    array.setflags(write=False)
    return array


def _unit_vector(value: ArrayLike, *, name: str) -> NDArray[np.float64]:
    vector = np.asarray(value, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a finite non-empty vector")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError(f"{name} must have non-zero norm")
    return _readonly(vector / norm)


@dataclass(frozen=True)
class OnlinePrototypeMember:
    """A complete, observed training patch retained by online memory."""

    variable_id: int
    start: int
    raw_patch: NDArray[np.float64]
    clean_key: NDArray[np.float64]

    def __post_init__(self) -> None:
        if (
            type(self.variable_id) is not int
            or type(self.start) is not int
            or self.variable_id < 0
            or self.start < 0
        ):
            raise ValueError("online prototype identity must be non-negative integers")
        raw_patch = np.asarray(self.raw_patch, dtype=np.float64)
        if raw_patch.ndim != 1 or raw_patch.size == 0:
            raise ValueError("online prototype raw_patch must be a non-empty vector")
        if not np.isfinite(raw_patch).all():
            raise ValueError("online prototype raw_patch must be finite")
        object.__setattr__(self, "raw_patch", _readonly(raw_patch))
        object.__setattr__(
            self,
            "clean_key",
            _unit_vector(self.clean_key, name="online prototype clean_key"),
        )

    @property
    def identity(self) -> PrototypeIdentity:
        return (self.variable_id, self.start)


@dataclass(frozen=True)
class OnlinePrototypeGroup:
    """A FIFO-bounded group whose center uses only current members."""

    variable_id: int
    group_id: int
    members: tuple[OnlinePrototypeMember, ...]

    def __post_init__(self) -> None:
        if (
            type(self.variable_id) is not int
            or type(self.group_id) is not int
            or self.variable_id < 0
            or self.group_id < 0
        ):
            raise ValueError("online prototype group identifiers must be non-negative integers")
        members = tuple(self.members)
        if not members:
            raise ValueError("online prototype groups must contain at least one member")
        if any(member.variable_id != self.variable_id for member in members):
            raise ValueError("online prototype group member variable does not match group")
        starts = [member.start for member in members]
        if len(starts) != len(set(starts)):
            raise ValueError("online prototype group members must have unique starts")
        object.__setattr__(self, "members", members)

    @property
    def center(self) -> NDArray[np.float64]:
        keys = np.stack([member.clean_key for member in self.members], axis=0)
        return _unit_vector(keys.mean(axis=0), name="online prototype group center")


@dataclass(frozen=True)
class OnlinePrototypeSnapshot:
    """Immutable state published for one online-memory epoch."""

    epoch: int
    groups: tuple[tuple[OnlinePrototypeGroup, ...], ...]
    seen: frozenset[PrototypeIdentity]
    diagnostics: dict[str, int | float]
    format_version: int = 1

    def __post_init__(self) -> None:
        if type(self.epoch) is not int or self.epoch < 0:
            raise ValueError("online prototype snapshot epoch must be non-negative")
        if self.format_version != 1:
            raise ValueError("unsupported online prototype snapshot version")
        groups = tuple(tuple(variable_groups) for variable_groups in self.groups)
        for variable_id, variable_groups in enumerate(groups):
            group_ids = [group.group_id for group in variable_groups]
            if len(group_ids) != len(set(group_ids)):
                raise ValueError("online prototype group ids must be unique per variable")
            if any(group.variable_id != variable_id for group in variable_groups):
                raise ValueError("online prototype snapshot group variable is inconsistent")
        normalized_seen = frozenset(
            (int(variable_id), int(start)) for variable_id, start in self.seen
        )
        diagnostics = {
            str(key): value
            for key, value in dict(self.diagnostics).items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        if len(diagnostics) != len(dict(self.diagnostics)):
            raise ValueError("online prototype diagnostics must contain numeric values")
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "seen", normalized_seen)
        object.__setattr__(self, "diagnostics", MappingProxyType(diagnostics))

    @property
    def member_count(self) -> int:
        return sum(
            len(group.members)
            for variable_groups in self.groups
            for group in variable_groups
        )


class OnlinePrototypeMemory:
    """Epoch-snapshot grouped memory that starts empty and accepts observed patches."""

    def __init__(
        self,
        *,
        patch_len: int,
        patch_stride: int,
        train_end: int,
        variable_count: int,
        group_similarity_threshold: float = 0.8,
        group_capacity: int = 16,
        max_groups_per_variable: int = 128,
        observed_only: bool = True,
    ) -> None:
        if (
            type(patch_len) is not int
            or type(patch_stride) is not int
            or type(train_end) is not int
            or type(variable_count) is not int
            or patch_len <= 0
            or patch_stride <= 0
            or train_end <= 0
            or variable_count <= 0
        ):
            raise ValueError("online prototype memory geometry must be positive integers")
        if not 0.0 < group_similarity_threshold <= 1.0:
            raise ValueError("online prototype group similarity threshold must lie in (0, 1]")
        if group_capacity <= 0 or max_groups_per_variable <= 0:
            raise ValueError("online prototype memory capacities must be positive")
        if type(observed_only) is not bool:
            raise ValueError("online prototype observed_only must be boolean")

        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.train_end = train_end
        self.variable_count = variable_count
        self.group_similarity_threshold = float(group_similarity_threshold)
        self.group_capacity = int(group_capacity)
        self.max_groups_per_variable = int(max_groups_per_variable)
        self.observed_only = observed_only
        self._pending: dict[PrototypeIdentity, OnlinePrototypeMember] = {}
        self._seen: set[PrototypeIdentity] = set()
        self._next_group_id = 0
        self._retrieval_dim: int | None = None
        self._diagnostic_counts: dict[str, int | float] = {
            "created_groups": 0,
            "joined_groups": 0,
            "evicted_members": 0,
            "evicted_groups": 0,
            "staged_observed": 0,
            "skipped_masked": 0,
            "causal_exclusions": 0,
        }
        self._snapshot = OnlinePrototypeSnapshot(
            epoch=0,
            groups=tuple(() for _ in range(variable_count)),
            seen=frozenset(),
            diagnostics=dict(self._diagnostic_counts),
        )

    def _validate_identity(self, variable_id: int, start: int) -> PrototypeIdentity:
        if (
            type(variable_id) is not int
            or type(start) is not int
            or variable_id < 0
            or variable_id >= self.variable_count
            or start < 0
        ):
            raise ValueError("online prototype identity is outside the configured range")
        if start + self.patch_len > self.train_end:
            raise ValueError("online prototype patch exceeds the training boundary")
        return variable_id, start

    def stage_patch(
        self,
        variable_id: int,
        absolute_start: int,
        raw_patch: ArrayLike,
        clean_key: ArrayLike,
        *,
        observed: bool,
    ) -> bool:
        if type(observed) is not bool:
            raise ValueError("online prototype observed flag must be boolean")
        identity = self._validate_identity(variable_id, absolute_start)
        if self.observed_only and not observed:
            self._diagnostic_counts["skipped_masked"] += 1
            return False
        if identity in self._pending or identity in self._seen:
            return False
        member = OnlinePrototypeMember(
            variable_id=identity[0],
            start=identity[1],
            raw_patch=np.asarray(raw_patch, dtype=np.float64),
            clean_key=np.asarray(clean_key, dtype=np.float64),
        )
        if member.raw_patch.size != self.patch_len:
            raise ValueError("online prototype raw_patch length does not match patch_len")
        if self._retrieval_dim is None:
            self._retrieval_dim = int(member.clean_key.size)
        elif member.clean_key.size != self._retrieval_dim:
            raise ValueError("online prototype clean_key dimension is inconsistent")
        self._pending[identity] = member
        self._diagnostic_counts["staged_observed"] += 1
        return True

    def rekey(self, clean_key_fn: Callable[[OnlinePrototypeMember], ArrayLike]) -> None:
        """Replace all retained and pending keys with a final-encoder snapshot.

        The callback is evaluated against the old immutable member objects.  All
        candidate members are validated before the published snapshot or pending
        buffer is replaced, so a failed re-key leaves memory unchanged.
        """
        if not callable(clean_key_fn):
            raise TypeError("online prototype clean_key_fn must be callable")

        members = [
            member
            for variable_groups in self._snapshot.groups
            for group in variable_groups
            for member in group.members
        ]
        members.extend(self._pending[identity] for identity in sorted(self._pending))
        if not members:
            return

        expected_dim = self._retrieval_dim
        replacements: dict[PrototypeIdentity, OnlinePrototypeMember] = {}
        for member in members:
            replacement = OnlinePrototypeMember(
                variable_id=member.variable_id,
                start=member.start,
                raw_patch=member.raw_patch,
                clean_key=clean_key_fn(member),
            )
            if expected_dim is None:
                expected_dim = int(replacement.clean_key.size)
            elif replacement.clean_key.size != expected_dim:
                raise ValueError("online prototype re-key dimension is inconsistent")
            replacements[member.identity] = replacement

        groups = tuple(
            tuple(
                OnlinePrototypeGroup(
                    variable_id=group.variable_id,
                    group_id=group.group_id,
                    members=tuple(replacements[member.identity] for member in group.members),
                )
                for group in variable_groups
            )
            for variable_groups in self._snapshot.groups
        )
        pending = {
            identity: replacements[identity]
            for identity in sorted(self._pending)
        }
        snapshot = OnlinePrototypeSnapshot(
            epoch=self._snapshot.epoch,
            groups=groups,
            seen=self._snapshot.seen,
            diagnostics=dict(self._snapshot.diagnostics),
        )
        self._pending = pending
        self._snapshot = snapshot
        self._retrieval_dim = expected_dim

    def rekey_batch(
        self,
        clean_key_fn: Callable[
            [tuple[OnlinePrototypeMember, ...]], ArrayLike
        ],
    ) -> None:
        """Replace all keys using one batched callback invocation."""

        if not callable(clean_key_fn):
            raise TypeError("online prototype batch clean_key_fn must be callable")
        members = tuple(
            member
            for variable_groups in self._snapshot.groups
            for group in variable_groups
            for member in group.members
        ) + tuple(self._pending[identity] for identity in sorted(self._pending))
        if not members:
            return
        clean_keys = np.asarray(clean_key_fn(members), dtype=np.float64)
        if clean_keys.ndim != 2 or clean_keys.shape[0] != len(members):
            raise ValueError(
                "online prototype batched re-key result must have shape [members, dim]"
            )
        replacements = {
            member.identity: clean_keys[index]
            for index, member in enumerate(members)
        }
        self.rekey(lambda member: replacements[member.identity])

    def publish_epoch(self, epoch: int) -> OnlinePrototypeSnapshot:
        if type(epoch) is not int or epoch != self._snapshot.epoch + 1:
            raise ValueError("online prototype epochs must be published consecutively")
        groups_by_variable = [list(groups) for groups in self._snapshot.groups]
        retrieval_dim = self._retrieval_dim
        if retrieval_dim is None and self._pending:
            retrieval_dim = next(iter(self._pending.values())).clean_key.size
        center_tables = [
            np.empty(
                (self.max_groups_per_variable, retrieval_dim or 0),
                dtype=np.float64,
            )
            for _ in range(self.variable_count)
        ]
        group_counts = [len(groups) for groups in groups_by_variable]
        for variable_id, variable_groups in enumerate(groups_by_variable):
            if variable_groups:
                center_tables[variable_id][: len(variable_groups)] = np.stack(
                    [group.center for group in variable_groups]
                )
        for identity in sorted(self._pending):
            member = self._pending[identity]
            variable_groups = groups_by_variable[member.variable_id]
            centers = center_tables[member.variable_id]
            group_count = group_counts[member.variable_id]
            best_index = -1
            best_similarity = -1.0
            if group_count:
                similarities = centers[:group_count] @ member.clean_key
                candidate_index = int(np.argmax(similarities))
                candidate_similarity = float(similarities[candidate_index])
                if candidate_similarity > best_similarity:
                    best_index = candidate_index
                    best_similarity = candidate_similarity
            if best_index >= 0 and best_similarity >= self.group_similarity_threshold:
                group = variable_groups[best_index]
                members = list(group.members)
                if len(members) >= self.group_capacity:
                    members.pop(0)
                    self._diagnostic_counts["evicted_members"] += 1
                members.append(member)
                variable_groups[best_index] = OnlinePrototypeGroup(
                    variable_id=group.variable_id,
                    group_id=group.group_id,
                    members=tuple(members),
                )
                centers[best_index] = variable_groups[best_index].center
                self._diagnostic_counts["joined_groups"] += 1
            else:
                if group_count >= self.max_groups_per_variable:
                    variable_groups.pop(0)
                    centers[: group_count - 1] = centers[1:group_count]
                    group_count -= 1
                    self._diagnostic_counts["evicted_groups"] += 1
                variable_groups.append(
                    OnlinePrototypeGroup(
                        variable_id=member.variable_id,
                        group_id=self._next_group_id,
                        members=(member,),
                    )
                )
                centers[group_count] = member.clean_key
                group_counts[member.variable_id] = group_count + 1
                self._next_group_id += 1
                self._diagnostic_counts["created_groups"] += 1
            self._seen.add(identity)

        self._pending.clear()
        self._snapshot = OnlinePrototypeSnapshot(
            epoch=epoch,
            groups=tuple(tuple(groups) for groups in groups_by_variable),
            seen=frozenset(self._seen),
            diagnostics=dict(self._diagnostic_counts),
        )
        return self._snapshot

    def snapshot(self) -> OnlinePrototypeSnapshot:
        return self._snapshot

    def query(
        self,
        variable_id: int,
        query_start: int,
        query_key: ArrayLike,
        *,
        max_items: int = 4,
    ) -> tuple[OnlinePrototypeMember, ...]:
        if type(max_items) is not int or max_items <= 0:
            raise ValueError("online prototype query max_items must be positive")
        if type(query_start) is not int or query_start < 0:
            raise ValueError("online prototype query start must be non-negative")
        if type(variable_id) is not int or not 0 <= variable_id < self.variable_count:
            raise ValueError("online prototype query variable is outside the configured range")
        normalized_query = _unit_vector(query_key, name="online prototype query_key")
        if self._retrieval_dim is not None and normalized_query.size != self._retrieval_dim:
            raise ValueError("online prototype query_key dimension is inconsistent")

        selected_members: tuple[OnlinePrototypeMember, ...] = ()
        selected_score = -np.inf
        for group in self._snapshot.groups[variable_id]:
            eligible = tuple(
                member
                for member in group.members
                if member.start + self.patch_len <= query_start
            )
            if not eligible:
                continue
            score = float(np.dot(normalized_query, group.center))
            if score > selected_score:
                selected_score = score
                selected_members = eligible
        ranked = sorted(
            selected_members,
            key=lambda member: (
                -float(np.dot(normalized_query, member.clean_key)),
                member.start,
            ),
        )
        return tuple(ranked[:max_items])

    def query_batch(
        self,
        variable_ids: ArrayLike,
        query_starts: ArrayLike,
        query_keys: ArrayLike,
        *,
        max_items: int = 4,
    ) -> tuple[tuple[OnlinePrototypeMember, ...], ...]:
        """Query many patches with vectorized group scoring per variable."""

        if type(max_items) is not int or max_items <= 0:
            raise ValueError("online prototype query max_items must be positive")
        variables = np.asarray(variable_ids)
        starts = np.asarray(query_starts)
        keys = np.asarray(query_keys, dtype=np.float64)
        if variables.ndim != 1 or starts.shape != variables.shape:
            raise ValueError("online prototype batch query metadata must be one-dimensional")
        if keys.ndim != 2 or keys.shape[0] != variables.size:
            raise ValueError("online prototype batch query keys must have shape [queries, dim]")
        if not np.issubdtype(variables.dtype, np.integer):
            raise ValueError("online prototype batch query variables must be integers")
        if not np.issubdtype(starts.dtype, np.integer):
            raise ValueError("online prototype batch query starts must be integers")
        variables = variables.astype(np.int64, copy=False)
        starts = starts.astype(np.int64, copy=False)
        if np.any((variables < 0) | (variables >= self.variable_count)):
            raise ValueError("online prototype batch query variable is outside the configured range")
        if np.any(starts < 0):
            raise ValueError("online prototype batch query start must be non-negative")
        if not np.isfinite(keys).all():
            raise ValueError("online prototype batch query keys must be finite")
        if self._retrieval_dim is not None and keys.shape[1] != self._retrieval_dim:
            raise ValueError("online prototype query_key dimension is inconsistent")
        norms = np.linalg.norm(keys, axis=1)
        if np.any(norms <= 1e-12):
            raise ValueError("online prototype batch query keys must have non-zero norm")
        normalized = keys / norms[:, None]

        results: list[tuple[OnlinePrototypeMember, ...]] = [
            () for _ in range(variables.size)
        ]
        for variable_id in range(self.variable_count):
            query_indices = np.flatnonzero(variables == variable_id)
            groups = self._snapshot.groups[variable_id]
            if query_indices.size == 0 or not groups:
                continue
            centers = np.stack([group.center for group in groups])
            first_causal_end = np.asarray(
                [
                    min(member.start + self.patch_len for member in group.members)
                    for group in groups
                ],
                dtype=np.int64,
            )
            scores = normalized[query_indices] @ centers.T
            causal = first_causal_end[None, :] <= starts[query_indices, None]
            scores = np.where(causal, scores, -np.inf)
            best_groups = scores.argmax(axis=1)
            has_causal_group = causal.any(axis=1)
            for local_index, query_index in enumerate(query_indices):
                if not has_causal_group[local_index]:
                    continue
                group = groups[int(best_groups[local_index])]
                eligible = (
                    member
                    for member in group.members
                    if member.start + self.patch_len <= starts[query_index]
                )
                ranked = sorted(
                    eligible,
                    key=lambda member: (
                        -float(np.dot(normalized[query_index], member.clean_key)),
                        member.start,
                    ),
                )
                results[int(query_index)] = tuple(ranked[:max_items])
        return tuple(results)

    def diagnostics(self) -> Mapping[str, int | float]:
        return dict(self._snapshot.diagnostics)

    def read_only_view(self) -> OnlinePrototypeSnapshot:
        return self._snapshot

    @staticmethod
    def _member_to_state(member: OnlinePrototypeMember) -> dict[str, object]:
        return {
            "variable_id": member.variable_id,
            "start": member.start,
            "raw_patch": np.array(member.raw_patch, dtype=np.float64, copy=True),
            "clean_key": np.array(member.clean_key, dtype=np.float64, copy=True),
        }

    @classmethod
    def _member_from_state(
        cls,
        state: Mapping[str, object],
        *,
        patch_len: int,
        variable_count: int,
        train_end: int,
    ) -> OnlinePrototypeMember:
        try:
            variable_id = int(state["variable_id"])
            start = int(state["start"])
            raw_patch = np.asarray(state["raw_patch"], dtype=np.float64)
            clean_key = np.asarray(state["clean_key"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid online prototype member state") from exc
        if (
            variable_id < 0
            or variable_id >= variable_count
            or start < 0
            or start + patch_len > train_end
        ):
            raise ValueError("online prototype member state is outside configured bounds")
        member = OnlinePrototypeMember(variable_id, start, raw_patch, clean_key)
        if member.raw_patch.size != patch_len:
            raise ValueError("online prototype member state has invalid patch length")
        return member

    @classmethod
    def _group_to_state(cls, group: OnlinePrototypeGroup) -> dict[str, object]:
        return {
            "variable_id": group.variable_id,
            "group_id": group.group_id,
            "members": [cls._member_to_state(member) for member in group.members],
        }

    @classmethod
    def _group_from_state(
        cls,
        state: Mapping[str, object],
        *,
        patch_len: int,
        variable_count: int,
        train_end: int,
        group_capacity: int,
    ) -> OnlinePrototypeGroup:
        try:
            variable_id = int(state["variable_id"])
            group_id = int(state["group_id"])
            members_state = state["members"]
            if not isinstance(members_state, (list, tuple)):
                raise TypeError("members must be a sequence")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid online prototype group state") from exc
        if variable_id < 0 or variable_id >= variable_count:
            raise ValueError("online prototype group state has invalid variable")
        if len(members_state) == 0 or len(members_state) > group_capacity:
            raise ValueError("online prototype group state has invalid capacity")
        members = tuple(
            cls._member_from_state(
                member_state,
                patch_len=patch_len,
                variable_count=variable_count,
                train_end=train_end,
            )
            for member_state in members_state
            if isinstance(member_state, Mapping)
        )
        if len(members) != len(members_state):
            raise ValueError("online prototype group member state must be mappings")
        return OnlinePrototypeGroup(variable_id, group_id, members)

    def state_dict(self) -> dict[str, object]:
        snapshot = self._snapshot
        return {
            "format_version": 1,
            "config": {
                "patch_len": self.patch_len,
                "patch_stride": self.patch_stride,
                "train_end": self.train_end,
                "variable_count": self.variable_count,
                "group_similarity_threshold": self.group_similarity_threshold,
                "group_capacity": self.group_capacity,
                "max_groups_per_variable": self.max_groups_per_variable,
                "observed_only": self.observed_only,
            },
            "snapshot": {
                "epoch": snapshot.epoch,
                "groups": [
                    [self._group_to_state(group) for group in variable_groups]
                    for variable_groups in snapshot.groups
                ],
                "seen": [list(identity) for identity in sorted(snapshot.seen)],
                "diagnostics": dict(snapshot.diagnostics),
            },
            "pending": [
                self._member_to_state(self._pending[identity])
                for identity in sorted(self._pending)
            ],
            "next_group_id": self._next_group_id,
            "retrieval_dim": self._retrieval_dim,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if not isinstance(state, Mapping) or state.get("format_version") != 1:
            raise ValueError("unsupported online prototype memory state version")
        config = state.get("config")
        snapshot_state = state.get("snapshot")
        pending_state = state.get("pending")
        if not isinstance(config, Mapping) or not isinstance(snapshot_state, Mapping):
            raise ValueError("online prototype memory state is incomplete")
        if not isinstance(pending_state, (list, tuple)):
            raise ValueError("online prototype pending state must be a sequence")
        expected_config = {
            "patch_len": self.patch_len,
            "patch_stride": self.patch_stride,
            "train_end": self.train_end,
            "variable_count": self.variable_count,
            "group_similarity_threshold": self.group_similarity_threshold,
            "group_capacity": self.group_capacity,
            "max_groups_per_variable": self.max_groups_per_variable,
            "observed_only": self.observed_only,
        }
        if dict(config) != expected_config:
            raise ValueError("online prototype memory state configuration mismatch")
        try:
            epoch = int(snapshot_state["epoch"])
            groups_state = snapshot_state["groups"]
            seen_state = snapshot_state["seen"]
            diagnostics_state = snapshot_state["diagnostics"]
            next_group_id = int(state["next_group_id"])
            retrieval_dim = state.get("retrieval_dim")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("online prototype memory state is incomplete") from exc
        if not isinstance(groups_state, (list, tuple)) or len(groups_state) != self.variable_count:
            raise ValueError("online prototype snapshot groups have invalid shape")
        if not isinstance(seen_state, (list, tuple)) or not isinstance(diagnostics_state, Mapping):
            raise ValueError("online prototype snapshot metadata is invalid")
        if retrieval_dim is not None:
            retrieval_dim = int(retrieval_dim)
            if retrieval_dim <= 0:
                raise ValueError("online prototype retrieval dimension must be positive")
        seen: set[PrototypeIdentity] = set()
        for identity in seen_state:
            if not isinstance(identity, (list, tuple)) or len(identity) != 2:
                raise ValueError("online prototype seen identity is invalid")
            normalized_identity = (int(identity[0]), int(identity[1]))
            self._validate_identity(*normalized_identity)
            seen.add(normalized_identity)
        groups: list[tuple[OnlinePrototypeGroup, ...]] = []
        all_group_ids: set[int] = set()
        all_member_ids: set[PrototypeIdentity] = set()
        for variable_groups_state in groups_state:
            if not isinstance(variable_groups_state, (list, tuple)):
                raise ValueError("online prototype variable groups must be sequences")
            variable_groups = tuple(
                self._group_from_state(
                    group_state,
                    patch_len=self.patch_len,
                    variable_count=self.variable_count,
                    train_end=self.train_end,
                    group_capacity=self.group_capacity,
                )
                for group_state in variable_groups_state
                if isinstance(group_state, Mapping)
            )
            if len(variable_groups) != len(variable_groups_state):
                raise ValueError("online prototype group state must be mappings")
            for group in variable_groups:
                if group.group_id in all_group_ids:
                    raise ValueError("online prototype group ids must be globally unique")
                all_group_ids.add(group.group_id)
                for member in group.members:
                    if member.identity in all_member_ids:
                        raise ValueError("online prototype member identity is duplicated")
                    all_member_ids.add(member.identity)
            groups.append(variable_groups)
        if not all_member_ids.issubset(seen):
            raise ValueError("online prototype snapshot members must be marked seen")
        pending: dict[PrototypeIdentity, OnlinePrototypeMember] = {}
        for member_state in pending_state:
            if not isinstance(member_state, Mapping):
                raise ValueError("online prototype pending member state must be a mapping")
            member = self._member_from_state(
                member_state,
                patch_len=self.patch_len,
                variable_count=self.variable_count,
                train_end=self.train_end,
            )
            if member.identity in seen or member.identity in pending:
                raise ValueError("online prototype pending identity is duplicated")
            pending[member.identity] = member
        member_dims = {
            member.clean_key.size
            for variable_groups in groups
            for group in variable_groups
            for member in group.members
        }
        member_dims.update(member.clean_key.size for member in pending.values())
        if retrieval_dim is None and member_dims:
            retrieval_dim = member_dims.pop()
        if member_dims and (len(member_dims) != 1 or retrieval_dim not in member_dims):
            raise ValueError("online prototype retrieval dimensions are inconsistent")
        if retrieval_dim is not None and member_dims and retrieval_dim not in member_dims:
            raise ValueError("online prototype retrieval dimension does not match members")
        max_seen_group_id = max(all_group_ids, default=-1)
        if next_group_id <= max_seen_group_id:
            raise ValueError("online prototype next group id is not ahead of existing groups")
        snapshot = OnlinePrototypeSnapshot(
            epoch=epoch,
            groups=tuple(groups),
            seen=frozenset(seen),
            diagnostics=dict(diagnostics_state),
        )
        self._pending = pending
        self._seen = seen
        self._next_group_id = next_group_id
        self._retrieval_dim = retrieval_dim
        self._diagnostic_counts = dict(snapshot.diagnostics)
        self._snapshot = snapshot

    def snapshot_hash(self) -> str:
        digest = sha256()
        snapshot = self._snapshot
        header = {
            "format_version": snapshot.format_version,
            "epoch": snapshot.epoch,
            "seen": [list(identity) for identity in sorted(snapshot.seen)],
            "groups": [
                {
                    "variable_id": group.variable_id,
                    "group_id": group.group_id,
                    "member_count": len(group.members),
                }
                for variable_groups in snapshot.groups
                for group in variable_groups
            ],
        }
        digest.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode())
        for variable_groups in snapshot.groups:
            for group in variable_groups:
                center = np.ascontiguousarray(group.center, dtype=np.float64)
                digest.update(center.tobytes())
                for member in group.members:
                    metadata = json.dumps(
                        {
                            "variable_id": member.variable_id,
                            "start": member.start,
                            "raw_shape": member.raw_patch.shape,
                            "key_shape": member.clean_key.shape,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                    digest.update(metadata)
                    digest.update(
                        np.ascontiguousarray(member.raw_patch, dtype=np.float64).tobytes()
                    )
                    digest.update(
                        np.ascontiguousarray(member.clean_key, dtype=np.float64).tobytes()
                    )
        return digest.hexdigest()
