from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class MatrixProfileResult:
    distances: NDArray[np.float64]
    neighbor_indices: NDArray[np.int64]
    pairwise_distances: NDArray[np.float64]


@dataclass(frozen=True)
class MotifGroup:
    occurrence_indices: tuple[int, ...]
    medoid_index: int
    mean_profile_distance: float


def _z_normalize(candidates: NDArray[np.float64]) -> NDArray[np.float64]:
    centered = candidates - candidates.mean(axis=1, keepdims=True)
    standard_deviation = candidates.std(axis=1, keepdims=True)
    return np.divide(
        centered,
        standard_deviation,
        out=np.zeros_like(centered),
        where=standard_deviation > 1e-12,
    )


def candidate_matrix_profile(
    candidates: ArrayLike,
    starts: ArrayLike,
    exclusion: int = 32,
) -> MatrixProfileResult:
    candidates_array = np.asarray(candidates, dtype=np.float64)
    starts_array = np.asarray(starts, dtype=np.int64)
    if candidates_array.ndim != 2:
        raise ValueError("candidates must have shape [candidate, time]")
    if starts_array.shape != (candidates_array.shape[0],):
        raise ValueError("starts must have one entry per candidate")
    if exclusion <= 0:
        raise ValueError("exclusion must be positive")
    if not np.isfinite(candidates_array).all():
        raise ValueError("candidates must be finite")

    count = candidates_array.shape[0]
    if count == 0:
        return MatrixProfileResult(
            distances=np.empty(0, dtype=np.float64),
            neighbor_indices=np.empty(0, dtype=np.int64),
            pairwise_distances=np.empty((0, 0), dtype=np.float64),
        )

    normalized = _z_normalize(candidates_array)
    squared_norm = np.einsum("ij,ij->i", normalized, normalized)
    squared = squared_norm[:, None] + squared_norm[None, :] - 2.0 * (
        normalized @ normalized.T
    )
    squared = np.maximum(squared, 0.0)
    pairwise = np.sqrt(squared)
    trivial = np.abs(starts_array[:, None] - starts_array[None, :]) < exclusion
    pairwise[trivial] = np.inf

    neighbor_indices = np.argmin(pairwise, axis=1).astype(np.int64)
    distances = pairwise[np.arange(count), neighbor_indices]
    no_neighbor = ~np.isfinite(distances)
    neighbor_indices[no_neighbor] = -1
    return MatrixProfileResult(distances, neighbor_indices, pairwise)


def discover_motif_groups(
    profile: MatrixProfileResult,
    percentile: float = 0.20,
    min_occurrences: int = 2,
) -> tuple[MotifGroup, ...]:
    if not 0.0 <= percentile <= 1.0:
        raise ValueError("percentile must be between zero and one")
    if min_occurrences < 2:
        raise ValueError("min_occurrences must be at least two")
    finite = np.isfinite(profile.distances)
    if not finite.any():
        return ()
    threshold = float(np.quantile(profile.distances[finite], percentile))
    seeds = np.flatnonzero(finite & (profile.distances <= threshold))

    adjacency: dict[int, set[int]] = {}
    for seed in seeds.tolist():
        neighbor = int(profile.neighbor_indices[seed])
        if neighbor < 0:
            continue
        adjacency.setdefault(seed, set()).add(neighbor)
        adjacency.setdefault(neighbor, set()).add(seed)

    groups: list[MotifGroup] = []
    visited: set[int] = set()
    for root in sorted(adjacency):
        if root in visited:
            continue
        stack = [root]
        component: list[int] = []
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            component.append(node)
            stack.extend(sorted(adjacency[node] - visited, reverse=True))
        occurrences = tuple(sorted(component))
        if len(occurrences) < min_occurrences:
            continue
        submatrix = profile.pairwise_distances[np.ix_(occurrences, occurrences)].copy()
        np.fill_diagonal(submatrix, 0.0)
        mean_distances = submatrix.sum(axis=1) / max(len(occurrences) - 1, 1)
        medoid_offset = int(np.argmin(mean_distances))
        finite_profiles = profile.distances[np.array(occurrences)]
        group_mean = float(finite_profiles[np.isfinite(finite_profiles)].mean())
        groups.append(
            MotifGroup(
                occurrence_indices=occurrences,
                medoid_index=occurrences[medoid_offset],
                mean_profile_distance=group_mean,
            )
        )
    return tuple(groups)
