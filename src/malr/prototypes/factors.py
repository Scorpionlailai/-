from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


FACTOR_COUNT = 7


def _readonly(array: ArrayLike, dtype: np.dtype) -> NDArray:
    result = np.array(array, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class FactorVector:
    values: NDArray[np.float64]
    available: NDArray[np.bool_]

    def __post_init__(self) -> None:
        values = _readonly(self.values, np.dtype(np.float64))
        available = _readonly(self.available, np.dtype(np.bool_))
        if values.shape != (FACTOR_COUNT,) or available.shape != (FACTOR_COUNT,):
            raise ValueError("factor values and availability must have shape [7]")
        if not np.isfinite(values).all():
            raise ValueError("factor values must be finite")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "available", available)


def _mean(values: NDArray[np.float64], valid: NDArray[np.bool_]) -> tuple[float, bool]:
    if not valid.any():
        return 0.0, False
    return float(values[valid].mean()), True


def _masked_slope(
    values: NDArray[np.float64], valid: NDArray[np.bool_], offset: int
) -> tuple[float, bool]:
    indices = np.flatnonzero(valid).astype(np.float64) + float(offset)
    if indices.size < 2:
        return 0.0, False
    selected = values[valid]
    centered_indices = indices - indices.mean()
    denominator = float(centered_indices @ centered_indices)
    if denominator <= 0.0:
        return 0.0, False
    slope = float(centered_indices @ (selected - selected.mean()) / denominator)
    return slope, True


def _difference_volatility(
    values: NDArray[np.float64], valid: NDArray[np.bool_]
) -> tuple[float, bool]:
    adjacent = valid[:-1] & valid[1:]
    if not adjacent.any():
        return 0.0, False
    differences = np.diff(values)[adjacent]
    return float(differences.std(ddof=0)), True


def compute_context_factors(
    segment: ArrayLike,
    valid: ArrayLike,
    center_slice: slice = slice(24, 40),
    train_std: float = 1.0,
) -> FactorVector:
    """Compute [level, trend, volatility] per side and a scaled boundary gap.

    Values in ``center_slice`` are never read. Unavailable dimensions contain a
    finite zero placeholder and are identified by ``available``.
    """

    values = np.asarray(segment, dtype=np.float64)
    availability = np.asarray(valid, dtype=np.bool_)
    if values.ndim != 1 or availability.shape != values.shape:
        raise ValueError("segment and valid must be matching one-dimensional arrays")
    if not np.isfinite(values).all():
        raise ValueError("segment values must be finite")
    start, stop, step = center_slice.indices(values.size)
    if step != 1 or start <= 0 or stop >= values.size or start >= stop:
        raise ValueError("center_slice must be a non-empty interior contiguous slice")
    if not np.isfinite(train_std) or train_std <= 0:
        raise ValueError("train_std must be finite and positive")

    left_values = values[:start]
    left_valid = availability[:start]
    right_values = values[stop:]
    right_valid = availability[stop:]

    left_mean = _mean(left_values, left_valid)
    left_slope = _masked_slope(left_values, left_valid, offset=0)
    left_volatility = _difference_volatility(left_values, left_valid)
    right_mean = _mean(right_values, right_valid)
    right_slope = _masked_slope(right_values, right_valid, offset=stop)
    right_volatility = _difference_volatility(right_values, right_valid)

    if left_valid.any() and right_valid.any():
        left_boundary = left_values[np.flatnonzero(left_valid)[-1]]
        right_boundary = right_values[np.flatnonzero(right_valid)[0]]
        boundary_gap = (float(right_boundary - left_boundary) / float(train_std), True)
    else:
        boundary_gap = (0.0, False)

    factors = (
        left_mean,
        left_slope,
        left_volatility,
        right_mean,
        right_slope,
        right_volatility,
        boundary_gap,
    )
    return FactorVector(
        values=np.array([item[0] for item in factors], dtype=np.float64),
        available=np.array([item[1] for item in factors], dtype=np.bool_),
    )
