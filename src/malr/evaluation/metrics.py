from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class ForecastMetrics:
    mse: float
    mae: float


def compute_metrics(prediction: ArrayLike, target: ArrayLike) -> ForecastMetrics:
    predicted = np.asarray(prediction, dtype=np.float64)
    actual = np.asarray(target, dtype=np.float64)
    if predicted.shape != actual.shape or predicted.size == 0:
        raise ValueError("prediction and target must have the same non-empty shape")
    if not np.isfinite(predicted).all() or not np.isfinite(actual).all():
        raise ValueError("prediction and target must be finite")
    error = predicted - actual
    return ForecastMetrics(
        mse=float(np.mean(error**2)),
        mae=float(np.mean(np.abs(error))),
    )


def degradation_recovery_rate(
    *,
    clean: float,
    naive: float,
    ours: float,
) -> float | None:
    values = np.asarray([clean, naive, ours], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("DRR inputs must be finite")
    denominator = naive - clean
    if denominator <= 0.0:
        return None
    return float((naive - ours) / denominator * 100.0)
