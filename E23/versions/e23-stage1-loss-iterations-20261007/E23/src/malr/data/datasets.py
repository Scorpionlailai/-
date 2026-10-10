from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Sequence

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from .windows import WindowRef


@dataclass(frozen=True)
class CleanSeries:
    values: np.ndarray
    columns: tuple[str, ...]
    timestamps: np.ndarray
    source_path: Path

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError("clean series values must be [time, variables]")
        if self.values.shape[1] != len(self.columns):
            raise ValueError("column names do not match value width")
        if self.values.shape[0] != self.timestamps.shape[0]:
            raise ValueError("timestamps do not match value length")


class ForecastSample(NamedTuple):
    clean_input: np.ndarray
    clean_target: np.ndarray
    ref: WindowRef


def load_clean_csv(
    path: Path,
    expected_n_vars: int,
    expected_length: int | None = None,
    *,
    allow_length_override: bool = False,
) -> CleanSeries:
    frame = pd.read_csv(path)
    if "date" not in frame.columns:
        raise ValueError("CSV must contain a date column")
    numeric_columns = tuple(column for column in frame.columns if column != "date")
    if len(numeric_columns) != expected_n_vars:
        raise ValueError(
            f"expected {expected_n_vars} variables, found {len(numeric_columns)}"
        )
    if (
        expected_length is not None
        and len(frame) != expected_length
        and not allow_length_override
    ):
        raise ValueError(f"expected {expected_length} rows, found {len(frame)}")
    try:
        # Keep source precision while fitting the train-only scaler, matching
        # sklearn's float64 statistics in the official PatchTST loader. Batches
        # are converted to float32 only after standardization.
        values = frame.loc[:, numeric_columns].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("all non-date columns must be numeric") from exc
    if not np.isfinite(values).all():
        raise ValueError("clean source CSV contains NaN or infinite values")
    timestamps = pd.to_datetime(frame["date"], errors="raise").to_numpy()
    return CleanSeries(
        values=np.ascontiguousarray(values),
        columns=numeric_columns,
        timestamps=timestamps,
        source_path=path.resolve(),
    )


class ForecastDataset(Dataset[ForecastSample]):
    def __init__(self, series: CleanSeries, refs: Sequence[WindowRef]) -> None:
        self.series = series
        self.refs = tuple(refs)
        for ref in self.refs:
            if ref.target_end > len(series.values):
                raise ValueError("window extends beyond the clean series")

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int) -> ForecastSample:
        ref = self.refs[index]
        return ForecastSample(
            clean_input=self.series.values[ref.input_start : ref.input_end].copy(),
            clean_target=self.series.values[
                ref.target_start : ref.target_end
            ].copy(),
            ref=ref,
        )
