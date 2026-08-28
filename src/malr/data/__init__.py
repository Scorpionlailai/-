"""Leakage-safe data protocols for stage-1 experiments."""

from .datasets import CleanSeries, ForecastDataset, ForecastSample, load_clean_csv
from .splits import DatasetProtocol, SplitBounds, protocol_for
from .windows import WindowRef, build_window_index

__all__ = [
    "CleanSeries",
    "DatasetProtocol",
    "ForecastDataset",
    "ForecastSample",
    "SplitBounds",
    "WindowRef",
    "build_window_index",
    "load_clean_csv",
    "protocol_for",
]

