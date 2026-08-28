"""Leakage-safe data protocols for stage-1 experiments."""

from .datasets import CleanSeries, ForecastDataset, ForecastSample, load_clean_csv
from .masks import (
    EpochMaskManager,
    MaskExpectation,
    MaskMetadata,
    generate_variable_missing_mask,
    load_fixed_mask,
    save_fixed_mask,
)
from .splits import DatasetProtocol, SplitBounds, protocol_for
from .windows import WindowRef, build_window_index

__all__ = [
    "CleanSeries",
    "DatasetProtocol",
    "ForecastDataset",
    "ForecastSample",
    "EpochMaskManager",
    "MaskExpectation",
    "MaskMetadata",
    "SplitBounds",
    "WindowRef",
    "build_window_index",
    "generate_variable_missing_mask",
    "load_clean_csv",
    "load_fixed_mask",
    "protocol_for",
    "save_fixed_mask",
]
