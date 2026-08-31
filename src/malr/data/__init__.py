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
from .normalization import (
    MaskedNormalization,
    TrainVariableStats,
    denormalize_forecast,
    fit_train_variable_stats,
    masked_window_normalize,
    standardize_values,
)
from .patching import (
    PATCH_MISSING,
    PATCH_OBSERVED,
    PATCH_PARTIAL,
    PatchBatch,
    patchify_with_state,
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
    "MaskedNormalization",
    "PATCH_MISSING",
    "PATCH_OBSERVED",
    "PATCH_PARTIAL",
    "PatchBatch",
    "SplitBounds",
    "TrainVariableStats",
    "WindowRef",
    "build_window_index",
    "denormalize_forecast",
    "fit_train_variable_stats",
    "generate_variable_missing_mask",
    "load_clean_csv",
    "load_fixed_mask",
    "masked_window_normalize",
    "patchify_with_state",
    "protocol_for",
    "save_fixed_mask",
    "standardize_values",
]
