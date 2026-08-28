from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from malr.data.datasets import ForecastDataset, load_clean_csv
from malr.data.splits import DatasetProtocol, SplitBounds, protocol_for
from malr.data.windows import WindowRef, build_window_index


def test_etth1_uses_custom_12_4_4_boundaries() -> None:
    protocol = protocol_for("ETTh1", 17420)

    assert protocol.train == SplitBounds(0, 8640)
    assert protocol.validation == SplitBounds(8640, 11520)
    assert protocol.test == SplitBounds(11520, 14400)


def test_weather_uses_standard_70_10_20_boundaries() -> None:
    protocol = protocol_for("Weather", 52696)

    assert protocol.train == SplitBounds(0, 36887)
    assert protocol.validation == SplitBounds(36887, 42157)
    assert protocol.test == SplitBounds(42157, 52696)


def test_validation_windows_are_owned_by_target_and_may_read_prior_history() -> None:
    protocol = protocol_for("ETTh1", 17420)
    refs = build_window_index(
        protocol,
        "validation",
        seq_len=96,
        pred_len=96,
    )

    first = refs[0]
    assert first.input_start == 8640 - 96
    assert first.input_end == 8640
    assert first.target_start == 8640
    assert first.target_end == 8640 + 96
    assert all(
        ref.target_start >= 8640 and ref.target_end <= 11520 for ref in refs
    )


def test_training_windows_keep_history_and_target_inside_train() -> None:
    protocol = DatasetProtocol(
        name="tiny",
        source_length=30,
        train=SplitBounds(0, 18),
        validation=SplitBounds(18, 24),
        test=SplitBounds(24, 30),
    )

    refs = build_window_index(protocol, "train", seq_len=4, pred_len=2)

    assert refs[0] == WindowRef(0, 4, 4, 6, "train")
    assert refs[-1] == WindowRef(12, 16, 16, 18, "train")
    assert len(refs) == 13


def test_clean_csv_loader_preserves_numeric_column_order(tmp_path: Path) -> None:
    path = tmp_path / "tiny.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=12, freq="h"),
            "b": np.arange(12, dtype=float) + 100,
            "a": np.arange(12, dtype=float),
        }
    ).to_csv(path, index=False)

    series = load_clean_csv(path, expected_n_vars=2, expected_length=12)

    assert series.columns == ("b", "a")
    np.testing.assert_array_equal(series.values[:, 0], np.arange(12) + 100)
    np.testing.assert_array_equal(series.values[:, 1], np.arange(12))


def test_forecast_dataset_returns_clean_absolute_slices(tmp_path: Path) -> None:
    path = tmp_path / "tiny.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=12, freq="h"),
            "x": np.arange(12, dtype=float),
        }
    ).to_csv(path, index=False)
    series = load_clean_csv(path, expected_n_vars=1, expected_length=12)
    dataset = ForecastDataset(
        series,
        [WindowRef(2, 6, 6, 9, "validation")],
    )

    sample = dataset[0]

    np.testing.assert_array_equal(sample.clean_input[:, 0], np.arange(2, 6))
    np.testing.assert_array_equal(sample.clean_target[:, 0], np.arange(6, 9))
    assert sample.ref.target_start == 6


def test_loader_rejects_nan_and_wrong_length(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=3, freq="h"),
            "x": [1.0, np.nan, 3.0],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="NaN"):
        load_clean_csv(path, expected_n_vars=1, expected_length=3)

    good = tmp_path / "good.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=3, freq="h"),
            "x": [1.0, 2.0, 3.0],
        }
    ).to_csv(good, index=False)
    with pytest.raises(ValueError, match="expected 4 rows"):
        load_clean_csv(good, expected_n_vars=1, expected_length=4)
