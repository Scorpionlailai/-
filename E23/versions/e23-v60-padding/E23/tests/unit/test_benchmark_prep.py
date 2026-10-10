from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from malr.data import benchmark_prep
from malr.data.benchmark_prep import (
    prepare_appliances,
    prepare_beijing_air_quality,
    prepare_exchange,
)
from malr.data.datasets import load_clean_csv


def test_prepare_exchange_converts_the_canonical_headerless_file(
    tmp_path: Path,
) -> None:
    source = tmp_path / "exchange_rate.txt"
    output = tmp_path / "exchange_rate.csv"
    values = np.arange(7588 * 8, dtype=np.float64).reshape(7588, 8)
    pd.DataFrame(values).to_csv(source, index=False, header=False)

    result = prepare_exchange(source, output)
    series = load_clean_csv(output, expected_n_vars=8, expected_length=7588)

    assert result["shape"] == [7588, 8]
    assert series.columns[0] == "currency_0"
    np.testing.assert_array_equal(series.values, values)


def test_prepare_appliances_removes_only_the_two_random_controls(
    tmp_path: Path,
) -> None:
    source = tmp_path / "energydata_complete.csv"
    output = tmp_path / "appliances_energy.csv"
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2016-01-01", periods=19735, freq="10min"),
            **{
                f"signal_{index}": np.arange(19735, dtype=np.float64) + index
                for index in range(26)
            },
            "rv1": np.zeros(19735),
            "rv2": np.ones(19735),
        }
    )
    frame.to_csv(source, index=False)

    result = prepare_appliances(source, output)
    series = load_clean_csv(output, expected_n_vars=26, expected_length=19735)

    assert result["dropped_columns"] == ["rv1", "rv2"]
    assert "rv1" not in series.columns and "rv2" not in series.columns
    assert series.columns == tuple(f"signal_{index}" for index in range(26))


def test_prepare_beijing_preserves_native_missing_mask_and_causally_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    timestamps = pd.date_range("2013-03-01", periods=35064, freq="h")
    frames = []
    for station_index in range(12):
        values = np.arange(35064, dtype=np.float64) + station_index
        values[0 if station_index == 0 else 100] = np.nan
        frames.append(
            pd.DataFrame(
                {
                    "year": timestamps.year,
                    "month": timestamps.month,
                    "day": timestamps.day,
                    "hour": timestamps.hour,
                    "PM2.5": values,
                    "station": f"station_{station_index:02d}",
                }
            )
        )
    monkeypatch.setattr(benchmark_prep, "_beijing_frames", lambda source: iter(frames))
    output = tmp_path / "beijing.csv"

    result = prepare_beijing_air_quality(tmp_path / "unused.zip", output)
    series = load_clean_csv(output, expected_n_vars=12, expected_length=35064)

    assert result["native_missing_rate"] == pytest.approx(12 / (35064 * 12))
    assert np.isfinite(series.values).all()
    sidecar = np.load(tmp_path / "beijing.natural_missing.npz")
    assert sidecar["observed_mask"].shape == (35064, 12)
    assert int((~sidecar["observed_mask"]).sum()) == 12
    assert str(sidecar["completion_policy"].item()) == "causal_ffill_then_train_median"
