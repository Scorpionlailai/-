"""Prepare external benchmark files for the strict MALR CSV loader.

ETT files already use the required ``date + numeric variables`` layout.  The
three remaining external sources need deterministic conversion before they are
used by the experiment configs.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile
from zipfile import ZipFile

import numpy as np
import pandas as pd


EXCHANGE_COLUMNS = tuple(f"currency_{index}" for index in range(8))
BEIJING_TRAIN_END = 24544


def _validate_prepared_frame(
    frame: pd.DataFrame, *, expected_n_vars: int, expected_length: int
) -> None:
    if list(frame.columns).count("date") != 1 or frame.columns[0] != "date":
        raise ValueError("prepared frame must begin with exactly one date column")
    if len(frame.columns) - 1 != expected_n_vars:
        raise ValueError(
            f"expected {expected_n_vars} prepared variables, "
            f"found {len(frame.columns) - 1}"
        )
    if len(frame) != expected_length:
        raise ValueError(f"expected {expected_length} rows, found {len(frame)}")
    pd.to_datetime(frame["date"], errors="raise")
    values = frame.iloc[:, 1:].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("prepared values must all be finite")


def _write_csv_atomic(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            frame.to_csv(temporary, index=False)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def prepare_exchange(source: Path, destination: Path) -> dict[str, object]:
    """Convert the canonical headerless eight-column exchange-rate file."""

    headerless = pd.read_csv(source, header=None, compression="infer")
    numeric = headerless.apply(pd.to_numeric, errors="coerce")
    if numeric.shape[1] == 8 and numeric.notna().all(axis=None):
        values = numeric
        dates = pd.date_range("1990-01-01", periods=len(values), freq="D")
    else:
        raw = pd.read_csv(source, compression="infer")
        if "date" not in raw.columns:
            raise ValueError(
                "exchange input must be the canonical headerless 8-column file "
                "or a CSV containing a date column"
            )
        dates = pd.to_datetime(raw.pop("date"), errors="raise")
        values = raw.apply(pd.to_numeric, errors="raise")
        if values.shape[1] != 8:
            raise ValueError("exchange input must contain exactly 8 variables")
    prepared = pd.DataFrame(values.to_numpy(dtype=np.float64), columns=EXCHANGE_COLUMNS)
    prepared.insert(0, "date", pd.DatetimeIndex(dates).astype(str))
    _validate_prepared_frame(prepared, expected_n_vars=8, expected_length=7588)
    _write_csv_atomic(prepared, destination)
    return {
        "dataset": "Exchange",
        "output": str(destination.resolve()),
        "shape": [7588, 8],
        "date_note": "canonical headerless input uses a synthetic monotonic daily index",
    }


def prepare_appliances(source: Path, destination: Path) -> dict[str, object]:
    """Drop the two published random controls and retain 26 signal channels."""

    if source.suffix.lower() == ".zip":
        with ZipFile(source) as archive:
            members = [
                name
                for name in archive.namelist()
                if name.lower().endswith("energydata_complete.csv")
            ]
            if len(members) != 1:
                raise ValueError(
                    "Appliances ZIP must contain exactly one energydata_complete.csv"
                )
            with archive.open(members[0]) as stream:
                raw = pd.read_csv(stream)
    else:
        raw = pd.read_csv(source)
    required = {"date", "rv1", "rv2"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"appliances input is missing columns: {sorted(missing)}")
    prepared = raw.drop(columns=["rv1", "rv2"]).copy()
    columns = [column for column in prepared.columns if column != "date"]
    prepared = prepared.loc[:, ["date", *columns]]
    prepared["date"] = pd.to_datetime(prepared["date"], errors="raise").astype(str)
    prepared.loc[:, columns] = prepared.loc[:, columns].apply(
        pd.to_numeric, errors="raise"
    )
    _validate_prepared_frame(prepared, expected_n_vars=26, expected_length=19735)
    _write_csv_atomic(prepared, destination)
    return {
        "dataset": "Appliances",
        "output": str(destination.resolve()),
        "shape": [19735, 26],
        "dropped_columns": ["rv1", "rv2"],
    }


def _beijing_frames_from_archive(archive: ZipFile) -> Iterable[pd.DataFrame]:
    members = sorted(
        name
        for name in archive.namelist()
        if name.lower().endswith(".csv") and "prsa_data_" in name.lower()
    )
    if members:
        for member in members:
            with archive.open(member) as stream:
                yield pd.read_csv(stream)
        return
    nested = sorted(
        name
        for name in archive.namelist()
        if name.lower().endswith(".zip") and "prsa2017_data" in name.lower()
    )
    if len(nested) != 1:
        raise ValueError(
            "Beijing ZIP must contain station CSV files or one PRSA2017 data ZIP"
        )
    with archive.open(nested[0]) as stream:
        with ZipFile(BytesIO(stream.read())) as inner:
            yield from _beijing_frames_from_archive(inner)


def _beijing_frames(source: Path) -> Iterable[pd.DataFrame]:
    if source.is_dir():
        paths = sorted(source.glob("*.csv"))
        if not paths:
            raise ValueError("Beijing source directory contains no CSV files")
        for path in paths:
            yield pd.read_csv(path)
        return
    if source.suffix.lower() != ".zip":
        raise ValueError("Beijing source must be the extracted directory or UCI ZIP")
    with ZipFile(source) as archive:
        yield from _beijing_frames_from_archive(archive)


def prepare_beijing_air_quality(
    source: Path,
    destination: Path,
    *,
    natural_mask_path: Path | None = None,
) -> dict[str, object]:
    """Pivot PM2.5 from 12 stations and preserve the native-missing sidecar.

    The current four artificial-missing protocols require a finite reference
    series. Native gaps are therefore completed causally by forward fill; any
    leading gap is filled by that station's observed training median.  The
    original observation mask is stored separately for the later native-gap
    evaluation protocol and is never discarded.
    """

    selected: list[pd.DataFrame] = []
    required = {"year", "month", "day", "hour", "PM2.5", "station"}
    for frame in _beijing_frames(source):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Beijing station file is missing: {sorted(missing)}")
        timestamps = pd.to_datetime(
            frame.loc[:, ["year", "month", "day", "hour"]].rename(
                columns={"year": "year", "month": "month", "day": "day", "hour": "hour"}
            ),
            errors="raise",
        )
        selected.append(
            pd.DataFrame(
                {
                    "date": timestamps,
                    "station": frame["station"].astype(str),
                    "PM2.5": pd.to_numeric(frame["PM2.5"], errors="coerce"),
                }
            )
        )
    combined = pd.concat(selected, ignore_index=True)
    if combined.duplicated(["date", "station"]).any():
        raise ValueError("Beijing input contains duplicate station timestamps")
    pivot = combined.pivot(index="date", columns="station", values="PM2.5").sort_index()
    if pivot.shape[1] != 12:
        raise ValueError(f"expected 12 Beijing stations, found {pivot.shape[1]}")
    complete_index = pd.date_range(pivot.index.min(), pivot.index.max(), freq="h")
    pivot = pivot.reindex(complete_index)
    if len(pivot) != 35064:
        raise ValueError(f"expected 35064 Beijing hours, found {len(pivot)}")

    naturally_observed = pivot.notna().to_numpy(dtype=bool)
    train_medians = pivot.iloc[:BEIJING_TRAIN_END].median(axis=0, skipna=True)
    if train_medians.isna().any():
        missing_stations = train_medians.index[train_medians.isna()].tolist()
        raise ValueError(
            f"Beijing stations have no observed training values: {missing_stations}"
        )
    completed = pivot.ffill().fillna(train_medians)
    prepared = completed.reset_index(names="date")
    prepared["date"] = prepared["date"].astype(str)
    prepared.columns = ["date", *[str(column) for column in pivot.columns]]
    _validate_prepared_frame(prepared, expected_n_vars=12, expected_length=35064)
    _write_csv_atomic(prepared, destination)

    sidecar = natural_mask_path or destination.with_suffix(".natural_missing.npz")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        sidecar,
        observed_mask=naturally_observed,
        stations=np.asarray([str(column) for column in pivot.columns]),
        dates=np.asarray(complete_index.astype(str)),
        completion_policy=np.asarray("causal_ffill_then_train_median"),
        train_end=np.asarray(BEIJING_TRAIN_END, dtype=np.int64),
    )
    return {
        "dataset": "BeijingAirQuality",
        "output": str(destination.resolve()),
        "natural_missing_sidecar": str(sidecar.resolve()),
        "shape": [35064, 12],
        "native_missing_rate": float(1.0 - naturally_observed.mean()),
        "completion_policy": "causal_ffill_then_train_median",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m malr.data.benchmark_prep")
    parser.add_argument(
        "dataset", choices=("Exchange", "Appliances", "BeijingAirQuality")
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--natural-mask-output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dataset == "Exchange":
        result = prepare_exchange(args.source, args.output)
    elif args.dataset == "Appliances":
        result = prepare_appliances(args.source, args.output)
    else:
        result = prepare_beijing_air_quality(
            args.source,
            args.output,
            natural_mask_path=args.natural_mask_output,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
