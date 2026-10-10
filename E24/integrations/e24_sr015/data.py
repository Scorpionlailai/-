"""Sealed, train-only-normalized dataset protocols for E24 migration runs."""
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from integrations.forecast_compare.protocol import P4Protocol
from malr.config import load_config
from malr.data.datasets import load_clean_csv
from malr.data.masks import (MaskExpectation, generate_variable_missing_mask,
                            load_fixed_mask, save_fixed_mask)
from malr.data.normalization import fit_train_variable_stats, standardize_values
from malr.data.splits import protocol_from_bounds
from malr.data.windows import build_window_index
from malr.evaluation.report import file_sha256

ROOT = Path(__file__).resolve().parents[2]
WEATHER_ID = "E24-weather-v79-sr015-bl8-seq96-pred96-all-eval-v1"
TEST_SEEDS = tuple(range(2026100801, 2026100821))


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def prepare_weather(artifacts: Path) -> Path:
    """Create the data/mask seal once, before training or test observation."""
    artifacts = Path(artifacts).resolve()
    if artifacts.exists():
        raise FileExistsError(artifacts)
    artifacts.mkdir(parents=True)
    template = yaml.safe_load((ROOT / "configs/etth1_seq96_pred96.yaml").read_text())
    old = template["data"]["datasets"].pop("ETTh1")
    old.update(path=str((ROOT / "data/raw/weather.csv").resolve()), n_vars=21,
               expected_length=52696, train_start=0, train_end=36887,
               validation_end=42157, test_end=52696, allow_length_override=False)
    template["data"]["datasets"]["Weather"] = old
    for section in ("model", "training"):
        template[section]["profiles"]["Weather"] = template[section]["profiles"].pop("ETTh1")
    template["missing"]["start_rate"] = .15
    config_path = artifacts / "weather_seq96_pred96.yaml"
    config_path.write_text(yaml.safe_dump(template, sort_keys=False))
    series = load_clean_csv(Path(old["path"]), 21, 52696)
    differences = np.diff(series.timestamps).astype("timedelta64[s]").astype(np.int64)
    if np.any(differences < 0):
        raise ValueError("Weather timestamps must not go backwards")
    anomalies = [dict(after_row=int(i), interval_seconds=int(differences[i]),
                      before=str(series.timestamps[i]), after=str(series.timestamps[i+1]))
                 for i in np.flatnonzero(differences != 600)]
    masks = artifacts / "masks"
    masks.mkdir()
    fixed = []
    for split, start, end, seeds in (
        ("validation", 36887 - 96, 42157, (2021,)),
        ("test", 42157 - 96, 52696, TEST_SEEDS),
    ):
        for seed in seeds:
            observed, metadata = generate_variable_missing_mask(
                end - start, 21, .15, 8, seed, dataset="Weather",
                split=split, support_start=start)
            path = masks / f"Weather_{split}_seed{seed}.npz"
            checksum = save_fixed_mask(path, observed, metadata)
            fixed.append(dict(split=split, seed=seed, path=str(path),
                              sha256=file_sha256(path), checksum=checksum,
                              support_start=start, support_end=end,
                              shape=list(observed.shape), final_missing_rate=metadata.final_rate))
    stats = fit_train_variable_stats(torch.from_numpy(series.values[:36887]), eps=0.)
    record = dict(protocol_id=WEATHER_ID, dataset="Weather", n_vars=21,
                  source_path=str(series.source_path), source_sha256=file_sha256(series.source_path),
                  config_path=str(config_path), config_file_sha256=file_sha256(config_path),
                  columns=list(series.columns), rows=52696, nominal_frequency_seconds=600,
                  timestamp_anomalies=anomalies, row_order_preserved=True,
                  splits=dict(train=[0,36887], validation=[36887,42157], test=[42157,52696]),
                  seq_len=96, pred_len=96, start_rate=.15, block_len=8,
                  overlap_allowed=True, evaluation_drop_last=False, train_drop_last=True,
                  validation_seed=2021, test_seeds=list(TEST_SEEDS), model_training_seed=2023,
                  train_scaler=dict(mean=stats.mean.tolist(), std=stats.std.tolist(), fit_rows=[0,36887]),
                  masks=fixed, created_before_training=True,
                  test_policy="All 20 predeclared masks; weights selected only on fixed validation.")
    manifest = artifacts / "dataset_manifest.json"
    _write_json(manifest, record)
    manifest.with_suffix(".sha256").write_text(file_sha256(manifest) + "\n")
    return manifest


def read_manifest(path: Path) -> dict:
    path = Path(path).resolve(strict=True)
    if file_sha256(path) != path.with_suffix(".sha256").read_text().strip():
        raise ValueError("dataset manifest seal mismatch")
    record = json.loads(path.read_text())
    if record["protocol_id"] != WEATHER_ID or record["test_seeds"] != list(TEST_SEEDS):
        raise ValueError("unexpected Weather protocol or test seeds")
    return record


def load_mask(record, item):
    path = Path(item["path"])
    if file_sha256(path) != item["sha256"]:
        raise ValueError(f"mask SHA mismatch: {path}")
    mask, _ = load_fixed_mask(path, MaskExpectation(
        dataset=record["dataset"], split=item["split"], seed=item["seed"],
        start_rate=record["start_rate"], block_len=record["block_len"],
        support_start=item["support_start"], support_end=item["support_end"],
        shape=tuple(item["shape"])))
    return mask


def load_dataset_protocol(manifest: Path, *, config_path: Path, artifacts: Path) -> P4Protocol:
    manifest = Path(manifest).resolve(strict=True)
    record = read_manifest(manifest)
    config_path = Path(config_path).resolve(strict=True)
    if (config_path != Path(record["config_path"])
            or file_sha256(config_path) != record["config_file_sha256"]
            or Path(artifacts).resolve() != manifest.parent):
        raise ValueError("dataset config/artifacts do not match the sealed manifest")
    config = load_config(config_path)
    dataset = config.data.datasets["Weather"]
    if (config.data.seq_len, config.data.pred_len, dataset.n_vars,
        dataset.train_start, dataset.train_end, dataset.validation_end, dataset.test_end,
        config.missing.start_rate, config.missing.block_len, config.missing.allow_block_overlap
        ) != (96,96,21,0,36887,42157,52696,.15,8,True):
        raise ValueError("Weather split or missing configuration changed")
    source = Path(dataset.path).resolve(strict=True)
    if source != Path(record["source_path"]) or file_sha256(source) != record["source_sha256"]:
        raise ValueError("Weather CSV differs from sealed identity")
    series = load_clean_csv(source, 21, 52696)
    stats = fit_train_variable_stats(torch.from_numpy(series.values[:dataset.train_end]), eps=0.)
    if stats.mean.tolist() != record["train_scaler"]["mean"] or stats.std.tolist() != record["train_scaler"]["std"]:
        raise ValueError("train-only scaler does not match manifest")
    standardized = np.ascontiguousarray(
        standardize_values(torch.from_numpy(series.values), stats).numpy(), dtype=np.float32)
    fallback = fit_train_variable_stats(torch.from_numpy(standardized[:dataset.train_end]), eps=0.)
    bounds = protocol_from_bounds("Weather", len(series.values), train_start=0,
                                 train_end=36887, validation_end=42157, test_end=52696)
    refs = {split: tuple(build_window_index(bounds, split, 96,96))
            for split in ("train", "validation", "test")}
    validation = next(x for x in record["masks"] if x["split"] == "validation")
    first_test = next(x for x in record["masks"] if x["split"] == "test")
    # Verify every mask before any training begins, including all 20 test fixtures.
    masks = { (x["split"], x["seed"]): load_mask(record,x) for x in record["masks"] }
    return P4Protocol(config, bounds, standardized, fallback, refs,
                      {"validation":masks[("validation",validation["seed"])],
                       "test":masks[("test",first_test["seed"])]},
                      record["source_sha256"],
                      {"validation":validation["sha256"], "test":first_test["sha256"]},
                      config_path, manifest.parent, training_seed=2023,
                      evaluation_drop_last=False, dataset_manifest_sha256=file_sha256(manifest))


def with_test_mask(protocol, record, item):
    # The checkpoint's original test identity remains bound to the sealed bank;
    # evaluate after loading its state, then replace only the iterator timeline.
    return replace(protocol, fixed_masks={**protocol.fixed_masks, "test":load_mask(record,item)})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-weather", type=Path, required=True)
    args = parser.parse_args()
    print(prepare_weather(args.prepare_weather))
