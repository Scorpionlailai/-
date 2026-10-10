"""Identity-checked forecast MSE/MAE and shared-reference DRR report."""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from typing import Mapping, Sequence

SOURCE_SHA256 = "f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066"
TEST_MASK_SHA256 = "f7974ff9b4228d69fe33d940e3722cd1a069766e2fa1387c6240757d5a8762d8"
TEST_REFS_SHA256 = "bc4edaa33e4eac3b25045340f31a32cb799d12206202d1a808c336101d95cdb4"
CLEAN_FINGERPRINT = "b5932e46c76ee3e89ccbc51bec219b521bb1c8b4ed085d063a8c1b89102148fe"
MISSING_FINGERPRINT = "292570de0560129a4bd96568f13fc19875e06e80fd7041a7e7f8d06a89710f10"
E22_RUN = "ETTh1-ours-dea387986927"
E22_CHECKPOINT_SHA256 = "9d96d5c74b72e6b3a5b517d59504ca73db3550a6d12181e4bf74ab084f103c24"


def _require(value: object, expected: object, label: str) -> None:
    if value != expected:
        raise ValueError(f"{label} mismatch: expected {expected!r}, got {value!r}")


def _metric(record: Mapping[str, object], name: str, label: str) -> float:
    value = float(record[name])
    if not isfinite(value) or value < 0:
        raise ValueError(f"{label} {name} is invalid")
    return value


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_p4_report(
    *, clean: Mapping[str, object], missing: Mapping[str, object],
    e22: Mapping[str, object], saits: Mapping[str, object],
    s4m: Mapping[str, object], s4m_clean: Mapping[str, object],
) -> dict[str, object]:
    records = {"PatchTST Clean": clean, "PatchTST Missing": missing,
               "E22": e22, "SAITS+Predictor": saits, "S4M": s4m,
               "S4M Clean": s4m_clean}
    for label, record in records.items():
        _require(record.get("source_csv_sha256"), SOURCE_SHA256, f"{label} source CSV")
        _require(record.get("metric_space"), "train_standardized", f"{label} metric space")
        _require(record.get("seed"), 2021, f"{label} seed")
        if record.get("dataset", "ETTh1") != "ETTh1":
            raise ValueError(f"{label} dataset mismatch")
        _metric(record, "mse", label)
        _metric(record, "mae", label)
    for label, record in (("PatchTST Clean", clean), ("PatchTST Missing", missing), ("E22", e22)):
        _require(record.get("clean_baseline_fingerprint"), CLEAN_FINGERPRINT,
                 f"{label} Clean fingerprint")
        _require(record.get("missing_baseline_fingerprint"), MISSING_FINGERPRINT,
                 f"{label} Missing fingerprint")
    for label, record, field in (
        ("PatchTST Missing", missing, "mask_sha256"),
        ("E22", e22, "mask_sha256"),
        ("SAITS+Predictor", saits, "test_mask_sha256"),
        ("S4M", s4m, "mask_sha256"),
    ):
        _require(record.get(field), TEST_MASK_SHA256, f"{label} test mask")
    _require(clean.get("mask_sha256"), None, "PatchTST Clean mask")
    _require(s4m_clean.get("mask_sha256"), None, "S4M Clean mask")
    _require(e22.get("run_id"), E22_RUN, "E22 run")
    _require(e22.get("checkpoint_sha256"), E22_CHECKPOINT_SHA256, "E22 checkpoint")
    _require(saits.get("stage"), "forecast", "SAITS selected stage")
    _require(saits.get("split"), "test", "SAITS selected split")
    _require(saits.get("windows"), 2688, "SAITS test windows")
    for key in ("checkpoint_sha256", "selection_manifest_sha256"):
        if not isinstance(saits.get(key), str) or len(str(saits[key])) != 64:
            raise ValueError(f"SAITS selected-test {key} missing")
    for label, record, expected_mask in (("S4M", s4m, TEST_MASK_SHA256),
                                          ("S4M Clean", s4m_clean, None)):
        _require(record.get("status"), "test_finalized", f"{label} selected-test status")
        _require(record.get("variant"), "clean" if expected_mask is None else "missing", f"{label} variant")
        identity = record.get("test_identity")
        if not isinstance(identity, Mapping):
            raise ValueError(f"{label} test identity missing")
        for key, expected in (
            ("source_csv_sha256", SOURCE_SHA256), ("mask_sha256", expected_mask),
            ("window_indices_sha256", TEST_REFS_SHA256), ("sample_count", 2688),
            ("metric_space", "train_standardized"),
        ):
            _require(identity.get(key), expected, f"{label} {key}")
    notes: dict[str, str | None] = {}
    gaps = {}
    for metric in ("mse", "mae"):
        gap = _metric(missing, metric, "PatchTST Missing") - _metric(clean, metric, "PatchTST Clean")
        gaps[metric] = gap
        notes[metric] = (
            "shared Missing-Clean gap <= 1e-12; DRR undefined" if gap <= 1e-12 else None
        )
    rows: list[dict[str, object]] = []
    unavailable: dict[str, list[str]] = {}
    for label, record in records.items():
        row: dict[str, object] = {
            "model": label, "mse": _metric(record, "mse", label), "mae": _metric(record, "mae", label),
            "trainable_parameters": record.get("trainable_parameters"),
            "training_seconds": record.get("training_seconds"),
            "checkpoint_sha256": record.get("checkpoint_sha256"),
            "best_epoch": record.get("best_epoch"),
            "best_val_mse": record.get("best_val_mse"),
            "best_val_mae": record.get("best_val_mae"),
            "selection_manifest": record.get("selection_manifest"),
            "selection_manifest_sha256": record.get("selection_manifest_sha256"),
            "source_artifact_path": record.get("source_artifact_path"),
            "model_config_sha256": record.get("model_config_sha256", record.get("config_hash")),
            "adapter_code_sha256": record.get("adapter_code_sha256"),
            "upstream_commit": record.get("upstream_commit"),
            "model_settings_json": record.get("model_settings_json"),
            "drr_interpretation": (
                "shared PatchTST reference only" if label == "S4M" else
                "auxiliary S4M clean diagnostic; no shared-reference DRR" if label == "S4M Clean" else
                "relative to cached PatchTST degradation"
            ),
        }
        for metric in ("mse", "mae"):
            row[f"drr_{metric}_percent"] = (
                None if gaps[metric] <= 1e-12 or label == "S4M Clean" else
                (float(missing[metric]) - float(record[metric])) / gaps[metric] * 100.0
            )
        rows.append(row)
        unavailable[label] = [key for key in (
            "best_epoch", "best_val_mse", "best_val_mae", "selection_manifest_sha256",
            "source_artifact_path", "model_config_sha256", "adapter_code_sha256",
            "upstream_commit", "model_settings_json",
        ) if row[key] is None]
    return {
        "title": "ETTh1 P4 exploratory single-seed forecast comparison",
        "protocol": {"dataset": "ETTh1", "seq_len": 96, "pred_len": 96,
                     "train_end": 8640, "validation_end": 11520, "test_end": 14400,
                     "test_windows": 2688, "source_csv_sha256": SOURCE_SHA256,
                     "test_mask_sha256": TEST_MASK_SHA256, "test_refs_sha256": TEST_REFS_SHA256,
                     "metric_space": "train_standardized", "seed": 2021},
        "drr_definition": "(cached PatchTST Missing - model) / (cached PatchTST Missing - cached PatchTST Clean) * 100%",
        "drr_notes": notes,
        "rows": rows,
        "unavailable_metadata": unavailable,
        "s4m_clean_diagnostic": {
            "mse_gap": float(s4m["mse"]) - float(s4m_clean["mse"]),
            "mae_gap": float(s4m["mae"]) - float(s4m_clean["mae"]),
            "interpretation": "S4M-specific clean-to-missing gap; not used for shared DRR",
        },
        "limitations": "Exploratory single-seed comparison; ETTh1 P4 test set participated in earlier E22 development.",
    }


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_baseline_cache(path: Path) -> dict:
    payload = _load_json(path)
    result = dict(payload["result"])
    training = payload["manifest"].get("training", {})
    result.update(
        best_epoch=training.get("best_epoch"),
        best_val_mse=training.get("best_val_mse"),
        best_val_mae=training.get("best_val_mae"),
        source_artifact_path=str(path.resolve()),
        model_config_sha256=result.get("config_hash"),
    )
    return result


def _load_e22_evaluation(path: Path) -> dict:
    result = _load_json(path)
    manifest_path = path.parent.parent / "runs" / f"{result['run_id']}.json"
    manifest = _load_json(manifest_path)
    _require(manifest.get("checkpoint_sha256"), result.get("checkpoint_sha256"), "E22 selected checkpoint")
    training = manifest.get("training", {})
    result.update(
        best_epoch=training.get("best_epoch"),
        best_val_mse=training.get("best_val_mse"),
        best_val_mae=training.get("best_val_mae"),
        selection_manifest=str(manifest_path.resolve()),
        selection_manifest_sha256=_sha256(manifest_path),
        source_artifact_path=str(path.resolve()),
        model_config_sha256=result.get("config_hash"),
    )
    return result


def _selected_s4m_result(output: Path, *, clean: bool) -> dict:
    selection_path = output / ("selection-official-clean.json" if clean else "selection-official.json")
    selection = _load_json(selection_path)
    if selection.get("test_metrics_seen") is not False:
        raise ValueError("S4M selection was not validation-only")
    evaluation_path = output / "evaluations" / f"{selection['selected_run_id']}.json"
    evaluation = _load_json(evaluation_path)
    _require(evaluation.get("run_id"), selection["selected_run_id"], "S4M selected run")
    _require(evaluation.get("checkpoint_sha256"), selection["checkpoint_sha256"], "S4M selected checkpoint")
    manifest_path = Path(selection["selected_manifest"])
    manifest = _load_json(manifest_path)
    _require(manifest.get("status"), "validation_finalized", "S4M validation selection")
    _require(manifest.get("checkpoint_sha256"), evaluation["checkpoint_sha256"], "S4M manifest checkpoint")
    evaluation.update(
        best_epoch=manifest.get("best_epoch"),
        best_val_mse=manifest.get("validation_mse"),
        best_val_mae=manifest.get("validation_mae"),
        selection_manifest=str(selection_path.resolve()),
        selection_manifest_sha256=_sha256(selection_path),
        source_artifact_path=str(evaluation_path.resolve()),
        model_config_sha256=evaluation.get("config_hash"),
        model_settings_json=json.dumps(manifest.get("hyperparameters"), sort_keys=True),
    )
    return evaluation


def _write_report(report: dict[str, object], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    (output / "comparison.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    rows = report["rows"]
    with (output / "comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [f"# {report['title']}", "", "| Model | MSE | MAE | DRR-MSE % | DRR-MAE % |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for row in rows:
        numbers = ["—" if row[key] is None else f"{row[key]:.6f}"
                   for key in ("mse", "mae", "drr_mse_percent", "drr_mae_percent")]
        lines.append("| " + " | ".join((str(row["model"]), *numbers)) + " |")
    lines.extend(("", str(report["limitations"]), "",
                  "S4M DRR uses the shared PatchTST references only; S4M Clean is auxiliary.", ""))
    (output / "comparison.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("clean-cache", "missing-cache", "e22-evaluation", "saits-evaluation", "s4m-output", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    saits = _load_json(args.saits_evaluation)
    selection_path = Path(saits["selection_manifest"])
    _require(_sha256(selection_path), saits["selection_manifest_sha256"], "SAITS selection manifest")
    selection = _load_json(selection_path)
    _require(selection.get("checkpoint_sha256"), saits.get("checkpoint_sha256"), "SAITS selected checkpoint")
    saits["source_artifact_path"] = str(args.saits_evaluation.resolve())
    saits["model_settings_json"] = json.dumps(selection.get("predictor_settings"), sort_keys=True)
    result = build_p4_report(
        clean=_load_baseline_cache(args.clean_cache),
        missing=_load_baseline_cache(args.missing_cache),
        e22=_load_e22_evaluation(args.e22_evaluation), saits=saits,
        s4m=_selected_s4m_result(args.s4m_output, clean=False),
        s4m_clean=_selected_s4m_result(args.s4m_output, clean=True),
    )
    _write_report(result, args.output)
    print(json.dumps({"report": str(args.output.resolve()), "rows": result["rows"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
