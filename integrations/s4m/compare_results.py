from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from hashlib import sha256
import io
import json
from pathlib import Path
from typing import Any

from malr.data.splits import protocol_for
from malr.data.windows import build_window_index


def degradation(*, clean: float, missing: float) -> dict[str, float]:
    if clean <= 0.0 or missing < 0.0:
        raise ValueError("degradation metrics must be non-negative and clean > 0")
    absolute = missing - clean
    return {
        "absolute": absolute,
        "relative_percent": absolute / clean * 100.0,
    }


def _require_equal(
    name: str,
    values: list[object],
    *,
    tolerance: float | None = None,
) -> object:
    first = values[0]
    for value in values[1:]:
        if tolerance is not None:
            if abs(float(value) - float(first)) <= tolerance:
                continue
        elif value == first:
            continue
        raise ValueError(f"comparison identity mismatch for {name}: {values}")
    return first


def _row(
    label: str,
    result: dict[str, Any],
    *,
    clean: dict[str, Any] | None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "label": label,
        "run_id": result["run_id"],
        "mse": float(result["mse"]),
        "mae": float(result["mae"]),
        "final_rate": float(result["final_rate"]),
        "mask_sha256": result.get("mask_sha256"),
        "degradation_mse_absolute": None,
        "degradation_mse_relative_percent": None,
        "degradation_mae_absolute": None,
        "degradation_mae_relative_percent": None,
    }
    if clean is not None:
        mse = degradation(clean=float(clean["mse"]), missing=float(result["mse"]))
        mae = degradation(clean=float(clean["mae"]), missing=float(result["mae"]))
        row.update(
            {
                "degradation_mse_absolute": mse["absolute"],
                "degradation_mse_relative_percent": mse["relative_percent"],
                "degradation_mae_absolute": mae["absolute"],
                "degradation_mae_relative_percent": mae["relative_percent"],
            }
        )
    return row


def compare_results(
    *,
    patchtst_manifest: dict[str, Any],
    patchtst_clean: dict[str, Any],
    patchtst_missing: dict[str, Any],
    s4m_clean: dict[str, Any],
    s4m_official: dict[str, Any],
    s4m_tuned: dict[str, Any],
    tuned_selection: dict[str, Any],
    expected_window_hash: str,
) -> dict[str, object]:
    results = [
        patchtst_clean,
        patchtst_missing,
        s4m_clean,
        s4m_official,
        s4m_tuned,
    ]
    manifest_metadata = dict(patchtst_manifest.get("metadata", {}))
    dataset = patchtst_manifest.get("dataset") or manifest_metadata.get("dataset")
    if dataset != "ETTh1":
        raise ValueError("the first fair comparison supports ETTh1 only")
    for key, expected in (
        ("seq_len", 96),
        ("pred_len", 96),
        ("train_end", 8640),
        ("validation_end", 11520),
        ("test_end", 14400),
    ):
        if manifest_metadata.get(key) != expected:
            raise ValueError(
                f"PatchTST manifest {key} must equal {expected}, "
                f"got {manifest_metadata.get(key)!r}"
            )

    _require_equal(
        "dataset",
        [result.get("dataset", dataset) for result in results],
    )
    source_hash = _require_equal(
        "source_csv_sha256",
        [result.get("source_csv_sha256") for result in results],
    )
    config_hash = _require_equal(
        "config_hash",
        [result.get("config_hash") for result in results],
    )
    _require_equal(
        "metric_space",
        [result.get("metric_space") for result in results],
    )
    if manifest_metadata.get("config_hash") != config_hash:
        raise ValueError("PatchTST manifest config_hash differs from evaluations")

    missing_results = [patchtst_missing, s4m_official, s4m_tuned]
    mask_hash = _require_equal(
        "mask_sha256",
        [result.get("mask_sha256") for result in missing_results],
    )
    final_rate = _require_equal(
        "final_rate",
        [result.get("final_rate") for result in missing_results],
        tolerance=1e-12,
    )
    for clean in (patchtst_clean, s4m_clean):
        if clean.get("mask_sha256") is not None or float(clean["final_rate"]) != 0.0:
            raise ValueError("clean result must not depend on a missing mask")

    for label, result in (
        ("S4M clean", s4m_clean),
        ("S4M official", s4m_official),
        ("S4M tuned", s4m_tuned),
    ):
        identity = dict(result.get("test_identity", {}))
        expected = {
            "dataset": "ETTh1",
            "split": "test",
            "split_start": 11520,
            "split_end": 14400,
            "seq_len": 96,
            "pred_len": 96,
            "sample_count": 2688,
            "window_indices_sha256": expected_window_hash,
            "metric_space": "train_standardized",
        }
        for key, value in expected.items():
            if identity.get(key) != value:
                raise ValueError(
                    f"{label} test identity mismatch for {key}: "
                    f"expected {value!r}, got {identity.get(key)!r}"
                )
        if identity.get("source_csv_sha256") != source_hash:
            raise ValueError(f"{label} source CSV identity mismatch")
        if identity.get("config_hash") != config_hash:
            raise ValueError(f"{label} config identity mismatch")

    if tuned_selection.get("selected_run_id") != s4m_tuned.get("run_id"):
        raise ValueError("S4M tuned evaluation is not the selected validation run")
    if tuned_selection.get("checkpoint_sha256") != s4m_tuned.get(
        "checkpoint_sha256"
    ):
        raise ValueError("S4M tuned checkpoint differs from selection")

    rows = [
        _row("PatchTST clean", patchtst_clean, clean=None),
        _row("PatchTST missing", patchtst_missing, clean=patchtst_clean),
        _row("S4M official clean", s4m_clean, clean=None),
        _row("S4M official missing", s4m_official, clean=s4m_clean),
        _row("S4M tuned missing", s4m_tuned, clean=s4m_clean),
    ]
    return {
        "identity": {
            "dataset": "ETTh1",
            "source_csv_sha256": source_hash,
            "config_hash": config_hash,
            "metric_space": "train_standardized",
            "mask_sha256": mask_hash,
            "final_rate": float(final_rate),
            "seq_len": 96,
            "pred_len": 96,
            "test_sample_count": 2688,
            "window_indices_sha256": expected_window_hash,
        },
        "rows": rows,
    }


def canonical_test_window_hash() -> str:
    protocol = protocol_for("ETTh1", 14400)
    refs = build_window_index(protocol, "test", 96, 96)
    refs = refs[: len(refs) - len(refs) % 128]
    content = json.dumps(
        [asdict(ref) for ref in refs],
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(content.encode("utf-8")).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_comparison(report: dict[str, object], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    rows = list(report["rows"])  # type: ignore[arg-type]
    columns = list(rows[0].keys())
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    (output / "comparison.csv").write_text(buffer.getvalue(), encoding="utf-8")
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                "N/A" if row[column] is None else str(row[column])
                for column in columns
            )
            + " |"
        )
    (output / "comparison.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare PatchTST and S4M")
    parser.add_argument("--patchtst-manifest", type=Path, required=True)
    parser.add_argument("--patchtst-clean", type=Path, required=True)
    parser.add_argument("--patchtst-missing", type=Path, required=True)
    parser.add_argument("--s4m-clean", type=Path, required=True)
    parser.add_argument("--s4m-official", type=Path, required=True)
    parser.add_argument("--s4m-tuned", type=Path, required=True)
    parser.add_argument("--tuned-selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = compare_results(
        patchtst_manifest=_load(args.patchtst_manifest),
        patchtst_clean=_load(args.patchtst_clean),
        patchtst_missing=_load(args.patchtst_missing),
        s4m_clean=_load(args.s4m_clean),
        s4m_official=_load(args.s4m_official),
        s4m_tuned=_load(args.s4m_tuned),
        tuned_selection=_load(args.tuned_selection),
        expected_window_hash=canonical_test_window_hash(),
    )
    write_comparison(report, args.output.resolve())
    print((args.output.resolve() / "comparison.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
