from argparse import Namespace
import json
from pathlib import Path
from typing import Any

import pytest

import malr.cli as cli


_PROVENANCE = {
    "dataset": "ETTh1",
    "config_hash": "a" * 64,
    "source_csv_sha256": "b" * 64,
    "seed": 2021,
    "validation_mask_sha256": "c" * 64,
    "validation_final_rate": 0.25,
    "train_start": 0,
    "train_end": 8640,
    "validation_end": 11520,
    "test_end": 14400,
    "seq_len": 96,
    "pred_len": 96,
    "metric_space": "train_standardized",
}


def _write_manifest(
    artifacts: Path,
    variant: str,
    best_val_mse: float,
    *,
    best_val_mae: float = 0.30,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_id = f"ETTh1-{variant}-fixture"
    manifest = {
        "run_id": run_id,
        "dataset": "ETTh1",
        "variant": variant,
        "metadata": dict(_PROVENANCE if metadata is None else metadata),
        "training": {
            "best_val_mse": best_val_mse,
            "best_val_mae": best_val_mae,
        },
    }
    path = artifacts / "runs" / f"{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def _args(artifacts: Path, run_id: str) -> Namespace:
    return Namespace(
        artifacts=artifacts,
        config=Path("unused.yaml"),
        dataset="ETTh1",
        run_id=run_id,
        device="cpu",
    )


def test_failed_validation_gate_blocks_before_test_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = _write_manifest(tmp_path, "missing", 0.40)
    ours = _write_manifest(tmp_path, "ours", 0.40)
    evaluated: list[str] = []

    def fake_test_evaluation(
        args: Namespace, manifest: dict[str, Any], *, allow_existing: bool
    ) -> dict[str, Any]:
        evaluated.append(str(manifest["variant"]))
        return {"variant": manifest["variant"]}

    monkeypatch.setattr(cli, "_evaluate_manifest", fake_test_evaluation)

    with pytest.raises(ValueError, match="ours best_val_mse=0.4.*missing best_val_mse=0.4.*blocked"):
        cli._command_evaluate(_args(tmp_path, str(ours["run_id"])))

    assert evaluated == []
    assert not (tmp_path / "evaluations").exists()
    assert missing["training"]["best_val_mse"] == 0.40


def test_matching_provenance_and_better_validation_evaluate_missing_then_ours(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_manifest(tmp_path, "missing", 0.40, best_val_mae=0.30)
    ours = _write_manifest(tmp_path, "ours", 0.35, best_val_mae=0.25)
    evaluated: list[tuple[str, bool]] = []
    selected_runs: list[str] = []

    real_latest_run_manifest = cli._latest_run_manifest

    def record_selected_run(
        artifacts: Path, dataset: str, variant: str
    ) -> dict[str, Any]:
        selected_runs.append(variant)
        return real_latest_run_manifest(artifacts, dataset, variant)

    def fake_test_evaluation(
        args: Namespace, manifest: dict[str, Any], *, allow_existing: bool
    ) -> dict[str, Any]:
        evaluated.append((str(manifest["variant"]), allow_existing))
        return {"variant": manifest["variant"], "mask_sha256": "d" * 64}

    monkeypatch.setattr(cli, "_latest_run_manifest", record_selected_run)
    monkeypatch.setattr(cli, "_evaluate_manifest", fake_test_evaluation)

    cli._command_evaluate(_args(tmp_path, str(ours["run_id"])))

    assert evaluated == [("missing", True), ("ours", False)]
    assert selected_runs == ["missing"]


def test_better_validation_mse_but_worse_mae_blocks_test_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_manifest(tmp_path, "missing", 0.40, best_val_mae=0.30)
    ours = _write_manifest(tmp_path, "ours", 0.35, best_val_mae=0.31)
    evaluated: list[str] = []

    def fake_test_evaluation(
        args: Namespace, manifest: dict[str, Any], *, allow_existing: bool
    ) -> dict[str, Any]:
        evaluated.append(str(manifest["variant"]))
        return {"variant": manifest["variant"]}

    monkeypatch.setattr(cli, "_evaluate_manifest", fake_test_evaluation)

    with pytest.raises(ValueError, match="best_val_mae=0.31.*best_val_mae=0.3.*blocked"):
        cli._command_evaluate(_args(tmp_path, str(ours["run_id"])))

    assert evaluated == []


def test_missing_provenance_field_blocks_before_test_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing_metadata = dict(_PROVENANCE)
    missing_metadata.pop("source_csv_sha256")
    _write_manifest(tmp_path, "missing", 0.40, metadata=missing_metadata)
    ours = _write_manifest(tmp_path, "ours", 0.35)
    evaluated: list[str] = []

    def fake_test_evaluation(
        args: Namespace, manifest: dict[str, Any], *, allow_existing: bool
    ) -> dict[str, Any]:
        evaluated.append(str(manifest["variant"]))
        return {"variant": manifest["variant"]}

    monkeypatch.setattr(cli, "_evaluate_manifest", fake_test_evaluation)

    with pytest.raises(ValueError, match="source_csv_sha256.*missing.*blocked"):
        cli._command_evaluate(_args(tmp_path, str(ours["run_id"])))

    assert evaluated == []


def test_mismatched_provenance_value_blocks_before_test_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing_metadata = dict(_PROVENANCE)
    missing_metadata["seed"] = 2022
    _write_manifest(tmp_path, "missing", 0.40, metadata=missing_metadata)
    ours = _write_manifest(tmp_path, "ours", 0.35)
    evaluated: list[str] = []

    def fake_test_evaluation(
        args: Namespace, manifest: dict[str, Any], *, allow_existing: bool
    ) -> dict[str, Any]:
        evaluated.append(str(manifest["variant"]))
        return {"variant": manifest["variant"], "mask_sha256": "d" * 64}

    monkeypatch.setattr(cli, "_evaluate_manifest", fake_test_evaluation)

    with pytest.raises(ValueError, match="seed: missing=2022, ours=2021.*blocked"):
        cli._command_evaluate(_args(tmp_path, str(ours["run_id"])))

    assert evaluated == []


def test_gated_evaluations_require_identical_test_mask_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_manifest(tmp_path, "missing", 0.40, best_val_mae=0.30)
    ours = _write_manifest(tmp_path, "ours", 0.35, best_val_mae=0.25)

    def fake_test_evaluation(
        args: Namespace, manifest: dict[str, Any], *, allow_existing: bool
    ) -> dict[str, Any]:
        return {
            "variant": manifest["variant"],
            "mask_sha256": "missing-mask" if allow_existing else "ours-mask",
        }

    monkeypatch.setattr(cli, "_evaluate_manifest", fake_test_evaluation)

    with pytest.raises(ValueError, match="test mask SHA-256 mismatch"):
        cli._command_evaluate(_args(tmp_path, str(ours["run_id"])))
