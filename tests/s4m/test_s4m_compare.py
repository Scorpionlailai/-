from __future__ import annotations

from copy import deepcopy

import pytest

from integrations.s4m.compare_results import compare_results, degradation


def _records() -> dict[str, dict]:
    patch_manifest = {
        "dataset": "ETTh1",
        "metadata": {
            "dataset": "ETTh1",
            "seq_len": 96,
            "pred_len": 96,
            "train_end": 8640,
            "validation_end": 11520,
            "test_end": 14400,
            "config_hash": "config",
        },
    }
    base = {
        "dataset": "ETTh1",
        "metric_space": "train_standardized",
        "source_csv_sha256": "source",
        "config_hash": "config",
    }
    test_identity = {
        "dataset": "ETTh1",
        "split": "test",
        "split_start": 11520,
        "split_end": 14400,
        "seq_len": 96,
        "pred_len": 96,
        "sample_count": 2688,
        "window_indices_sha256": "canonical-window-hash",
        "metric_space": "train_standardized",
        "mask_sha256": "mask",
        "final_rate": 0.409,
        "source_csv_sha256": "source",
        "config_hash": "config",
    }
    return {
        "patchtst_manifest": patch_manifest,
        "patchtst_clean": {
            **base,
            "run_id": "patch-clean",
            "mse": 0.4,
            "mae": 0.4,
            "final_rate": 0.0,
            "mask_sha256": None,
        },
        "patchtst_missing": {
            **base,
            "run_id": "patch-missing",
            "mse": 0.5,
            "mae": 0.45,
            "final_rate": 0.409,
            "mask_sha256": "mask",
        },
        "s4m_clean": {
            **base,
            "run_id": "s4m-clean",
            "mse": 0.42,
            "mae": 0.41,
            "final_rate": 0.0,
            "mask_sha256": None,
            "test_identity": {**test_identity, "mask_sha256": None, "final_rate": 0.0},
        },
        "s4m_official": {
            **base,
            "run_id": "s4m-official",
            "mse": 0.46,
            "mae": 0.43,
            "final_rate": 0.409,
            "mask_sha256": "mask",
            "checkpoint_sha256": "official-checkpoint",
            "test_identity": test_identity,
        },
        "s4m_tuned": {
            **base,
            "run_id": "s4m-tuned",
            "mse": 0.44,
            "mae": 0.42,
            "final_rate": 0.409,
            "mask_sha256": "mask",
            "checkpoint_sha256": "tuned-checkpoint",
            "test_identity": test_identity,
        },
        "tuned_selection": {
            "selected_run_id": "s4m-tuned",
            "checkpoint_sha256": "tuned-checkpoint",
            "test_metrics_seen": False,
        },
        "expected_window_hash": "canonical-window-hash",
    }


def test_absolute_and_relative_degradation() -> None:
    assert degradation(clean=0.4, missing=0.5) == {
        "absolute": pytest.approx(0.1),
        "relative_percent": pytest.approx(25.0),
    }


def test_comparison_rejects_different_test_mask() -> None:
    records = _records()
    records["s4m_tuned"] = deepcopy(records["s4m_tuned"])
    records["s4m_tuned"]["mask_sha256"] = "different"

    with pytest.raises(ValueError, match="mask_sha256"):
        compare_results(**records)


def test_comparison_rejects_non_selected_tuned_checkpoint() -> None:
    records = _records()
    records["tuned_selection"] = {
        **records["tuned_selection"],
        "selected_run_id": "another-run",
    }

    with pytest.raises(ValueError, match="selected"):
        compare_results(**records)


def test_comparison_reports_raw_metrics_and_degradation() -> None:
    report = compare_results(**_records())

    rows = {row["label"]: row for row in report["rows"]}
    assert rows["PatchTST missing"]["mse"] == 0.5
    assert rows["S4M tuned missing"]["mse"] == 0.44
    assert rows["PatchTST missing"]["degradation_mse_relative_percent"] == pytest.approx(25.0)
    assert rows["S4M tuned missing"]["degradation_mse_absolute"] == pytest.approx(0.02)
    assert report["identity"]["mask_sha256"] == "mask"
