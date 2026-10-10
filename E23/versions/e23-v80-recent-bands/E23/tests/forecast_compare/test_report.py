import pytest

from integrations.forecast_compare.report import build_p4_report


SOURCE = "f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066"
MASK = "f7974ff9b4228d69fe33d940e3722cd1a069766e2fa1387c6240757d5a8762d8"


def records():
    base = {"dataset": "ETTh1", "source_csv_sha256": SOURCE,
            "metric_space": "train_standardized", "seed": 2021,
            "clean_baseline_fingerprint": "b5932e46c76ee3e89ccbc51bec219b521bb1c8b4ed085d063a8c1b89102148fe",
            "missing_baseline_fingerprint": "292570de0560129a4bd96568f13fc19875e06e80fd7041a7e7f8d06a89710f10"}
    clean = {**base, "mse": 0.39158731901726757, "mae": 0.406349116171839,
             "mask_sha256": None, "config_hash": "baseline"}
    missing = {**base, "mse": 0.5196692808415073, "mae": 0.4668949174772855,
               "mask_sha256": MASK, "config_hash": "baseline"}
    e22 = {**base, "mse": 0.44319546087528844, "mae": 0.4332482160624463,
           "mask_sha256": MASK, "config_hash": "e22", "run_id": "ETTh1-ours-dea387986927",
           "checkpoint_sha256": "9d96d5c74b72e6b3a5b517d59504ca73db3550a6d12181e4bf74ab084f103c24"}
    saits = {"mse": 0.46, "mae": 0.44, "source_csv_sha256": SOURCE,
             "test_mask_sha256": MASK, "metric_space": "train_standardized", "windows": 2688,
             "checkpoint_sha256": "a" * 64, "selection_manifest_sha256": "b" * 64,
             "stage": "forecast", "split": "test", "seed": 2021,
             "model_config_sha256": "d" * 64, "adapter_code_sha256": "e" * 64,
             "best_epoch": 18, "best_val_mse": 0.81,
             "selection_manifest": "/selected/saits-stage2.json"}
    identity = {"source_csv_sha256": SOURCE, "mask_sha256": MASK,
                "window_indices_sha256": "bc4edaa33e4eac3b25045340f31a32cb799d12206202d1a808c336101d95cdb4",
                "sample_count": 2688, "metric_space": "train_standardized"}
    s4m = {"mse": 0.47, "mae": 0.45, "source_csv_sha256": SOURCE,
           "mask_sha256": MASK, "metric_space": "train_standardized",
           "test_identity": identity, "status": "test_finalized", "variant": "missing",
           "checkpoint_sha256": "c" * 64, "seed": 2021,
           "best_epoch": 12, "best_val_mse": 0.9,
           "selection_manifest_sha256": "f" * 64,
           "source_artifact_path": "/selected/s4m-evaluation.json"}
    s4m_clean = {**s4m, "mse": 0.4, "mae": 0.42, "mask_sha256": None,
                 "variant": "clean", "test_identity": {**identity, "mask_sha256": None}}
    return clean, missing, e22, saits, s4m, s4m_clean


def report(*items):
    return build_p4_report(**dict(zip(("clean", "missing", "e22", "saits", "s4m", "s4m_clean"), items)))


def test_shared_reference_drr():
    result = report(*records())
    row = next(row for row in result["rows"] if row["model"] == "E22")
    assert row["drr_mse_percent"] == pytest.approx(59.706939897719515)
    assert row["drr_mae_percent"] == pytest.approx(55.57231168697484)


def test_near_zero_gap_returns_null():
    clean, missing, *others = records()
    missing["mse"] = clean["mse"] + 1e-13
    result = report(clean, missing, *others)
    row = next(row for row in result["rows"] if row["model"] == "E22")
    assert row["drr_mse_percent"] is None
    assert "<= 1e-12" in result["drr_notes"]["mse"]


def test_report_rejects_wrong_mask_or_space():
    items = list(records())
    items[3]["test_mask_sha256"] = "x" * 64
    with pytest.raises(ValueError, match="mask"):
        report(*items)
    items = list(records())
    items[4]["metric_space"] = "raw"
    with pytest.raises(ValueError, match="metric space"):
        report(*items)


def test_different_config_hash_is_allowed():
    assert len(report(*records())["rows"]) == 6


def test_s4m_drr_is_labeled_reference_only():
    result = report(*records())
    row = next(row for row in result["rows"] if row["model"] == "S4M")
    assert row["drr_interpretation"] == "shared PatchTST reference only"
    assert result["s4m_clean_diagnostic"]["mse_gap"] == pytest.approx(0.07)
    assert next(row for row in result["rows"] if row["model"] == "S4M Clean")["drr_mse_percent"] is None


def test_report_preserves_validation_and_code_audit_fields():
    result = report(*records())
    saits = next(row for row in result["rows"] if row["model"] == "SAITS+Predictor")
    assert saits["best_epoch"] == 18
    assert saits["best_val_mse"] == 0.81
    assert saits["selection_manifest_sha256"] == "b" * 64
    assert saits["model_config_sha256"] == "d" * 64
    assert saits["adapter_code_sha256"] == "e" * 64
    s4m = next(row for row in result["rows"] if row["model"] == "S4M")
    assert s4m["source_artifact_path"] == "/selected/s4m-evaluation.json"
    assert s4m["best_epoch"] == 12
    assert "best_epoch" in result["unavailable_metadata"]["PatchTST Clean"]
