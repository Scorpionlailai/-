from pathlib import Path

import numpy as np
import pytest

from malr.evaluation.metrics import compute_metrics, degradation_recovery_rate
from malr.evaluation.report import ResultRecord, write_result_record


def test_metrics_and_drr_share_predictions():
    prediction = np.array([0.0, 2.0])
    target = np.array([1.0, 0.0])
    metrics = compute_metrics(prediction, target)

    assert metrics.mse == pytest.approx(2.5)
    assert metrics.mae == pytest.approx(1.5)
    assert degradation_recovery_rate(clean=1.0, naive=3.0, ours=2.0) == pytest.approx(50.0)


def test_nonpositive_drr_denominator_returns_none():
    assert degradation_recovery_rate(clean=2.0, naive=2.0, ours=1.5) is None
    assert degradation_recovery_rate(clean=3.0, naive=2.0, ours=1.5) is None


def _record(run_id: str = "fixture-run") -> ResultRecord:
    return ResultRecord(
        run_id=run_id,
        dataset="ETTh1",
        variant="ours",
        ablations={
            "use_temporal": True,
            "use_variable": True,
            "use_historical": True,
            "use_recovery_loss": True,
        },
        mse=1.0,
        mae=0.5,
        drr_mse=None,
        drr_mae=None,
        final_rate=0.25,
        trainable_parameters=10,
        training_seconds=2.0,
        metric_space="train_standardized",
        config_hash="a" * 64,
        checkpoint_sha256="b" * 64,
        mask_sha256="c" * 64,
        bank_sha256=None,
        source_csv_sha256="d" * 64,
        seed=2021,
        timestamp_utc="2026-08-28T00:00:00+00:00",
    )


def test_result_is_immutable_and_second_final_evaluation_is_rejected(tmp_path: Path):
    record = _record()
    path = tmp_path / "fixture-run.json"
    write_result_record(record, path)

    with pytest.raises(ValueError, match="already been evaluated"):
        write_result_record(record, path)
    with pytest.raises(Exception):
        record.mse = 2.0

    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "N/A" in summary
