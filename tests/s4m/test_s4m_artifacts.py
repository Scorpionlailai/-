from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from integrations.s4m.overlay.malr_fair.artifacts import (
    MetricAccumulator,
    authorize_test,
    file_sha256,
    write_evaluation_result,
    write_training_manifest,
)


def test_metrics_are_global_not_mean_of_batch_means() -> None:
    accumulator = MetricAccumulator()
    accumulator.update(torch.tensor([0.0]), torch.tensor([2.0]))
    accumulator.update(
        torch.tensor([0.0, 0.0, 0.0]), torch.tensor([1.0, 1.0, 1.0])
    )

    assert accumulator.compute() == {"mse": 1.75, "mae": 1.25}


def test_metrics_reject_non_finite_values() -> None:
    accumulator = MetricAccumulator()

    with pytest.raises(ValueError, match="finite"):
        accumulator.update(torch.tensor([float("nan")]), torch.tensor([0.0]))


def test_training_manifest_is_immutable(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    payload = {
        "run_id": "candidate-a",
        "status": "validation_finalized",
        "validation_mse": 0.7,
    }
    write_training_manifest(path, payload)

    assert json.loads(path.read_text(encoding="utf-8")) == payload
    with pytest.raises(ValueError, match="already exists"):
        write_training_manifest(path, payload)


def test_test_guard_rejects_non_selected_run(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint")
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "selected_run_id": "best",
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": file_sha256(checkpoint),
                "test_metrics_seen": False,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="selected"):
        authorize_test(selection, "not-best", checkpoint)


def test_test_guard_verifies_checkpoint_identity(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint")
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "selected_run_id": "best",
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": file_sha256(checkpoint),
                "test_metrics_seen": False,
            }
        ),
        encoding="utf-8",
    )

    authorized = authorize_test(selection, "best", checkpoint)
    assert authorized["selected_run_id"] == "best"
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        authorize_test(selection, "best", checkpoint)


def test_evaluation_result_requires_authorization(tmp_path: Path) -> None:
    result = tmp_path / "evaluation.json"
    payload = {"run_id": "best", "status": "test_finalized", "mse": 0.5}

    with pytest.raises(ValueError, match="authorized"):
        write_evaluation_result(result, payload, authorized=False)
    write_evaluation_result(result, payload, authorized=True)
    assert json.loads(result.read_text(encoding="utf-8"))["mse"] == 0.5
