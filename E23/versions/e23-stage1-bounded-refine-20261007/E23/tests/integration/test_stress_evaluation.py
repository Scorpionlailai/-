from __future__ import annotations

from argparse import Namespace
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

import malr.cli as cli
from malr.config import load_config
from malr.data.masks import generate_variable_missing_mask, save_fixed_mask
from malr.data.splits import protocol_for
from malr.evaluation.evaluator import OnlineEvaluation
from malr.evaluation.metrics import ForecastMetrics
from malr.evaluation.report import file_sha256

from test_training_provenance import _series


@pytest.mark.parametrize(
    ("target_rate", "target_block_len"),
    ((0.15, 8), (0.10, 5)),
)
def test_stress_evaluation_records_original_checkpoint_and_new_fixed_mask(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    target_rate: float,
    target_block_len: int,
) -> None:
    source = load_config(Path("configs/stage1_seq96_dynamic_temporal_repair.yaml"))
    target = replace(
        source,
        missing=replace(
            source.missing, start_rate=target_rate, block_len=target_block_len
        ),
    )
    series = _series(tmp_path)
    protocol = protocol_for("ETTh1", len(series.values))
    source_metadata = cli._base_metadata(source, series, protocol, "ETTh1")
    source_metadata.update(
        {
            "training_stage": "forecast",
            "online_mode": True,
            "validation_mask_sha256": "a" * 64,
        }
    )
    checkpoint = tmp_path / "source-stage2.pt"
    checkpoint.write_bytes(b"fixture checkpoint")
    source_manifest = tmp_path / "source-run.json"
    source_manifest.write_text(
        json.dumps(
            {
                "run_id": "ETTh1-ours-source",
                "dataset": "ETTh1",
                "variant": "ours",
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": file_sha256(checkpoint),
                "metadata": source_metadata,
            }
        ),
        encoding="utf-8",
    )
    artifacts = tmp_path / "target-artifacts"
    support_start, support_end, expected = cli._mask_identity(
        target, "ETTh1", protocol, "validation"
    )
    mask, metadata = generate_variable_missing_mask(
        support_end - support_start,
        7,
        target_rate,
        target_block_len,
        2021,
        dataset="ETTh1",
        split="validation",
        support_start=support_start,
    )
    assert mask.shape == expected.shape
    mask_path = cli._fixed_mask_path(artifacts, "ETTh1", "validation")
    save_fixed_mask(mask_path, mask, metadata)

    def load_context(path: Path, dataset: str):
        assert dataset == "ETTh1"
        return (source if path.name == "source.yaml" else target), series, protocol

    def score(_model, batches, *, device):
        assert device == "cpu"
        first = next(iter(batches))
        assert np.isfinite(first.patch_values.numpy()).all()
        return OnlineEvaluation(
            metrics=ForecastMetrics(mse=0.77, mae=0.55),
            imputation_mse=0.32,
            imputation_mae=0.40,
            diagnostics={},
            inference_ablations={
                "without_temporal": ForecastMetrics(mse=0.81, mae=0.58)
            },
        )

    monkeypatch.setattr(cli, "_load_context", load_context)
    monkeypatch.setattr(cli, "load_online_checkpoint", lambda *_a, **_k: {"online_state": {}})
    monkeypatch.setattr(cli, "_restore_online_model_for_evaluation", lambda *_a, **_k: (object(), {}))
    monkeypatch.setattr(cli, "evaluate_online_model", score)
    args = Namespace(
        config=tmp_path / "target.yaml",
        source_config=tmp_path / "source.yaml",
        source_run_manifest=source_manifest,
        dataset="ETTh1",
        artifacts=artifacts,
        split="validation",
        device="cpu",
    )

    cli._command_stress_evaluate(args)
    progress_output = capsys.readouterr().err
    assert "phase=validation ETTh1 ours zero-shot" in progress_output
    assert "0/" in progress_output

    record_path = (
        artifacts / "stress_evaluations" / "ETTh1-ours-source_validation.json"
    )
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["kind"] == "zero_shot_missing_stress"
    assert record["source_checkpoint_sha256"] == file_sha256(checkpoint)
    assert record["target_mask_sha256"] == file_sha256(mask_path)
    assert record["target_missing_start_rate"] == target_rate
    assert record["target_missing_block_len"] == target_block_len
    assert record["final_missing_rate"] == pytest.approx(metadata.final_rate)
    assert record["mse"] == 0.77
    assert record["mae"] == 0.55
    assert record["inference_ablations"]["without_temporal"]["mse"] == 0.81

    with pytest.raises(ValueError, match="already exists"):
        cli._command_stress_evaluate(args)
