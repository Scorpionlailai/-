import json
from pathlib import Path

import numpy as np
import pytest

import malr.cli as cli
from malr.config import load_config
from malr.data.datasets import CleanSeries
from malr.data.masks import generate_variable_missing_mask, save_fixed_mask
from malr.data.splits import protocol_for
from malr.evaluation.report import file_sha256
from malr.training.checkpoints import load_checkpoint, save_checkpoint
from malr.training.trainer import Trainer, TrainingResult


def _series(tmp_path: Path) -> CleanSeries:
    length = 14400
    time = np.linspace(0.0, 1.0, length, dtype=np.float64)
    values = np.stack([time + variable for variable in range(7)], axis=1)
    source_path = tmp_path / "ETTh1.csv"
    source_path.write_text("fixture source csv\n", encoding="utf-8")
    return CleanSeries(
        values=values,
        columns=tuple(f"v{index}" for index in range(7)),
        timestamps=np.arange(length),
        source_path=source_path,
    )


def _skip_expensive_training(
    self: Trainer, train_batches_for_epoch: object, validation_batches: object
) -> TrainingResult:
    save_checkpoint(
        self.checkpoint_path,
        self.model.state_dict(),
        self.tracker.metadata,
        epoch=0,
        val_mse=0.5,
    )
    return TrainingResult(
        best_checkpoint=self.checkpoint_path,
        best_val_mse=0.5,
        best_epoch=0,
        epochs_completed=1,
        training_seconds=0.0,
        trainable_parameters=sum(parameter.numel() for parameter in self.model.parameters()),
    )


@pytest.mark.parametrize("variant", ["clean", "missing"])
def test_training_persists_pre_test_provenance_for_clean_and_missing_variants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
) -> None:
    config = load_config(Path("configs/stage1.yaml"))
    series = _series(tmp_path)
    protocol = protocol_for("ETTh1", len(series.values))
    artifacts = tmp_path / "artifacts"
    validation_mask_sha256 = None
    validation_final_rate = 0.0
    if variant == "missing":
        support_start, support_end, _ = cli._mask_identity(
            config, "ETTh1", protocol, "validation"
        )
        mask, mask_metadata = generate_variable_missing_mask(
            support_end - support_start,
            series.values.shape[1],
            config.missing.start_rate,
            config.missing.block_len,
            config.training.seed,
            dataset="ETTh1",
            split="validation",
            support_start=support_start,
        )
        mask_path = cli._fixed_mask_path(artifacts, "ETTh1", "validation")
        save_fixed_mask(mask_path, mask, mask_metadata)
        validation_mask_sha256 = file_sha256(mask_path)
        validation_final_rate = mask_metadata.final_rate

    monkeypatch.setattr(Trainer, "fit", _skip_expensive_training)

    result = cli.run_training(
        config,
        series,
        protocol,
        "ETTh1",
        variant,
        artifacts,
        device="cpu",
    )

    manifest = json.loads(Path(result["run_manifest"]).read_text(encoding="utf-8"))
    checkpoint = load_checkpoint(Path(result["best_checkpoint"]))
    expected = {
        "source_csv_sha256": file_sha256(series.source_path),
        "seed": config.training.seed,
        "validation_mask_sha256": validation_mask_sha256,
        "validation_final_rate": validation_final_rate,
    }
    for metadata in (manifest["metadata"], checkpoint["metadata"]):
        assert {key: metadata[key] for key in expected} == expected
        assert metadata["train_start"] == protocol.train.start
        assert metadata["train_end"] == protocol.train.end
        assert metadata["validation_end"] == protocol.validation.end
        assert metadata["test_end"] == protocol.test.end
