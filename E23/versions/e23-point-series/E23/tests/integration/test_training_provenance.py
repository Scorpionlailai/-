import json
from pathlib import Path

import numpy as np
import pytest

import malr.cli as cli
from malr.config import config_sha256, load_config
from malr.data.datasets import CleanSeries
from malr.data.masks import generate_variable_missing_mask, save_fixed_mask
from malr.data.splits import protocol_for
from malr.evaluation.report import file_sha256
from malr.prototypes.bank import PrototypeBank, save_bank, source_checksum
from malr.training.checkpoints import load_checkpoint, save_checkpoint
from malr.training.trainer import Trainer, TrainingResult


FILTER_DIAGNOSTICS = {
    "temporal_keep_rate": 0.5,
    "temporal_zero_edge_rate": 0.25,
    "variable_keep_rate": 0.4,
    "variable_zero_edge_rate": 0.1,
    "historical_keep_rate": 0.2,
    "historical_zero_edge_rate": 0.3,
}


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
    recovery = getattr(self.model, "recovery", None)
    filtered = recovery is not None and recovery.edge_filter is not None
    diagnostics = dict(FILTER_DIAGNOSTICS) if filtered else {}
    save_checkpoint(
        self.checkpoint_path,
        self.model.state_dict(),
        self.tracker.metadata,
        epoch=0,
        val_mse=0.5,
        val_mae=0.4,
        validation_diagnostics=diagnostics,
    )
    return TrainingResult(
        best_checkpoint=self.checkpoint_path,
        best_val_mse=0.5,
        best_val_mae=0.4,
        best_val_diagnostics=diagnostics,
        best_epoch=0,
        epochs_completed=1,
        training_seconds=0.0,
        trainable_parameters=sum(parameter.numel() for parameter in self.model.parameters()),
    )


def test_training_reports_train_and_validation_batch_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = load_config(Path("configs/stage1.yaml"))
    series = _series(tmp_path)
    protocol = protocol_for("ETTh1", len(series.values))

    def consume_one_batch(self, train_batches_for_epoch, validation_batches):
        next(iter(train_batches_for_epoch(0)))
        next(iter(validation_batches()))
        return _skip_expensive_training(self, train_batches_for_epoch, validation_batches)

    monkeypatch.setattr(Trainer, "fit", consume_one_batch)
    cli.run_training(config, series, protocol, "ETTh1", "clean", tmp_path / "artifacts", device="cpu")

    progress_output = capsys.readouterr().err
    assert "phase=train ETTh1 clean epoch=1/" in progress_output
    assert "phase=validation ETTh1 clean epoch=1/" in progress_output


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
    assert checkpoint["validation_diagnostics"] == {}
    assert manifest["training"]["best_val_diagnostics"] == {}
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


def test_filtered_ours_persists_best_validation_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(
        Path("configs/stage1_sr010_replacement_filter_no_recloss.yaml")
    )
    series = _series(tmp_path)
    protocol = protocol_for("ETTh1", len(series.values))
    artifacts = tmp_path / "filtered-artifacts"
    support_start, support_end, _ = cli._mask_identity(
        config, "ETTh1", protocol, "validation"
    )
    mask, metadata = generate_variable_missing_mask(
        support_end - support_start,
        series.values.shape[1],
        config.missing.start_rate,
        config.missing.block_len,
        config.training.seed,
        dataset="ETTh1",
        split="validation",
        support_start=support_start,
    )
    save_fixed_mask(
        cli._fixed_mask_path(artifacts, "ETTh1", "validation"),
        mask,
        metadata,
    )
    standardized, _ = cli._standardized_context(series, protocol)
    bank = PrototypeBank(
        dataset="ETTh1",
        train_end=protocol.train.end,
        source_checksum=source_checksum(standardized[: protocol.train.end]),
        config_sha256=config_sha256(config),
        records=tuple(() for _ in range(series.values.shape[1])),
        factor_mean=np.zeros((series.values.shape[1], 7)),
        factor_std=np.ones((series.values.shape[1], 7)),
    )
    bank_path = tmp_path / "bank.npz"
    save_bank(bank, bank_path)
    monkeypatch.setattr(Trainer, "fit", _skip_expensive_training)

    result = cli.run_training(
        config,
        series,
        protocol,
        "ETTh1",
        "ours",
        artifacts,
        device="cpu",
        bank_path=bank_path,
    )

    checkpoint = load_checkpoint(Path(result["best_checkpoint"]))
    manifest = json.loads(Path(result["run_manifest"]).read_text(encoding="utf-8"))
    assert checkpoint["validation_diagnostics"] == FILTER_DIAGNOSTICS
    assert manifest["training"]["best_val_diagnostics"] == FILTER_DIAGNOSTICS
