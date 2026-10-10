from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import malr.cli as cli
from malr.config import config_sha256, load_config
from malr.data.masks import generate_variable_missing_mask, save_fixed_mask
from malr.data.repaired import RepairedSplit, save_repaired_split
from malr.data.splits import protocol_for
from malr.evaluation.evaluator import OnlineEvaluation
from malr.evaluation.metrics import ForecastMetrics
from malr.evaluation.report import file_sha256, write_json_atomic
from malr.models.online_stage1 import OnlineStage1Model
from malr.models.online_stage1_v2 import RelationV2OnlineStage1Model
from malr.training.checkpoints import load_checkpoint, save_checkpoint
from malr.training.online_checkpoint import save_online_checkpoint
from malr.training.online_trainer import OnlineTrainingResult
from malr.training.trainer import TrainingResult

from test_training_provenance import _series


_GATE_PROVENANCE = {
    "dataset": "ETTh1",
    "config_hash": "a" * 64,
    "source_csv_sha256": "b" * 64,
    "seed": 2021,
    "validation_mask_sha256": "c" * 64,
    "validation_final_rate": 0.10,
    "train_start": 0,
    "train_end": 8640,
    "validation_end": 11520,
    "test_end": 14400,
    "seq_len": 96,
    "pred_len": 96,
    "metric_space": "train_standardized",
}


def _trusted_e20_metadata() -> dict[str, object]:
    return {
        **_GATE_PROVENANCE,
        "stage2_route_teacher": "frozen_stage1",
        "stage2_route_teacher_checksum_verified": True,
        "stage2_route_teacher_invariance_verified": True,
        "stage2_validation_diagnostics": {
            "route_repair_regret": 0.125,
            "route_repair_regret_positive_rate": 0.25,
        },
        "stage2_validation_inference_ablations": {
            name: {"mse": 0.8, "mae": 0.7}
            for name in (
                "without_historical",
                "without_temporal",
                "without_variable",
                "historical_only",
            )
        },
    }


def _trusted_e21_metadata() -> dict[str, object]:
    metadata = _trusted_e20_metadata()
    metadata.update(
        {
            "stage2_route_teacher": "frozen_stage1_hv_fusion",
            "stage2_forecast_fusion": "learned_hv_none_from_forecast_mse",
            "stage2_temporal_forecast_weight": 0,
        }
    )
    return metadata


def _validation_gate_manifest(
    metadata: dict[str, object], *, mse: float, mae: float
) -> dict[str, object]:
    return {
        "metadata": metadata,
        "training": {"best_val_mse": mse, "best_val_mae": mae},
    }


def _online_evaluation_artifacts(
    tmp_path: Path,
    monkeypatch,
    *,
    frozen_teacher: bool,
    hv_fusion: bool = False,
) -> tuple[SimpleNamespace, dict[str, object], list[RelationV2OnlineStage1Model]]:
    """Create a complete online checkpoint while mocking only forecast scoring."""

    config = load_config(Path("configs/stage1_seq96_relational_recovery.yaml"))
    series = _series(tmp_path)
    protocol = protocol_for("ETTh1", len(series.values))
    artifacts = tmp_path / "artifacts"
    _, _, expected = cli._mask_identity(config, "ETTh1", protocol, "test")
    test_mask, test_metadata = generate_variable_missing_mask(
        expected.shape[0],
        expected.shape[1],
        expected.start_rate,
        expected.block_len,
        expected.seed,
        dataset="ETTh1",
        split="test",
        support_start=expected.support_start,
    )
    save_fixed_mask(
        cli._fixed_mask_path(artifacts, "ETTh1", "test"),
        test_mask,
        test_metadata,
    )
    metadata = cli._base_metadata(config, series, protocol, "ETTh1")
    metadata.update(
        {
            "variant": "ours",
            "online_mode": True,
            "training_stage": "forecast",
            "stage2_loss": "forecast_mse_only",
        }
    )
    model = cli.build_online_experiment_model(config, "ETTh1", protocol)
    assert isinstance(model, RelationV2OnlineStage1Model)
    source_checkpoint = tmp_path / "stage1-source.pt"
    source_checkpoint.write_bytes(b"stage-1 source")
    if hv_fusion and not frozen_teacher:
        raise ValueError("hv_fusion requires frozen_teacher")
    if frozen_teacher:
        teacher_mode = (
            "frozen_stage1_hv_fusion" if hv_fusion else "frozen_stage1"
        )
        metadata.update(
            {
                "stage2_route_teacher": teacher_mode,
                "stage2_route_teacher_sha256": model.install_frozen_route_teacher(),
                "stage2_route_formula": (
                    "argmax_positive_teacher_branch_gain_else_none"
                ),
                "stage2_route_teacher_gradient": "disabled",
                "stage1_checkpoint": str(source_checkpoint),
                "stage1_checkpoint_sha256": file_sha256(source_checkpoint),
                "stage2_route_teacher_checksum_verified": True,
                "stage2_route_teacher_invariance_verified": True,
                "stage2_validation_diagnostics": {
                    "route_repair_regret": 0.0,
                    "route_repair_regret_positive_rate": 0.0,
                },
                "stage2_validation_inference_ablations": {
                    name: {"mse": 0.5, "mae": 0.4}
                    for name in (
                        "without_historical",
                        "without_temporal",
                        "without_variable",
                        "historical_only",
                    )
                },
            }
        )
        if hv_fusion:
            model.install_forecast_hv_fusion()
            model.configure_forecast_finetuning(
                frozen_route_teacher=True,
                forecast_fusion="teacher_hv",
            )
            metadata.update(
                {
                    "stage2_forecast_fusion": "learned_hv_none_from_forecast_mse",
                    "stage2_temporal_forecast_weight": 0,
                }
            )
    state = model.online_state()
    checkpoint_path = tmp_path / (
        "e21.pt" if hv_fusion else "e20.pt" if frozen_teacher else "e19.pt"
    )
    save_online_checkpoint(
        checkpoint_path,
        model.state_dict(),
        state["memory_state"],
        state["ema_state"],
        state["relation_prior_state"],
        metadata,
        epoch=0,
        config_hash=str(metadata["config_hash"]),
        data_hash=str(metadata["train_checksum"]),
        mask_hash=file_sha256(cli._fixed_mask_path(artifacts, "ETTh1", "test")),
        memory_hash=str(state["memory_hash"]),
    )
    manifest: dict[str, object] = {
        "format_version": 1,
        "run_id": (
            "e21-roundtrip"
            if hv_fusion
            else "e20-roundtrip"
            if frozen_teacher
            else "e19-roundtrip"
        ),
        "dataset": "ETTh1",
        "variant": "ours",
        "online_mode": True,
        "training_pipeline": "supervised_relation_finetune",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "metadata": metadata,
        "training": {"trainable_parameters": 1, "training_seconds": 0.0},
        "ablations": asdict(config.ablations),
    }
    restored_models: list[RelationV2OnlineStage1Model] = []

    def fake_evaluate_online(model, _batches, *, device):
        assert device == "cpu"
        assert isinstance(model, RelationV2OnlineStage1Model)
        next(iter(_batches))
        restored_models.append(model)
        return OnlineEvaluation(
            metrics=ForecastMetrics(mse=0.5, mae=0.4),
            imputation_mse=0.0,
            imputation_mae=0.0,
            diagnostics={},
            inference_ablations={},
        )

    monkeypatch.setattr(cli, "_load_context", lambda _path, _dataset: (config, series, protocol))
    monkeypatch.setattr(cli, "evaluate_online_model", fake_evaluate_online)
    args = SimpleNamespace(
        artifacts=artifacts,
        config=tmp_path / "config.yaml",
        dataset="ETTh1",
        device="cpu",
    )
    return args, manifest, restored_models


def test_online_config_builds_ours_without_bank_or_clean_checkpoint(tmp_path: Path, monkeypatch) -> None:
    config = load_config(Path("configs/stage1_seq96_online_memory.yaml"))
    series = _series(tmp_path)
    protocol = protocol_for("ETTh1", len(series.values))
    artifacts = tmp_path / "artifacts"
    support_start, support_end, _ = cli._mask_identity(config, "ETTh1", protocol, "validation")
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
    save_fixed_mask(cli._fixed_mask_path(artifacts, "ETTh1", "validation"), mask, metadata)

    def fake_fit(self, _train, _validation, *, config_hash, data_hash, mask_hash):
        state = self.model.online_state()
        save_online_checkpoint(
            self.checkpoint_path,
            self.model.state_dict(),
            state["memory_state"],
            state["ema_state"],
            state["relation_prior_state"],
            self.metadata,
            epoch=0,
            config_hash=config_hash,
            data_hash=data_hash,
            mask_hash=mask_hash,
            memory_hash=state["memory_hash"],
        )
        return OnlineTrainingResult(self.checkpoint_path, 1.0, 1.0, 0, 1, 0.0, 1)

    monkeypatch.setattr(cli.OnlineTrainer, "fit", fake_fit)
    result = cli.run_training(
        config,
        series,
        protocol,
        "ETTh1",
        "ours",
        artifacts,
        device="cpu",
    )
    manifest = json.loads(Path(result["run_manifest"]).read_text(encoding="utf-8"))
    assert manifest["online_mode"] is True
    assert manifest["bank"] is None
    assert manifest["clean_checkpoint"] is None


def test_cli_builds_protocol_selected_online_model() -> None:
    """The config protocol selects its matching online relation implementation."""

    legacy = load_config(Path("configs/stage1_seq96_online_memory_bl8_legacy.yaml"))
    v2 = load_config(Path("configs/stage1_seq96_relational_recovery.yaml"))
    protocol = protocol_for("ETTh1", 14400)

    assert isinstance(cli.build_online_experiment_model(legacy, "ETTh1", protocol), OnlineStage1Model)
    assert isinstance(
        cli.build_online_experiment_model(v2, "ETTh1", protocol),
        RelationV2OnlineStage1Model,
    )


def test_train_parser_accepts_the_explicit_frozen_stage1_teacher_mode() -> None:
    args = cli.build_parser().parse_args(
        [
            "train",
            "--dataset",
            "ETTh1",
            "--variant",
            "ours",
            "--stage",
            "forecast",
            "--init-checkpoint",
            "stage1.pt",
            "--stage2-route-teacher",
            "frozen_stage1",
        ]
    )

    assert args.stage2_route_teacher == "frozen_stage1"


def test_train_parser_accepts_the_e21_hv_forecast_fusion_mode() -> None:
    args = cli.build_parser().parse_args(
        [
            "train",
            "--dataset",
            "ETTh1",
            "--variant",
            "ours",
            "--stage",
            "forecast",
            "--init-checkpoint",
            "stage1.pt",
            "--stage2-route-teacher",
            "frozen_stage1_hv_fusion",
        ]
    )

    assert args.stage2_route_teacher == "frozen_stage1_hv_fusion"


def test_train_parser_accepts_stage1_runtime_speed_options() -> None:
    args = cli.build_parser().parse_args(
        [
            "train", "--dataset", "Weather", "--variant", "ours",
            "--stage", "repair", "--repair-batch-size", "24",
            "--repair-matmul-precision", "medium",
        ]
    )

    assert args.repair_batch_size == 24
    assert args.repair_matmul_precision == "medium"


def test_stage1_runtime_options_change_only_training_batches_and_restore_precision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(Path("configs/stage1_seq96_relational_recovery.yaml"))
    series = _series(tmp_path)
    protocol = protocol_for("ETTh1", len(series.values))
    artifacts = tmp_path / "artifacts"
    support_start, support_end, _ = cli._mask_identity(config, "ETTh1", protocol, "validation")
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
    save_fixed_mask(cli._fixed_mask_path(artifacts, "ETTh1", "validation"), mask, mask_metadata)
    original_precision = torch.get_float32_matmul_precision()
    expected_validation_batch = config.training.for_dataset("ETTh1").batch_size

    def fake_fit(self, train_batches, validation_batches, **kwargs):
        assert torch.get_float32_matmul_precision() == "medium"
        assert next(train_batches(1)).forecast_target.shape[0] == 24
        assert next(validation_batches()).forecast_target.shape[0] == expected_validation_batch
        state = self.model.online_state()
        save_online_checkpoint(
            self.checkpoint_path,
            self.model.state_dict(),
            state["memory_state"],
            state["ema_state"],
            state["relation_prior_state"],
            self.metadata,
            epoch=1,
            config_hash=kwargs["config_hash"],
            data_hash=kwargs["data_hash"],
            mask_hash=kwargs["mask_hash"],
            memory_hash=state["memory_hash"],
        )
        return OnlineTrainingResult(self.checkpoint_path, 1.0, 1.0, 1, 1, 0.0, 1)

    monkeypatch.setattr(cli.OnlineTrainer, "fit", fake_fit)
    result = cli.run_training(
        config, series, protocol, "ETTh1", "ours", artifacts,
        device="cpu", stage="repair", repair_batch_size=24,
        repair_matmul_precision="medium",
    )

    assert torch.get_float32_matmul_precision() == original_precision
    manifest = json.loads(Path(result["run_manifest"]).read_text(encoding="utf-8"))
    assert manifest["metadata"]["config_hash"] == config_sha256(config)
    assert manifest["metadata"]["stage1_runtime"] == {
        "train_batch_size": 24,
        "validation_batch_size": expected_validation_batch,
        "matmul_precision": "medium",
    }


def test_stage1_runtime_options_reject_forecast_stage_before_loading_checkpoint(
    tmp_path: Path,
) -> None:
    config = load_config(Path("configs/stage1_seq96_relational_recovery.yaml"))
    series = _series(tmp_path)
    protocol = protocol_for("ETTh1", len(series.values))

    with pytest.raises(ValueError, match="only supported for ours --stage repair"):
        cli.run_training(
            config, series, protocol, "ETTh1", "ours", tmp_path / "artifacts",
            device="cpu", stage="forecast", init_checkpoint_path=Path("missing.pt"),
            repair_batch_size=24,
        )


@pytest.mark.parametrize(
    ("config_path", "variant", "stage", "init_checkpoint", "message"),
    (
        (
            "configs/stage1_seq96_relational_recovery.yaml",
            "clean",
            "forecast",
            Path("unused-stage1.pt"),
            "requires relation-v2 ours --stage forecast",
        ),
        (
            "configs/stage1_seq96_relational_recovery.yaml",
            "missing",
            "forecast",
            Path("unused-stage1.pt"),
            "requires relation-v2 ours --stage forecast",
        ),
        (
            "configs/stage1_seq96_relational_recovery.yaml",
            "ours",
            "repair",
            None,
            "requires relation-v2 ours --stage forecast",
        ),
        (
            "configs/stage1_seq96_online_memory_bl8_legacy.yaml",
            "ours",
            "forecast",
            Path("unused-stage1.pt"),
            "requires relation-v2 ours --stage forecast",
        ),
        (
            "configs/stage1_seq96_relational_recovery.yaml",
            "ours",
            "forecast",
            None,
            "requires --init-checkpoint",
        ),
    ),
)
def test_frozen_stage1_teacher_rejects_invalid_training_modes_before_artifact_access(
    tmp_path: Path,
    config_path: str,
    variant: str,
    stage: str,
    init_checkpoint: Path | None,
    message: str,
) -> None:
    config = load_config(Path(config_path))
    series = _series(tmp_path)
    protocol = protocol_for("ETTh1", len(series.values))

    with pytest.raises(ValueError, match=message):
        cli.run_training(
            config,
            series,
            protocol,
            "ETTh1",
            variant,
            tmp_path / "artifacts",
            device="cpu",
            stage=stage,
            init_checkpoint_path=init_checkpoint,
            stage2_route_teacher="frozen_stage1",
        )


@pytest.mark.parametrize(
    "stage2_teacher_mode",
    ("frozen_stage1", "frozen_stage1_hv_fusion"),
)
def test_v2_cli_loads_repair_checkpoint_for_supervised_forecast_finetuning(
    tmp_path: Path, monkeypatch, stage2_teacher_mode: str
) -> None:
    config = load_config(Path("configs/stage1_seq96_relational_recovery.yaml"))
    series = _series(tmp_path)
    protocol = protocol_for("ETTh1", len(series.values))
    artifacts = tmp_path / "artifacts"
    support_start, support_end, _ = cli._mask_identity(config, "ETTh1", protocol, "validation")
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
    save_fixed_mask(cli._fixed_mask_path(artifacts, "ETTh1", "validation"), mask, metadata)
    calls: list[dict[str, object]] = []
    stage1_checkpoint: list[Path] = []
    validation_evaluations: list[str] = []

    def fake_validation_evaluation(model, batches, *, device):
        assert device == "cpu"
        assert isinstance(model, RelationV2OnlineStage1Model)
        first_batch = next(iter(batches))
        assert first_batch.forecast_target.shape[-1] == config.data.pred_len
        validation_evaluations.append("validation")
        return OnlineEvaluation(
            metrics=ForecastMetrics(mse=0.81, mae=0.9),
            imputation_mse=0.2,
            imputation_mae=0.1,
            diagnostics={
                "route_repair_regret": np.float64(0.125),
                "route_repair_regret_positive_rate": np.float32(0.25),
            },
            inference_ablations={
                "without_historical": ForecastMetrics(mse=0.82, mae=0.91),
                "without_temporal": ForecastMetrics(mse=0.83, mae=0.92),
                "without_variable": ForecastMetrics(mse=0.84, mae=0.93),
                "historical_only": ForecastMetrics(mse=0.85, mae=0.94),
            },
        )

    def fake_fit(self, _train, _validation, **kwargs):
        calls.append(kwargs)
        if kwargs.get("objective") == "forecast":
            assert self.model.has_frozen_route_teacher
            assert self.model.route_teacher is not None
            assert not self.model.route_teacher.training
            assert all(not parameter.requires_grad for parameter in self.model.route_teacher.parameters())
            assert all(
                parameter.requires_grad
                for parameter in self.model.historical_operator.message_head.parameters()
            )
            if stage2_teacher_mode == "frozen_stage1_hv_fusion":
                assert self.model.has_forecast_hv_fusion
                assert self.model.forecast_fusion is not None
                assert all(
                    parameter.requires_grad
                    for parameter in self.model.forecast_fusion.parameters()
                )
                assert all(
                    not parameter.requires_grad
                    for parameter in self.model.temporal_operator.candidate_projector.parameters()
                )
        state = self.model.online_state()
        checkpoint_epoch = 17 if kwargs.get("objective") == "repair" else state["epoch"]
        save_online_checkpoint(
            self.checkpoint_path,
            self.model.state_dict(),
            state["memory_state"],
            state["ema_state"],
            state["relation_prior_state"],
            self.metadata,
            epoch=checkpoint_epoch,
            config_hash=kwargs["config_hash"],
            data_hash=kwargs["data_hash"],
            mask_hash=kwargs["mask_hash"],
            memory_hash=state["memory_hash"],
        )
        if kwargs.get("objective") == "forecast":
            with stage1_checkpoint[0].open("ab") as stream:
                stream.write(b"changed after Stage-2 initialization")
        # Simulate a repair stage whose best paired checkpoint is earlier than
        # the configured stage length.  Stage 2 must continue from that
        # restored online epoch, not from the nominal repair epoch count.
        best_epoch = 17 if kwargs.get("objective") == "repair" else 1
        return OnlineTrainingResult(self.checkpoint_path, 1.0, 1.0, best_epoch, 1, 0.0, 1)

    monkeypatch.setattr(cli.OnlineTrainer, "fit", fake_fit)
    monkeypatch.setattr(cli, "evaluate_online_model", fake_validation_evaluation)
    repair_result = cli.run_training(
        config,
        series,
        protocol,
        "ETTh1",
        "ours",
        artifacts,
        device="cpu",
        stage="repair",
    )

    assert [call["objective"] for call in calls] == ["repair"]
    assert repair_result["training_stage"] == "repair"
    assert Path(repair_result["best_checkpoint"]).name == "ETTh1_ours_stage1.pt"
    stage1_checkpoint.append(Path(repair_result["best_checkpoint"]))

    result = cli.run_training(
        config,
        series,
        protocol,
        "ETTh1",
        "ours",
        artifacts,
        device="cpu",
        stage="forecast",
        init_checkpoint_path=Path(repair_result["best_checkpoint"]),
        stage2_route_teacher=stage2_teacher_mode,
    )

    assert [call["objective"] for call in calls] == ["repair", "forecast"]
    assert calls[1]["validation_objective"] == "forecast"
    assert calls[1]["epoch_offset"] == 0
    assert calls[1]["model_epoch_offset"] == 17
    assert calls[1]["freeze_online_state"] is True
    assert result["training_stage"] == "forecast"
    assert Path(result["best_checkpoint"]).name == "ETTh1_ours_stage2.pt"
    manifest = json.loads(Path(result["run_manifest"]).read_text(encoding="utf-8"))
    assert manifest["training_pipeline"] == "supervised_relation_finetune"
    assert manifest["metadata"]["stage1_parameters_loaded"] is True
    assert manifest["metadata"]["stage2_loss"] == "forecast_mse_only"
    assert manifest["metadata"]["stage2_memory_updates"] is False
    assert manifest["metadata"]["stage2_route_teacher"] == stage2_teacher_mode
    assert len(manifest["metadata"]["stage2_route_teacher_sha256"]) == 64
    assert manifest["metadata"]["stage2_route_formula"] == (
        "argmax_positive_teacher_branch_gain_else_none"
    )
    assert manifest["metadata"]["stage2_route_teacher_gradient"] == "disabled"
    assert manifest["metadata"]["stage2_route_teacher_checksum_verified"] is True
    assert manifest["metadata"]["stage2_route_teacher_invariance_verified"] is True
    assert manifest["metadata"]["stage2_validation_diagnostics"] == {
        "route_repair_regret": 0.125,
        "route_repair_regret_positive_rate": 0.25,
    }
    assert set(manifest["metadata"]["stage2_validation_inference_ablations"]) == {
        "without_historical",
        "without_temporal",
        "without_variable",
        "historical_only",
    }
    assert validation_evaluations == ["validation"]
    assert "frozen_stage1_route_teacher" in manifest["metadata"]["stage2_frozen"]
    if stage2_teacher_mode == "frozen_stage1_hv_fusion":
        assert manifest["metadata"]["stage2_forecast_fusion"] == (
            "learned_hv_none_from_forecast_mse"
        )
        assert manifest["metadata"]["stage2_temporal_forecast_weight"] == 0
        assert manifest["metadata"][
            "stage2_forecast_fusion_trainable_parameters"
        ] > 0
        assert manifest["metadata"][
            "stage2_forecast_fusion_frozen_parameters"
        ] == 0
        assert "forecast_fusion" in manifest["metadata"]["stage2_updates"]
        assert "temporal_student" in manifest["metadata"]["stage2_frozen"]
    assert manifest["stage1_checkpoint_sha256"] == manifest["metadata"][
        "stage1_checkpoint_sha256"
    ]


def test_e20_validation_gate_uses_fixed_mse_threshold_and_reports_mae_only() -> None:
    missing = _validation_gate_manifest(
        dict(_GATE_PROVENANCE), mse=0.40, mae=0.30
    )

    with pytest.raises(ValueError, match="best_val_mse=0.85.*0.8150254474"):
        cli._validate_ours_evaluation_gate(
            _validation_gate_manifest(_trusted_e20_metadata(), mse=0.85, mae=0.10),
            missing,
        )

    cli._validate_ours_evaluation_gate(
        _validation_gate_manifest(_trusted_e20_metadata(), mse=0.81, mae=0.90),
        missing,
    )


def test_e21_validation_gate_reuses_the_fixed_e20_mse_threshold() -> None:
    missing = _validation_gate_manifest(
        dict(_GATE_PROVENANCE), mse=0.40, mae=0.30
    )

    with pytest.raises(ValueError, match="best_val_mse=0.85.*0.8150254474"):
        cli._validate_ours_evaluation_gate(
            _validation_gate_manifest(_trusted_e21_metadata(), mse=0.85, mae=0.10),
            missing,
        )

    cli._validate_ours_evaluation_gate(
        _validation_gate_manifest(_trusted_e21_metadata(), mse=0.81, mae=0.90),
        missing,
    )


def test_higher_missing_rate_uses_same_protocol_missing_validation_gate() -> None:
    missing_metadata = {**_GATE_PROVENANCE, "missing_start_rate": 0.15, "missing_block_len": 8}
    ours_metadata = {**_trusted_e21_metadata(), "missing_start_rate": 0.15, "missing_block_len": 8}
    missing = _validation_gate_manifest(missing_metadata, mse=1.20, mae=0.90)
    ours = _validation_gate_manifest(ours_metadata, mse=1.10, mae=0.80)

    cli._validate_ours_evaluation_gate(ours, missing)

    worse = _validation_gate_manifest(ours_metadata, mse=1.30, mae=0.80)
    with pytest.raises(ValueError, match="missing best_val_mse=1.2"):
        cli._validate_ours_evaluation_gate(worse, missing)


@pytest.mark.parametrize(
    "best_val_mse",
    (False, "0.81", None, float("nan"), float("inf"), float("-inf")),
)
def test_e20_validation_gate_rejects_non_real_or_non_finite_mse(
    best_val_mse: object,
) -> None:
    missing = _validation_gate_manifest(
        dict(_GATE_PROVENANCE), mse=0.40, mae=0.30
    )
    ours = _validation_gate_manifest(
        _trusted_e20_metadata(), mse=0.81, mae=0.90
    )
    ours["training"]["best_val_mse"] = best_val_mse

    with pytest.raises(ValueError, match="real finite best_val_mse"):
        cli._validate_ours_evaluation_gate(ours, missing)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda metadata: metadata.pop("stage2_route_teacher_checksum_verified"), "checksum_verified"),
        (lambda metadata: metadata.update(stage2_route_teacher_checksum_verified=False), "checksum_verified"),
        (lambda metadata: metadata.pop("stage2_route_teacher_invariance_verified"), "invariance_verified"),
        (lambda metadata: metadata.update(stage2_route_teacher_invariance_verified=False), "invariance_verified"),
        (lambda metadata: metadata.pop("stage2_validation_diagnostics"), "stage2_validation_diagnostics"),
        (
            lambda metadata: metadata["stage2_validation_diagnostics"].pop(
                "route_repair_regret"
            ),
            "route_repair_regret",
        ),
        (
            lambda metadata: metadata["stage2_validation_inference_ablations"].pop(
                "without_historical"
            ),
            "without_historical",
        ),
    ),
)
def test_e20_validation_gate_rejects_untrusted_post_training_metadata(
    mutation, message: str
) -> None:
    metadata = _trusted_e20_metadata()
    mutation(metadata)
    missing = _validation_gate_manifest(
        dict(_GATE_PROVENANCE), mse=0.40, mae=0.30
    )

    with pytest.raises(ValueError, match=message):
        cli._validate_ours_evaluation_gate(
            _validation_gate_manifest(metadata, mse=0.81, mae=0.90),
            missing,
        )


def test_evaluation_reconstructs_and_validates_the_serialized_route_teacher(
    tmp_path: Path,
) -> None:
    config = load_config(Path("configs/stage1_seq96_relational_recovery.yaml"))
    protocol = protocol_for("ETTh1", 14400)
    source = cli.build_online_experiment_model(config, "ETTh1", protocol)
    assert isinstance(source, RelationV2OnlineStage1Model)
    teacher_sha256 = source.install_frozen_route_teacher()
    checkpoint = {
        "model_state": source.state_dict(),
        "online_state": source.online_state(),
    }
    source_checkpoint = tmp_path / "stage1.pt"
    source_checkpoint.write_bytes(b"stage-1 provenance")
    manifest = {
        "metadata": {
            "stage2_route_teacher": "frozen_stage1",
            "stage2_route_teacher_sha256": teacher_sha256,
            "stage2_route_formula": "argmax_positive_teacher_branch_gain_else_none",
            "stage2_route_teacher_gradient": "disabled",
            "stage2_loss": "forecast_mse_only",
            "stage1_checkpoint": str(source_checkpoint),
            "stage1_checkpoint_sha256": file_sha256(source_checkpoint),
        }
    }

    restored, _ = cli._restore_online_model_for_evaluation(
        config,
        "ETTh1",
        protocol,
        checkpoint,
        manifest,
        device="cpu",
    )

    assert isinstance(restored, RelationV2OnlineStage1Model)
    assert restored.route_teacher is not None
    assert restored.route_teacher.state_sha256() == teacher_sha256
    assert not restored.route_teacher.training
    assert all(not parameter.requires_grad for parameter in restored.route_teacher.parameters())

    corrupted_state = {name: value.clone() for name, value in checkpoint["model_state"].items()}
    teacher_key = next(name for name in corrupted_state if name.startswith("route_teacher."))
    corrupted_state[teacher_key].view(-1)[0] += 1.0
    with pytest.raises(ValueError, match="frozen route teacher checksum mismatch"):
        cli._restore_online_model_for_evaluation(
            config,
            "ETTh1",
            protocol,
            {**checkpoint, "model_state": corrupted_state},
            manifest,
            device="cpu",
        )


def test_evaluation_reconstructs_teacher_and_forecast_fusion_strictly(
    tmp_path: Path,
) -> None:
    config = load_config(Path("configs/stage1_seq96_relational_recovery.yaml"))
    protocol = protocol_for("ETTh1", 14400)
    source = cli.build_online_experiment_model(config, "ETTh1", protocol)
    assert isinstance(source, RelationV2OnlineStage1Model)
    teacher_sha256 = source.install_frozen_route_teacher()
    source.install_forecast_hv_fusion()
    source.configure_forecast_finetuning(
        frozen_route_teacher=True,
        forecast_fusion="teacher_hv",
    )
    checkpoint = {
        "model_state": source.state_dict(),
        "online_state": source.online_state(),
    }
    assert any(name.startswith("forecast_fusion.") for name in checkpoint["model_state"])
    source_checkpoint = tmp_path / "stage1.pt"
    source_checkpoint.write_bytes(b"stage-1 provenance")
    manifest = {
        "metadata": {
            "stage2_route_teacher": "frozen_stage1_hv_fusion",
            "stage2_route_teacher_sha256": teacher_sha256,
            "stage2_route_formula": "argmax_positive_teacher_branch_gain_else_none",
            "stage2_route_teacher_gradient": "disabled",
            "stage2_loss": "forecast_mse_only",
            "stage2_forecast_fusion": "learned_hv_none_from_forecast_mse",
            "stage2_temporal_forecast_weight": 0,
            "stage1_checkpoint": str(source_checkpoint),
            "stage1_checkpoint_sha256": file_sha256(source_checkpoint),
        }
    }

    restored, _ = cli._restore_online_model_for_evaluation(
        config,
        "ETTh1",
        protocol,
        checkpoint,
        manifest,
        device="cpu",
    )

    assert isinstance(restored, RelationV2OnlineStage1Model)
    assert restored.has_frozen_route_teacher
    assert restored.has_forecast_hv_fusion
    assert restored.route_teacher is not None
    assert restored.route_teacher.state_sha256() == teacher_sha256
    assert restored._forecast_fusion_mode == "teacher_hv"


def test_evaluation_rejects_e21_checkpoint_without_forecast_fusion_state(
    tmp_path: Path,
) -> None:
    config = load_config(Path("configs/stage1_seq96_relational_recovery.yaml"))
    protocol = protocol_for("ETTh1", 14400)
    source = cli.build_online_experiment_model(config, "ETTh1", protocol)
    assert isinstance(source, RelationV2OnlineStage1Model)
    teacher_sha256 = source.install_frozen_route_teacher()
    source_checkpoint = tmp_path / "stage1.pt"
    source_checkpoint.write_bytes(b"stage-1 provenance")
    manifest = {
        "metadata": {
            "stage2_route_teacher": "frozen_stage1_hv_fusion",
            "stage2_route_teacher_sha256": teacher_sha256,
            "stage2_route_formula": "argmax_positive_teacher_branch_gain_else_none",
            "stage2_route_teacher_gradient": "disabled",
            "stage2_loss": "forecast_mse_only",
            "stage2_forecast_fusion": "learned_hv_none_from_forecast_mse",
            "stage2_temporal_forecast_weight": 0,
            "stage1_checkpoint": str(source_checkpoint),
            "stage1_checkpoint_sha256": file_sha256(source_checkpoint),
        }
    }

    with pytest.raises(
        ValueError,
        match="frozen route teacher metadata contract mismatch: forecast_fusion_state",
    ):
        cli._restore_online_model_for_evaluation(
            config,
            "ETTh1",
            protocol,
            {
                "model_state": source.state_dict(),
                "online_state": source.online_state(),
            },
            manifest,
            device="cpu",
        )


def test_evaluation_keeps_legacy_online_checkpoints_teacher_free() -> None:
    config = load_config(Path("configs/stage1_seq96_relational_recovery.yaml"))
    protocol = protocol_for("ETTh1", 14400)
    source = cli.build_online_experiment_model(config, "ETTh1", protocol)
    assert isinstance(source, RelationV2OnlineStage1Model)

    restored, _ = cli._restore_online_model_for_evaluation(
        config,
        "ETTh1",
        protocol,
        {"model_state": source.state_dict(), "online_state": source.online_state()},
        {"metadata": {}},
        device="cpu",
    )

    assert isinstance(restored, RelationV2OnlineStage1Model)
    assert not restored.has_frozen_route_teacher


def test_evaluation_rejects_a_moved_source_checkpoint_only_when_present_and_changed(
    tmp_path: Path,
) -> None:
    config = load_config(Path("configs/stage1_seq96_relational_recovery.yaml"))
    protocol = protocol_for("ETTh1", 14400)
    source = cli.build_online_experiment_model(config, "ETTh1", protocol)
    assert isinstance(source, RelationV2OnlineStage1Model)
    teacher_sha256 = source.install_frozen_route_teacher()
    checkpoint = {"model_state": source.state_dict(), "online_state": source.online_state()}
    moved_source = tmp_path / "moved-stage1.pt"
    manifest = {
        "metadata": {
            "stage2_route_teacher": "frozen_stage1",
            "stage2_route_teacher_sha256": teacher_sha256,
            "stage2_route_formula": "argmax_positive_teacher_branch_gain_else_none",
            "stage2_route_teacher_gradient": "disabled",
            "stage2_loss": "forecast_mse_only",
            "stage1_checkpoint": str(moved_source),
            "stage1_checkpoint_sha256": "a" * 64,
        }
    }

    restored, _ = cli._restore_online_model_for_evaluation(
        config, "ETTh1", protocol, checkpoint, manifest, device="cpu"
    )
    assert isinstance(restored, RelationV2OnlineStage1Model)

    moved_source.write_bytes(b"different checkpoint")
    with pytest.raises(ValueError, match="Stage-1 source checkpoint checksum mismatch"):
        cli._restore_online_model_for_evaluation(
            config, "ETTh1", protocol, checkpoint, manifest, device="cpu"
        )


def test_evaluate_manifest_round_trips_an_e20_teacher_checkpoint(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    args, manifest, restored_models = _online_evaluation_artifacts(
        tmp_path, monkeypatch, frozen_teacher=True
    )

    record = cli._evaluate_manifest(args, manifest, allow_existing=False)
    assert "phase=test ETTh1 ours" in capsys.readouterr().err

    assert record["mse"] == 0.5
    assert len(restored_models) == 1
    restored = restored_models[0]
    assert restored.route_teacher is not None
    assert restored.route_teacher.state_sha256() == manifest["metadata"][
        "stage2_route_teacher_sha256"
    ]
    assert all(not parameter.requires_grad for parameter in restored.route_teacher.parameters())


def test_evaluate_manifest_round_trips_an_e21_teacher_fusion_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    args, manifest, restored_models = _online_evaluation_artifacts(
        tmp_path,
        monkeypatch,
        frozen_teacher=True,
        hv_fusion=True,
    )

    record = cli._evaluate_manifest(args, manifest, allow_existing=False)

    assert record["mse"] == 0.5
    assert len(restored_models) == 1
    restored = restored_models[0]
    assert restored.route_teacher is not None
    assert restored.forecast_fusion is not None
    assert restored._forecast_fusion_mode == "teacher_hv"
    assert restored.route_teacher.state_sha256() == manifest["metadata"][
        "stage2_route_teacher_sha256"
    ]


def test_evaluate_manifest_reaches_the_internal_teacher_checksum_gate(
    tmp_path: Path, monkeypatch
) -> None:
    args, manifest, _ = _online_evaluation_artifacts(
        tmp_path, monkeypatch, frozen_teacher=True
    )
    original_path = Path(str(manifest["checkpoint"]))
    payload = torch.load(original_path, weights_only=False)
    teacher_key = next(
        name for name in payload["model_state"] if name.startswith("route_teacher.")
    )
    payload["model_state"][teacher_key].view(-1)[0] += 1.0
    corrupt_path = tmp_path / "e20-corrupt.pt"
    torch.save(payload, corrupt_path)
    manifest["checkpoint"] = str(corrupt_path)
    manifest["checkpoint_sha256"] = file_sha256(corrupt_path)
    manifest["run_id"] = "e20-corrupt"

    with pytest.raises(ValueError, match="frozen route teacher checksum mismatch"):
        cli._evaluate_manifest(args, manifest, allow_existing=False)


def test_evaluate_manifest_keeps_a_legacy_e19_checkpoint_teacher_free(
    tmp_path: Path, monkeypatch
) -> None:
    args, manifest, restored_models = _online_evaluation_artifacts(
        tmp_path, monkeypatch, frozen_teacher=False
    )

    cli._evaluate_manifest(args, manifest, allow_existing=False)

    assert len(restored_models) == 1
    assert not restored_models[0].has_frozen_route_teacher


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    (
        ("stage2_route_teacher", None),
        ("stage2_route_teacher", "mutable_stage1"),
        ("stage2_route_teacher_sha256", None),
        ("stage2_route_formula", None),
        ("stage2_route_formula", "argmax_teacher_gain"),
        ("stage2_route_teacher_gradient", None),
        ("stage2_route_teacher_gradient", "enabled"),
        ("stage2_loss", None),
        ("stage2_loss", "forecast_plus_route"),
        ("stage2_route_teacher_checksum_verified", None),
        ("stage2_route_teacher_checksum_verified", False),
        ("stage2_route_teacher_invariance_verified", None),
        ("stage2_route_teacher_invariance_verified", False),
        ("stage2_validation_diagnostics", None),
        ("stage2_validation_inference_ablations", None),
    ),
)
def test_evaluate_manifest_rejects_an_incomplete_e20_contract(
    tmp_path: Path,
    monkeypatch,
    field: str,
    invalid_value: object,
) -> None:
    args, manifest, _ = _online_evaluation_artifacts(
        tmp_path, monkeypatch, frozen_teacher=True
    )
    metadata = manifest["metadata"]
    assert isinstance(metadata, dict)
    if invalid_value is None:
        metadata.pop(field)
    else:
        metadata[field] = invalid_value

    with pytest.raises(
        ValueError, match=f"frozen route teacher metadata contract mismatch: {field}"
    ):
        cli._evaluate_manifest(args, manifest, allow_existing=False)


def test_legacy_repaired_forecast_helper_contains_only_predictor_and_mask_path(
    tmp_path: Path, monkeypatch
) -> None:
    config = load_config(Path("configs/stage1_seq96_relational_recovery.yaml"))
    series = _series(tmp_path)
    protocol = protocol_for("ETTh1", len(series.values))
    artifacts = tmp_path / "artifacts"
    support_start, support_end, expected = cli._mask_identity(
        config, "ETTh1", protocol, "validation"
    )
    validation_mask, validation_metadata = generate_variable_missing_mask(
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
        validation_mask,
        validation_metadata,
    )
    metadata = cli._base_metadata(config, series, protocol, "ETTh1")
    split_entries = {}
    for index, split_name in enumerate(("train", "validation", "test")):
        values = np.zeros((2, 96, 7), dtype=np.float32)
        mask = np.ones_like(values, dtype=np.bool_)
        mask[:, index, :] = False
        repaired = RepairedSplit(
            values=values,
            observed_mask=mask,
            query_mean=np.zeros((2, 7), dtype=np.float32),
            query_std=np.ones((2, 7), dtype=np.float32),
            targets=np.zeros((2, 7, 96), dtype=np.float32),
            input_starts=np.asarray([0, 1], dtype=np.int64),
        )
        path = tmp_path / f"{split_name}.npz"
        save_repaired_split(path, repaired)
        split_entries[split_name] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "window_count": 2,
            "missing_rate": repaired.missing_rate,
            "source_mask_sha256": None,
        }
    handoff_path = tmp_path / "handoff.json"
    write_json_atomic(
        handoff_path,
        {
            "format_version": 1,
            "kind": "repaired_windows",
            "metadata": metadata,
            "repair_checkpoint": "stage1.pt",
            "repair_checkpoint_sha256": "repair-sha",
            "splits": split_entries,
        },
    )

    def fake_fit(self, _train, _validation):
        save_checkpoint(
            self.tracker.path,
            self.model.state_dict(),
            self.tracker.metadata,
            epoch=1,
            val_mse=0.5,
            val_mae=0.4,
            validation_diagnostics={},
        )
        return TrainingResult(
            self.tracker.path, 0.5, 0.4, {}, 1, 1, 0.0,
            sum(parameter.numel() for parameter in self.model.parameters()),
        )

    monkeypatch.setattr(cli.RepairedForecastTrainer, "fit", fake_fit)
    result = cli._run_repaired_forecast_training(
        config,
        "ETTh1",
        artifacts,
        device="cpu",
        metadata=metadata,
        run_metadata={**metadata, "variant": "ours"},
        repaired_data_path=handoff_path,
    )

    checkpoint = load_checkpoint(Path(result["best_checkpoint"]))
    assert "mask_projection.weight" in checkpoint["model_state"]
    assert not any("relation" in key for key in checkpoint["model_state"])
    assert checkpoint["metadata"]["online_mode"] is False
    assert checkpoint["metadata"]["stage1_parameters_loaded"] is False
