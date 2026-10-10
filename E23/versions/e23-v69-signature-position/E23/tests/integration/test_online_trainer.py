from __future__ import annotations

from pathlib import Path

import torch

from malr.models.online_stage1_v2 import RelationV2OnlineStage1Model
from malr.training.online_checkpoint import load_online_checkpoint
from malr.training.online_trainer import OnlineTrainer
from malr.training.trainer import PreparedBatch

from test_online_stage1 import _fixture
from test_online_stage1_v2 import _v2_fixture


def _batch(data: dict[str, torch.Tensor]) -> PreparedBatch:
    return PreparedBatch(
        patch_values=data["values"],
        patch_mask=data["mask"],
        rho=data["rho"],
        state=data["state"],
        query_mean=data["query_mean"],
        query_std=data["query_std"],
        forecast_target=data["forecast_target"],
        clean_raw_patches=data["clean"],
        patch_valid=torch.ones_like(data["state"], dtype=torch.bool),
        raw_context=torch.zeros(1, 96, 2),
        context_mask=torch.ones(1, 96, 2, dtype=torch.bool),
        train_std=torch.ones(1, 2),
        patch_starts=data["starts"],
    )


def make_v2_trainer(
    model: RelationV2OnlineStage1Model,
    tmp_path: Path,
    *,
    max_epochs: int,
    learning_rate: float = 1e-4,
) -> OnlineTrainer:
    return OnlineTrainer(
        model,
        checkpoint_path=tmp_path / "v2-online.pt",
        checkpoint_metadata={"dataset": "ETTh1", "variant": "ours"},
        max_epochs=max_epochs,
        patience=max_epochs,
        learning_rate=learning_rate,
    )


def _statistics_snapshot(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().clone()
        for name, value in module.state_dict().items()
        if isinstance(value, torch.Tensor)
    }


def test_v2_trainer_changes_only_relation_lr_in_calibration(tmp_path: Path) -> None:
    """A calibration epoch must slow only the relation-side parameters."""

    model, _ = _v2_fixture()
    trainer = make_v2_trainer(model, tmp_path, max_epochs=100, learning_rate=1e-4)
    before = trainer.optimizer

    trainer._configure_epoch(81)

    rates = {group["name"]: group["lr"] for group in trainer.optimizer.param_groups}
    assert rates == {"backbone": 1e-4, "relations": 1e-5}
    parameter_ids = [
        id(parameter)
        for group in trainer.optimizer.param_groups
        for parameter in group["params"]
    ]
    expected_ids = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    assert len(parameter_ids) == len(set(parameter_ids))
    assert set(parameter_ids) == expected_ids
    assert trainer.optimizer is before


def test_v2_trainer_uses_configured_calibration_relation_lr_scale(tmp_path: Path) -> None:
    """Calibration must honor the relation learning-rate scale from config."""

    model, _ = _v2_fixture()
    trainer = OnlineTrainer(
        model,
        checkpoint_path=tmp_path / "v2-online-configured-scale.pt",
        checkpoint_metadata={"dataset": "ETTh1", "variant": "ours"},
        max_epochs=100,
        patience=100,
        learning_rate=1e-4,
        calibration_relation_lr_scale=0.25,
    )

    trainer._configure_epoch(81)

    rates = {group["name"]: group["lr"] for group in trainer.optimizer.param_groups}
    assert rates == {"backbone": 1e-4, "relations": 2.5e-5}


def test_teacher_hv_optimizer_includes_and_updates_forecast_fusion(
    tmp_path: Path,
) -> None:
    """The production optimizer must own every enabled E21 fusion parameter."""

    torch.manual_seed(131)
    model, data = _v2_fixture()
    model.install_frozen_route_teacher()
    model.install_forecast_hv_fusion()
    model.configure_forecast_finetuning(
        frozen_route_teacher=True,
        forecast_fusion="teacher_hv",
    )
    assert model.forecast_fusion is not None
    fusion_parameters = list(model.forecast_fusion.parameters())
    before = [parameter.detach().clone() for parameter in fusion_parameters]

    trainer = make_v2_trainer(
        model,
        tmp_path,
        max_epochs=1,
        learning_rate=1.0e-2,
    )
    optimizer_parameter_ids = {
        id(parameter)
        for group in trainer.optimizer.param_groups
        for parameter in group["params"]
    }

    assert {id(parameter) for parameter in fusion_parameters} <= optimizer_parameter_ids
    assert not {
        id(parameter)
        for parameter in model.temporal_operator.candidate_projector.parameters()
    } & optimizer_parameter_ids
    assert not {
        id(parameter)
        for parameter in model.temporal_operator.fusion_gate.parameters()
    } & optimizer_parameter_ids

    trainer._train_epoch(
        [_batch(data)],
        epoch=1,
        objective="forecast",
        model_epoch_index=100,
    )

    assert any(
        not torch.equal(previous, parameter)
        for previous, parameter in zip(before, fusion_parameters, strict=True)
    )


def test_v2_validation_does_not_change_variable_or_route_statistics(tmp_path: Path) -> None:
    """Validation must not train the V2 relation or route statistics."""

    model, data = _v2_fixture()
    trainer = make_v2_trainer(model, tmp_path, max_epochs=1)
    before_variable = _statistics_snapshot(model.variable_statistics)
    before_route = _statistics_snapshot(model.route_statistics)

    trainer._validation([_batch(data)])

    for name, value in before_variable.items():
        torch.testing.assert_close(value, _statistics_snapshot(model.variable_statistics)[name])
    for name, value in before_route.items():
        torch.testing.assert_close(value, _statistics_snapshot(model.route_statistics)[name])


def test_v2_training_step_updates_variable_and_route_statistics(tmp_path: Path) -> None:
    """Removing unified trainer statistics updates would leave V2 priors empty."""

    model, data = _v2_fixture()
    trainer = make_v2_trainer(model, tmp_path, max_epochs=1)
    model.begin_epoch(1)

    trainer._train_epoch([_batch(data)], epoch=1)

    assert model.variable_statistics.count.sum().item() > 0
    assert model.route_statistics.count.sum().item() > 0


def test_online_trainer_saves_and_restores_paired_state(tmp_path: Path) -> None:
    model, data = _fixture()
    path = tmp_path / "best-online.pt"
    trainer = OnlineTrainer(
        model,
        checkpoint_path=path,
        checkpoint_metadata={"dataset": "ETTh1", "variant": "ours-online"},
        max_epochs=1,
        patience=1,
        learning_rate=1e-3,
    )
    batch = _batch(data)
    result = trainer.fit(
        lambda _epoch: [batch],
        lambda: [batch],
        config_hash="c" * 64,
        data_hash="d" * 64,
        mask_hash="m" * 64,
    )
    assert result.best_epoch == 1
    payload = load_online_checkpoint(path, expected_metadata={"dataset": "ETTh1"})
    assert payload["online_state"]["epoch"] == 1
    assert model.memory.snapshot().epoch == 1


def test_online_trainer_reports_live_batch_and_validation_progress(
    tmp_path: Path,
    capsys: object,
) -> None:
    model, data = _fixture()
    trainer = OnlineTrainer(
        model,
        checkpoint_path=tmp_path / "progress-online.pt",
        checkpoint_metadata={"dataset": "ETTh1", "variant": "ours-online"},
        max_epochs=1,
        patience=1,
        learning_rate=1e-3,
    )
    batch = _batch(data)

    trainer.fit(
        lambda _epoch: [batch],
        lambda: [batch],
        config_hash="c" * 64,
        data_hash="d" * 64,
        mask_hash="m" * 64,
    )

    output = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "epoch=1/1 phase=train start" in output
    assert "epoch=1/1 phase=train batch=1" in output
    assert "epoch=1/1 phase=validation" in output


def test_online_trainer_runs_repair_then_forecast_objectives(
    tmp_path: Path,
    capsys: object,
) -> None:
    model, data = _v2_fixture()
    batch = _batch(data)
    stage1 = OnlineTrainer(
        model,
        checkpoint_path=tmp_path / "stage1.pt",
        checkpoint_metadata={"dataset": "ETTh1", "variant": "ours"},
        max_epochs=1,
        patience=1,
        learning_rate=1e-4,
    )

    repair_result = stage1.fit(
        lambda _epoch: [batch],
        lambda: [batch],
        config_hash="c" * 64,
        data_hash="d" * 64,
        mask_hash="m" * 64,
        objective="repair",
        validation_objective="repair",
        stage_label="repair",
    )
    assert repair_result.best_checkpoint == tmp_path / "stage1.pt"
    assert model.memory.snapshot().epoch == 1

    stage2 = OnlineTrainer(
        model,
        checkpoint_path=tmp_path / "stage2.pt",
        checkpoint_metadata={"dataset": "ETTh1", "variant": "ours"},
        max_epochs=1,
        patience=1,
        learning_rate=1e-4,
    )
    forecast_result = stage2.fit(
        lambda _epoch: [batch],
        lambda: [batch],
        config_hash="c" * 64,
        data_hash="d" * 64,
        mask_hash="m" * 64,
        objective="forecast",
        validation_objective="forecast",
        epoch_offset=1,
        stage_label="forecast",
    )

    assert forecast_result.best_checkpoint == tmp_path / "stage2.pt"
    assert model.memory.snapshot().epoch == 2
    output = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "epoch=1/1 phase=repair start" in output
    assert "epoch=1/1 phase=repair-validation" in output
    assert "epoch=2/2 phase=forecast start" in output
    assert "epoch=2/2 phase=validation" in output


def test_forecast_fine_tuning_freezes_online_state_when_requested(tmp_path: Path) -> None:
    """Stage 2 must not advance the Stage-1 memory/EMA snapshot."""

    model, data = _v2_fixture()
    batch = _batch(data)
    model.begin_epoch(1)
    model.end_epoch(1)
    before = model.online_state()
    trainer = OnlineTrainer(
        model,
        checkpoint_path=tmp_path / "frozen-stage2.pt",
        checkpoint_metadata={"dataset": "ETTh1", "variant": "ours"},
        max_epochs=1,
        patience=1,
        learning_rate=1e-4,
    )

    trainer.fit(
        lambda _epoch: [batch],
        lambda: [batch],
        config_hash="c" * 64,
        data_hash="d" * 64,
        mask_hash="m" * 64,
        objective="forecast",
        validation_objective="forecast",
        model_epoch_offset=1,
        stage_label="forecast",
        freeze_online_state=True,
    )
    after = model.online_state()
    assert after["epoch"] == before["epoch"] == 1
    assert after["memory_hash"] == before["memory_hash"]
    assert after["ema_hash"] == before["ema_hash"]
    assert after["relation_prior_hash"] == before["relation_prior_hash"]
