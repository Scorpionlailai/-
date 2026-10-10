from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from malr.models.patchtst import PatchTSTCore
from malr.models.stage1 import ModelVariant, Stage1Model, initialize_from_clean_checkpoint
from malr.training.trainer import (
    PreparedBatch,
    Trainer,
    patchtst_type3_learning_rate,
)
from malr.cli import build_experiment_model, build_parser
from malr.config import load_config
from tests.unit.test_teacher_and_losses import _empty_bank, _tiny_ours


def _core() -> PatchTSTCore:
    return PatchTSTCore(
        c_in=1,
        seq_len=32,
        pred_len=4,
        patch_len=16,
        stride=8,
        d_model=8,
        n_heads=2,
        d_ff=16,
        n_layers=1,
        dropout=0.0,
    )


def _model(variant: str) -> Stage1Model:
    if variant == "ours":
        return _tiny_ours()
    core = _core()
    model = Stage1Model(core, ModelVariant(variant))
    if variant == "missing_pretrained":
        initialize_from_clean_checkpoint(model, deepcopy(core.state_dict()))
    return model


def _batch(model: Stage1Model) -> PreparedBatch:
    batch_size = 2
    patches = model.core.patch_num
    values = torch.randn(batch_size, 1, patches, 16)
    mask = torch.ones_like(values, dtype=torch.bool)
    state = torch.zeros(batch_size, 1, patches, dtype=torch.long)
    if model.variant is not ModelVariant.CLEAN:
        mask[:, :, 1] = False
        values[:, :, 1] = 0.0
        state[:, :, 1] = 2
    return PreparedBatch(
        patch_values=values,
        patch_mask=mask,
        rho=mask.float().mean(-1),
        state=state,
        query_mean=torch.zeros(batch_size, 1),
        query_std=torch.ones(batch_size, 1),
        forecast_target=torch.randn(batch_size, 1, 4),
        clean_raw_patches=torch.randn_like(values),
        patch_valid=torch.ones_like(state, dtype=torch.bool),
        raw_context=torch.zeros(batch_size, 32, 1),
        context_mask=torch.ones(batch_size, 32, 1, dtype=torch.bool),
        train_std=torch.ones(1),
    )


@pytest.mark.parametrize("variant", ["clean", "missing", "missing_pretrained", "ours"])
def test_every_primary_variant_runs_one_epoch(tmp_path: Path, variant: str):
    model = _model(variant)
    batch = _batch(model)
    epochs_requested = []

    def train_batches(epoch: int):
        epochs_requested.append(epoch)
        return [batch]

    trainer = Trainer(
        model,
        checkpoint_path=tmp_path / f"{variant}.pt",
        checkpoint_metadata={"dataset": "fixture", "variant": variant},
        max_epochs=1,
        patience=1,
        learning_rate=1e-3,
    )
    result = trainer.fit(train_batches, lambda: [batch])

    assert result.best_checkpoint.exists()
    assert result.best_val_mse >= 0.0
    assert epochs_requested == [0]
    assert result.test_evaluated is False


def test_ablation_switch_can_disable_recovery_loss(tmp_path: Path):
    model = _tiny_ours()
    batch = _batch(model)
    trainer = Trainer(
        model,
        checkpoint_path=tmp_path / "ablation.pt",
        checkpoint_metadata={"dataset": "fixture", "variant": "ours"},
        max_epochs=1,
        patience=1,
        use_recovery_loss=False,
    )
    result = trainer.fit(lambda epoch: [batch], lambda: [batch])
    assert result.best_checkpoint.exists()


def test_trainer_forwards_filter_keep_weight(tmp_path: Path):
    model = _tiny_ours(fusion="replacement_mlp", filtered=True)
    batch = _batch(model)
    output_bias = model.recovery.edge_filter.temporal.network[-1].bias
    initial_bias = output_bias.detach().clone()
    trainer = Trainer(
        model,
        checkpoint_path=tmp_path / "filtered.pt",
        checkpoint_metadata={"dataset": "fixture", "variant": "ours"},
        max_epochs=1,
        patience=1,
        learning_rate=1e-3,
        lambda_keep=1e-4,
        use_recovery_loss=False,
    )

    trainer.fit(lambda epoch: [batch], lambda: [batch])

    assert output_bias.detach() < initial_bias


def test_filter_uses_constant_independent_lr_without_moving_patchtst_head(
    tmp_path: Path,
) -> None:
    model = _tiny_ours(fusion="replacement_mlp", filtered=True)
    batch = _batch(model)
    trainer = Trainer(
        model,
        checkpoint_path=tmp_path / "independent_filter_lr.pt",
        checkpoint_metadata={"dataset": "fixture", "variant": "ours"},
        max_epochs=4,
        patience=4,
        learning_rate=1.0e-4,
        filter_learning_rate=1.0e-3,
        lr_schedule="type3",
        use_recovery_loss=False,
    )

    groups = {
        group["name"]: group for group in trainer.optimizer.param_groups
    }
    base_parameter_ids = {id(parameter) for parameter in groups["base"]["params"]}
    filter_parameter_ids = {
        id(parameter) for parameter in groups["filter"]["params"]
    }
    expected_filter_ids = {
        id(parameter)
        for parameter in model.recovery.edge_filter.parameters()
        if parameter.requires_grad
    }
    prediction_head_ids = {
        id(parameter)
        for parameter in model.core.head.parameters()
        if parameter.requires_grad
    }

    assert filter_parameter_ids == expected_filter_ids
    assert prediction_head_ids <= base_parameter_ids
    assert prediction_head_ids.isdisjoint(filter_parameter_ids)

    trainer.fit(lambda epoch: [batch], lambda: [batch])

    assert groups["base"]["lr"] == pytest.approx(9.0e-5)
    assert groups["filter"]["lr"] == pytest.approx(1.0e-3)


def test_node_gate_uses_the_constant_filter_learning_rate(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model = _tiny_ours(
        fusion="replacement_mlp",
        filtered=True,
        node_gate_enabled=True,
    )
    batch = _batch(model)
    trainer = Trainer(
        model,
        checkpoint_path=tmp_path / "node_gate_lr.pt",
        checkpoint_metadata={"dataset": "fixture", "variant": "ours"},
        max_epochs=4,
        patience=4,
        learning_rate=1.0e-4,
        filter_learning_rate=1.0e-3,
        lr_schedule="type3",
        use_recovery_loss=False,
    )

    groups = {
        group["name"]: group for group in trainer.optimizer.param_groups
    }
    assert model.recovery is not None
    assert model.recovery.node_gate is not None
    assert {id(parameter) for parameter in groups["node_gate"]["params"]} == {
        id(parameter) for parameter in model.recovery.node_gate.parameters()
    }

    trainer.fit(lambda epoch: [batch], lambda: [batch])
    output = capsys.readouterr().out

    assert groups["base"]["lr"] == pytest.approx(9.0e-5)
    assert groups["filter"]["lr"] == pytest.approx(1.0e-3)
    assert groups["node_gate"]["lr"] == pytest.approx(1.0e-3)
    assert "node_gate_mean=" in output
    assert "node_gate_active=" in output


def test_training_log_reports_loss_components_and_both_learning_rates(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model = _tiny_ours(fusion="replacement_mlp", filtered=True)
    batch = _batch(model)
    trainer = Trainer(
        model,
        checkpoint_path=tmp_path / "filter_log.pt",
        checkpoint_metadata={"dataset": "fixture", "variant": "ours"},
        max_epochs=1,
        patience=1,
        learning_rate=1.0e-4,
        filter_learning_rate=1.0e-3,
        lambda_keep=1.0e-3,
        use_recovery_loss=False,
    )

    trainer.fit(lambda epoch: [batch], lambda: [batch])
    output = capsys.readouterr().out

    assert "train_forecast_mse=" in output
    assert "train_recovery_mse=" in output
    assert "train_filter_keep=" in output
    assert "train_weighted_filter_keep=" in output
    assert "base_lr=0.0001" in output
    assert "filter_lr=0.001" in output


def test_training_log_reports_imputation_loss_components(tmp_path: Path, capsys):
    model = _tiny_ours(reconstruction_enabled=True)
    batch = _batch(model)
    context_mask = torch.ones(2, 32, 1, dtype=torch.bool)
    context_mask[:, 8:16] = False
    batch = replace(
        batch,
        context_mask=context_mask,
        imputation_target=torch.randn(2, 32, 1),
    )
    trainer = Trainer(
        model,
        checkpoint_path=tmp_path / "imputation_log.pt",
        checkpoint_metadata={"dataset": "fixture", "variant": "ours"},
        max_epochs=1,
        patience=1,
        lambda_imputation_mse=0.1,
        lambda_imputation_mae=0.1,
        use_recovery_loss=False,
    )

    trainer.fit(lambda epoch: [batch], lambda: [batch])
    output = capsys.readouterr().out

    assert "train_imputation_mse=" in output
    assert "train_imputation_mae=" in output
    assert "train_weighted_imputation_mse=" in output
    assert "train_weighted_imputation_mae=" in output


def test_validation_aggregates_filter_diagnostics(tmp_path: Path):
    model = _tiny_ours(fusion="replacement_mlp", filtered=True)
    batch = _batch(model)
    trainer = Trainer(
        model,
        checkpoint_path=tmp_path / "diagnostics.pt",
        checkpoint_metadata={"dataset": "fixture", "variant": "ours"},
    )

    diagnostics = trainer.validation_metrics([batch]).diagnostics

    assert diagnostics["temporal_candidate_availability"] == pytest.approx(1.0)
    assert diagnostics["temporal_keep_rate"] == pytest.approx(2.0 / 3.0)
    assert diagnostics["temporal_zero_edge_rate"] == pytest.approx(0.0)
    assert diagnostics["temporal_mean_confidence"] == pytest.approx(0.9)
    assert diagnostics["temporal_mean_retained_confidence"] == pytest.approx(0.9)
    assert diagnostics["temporal_mean_retained_rho_eq_0"] == pytest.approx(2.0)
    assert diagnostics["temporal_threshold"] == pytest.approx(0.5)
    assert diagnostics["temporal_max_edges"] == pytest.approx(2.0)


def test_validation_aggregates_node_gate_diagnostics(tmp_path: Path) -> None:
    model = _tiny_ours(
        fusion="replacement_mlp",
        filtered=True,
        node_gate_enabled=True,
    )
    assert model.recovery is not None
    assert model.recovery.node_gate is not None
    with torch.no_grad():
        model.recovery.node_gate.bias.fill_(0.25)
    batch = _batch(model)
    trainer = Trainer(
        model,
        checkpoint_path=tmp_path / "node_gate_diagnostics.pt",
        checkpoint_metadata={"dataset": "fixture", "variant": "ours"},
    )

    diagnostics = trainer.validation_metrics([batch]).diagnostics

    assert diagnostics["node_gate_mean"] == pytest.approx(0.25)
    assert diagnostics["node_gate_active_rate"] == pytest.approx(1.0)
    assert diagnostics["node_gate_missing_mean"] == pytest.approx(0.25)


def test_cli_exposes_the_agreed_stage1_workflow():
    parser = build_parser()
    actions = next(
        action for action in parser._actions if action.dest == "command"
    )
    assert {
        "validate-data",
        "make-eval-masks",
        "train-clean",
        "build-bank",
        "train",
        "evaluate",
    } <= set(actions.choices)


def test_ours_builds_without_clean_checkpoint_when_historical_is_disabled():
    config = load_config(Path("configs/stage1_no_historical.yaml"))

    model = build_experiment_model(config, "ETTh1", "ours")

    assert model.variant is ModelVariant.OURS


def test_residual_config_builds_residual_recovery_without_clean_initialization() -> None:
    config = load_config(Path("configs/stage1_sr010_residual_no_recloss.yaml"))
    config = replace(
        config,
        ablations=replace(config.ablations, use_historical=False),
    )

    model = build_experiment_model(config, "ETTh1", "ours")

    assert model.recovery.fusion_mode == "residual_mlp"
    assert model.recovery.residual_mlp is not None
    assert torch.count_nonzero(model.recovery.residual_mlp[-1].weight) == 0


def test_global_replacement_config_builds_enabled_refiner() -> None:
    config = load_config(
        Path("configs/stage1_sr010_replacement_global_no_recloss.yaml")
    )
    config = replace(
        config,
        ablations=replace(config.ablations, use_historical=False),
    )

    model = build_experiment_model(config, "ETTh1", "ours")

    assert model.recovery.fusion_mode == "replacement_mlp"
    assert model.recovery.replacement_mlp is not None
    assert model.recovery.global_refiner is not None


def test_imputation_config_builds_reconstruction_head_without_clean_initialization():
    config = load_config(
        Path("configs/stage1_sr010_replacement_filter_imputation.yaml")
    )
    model = build_experiment_model(
        config,
        "ETTh1",
        "ours",
        bank=_empty_bank(7),
    )

    assert model.recovery.reconstruction_head is not None
    assert model._clean_initialized is True


def test_patchtst_type3_learning_rate_matches_official_schedule():
    base = 1.0e-4

    assert patchtst_type3_learning_rate(base, 1) == pytest.approx(base)
    assert patchtst_type3_learning_rate(base, 2) == pytest.approx(base)
    assert patchtst_type3_learning_rate(base, 3) == pytest.approx(base)
    assert patchtst_type3_learning_rate(base, 4) == pytest.approx(base * 0.9)
    assert patchtst_type3_learning_rate(base, 5) == pytest.approx(base * 0.9**2)
