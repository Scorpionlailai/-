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
from tests.unit.test_teacher_and_losses import _tiny_ours


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


def test_patchtst_type3_learning_rate_matches_official_schedule():
    base = 1.0e-4

    assert patchtst_type3_learning_rate(base, 1) == pytest.approx(base)
    assert patchtst_type3_learning_rate(base, 2) == pytest.approx(base)
    assert patchtst_type3_learning_rate(base, 3) == pytest.approx(base)
    assert patchtst_type3_learning_rate(base, 4) == pytest.approx(base * 0.9)
    assert patchtst_type3_learning_rate(base, 5) == pytest.approx(base * 0.9**2)
