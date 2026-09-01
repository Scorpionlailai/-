from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from malr.models.patchtst import PatchTSTCore
from malr.models.recovery import MissingAwareRecovery
from malr.models.relations import HistoricalRelation
from malr.models.stage1 import ModelVariant, Stage1Model, initialize_from_clean_checkpoint
from malr.prototypes.bank import PrototypeBank, source_checksum
from malr.training.checkpoints import load_checkpoint, save_checkpoint
from malr.training.losses import Stage1Loss, forecast_mse, latent_recovery_mse


def test_recovery_loss_covers_only_damaged_nonpadding_patches():
    output = torch.tensor([[[[1.0], [2.0], [3.0]]]])
    target = torch.zeros_like(output)
    state = torch.tensor([[[0, 1, 2]]])
    valid = torch.tensor([[[True, True, False]]])

    loss = latent_recovery_mse(output, target, state, valid)

    assert loss.item() == pytest.approx(4.0)


def test_empty_recovery_loss_is_differentiable_zero():
    output = torch.randn(2, 1, 3, 4, requires_grad=True)
    target = torch.zeros_like(output)
    state = torch.zeros(2, 1, 3, dtype=torch.long)
    valid = torch.ones_like(state, dtype=torch.bool)

    loss = latent_recovery_mse(output, target, state, valid)
    loss.backward()

    assert loss.item() == 0.0
    torch.testing.assert_close(output.grad, torch.zeros_like(output))


def test_stage1_total_contains_mse_not_mae():
    prediction = torch.tensor([0.0, 2.0])
    target = torch.tensor([0.0, 0.0])
    recovery = prediction.sum() * 0 + 3.0
    result = Stage1Loss.create(prediction, target, recovery, damaged_patch_count=1, lambda_rec=2.0)

    assert forecast_mse(prediction, target).item() == pytest.approx(2.0)
    assert result.total.item() == pytest.approx(8.0)


def _empty_bank(variable_count: int = 1) -> PrototypeBank:
    source = np.zeros((64, variable_count), dtype=np.float64)
    return PrototypeBank(
        dataset="fixture",
        train_end=64,
        source_checksum=source_checksum(source),
        config_sha256="fixture",
        records=tuple(() for _ in range(variable_count)),
        factor_mean=np.zeros((variable_count, 7)),
        factor_std=np.ones((variable_count, 7)),
    )


def _tiny_ours(
    *,
    clean_initialized: bool = False,
    fusion: str = "additive",
    global_refiner_enabled: bool = False,
) -> Stage1Model:
    core = PatchTSTCore(
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
    historical = HistoricalRelation(_empty_bank(), topk=4, d_model=8)
    recovery = MissingAwareRecovery(
        c_in=1,
        patch_num=core.patch_num,
        patch_len=16,
        d_model=8,
        temporal_topk=2,
        variable_topk=1,
        historical_relation=historical,
        use_historical=False,
        fusion=fusion,
        global_refiner_enabled=global_refiner_enabled,
        global_refiner_heads=2,
    )
    model = Stage1Model(core, ModelVariant.OURS, recovery=recovery)
    if clean_initialized:
        initialize_from_clean_checkpoint(model, deepcopy(core.state_dict()))
    return model


def test_ours_forward_runs_from_scratch_without_clean_checkpoint():
    model = _tiny_ours(clean_initialized=False)
    patch_count = model.core.patch_num
    state = torch.full((1, 1, patch_count), 2, dtype=torch.long)
    mask = torch.zeros(1, 1, patch_count, 16, dtype=torch.bool)
    values = torch.zeros(1, 1, patch_count, 16)

    output = model(
        values,
        mask,
        torch.zeros(1, 1, patch_count),
        state,
        torch.zeros(1, 1),
        torch.ones(1, 1),
    )

    assert output.forecast.shape == (1, 1, 4)


def test_recovery_loss_target_uses_current_patch_projection():
    model = _tiny_ours(clean_initialized=False)
    with torch.no_grad():
        model.core.W_P.weight.fill_(1.0)
        model.core.W_P.bias.zero_()
    clean_raw = torch.ones((1, 1, model.core.patch_num, 16))
    mean = torch.zeros((1, 1))
    std = torch.ones((1, 1))

    target = model.teacher_target(clean_raw, mean, std)

    torch.testing.assert_close(target, torch.full_like(target, 16.0))


def test_teacher_target_uses_query_normalization_and_is_detached():
    model = _tiny_ours()
    with torch.no_grad():
        model.core.W_P.weight.fill_(0.5)
        model.core.W_P.bias.zero_()
    clean_raw = torch.full((1, 1, 4, 16), 12.0)
    mean = torch.tensor([[10.0]])
    std = torch.tensor([[2.0]])

    target = model.teacher_target(clean_raw, mean, std)

    torch.testing.assert_close(target, torch.full_like(target, 8.0))
    assert not target.requires_grad
    assert all(parameter.requires_grad for parameter in model.teacher_projection.parameters())


def test_evaluation_rejects_clean_teacher_input():
    model = _tiny_ours().eval()
    patch_count = model.core.patch_num
    with pytest.raises(ValueError, match="training-only"):
        model(
            torch.zeros(1, 1, patch_count, 16),
            torch.zeros(1, 1, patch_count, 16, dtype=torch.bool),
            torch.zeros(1, 1, patch_count),
            torch.full((1, 1, patch_count), 2, dtype=torch.long),
            torch.zeros(1, 1),
            torch.ones(1, 1),
            clean_raw_patches=torch.zeros(1, 1, patch_count, 16),
        )


def test_checkpoint_metadata_is_validated(tmp_path: Path):
    path = tmp_path / "clean.pt"
    save_checkpoint(path, {"weight": torch.ones(1)}, {"dataset": "ETTh1", "hash": "abc"})
    restored = load_checkpoint(path, expected_metadata={"dataset": "ETTh1", "hash": "abc"})
    torch.testing.assert_close(restored["model_state"]["weight"], torch.ones(1))

    with pytest.raises(ValueError, match="metadata"):
        load_checkpoint(path, expected_metadata={"dataset": "Weather"})
