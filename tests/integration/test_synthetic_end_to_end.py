from pathlib import Path

import numpy as np
import torch

from malr.cli import build_experiment_model
from malr.config import load_config
from malr.data.masks import EpochMaskManager, generate_variable_missing_mask
from malr.data.normalization import fit_train_variable_stats
from malr.data.patching import PATCH_MISSING, PATCH_OBSERVED, PATCH_PARTIAL
from malr.data.splits import DatasetProtocol, SplitBounds
from malr.data.windows import WindowRef, build_window_index
from malr.training.data import iter_prepared_batches
from malr.training.trainer import Trainer
from tests.unit.test_teacher_and_losses import _empty_bank, _tiny_ours


def test_filtered_ours_runs_one_joint_optimization_step() -> None:
    config = load_config(
        Path("configs/stage1_sr010_replacement_filter_no_recloss.yaml")
    )
    model = build_experiment_model(
        config,
        "ETTh1",
        "ours",
        bank=_empty_bank(variable_count=7),
    ).train()
    patch_count = model.core.patch_num
    values = torch.randn(2, 7, patch_count, config.patch.length)
    mask = torch.ones_like(values, dtype=torch.bool)
    mask[:, 0, 2] = False
    mask[:, 1, 3, :8] = False
    values = torch.where(mask, values, torch.zeros_like(values))
    rho = mask.float().mean(dim=-1)
    state = torch.full((2, 7, patch_count), PATCH_OBSERVED, dtype=torch.long)
    state[:, 0, 2] = PATCH_MISSING
    state[:, 1, 3] = PATCH_PARTIAL
    forecast_target = torch.randn(2, 7, config.data.pred_len)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    result = model.training_objective(
        values,
        mask,
        rho,
        state,
        torch.zeros(2, 7),
        torch.ones(2, 7),
        forecast_target,
        None,
        torch.ones_like(state, dtype=torch.bool),
        raw_context=torch.zeros(2, config.data.seq_len, 7),
        context_mask=torch.ones(
            2, config.data.seq_len, 7, dtype=torch.bool
        ),
        train_std=torch.ones(7),
        use_recovery_loss=False,
        lambda_keep=config.relations.filter.lambda_keep,
    )
    optimizer.zero_grad(set_to_none=True)
    result.loss.total.backward()
    optimizer.step()

    assert torch.isfinite(result.loss.total)
    assert result.loss.filter_keep.item() > 0.0
    assert result.output.recovery is not None
    assert result.output.recovery.filter_keep_loss.requires_grad
    assert result.output.forecast.shape == forecast_target.shape
    assert result.output.recovery.latents.shape[-1] == model.core.d_model


def test_cpu_synthetic_raw_series_to_checkpoint_acceptance(tmp_path: Path):
    time = np.arange(128, dtype=np.float32)
    series = (np.sin(time / 5.0) + 0.01 * time)[:, None]
    protocol = DatasetProtocol(
        "synthetic",
        source_length=128,
        train=SplitBounds(0, 64),
        validation=SplitBounds(64, 96),
        test=SplitBounds(96, 128),
    )
    train_refs = build_window_index(protocol, "train", 32, 4)[:2]
    validation_refs = build_window_index(protocol, "validation", 32, 4)[:2]
    stats = fit_train_variable_stats(torch.from_numpy(series[:64]))
    manager = EpochMaskManager(64, 1, 0.06, 5, 2021)
    validation_mask, _ = generate_variable_missing_mask(
        64, 1, 0.06, 5, 2021, support_start=32
    )
    model = _tiny_ours(fusion="residual_mlp")
    trainer = Trainer(
        model,
        checkpoint_path=tmp_path / "synthetic_ours.pt",
        checkpoint_metadata={"dataset": "synthetic", "variant": "ours"},
        max_epochs=1,
        patience=1,
        device="cpu",
    )

    def train_batches(epoch: int):
        return iter_prepared_batches(
            series,
            train_refs,
            manager.mask_for_epoch(epoch),
            mask_support_start=0,
            train_stats=stats,
            variant="ours",
            batch_size=2,
            patch_len=16,
            patch_stride=8,
            shuffle_seed=2021 + epoch,
        )

    def validation_batches():
        return iter_prepared_batches(
            series,
            validation_refs,
            validation_mask,
            mask_support_start=32,
            train_stats=stats,
            variant="ours",
            batch_size=2,
            patch_len=16,
            patch_stride=8,
        )

    result = trainer.fit(train_batches, validation_batches)

    assert result.best_checkpoint.exists()
    assert np.isfinite(result.best_val_mse)
    assert result.test_evaluated is False


def test_official_drop_last_discards_only_the_incomplete_batch():
    series = np.arange(64, dtype=np.float32)[:, None]
    refs = [
        WindowRef(0, 32, 32, 36, "train"),
        WindowRef(1, 33, 33, 37, "train"),
        WindowRef(2, 34, 34, 38, "train"),
    ]
    stats = fit_train_variable_stats(torch.from_numpy(series), eps=0.0)

    batches = list(
        iter_prepared_batches(
            series,
            refs,
            np.zeros_like(series, dtype=bool),
            mask_support_start=0,
            train_stats=stats,
            variant="clean",
            batch_size=2,
            patch_len=16,
            patch_stride=8,
            drop_last=True,
        )
    )

    assert len(batches) == 1
    assert batches[0].forecast_target.shape[0] == 2
    assert not batches[0].state.any()
