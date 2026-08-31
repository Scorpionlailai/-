from pathlib import Path

import numpy as np
import torch

from malr.data.masks import EpochMaskManager, generate_variable_missing_mask
from malr.data.normalization import fit_train_variable_stats
from malr.data.splits import DatasetProtocol, SplitBounds
from malr.data.windows import WindowRef, build_window_index
from malr.training.data import iter_prepared_batches
from malr.training.trainer import Trainer
from tests.unit.test_teacher_and_losses import _tiny_ours


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
    model = _tiny_ours()
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
