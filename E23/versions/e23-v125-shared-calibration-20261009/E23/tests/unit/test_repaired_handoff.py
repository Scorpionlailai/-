from pathlib import Path

import numpy as np
import torch

from malr.data.repaired import (
    RepairedSplit,
    iter_repaired_batches,
    load_repaired_split,
    save_repaired_split,
)
from malr.models.patchtst import PatchTSTCore
from malr.models.reconstruction import aggregate_patch_predictions
from malr.models.repaired_forecast import MaskAwareForecastModel


def _split() -> RepairedSplit:
    values = np.arange(8, dtype=np.float32).reshape(1, 8, 1)
    values[0, 3, 0] = 77.0
    mask = np.ones_like(values, dtype=np.bool_)
    mask[0, 3, 0] = False
    return RepairedSplit(
        values=values,
        observed_mask=mask,
        query_mean=np.zeros((1, 1), dtype=np.float32),
        query_std=np.ones((1, 1), dtype=np.float32),
        targets=np.zeros((1, 1, 2), dtype=np.float32),
        input_starts=np.asarray([10], dtype=np.int64),
    )


def test_repaired_split_round_trip_preserves_mask_and_reconstructed_values(
    tmp_path: Path,
) -> None:
    path = tmp_path / "repaired.npz"
    original = _split()

    save_repaired_split(path, original)
    loaded = load_repaired_split(path)

    np.testing.assert_array_equal(loaded.observed_mask, original.observed_mask)
    np.testing.assert_allclose(loaded.values, original.values)
    assert loaded.values[0, 3, 0] == 77.0
    assert not loaded.observed_mask[0, 3, 0]


def test_repaired_batch_does_not_zero_values_at_originally_missing_coordinates() -> None:
    batch = next(
        iter(
            iter_repaired_batches(
                _split(), batch_size=1, patch_len=4, patch_stride=2
            )
        )
    )

    reconstructed, _ = aggregate_patch_predictions(
        batch.patch_values,
        window_len=8,
        patch_len=4,
        stride=2,
        validity=batch.patch_validity,
    )

    assert reconstructed[0, 3, 0] == 77.0
    assert (~batch.patch_mask & batch.patch_validity).any()


def test_forecast_model_has_an_explicit_trainable_mask_path() -> None:
    core = PatchTSTCore(
        c_in=1,
        seq_len=8,
        pred_len=2,
        patch_len=4,
        stride=2,
        d_model=4,
        n_heads=1,
        d_ff=8,
        n_layers=1,
        dropout=0.0,
    )
    model = MaskAwareForecastModel(core)
    batch = next(
        iter(
            iter_repaired_batches(
                _split(), batch_size=1, patch_len=4, patch_stride=2
            )
        )
    )
    with torch.no_grad():
        model.mask_projection.weight.fill_(0.25)

    missing_forecast = model(
        batch.patch_values,
        batch.patch_mask,
        batch.query_mean,
        batch.query_std,
        patch_validity=batch.patch_validity,
    )
    observed_forecast = model(
        batch.patch_values,
        torch.ones_like(batch.patch_mask),
        batch.query_mean,
        batch.query_std,
        patch_validity=batch.patch_validity,
    )

    assert model.mask_projection.weight.requires_grad
    assert not torch.allclose(missing_forecast, observed_forecast)
