from pathlib import Path

import torch
from torch import nn

from integrations.forecast_compare.protocol import P4Batch
from integrations.forecast_compare.t1_p4 import (
    T1ForecastModel, load_official_t1, t1_inputs, t1_repair_loss,
)


def sample_batch() -> P4Batch:
    mask = torch.ones(2, 96, 7, dtype=torch.bool)
    mask[:, 10:18, 0] = False
    return P4Batch(
        normalized_observed=torch.where(mask, torch.ones(2, 96, 7), 0.0),
        normalized_target=torch.full((2, 96, 7), 2.0),
        observed_mask=mask,
        query_mean=torch.full((2, 7), 3.0),
        query_std=torch.full((2, 7), 2.0),
        forecast_target=torch.zeros(2, 7, 96),
    )


def test_t1_inputs_hide_all_missing_targets():
    batch = sample_batch()
    values, mask = t1_inputs(batch)
    assert values[0, 10:18, 0].count_nonzero() == 0
    assert mask[0, 10:18, 0].count_nonzero() == 0
    assert values[0, 0, 0].item() == 5.0


def test_t1_repair_loss_only_scores_missing_coordinates():
    class Constant(nn.Module):
        def forward(self, values, *args, mask=None):
            return torch.full_like(values, 7.0)

    # Clean target is 2 * 2 + 3 = 7 in train-standardized units.
    assert t1_repair_loss(Constant(), sample_batch()).item() == 0.0


def test_official_t1_and_shared_head_forward():
    model = load_official_t1(Path('artifacts/upstream/T1'), device=torch.device('cpu'))
    batch = sample_batch()
    with torch.no_grad():
        prediction = model(*t1_inputs(batch), mask=t1_inputs(batch)[1])
    assert prediction.shape == (2, 96, 7)
    assert torch.isfinite(prediction).all()

    class CaptureHead(nn.Module):
        def forward(self, values, mask, mean, std, *, patch_validity):
            assert values.shape[0] == 2
            assert mean.shape == std.shape == (2, 7)
            return torch.zeros(2, 7, 96)

    forecast = T1ForecastModel(model, CaptureHead())
    with torch.no_grad():
        assert forecast(batch).shape == (2, 7, 96)
