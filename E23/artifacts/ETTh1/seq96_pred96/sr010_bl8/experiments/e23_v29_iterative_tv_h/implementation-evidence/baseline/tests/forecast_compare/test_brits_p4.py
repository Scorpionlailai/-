from pathlib import Path

import torch
from torch import nn

from integrations.forecast_compare.brits_p4 import (
    BritsForecastModel, brits_deltas, brits_inputs, load_brits,
)
from integrations.forecast_compare.protocol import P4Batch


def sample_batch() -> P4Batch:
    mask = torch.ones(2, 96, 7, dtype=torch.bool)
    mask[:, 10:18, 0] = False
    return P4Batch(
        normalized_observed=torch.where(mask, torch.ones(2, 96, 7), 0.0),
        normalized_target=torch.full((2, 96, 7), 9.0),
        observed_mask=mask,
        query_mean=torch.full((2, 7), 3.0), query_std=torch.full((2, 7), 2.0),
        forecast_target=torch.zeros(2, 7, 96),
    )


def test_brits_deltas_match_official_rule():
    mask = torch.tensor([[[1.0], [0.0], [0.0], [1.0], [0.0]]])
    assert brits_deltas(mask).flatten().tolist() == [0.0, 1.0, 2.0, 1.0, 2.0]


def test_brits_inputs_keep_hidden_targets_out_of_model_inputs():
    batch = sample_batch()
    data = brits_inputs(batch, supervised=True)
    assert data['forward']['X'][0, 10:18, 0].count_nonzero() == 0
    assert data['forward']['X'][0, 0, 0].item() == 5.0
    assert data['backward']['X'][0, -18:-10, 0].count_nonzero() == 0
    assert data['forward']['deltas'].shape == (2, 96, 7)
    assert data['indicating_mask'][0, 10:18, 0].all()
    assert data['X_holdout'][0, 10, 0].item() == 21.0


def test_brits_official_model_and_shared_head_forward():
    model = load_brits(Path('/home/xyw/paper/SAITS'), device=torch.device('cpu'))
    batch = sample_batch()
    with torch.no_grad():
        repaired, _ = model.impute(brits_inputs(batch, supervised=False))
    assert repaired.shape == (2, 96, 7)
    assert torch.isfinite(repaired).all()

    class CaptureHead(nn.Module):
        def forward(self, values, mask, mean, std, *, patch_validity):
            assert values.shape[0] == 2
            return torch.zeros(2, 7, 96)

    with torch.no_grad():
        assert BritsForecastModel(model, CaptureHead())(batch).shape == (2, 7, 96)


def test_brits_forecast_converts_global_repair_back_to_query_scale():
    class ConstantImputer(nn.Module):
        def impute(self, data):
            return torch.full_like(data['forward']['X'], 7.0), []

    class CheckHead(nn.Module):
        def forward(self, values, mask, mean, std, *, patch_validity):
            assert (values == 2.0).any()  # (global 7 - query mean 3) / std 2
            assert (values == 1.0).any()  # exact observed value retained
            return torch.zeros(2, 7, 96)

    assert BritsForecastModel(ConstantImputer(), CheckHead())(sample_batch()).shape == (2, 7, 96)
