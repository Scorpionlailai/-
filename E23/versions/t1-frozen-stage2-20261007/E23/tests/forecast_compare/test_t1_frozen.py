from types import SimpleNamespace
import copy

import torch
from torch import nn

from integrations.forecast_compare.t1_frozen import FrozenT1ForecastModel


class ToyImputer(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.BatchNorm1d(7)
        self.dropout = nn.Dropout(.5)
        self.bias = nn.Parameter(torch.ones(7))

    def forward(self, values, *, mask):
        return self.dropout(self.norm(values.transpose(1, 2)).transpose(1, 2)) + self.bias


class ToyPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.))

    def forward(self, values, mask, mean, std, *, patch_validity):
        self.inputs = (values.detach().clone(), mask.clone(), patch_validity.clone())
        return self.weight * values.mean((2, 3))


def fixture():
    torch.manual_seed(4)
    values = torch.randn(4, 96, 7)
    mask = torch.rand_like(values) > .4
    return SimpleNamespace(normalized_observed=values.masked_fill(~mask, 0),
                           observed_mask=mask, query_mean=torch.zeros(4, 7),
                           query_std=torch.ones(4, 7), normalized_target=values,
                           forecast_target=torch.randn(4, 7))


def test_freeze_includes_parameters_buffers_and_training_mode():
    model = FrozenT1ForecastModel(ToyImputer(), ToyPredictor())
    initial = {k: v.clone() for k, v in model.imputer.state_dict().items()}
    model.train()
    assert model.predictor.training
    assert all(not module.training for module in model.imputer.modules())
    assert all(not p.requires_grad for p in model.imputer.parameters())
    opt = torch.optim.Adam(model.predictor.parameters(), lr=.01)
    before = model.predictor.weight.detach().clone()
    for _ in range(3):
        opt.zero_grad()
        model(fixture()).square().mean().backward()
        opt.step()
    assert not torch.equal(before, model.predictor.weight)
    assert all(torch.equal(v, model.imputer.state_dict()[k]) for k, v in initial.items())
    assert all(p.grad is None for p in model.imputer.parameters())


def test_input_padding_masks_and_targets_are_separate():
    from malr.data.patching import patchify_preserving_values

    batch = fixture()
    model = FrozenT1ForecastModel(ToyImputer(), ToyPredictor()).eval()
    first = model(batch).detach().clone()
    values, mask, valid = model.predictor.inputs
    original = patchify_preserving_values(batch.normalized_observed, batch.observed_mask, 16, 8)
    assert torch.equal(mask, original.effective_mask)
    assert torch.equal(valid, original.validity)
    assert torch.equal(values[mask & valid], original.values[mask & valid])
    assert torch.equal(values[:, :, -1, -8:], values[:, :, -1, 7:8].expand(-1, -1, 8))
    assert values[~mask & valid].abs().sum() > 0
    batch.normalized_target.fill_(999)
    batch.forecast_target.fill_(-999)
    assert torch.equal(first, model(batch))


def test_frozen_forward_matches_existing_t1_interface_in_eval():
    from integrations.forecast_compare.t1_p4 import T1ForecastModel

    frozen = FrozenT1ForecastModel(ToyImputer(), ToyPredictor()).eval()
    existing = T1ForecastModel(copy.deepcopy(frozen.imputer), copy.deepcopy(frozen.predictor)).eval()
    batch = fixture()
    assert torch.equal(frozen(batch), existing(batch))
    assert all(torch.equal(left, right) for left, right in zip(frozen.predictor.inputs, existing.predictor.inputs))
