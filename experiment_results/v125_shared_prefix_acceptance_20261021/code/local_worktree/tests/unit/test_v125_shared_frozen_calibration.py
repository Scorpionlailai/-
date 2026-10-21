"""Catch duplicated-prefix reuse that changes gradients, RNG or model state."""
import importlib.util
from types import SimpleNamespace

import pytest
import torch
from torch import nn


class Member(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 7), nn.Dropout(.4), nn.Linear(7, 4))

    def forward(self, explicit, router, points, mean, std):
        return SimpleNamespace(normalized_sequence=self.net(explicit.normalized_sequence)), mean * 0


class Pair(nn.Module):
    def __init__(self):
        super().__init__()
        self.members = nn.ModuleList([Member(), Member()])

    def forward(self, *args):
        result = [m(*args)[0].normalized_sequence for m in self.members]
        return SimpleNamespace(normalized_sequence=sum(result) / 2), args[3] * 0


class Repair(nn.Module):
    def __init__(self):
        super().__init__()
        self.prefix = nn.Linear(4, 4)
        self.recent_mean_projector = Pair()
        self.random_prefix = False

    def forward(self, values):
        x = self.prefix(values)
        if self.random_prefix:
            x = x + torch.rand_like(x)
        return self.recent_mean_projector(SimpleNamespace(normalized_sequence=x),
            None, None, torch.zeros_like(x), torch.ones_like(x))


def capture_function():
    # An intentional missing-feature assertion gives a readable RED run.
    name = 'integrations.e23.shared_frozen_calibration'
    assert importlib.util.find_spec(name) is not None, 'frozen-prefix capture not implemented'
    from integrations.e23.shared_frozen_calibration import capture_frozen_calibration_inputs
    return capture_frozen_calibration_inputs


def make_model():
    torch.manual_seed(41)
    model = Repair().eval().requires_grad_(False)
    for member in model.recent_mean_projector.members:
        member.train().requires_grad_(True)
    return model


def test_shared_prefix_preserves_independent_dropout_outputs_and_main_gradients():
    capture = capture_function()
    model = make_model()
    values = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 10
    states = {k: v.clone() for k, v in model.state_dict().items()}
    shared = capture(model, lambda: model(values))
    assert not shared[0].normalized_sequence.requires_grad
    for index, member in enumerate(model.recent_mean_projector.members):
        torch.manual_seed(90 + index)
        args = (SimpleNamespace(normalized_sequence=model.prefix(values)),
                None, None, torch.zeros_like(values), torch.ones_like(values))
        reference = member(*args)[0].normalized_sequence
        expected_rng = torch.get_rng_state().clone()
        expected_grads = torch.autograd.grad(reference.abs().mean(), tuple(member.parameters()))
        torch.manual_seed(90 + index)
        actual = member(*shared)[0].normalized_sequence
        actual_grads = torch.autograd.grad(actual.abs().mean(), tuple(member.parameters()))
        assert torch.equal(actual, reference)
        assert torch.equal(torch.get_rng_state(), expected_rng)
        assert all(torch.equal(a, b) for a, b in zip(actual_grads, expected_grads))
    assert all(torch.equal(v, states[k]) for k, v in model.state_dict().items())
    assert 'forward' not in model.recent_mean_projector.__dict__


def test_rejects_live_prefix_gradients_and_restores_forward_on_failure():
    capture = capture_function()
    model = make_model()
    model.prefix.requires_grad_(True)
    with pytest.raises(ValueError, match='frozen'):
        capture(model, lambda: model(torch.ones(3, 4)))
    model.prefix.requires_grad_(False)
    def broken():
        model(torch.ones(3, 4))
        raise RuntimeError('actual prefix failure')
    with pytest.raises(RuntimeError, match='actual prefix failure'):
        capture(model, broken)
    assert 'forward' not in model.recent_mean_projector.__dict__
    assert model(torch.ones(3, 4))[0].normalized_sequence.shape == (3, 4)


def test_rejects_stochastic_prefix_instead_of_changing_branch_rng():
    capture = capture_function()
    model = make_model()
    model.random_prefix = True
    with pytest.raises(ValueError, match='RNG'):
        capture(model, lambda: model(torch.ones(3, 4)))
    assert 'forward' not in model.recent_mean_projector.__dict__
