from dataclasses import replace
from types import SimpleNamespace
import torch
from malr.models.e23.global_dense import GlobalDenseRefiner


def inputs():
    observed = torch.zeros(2,96,3,dtype=torch.bool)
    observed[:,::9,:2] = True
    valid = torch.ones_like(observed)
    valid[:,-2:] = False
    explicit = SimpleNamespace(sequence_validity=valid, sequence_observed=observed,
                               observed_sequence=torch.zeros(2,96,3))
    return torch.full((2,96,3),12.), explicit, torch.zeros(2,3), torch.ones(2,3)


def test_zero_gate_preserves_initial_output_and_adds_only_small_projection():
    torch.manual_seed(17)
    original = GlobalDenseRefiner(3).eval()
    torch.manual_seed(17)
    candidate = GlobalDenseRefiner(3, contextual_gate=True).eval()
    for name, value in original.state_dict().items():
        assert torch.equal(value, candidate.state_dict()[name])
    assert torch.equal(original(*inputs()), candidate(*inputs()))
    assert candidate.parameter_count-original.parameter_count == 65*3


def test_contextual_gate_can_correct_large_false_events_without_visible_changes():
    candidate = GlobalDenseRefiner(3, contextual_gate=True).eval()
    with torch.no_grad():
        candidate.guide_gate.bias.fill_(2.)
    sequence, explicit, mean, std = inputs()
    out = candidate(sequence, explicit, mean, std)
    active = explicit.sequence_validity & ~explicit.sequence_observed
    assert out[:,:,0][active[:,:,0]].max() < 1.
    assert torch.equal(out[explicit.sequence_observed],sequence[explicit.sequence_observed])
    assert torch.equal(out[~explicit.sequence_validity],sequence[~explicit.sequence_validity])
    # Third variable has no visible values: don't manufacture a zero guide.
    assert torch.equal(out[:,:,2],sequence[:,:,2])


def test_zero_initialized_gate_has_a_finite_learning_signal():
    candidate = GlobalDenseRefiner(3, contextual_gate=True)
    sequence, explicit, mean, std = inputs()
    out = candidate(sequence, explicit, mean, std)
    out.square().mean().backward()
    grad = candidate.guide_gate.weight.grad
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0
