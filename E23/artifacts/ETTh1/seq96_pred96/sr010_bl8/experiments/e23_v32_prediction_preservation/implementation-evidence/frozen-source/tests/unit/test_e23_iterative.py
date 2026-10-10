from dataclasses import replace

import numpy as np
import pytest
import torch

from malr.models.e23.historical import HistoricalExpert
from malr.models.e23.temporal import MultiScaleTemporalExpert
from malr.models.e23.variable import SparseVariableExpert
from malr.models.e23.router import ExpertRouter
from malr.prototypes.online_bank import OnlinePrototypeMemory
from tests.unit.test_e23_variable import _inputs


def _parts(rounds=3):
    from malr.models.e23.iterative import IterativeTVHController
    torch.manual_seed(30)
    inputs = _inputs(variables=3)
    inputs = replace(inputs, patch_starts=inputs.patch_starts + 32)
    h = HistoricalExpert(6, 8, 8, 3, similarity_threshold=-1., contextual=True)
    memory = OnlinePrototypeMemory(patch_len=6, patch_stride=3, train_end=32, variable_count=3)
    for variable in range(3):
        memory.stage_patch(variable, 0, np.linspace(-1, 1, 6), np.eye(8)[0], observed=True)
    h.refresh_snapshot(memory.publish_epoch(1))
    parts = dict(historical=h, variable=SparseVariableExpert(3, 6, 8,
                 selection="window_context_topk_v1", contextual=True),
                 temporal=MultiScaleTemporalExpert(6, 8, contextual=True), router=ExpertRouter(8),
                 epoch=1, explore_variable=False, specialist_exposure=0., specialist_seed=7)
    return IterativeTVHController(8, rounds), inputs, parts


def test_iterative_round_uses_previous_written_state(monkeypatch):
    # Catches starting each round from z0 or detaching the recurrent state.
    controller, inputs, parts = _parts()
    seen = []
    method = parts["temporal"].from_cache
    def capture(arg, *args, **kwargs):
        seen.append(arg.base_latents)
        return method(arg, *args, **kwargs)
    monkeypatch.setattr(parts["temporal"], "from_cache", capture)
    out = controller(inputs, **parts)
    assert len(seen) == 3
    torch.testing.assert_close(seen[1], out.iterations[0].latents, rtol=0, atol=0)
    assert seen[1].grad_fn is not None
    assert not torch.equal(out.iterations[0].latents, inputs.base_latents)


def test_t_preview_changes_variable_target_and_source_inputs(monkeypatch):
    controller, inputs, parts = _parts(1)
    seen = []
    method = parts["variable"].from_cache
    def capture(arg, *args, **kwargs):
        seen.append(arg.base_latents)
        return method(arg, *args, **kwargs)
    monkeypatch.setattr(parts["variable"], "from_cache", capture)
    out = controller(inputs, **parts)
    assert len(seen) == 1
    assert not torch.equal(seen[0], inputs.base_latents)
    delta = out.router.deltas["temporal"]
    expected = inputs.base_latents + out.iterations[0].preview_gates[..., 0, None] * delta
    torch.testing.assert_close(seen[0], expected)


def test_v_preview_changes_historical_condition(monkeypatch):
    controller, inputs, parts = _parts(1)
    seen = {}
    for name in ("variable", "historical"):
        method = parts[name].from_cache
        def capture(arg, *args, _method=method, _name=name, **kwargs):
            seen[_name] = arg.base_latents
            return _method(arg, *args, **kwargs)
        monkeypatch.setattr(parts[name], "from_cache", capture)
    out = controller(inputs, **parts)
    assert not torch.equal(seen["historical"], seen["variable"])
    torch.testing.assert_close(seen["historical"], seen["variable"] +
        out.iterations[0].preview_gates[..., 1, None] * out.router.deltas["variable"])


@pytest.mark.parametrize("disabled", ["temporal", "variable"])
def test_disabled_expert_cannot_leak_through_previews(monkeypatch, disabled):
    controller, inputs, parts = _parts()
    expected = controller(inputs, **parts, disabled_experts=frozenset([disabled]))
    method = parts["router"].expert_delta
    def corrupt(arg, name, item):
        delta = method(arg, name, item)
        return delta + 1000 if name == disabled else delta
    monkeypatch.setattr(parts["router"], "expert_delta", corrupt)
    actual = controller(inputs, **parts, disabled_experts=frozenset([disabled]))
    torch.testing.assert_close(actual.router.latents, expected.router.latents, rtol=0, atol=0)


def test_specialist_mask_is_sampled_once_and_applies_before_previews(monkeypatch):
    import malr.models.e23.iterative as module
    controller, inputs, parts = _parts()
    calls = []
    sample = module.sample_specialist_exposure
    def counted(*args, **kwargs):
        calls.append(1)
        return sample(*args, **kwargs)
    monkeypatch.setattr(module, "sample_specialist_exposure", counted)
    parts["specialist_exposure"] = 1.
    out = controller(inputs, **parts)
    assert calls == [1]
    for item in out.iterations:
        assert torch.equal(item.allowed, out.iterations[0].allowed)
        assert item.allowed.sum(-1).eq(1).all()
        assert not item.preview_gates[..., 0][~item.allowed[..., 2]].any()
        assert not item.preview_gates[..., 1][~item.allowed[..., 1]].any()


def test_iterative_writeback_counts_each_delta_once():
    controller, inputs, parts = _parts(1)
    out = controller(inputs, **parts)
    expected = inputs.base_latents
    for i, name in enumerate(("historical", "variable", "temporal"), 1):
        expected = expected + out.router.weights[..., i, None] * out.router.deltas[name]
    torch.testing.assert_close(out.router.latents, expected, rtol=0, atol=0)


def test_specialist_variable_target_does_not_read_temporal_source_preview(monkeypatch):
    # Catches T leaking into a V-only target through another variable's state.
    import malr.models.e23.iterative as module
    controller, inputs, parts = _parts(1)
    def controlled(damaged, availability, **kwargs):
        allowed = torch.zeros_like(availability)
        allowed[:, 0, :, 1] = True
        allowed[:, 1, :, 2] = True
        allowed[:, 2, :, 0] = True
        return allowed & availability, torch.ones_like(damaged)
    monkeypatch.setattr(module, "sample_specialist_exposure", controlled)
    parts["specialist_exposure"] = 1.
    with torch.no_grad():
        parts["variable"].source_context_projection.weight.copy_(torch.eye(8))
    first = controller(inputs, **parts)
    method = parts["router"].expert_delta
    def changed(arg, name, item):
        delta = method(arg, name, item)
        return -100 * delta if name == "temporal" else delta
    monkeypatch.setattr(parts["router"], "expert_delta", changed)
    second = controller(inputs, **parts)
    torch.testing.assert_close(first.evidence["variable"].message[:, 0],
                               second.evidence["variable"].message[:, 0], rtol=0, atol=0)
