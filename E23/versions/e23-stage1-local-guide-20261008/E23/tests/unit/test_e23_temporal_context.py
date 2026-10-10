from dataclasses import replace

import torch

from malr.models.e23.temporal import MultiScaleTemporalExpert
from tests.unit.test_e23_temporal import _inputs


def test_temporal_context_reads_updated_source_states_without_reencoding():
    # Catches static source messages and repeated raw source encoding.
    inputs = _inputs(11, visible_patches=[0])
    expert = MultiScaleTemporalExpert(6, 8, contextual=True)
    with torch.no_grad():
        expert.source_context_projection.weight.copy_(torch.eye(8))
    calls = []
    handle = expert.source_encoder.register_forward_hook(lambda *_: calls.append(1))
    cache = expert.prepare_sources(inputs)
    first = expert.from_cache(inputs, cache)
    state = inputs.base_latents.clone()
    state[:, :, 0] *= -1
    second = expert.from_cache(replace(inputs, base_latents=state), cache)
    third = expert.from_cache(inputs, cache)
    handle.remove()
    assert calls == [1]
    assert not torch.allclose(first.message[:, :, 9], second.message[:, :, 9])
    torch.testing.assert_close(first.message, third.message)
    first.message.square().mean().backward()
    grad = expert.source_context_projection.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_temporal_context_ignores_hidden_and_invalid_raw_values():
    inputs = _inputs(11)
    valid = inputs.patch_validity.clone()
    valid[..., -1] = False
    values = inputs.patch_values.detach().clone().requires_grad_()
    inputs = replace(inputs, patch_values=values, patch_validity=valid)
    expert = MultiScaleTemporalExpert(6, 8, contextual=True)
    first = expert.from_cache(inputs, expert.prepare_sources(inputs))
    altered = values.detach().clone()
    altered[~(valid & inputs.patch_mask)] = 1e5
    other = replace(inputs, patch_values=altered)
    second = expert.from_cache(other, expert.prepare_sources(other))
    torch.testing.assert_close(first.message, second.message)
    torch.testing.assert_close(first.quality, second.quality)
    first.message.square().mean().backward()
    assert torch.equal(values.grad[~valid], torch.zeros_like(values.grad[~valid]))


def test_temporal_context_cannot_create_an_observed_source():
    inputs = _inputs(11, visible_patches=[])
    expert = MultiScaleTemporalExpert(6, 8, contextual=True)
    with torch.no_grad():
        expert.source_context_projection.weight.fill_(20.)
    actual = expert.from_cache(inputs, expert.prepare_sources(inputs))
    assert not actual.available.any()
    assert not actual.candidates.valid.any()
    assert torch.equal(actual.message, torch.zeros_like(actual.message))


def test_temporal_single_expert_uses_original_sources_when_context_is_disabled():
    inputs = _inputs(11, visible_patches=[0])
    expert = MultiScaleTemporalExpert(6, 8, contextual=True)
    with torch.no_grad():
        expert.source_context_projection.weight.copy_(torch.eye(8))
    cache = expert.prepare_sources(inputs)
    mask = torch.zeros_like(inputs.damaged)
    first = expert.from_cache(inputs, cache, source_context_allowed=mask)
    state = inputs.base_latents.clone()
    state[:, :, 0] *= -10
    second = expert.from_cache(replace(inputs, base_latents=state), cache,
                               source_context_allowed=mask)
    torch.testing.assert_close(first.message[:, :, 9], second.message[:, :, 9], rtol=0, atol=0)
