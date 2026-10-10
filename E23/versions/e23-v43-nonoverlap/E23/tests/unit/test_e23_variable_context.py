from dataclasses import replace

import torch

from malr.models.e23.variable import SparseVariableExpert
from tests.unit.test_e23_variable import _inputs


def _expert(variables=3):
    return SparseVariableExpert(variables, 6, 8, selection="window_context_topk_v1", contextual=True)


def test_variable_context_reads_both_updated_target_and_source_states():
    # Catches using only the old raw sources or ignoring the sequential target.
    inputs = _inputs(variables=3)
    expert = _expert()
    with torch.no_grad():
        expert.source_context_projection.weight.copy_(torch.eye(8))
    cache = expert.prepare_sources(inputs, epoch=0, explore=False)
    first = expert.from_cache(inputs, cache)
    target = inputs.base_latents.clone()
    target[:, 0] *= -1
    from_target = expert.from_cache(replace(inputs, base_latents=target), cache)
    source = inputs.base_latents.clone()
    source[:, 1] *= -1
    from_source = expert.from_cache(replace(inputs, base_latents=source), cache)
    assert not torch.allclose(first.message[:, 0], from_target.message[:, 0])
    assert not torch.allclose(first.message[:, 0], from_source.message[:, 0])
    first.message.square().mean().backward()
    grad = expert.source_context_projection.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_variable_context_keeps_source_ids_and_original_validity():
    inputs = _inputs()
    expert = _expert(7)
    cache = expert.prepare_sources(inputs, epoch=4, explore=True)
    first = expert.from_cache(inputs, cache)
    other = replace(inputs, base_latents=inputs.base_latents * -2)
    second = expert.from_cache(other, cache)
    assert torch.equal(first.provenance["source_variables"], second.provenance["source_variables"])
    assert torch.equal(first.candidates.valid, second.candidates.valid)
    assert first.provenance["candidate_count"].max() <= 12
    assert second.candidates.weights[~second.candidates.valid].sum() == 0
    empty_mask = inputs.patch_mask.clone()
    empty_mask[:, 1:] = False
    empty_inputs = replace(inputs, patch_mask=empty_mask)
    empty = expert.from_cache(empty_inputs, expert.prepare_sources(empty_inputs, epoch=4, explore=False))
    assert not empty.available[:, 0].any()
    assert torch.equal(empty.message[:, 0], torch.zeros_like(empty.message[:, 0]))


def test_variable_context_single_channel_has_zero_available_evidence():
    expert = _expert(1)
    inputs = _inputs(variables=1)
    out = expert.from_cache(inputs, expert.prepare_sources(inputs, epoch=0, explore=False))
    assert not out.available.any()
    assert not out.candidates.valid.any()
    assert torch.isfinite(out.message).all()
