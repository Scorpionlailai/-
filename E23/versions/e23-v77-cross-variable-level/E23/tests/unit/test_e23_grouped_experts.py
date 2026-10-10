from dataclasses import replace

import numpy as np
import pytest
import torch

from malr.models.e23.historical import HistoricalExpert
from malr.models.e23.iterative import IterativeTVHController
from malr.prototypes.online_bank import OnlinePrototypeMemory
from tests.unit.test_e23_variable import _inputs


def parts(variables=7, groups=2, dtype=torch.float64):
    from malr.models.e23.historical_candidates import HistoricalCandidateProvider
    from malr.models.e23.stacked_group import TVHGroup
    torch.manual_seed(30)
    inputs = _inputs(variables=variables)
    inputs = replace(inputs, patch_starts=inputs.patch_starts+32,
                     base_latents=inputs.base_latents.to(dtype), patch_values=inputs.patch_values.to(dtype),
                     rho=inputs.rho.to(dtype), query_mean=inputs.query_mean.to(dtype), query_std=inputs.query_std.to(dtype))
    provider = HistoricalCandidateProvider(6, 8, 8, variables, similarity_threshold=-1).to(dtype=dtype)
    memory = OnlinePrototypeMemory(patch_len=6, patch_stride=3, train_end=32, variable_count=variables)
    for c in range(variables):
        memory.stage_patch(c, 0, np.linspace(-1, 1, 6), np.eye(8)[0], observed=True)
    provider.refresh_snapshot(memory.publish_epoch(1))
    networks = [TVHGroup(variables, 6, 8).to(dtype=dtype) for _ in range(groups)]
    # Exercise all contextual projections rather than relying on zero initialization.
    with torch.no_grad():
        for group in networks:
            group.historical.context_query.weight.normal_(std=.1)
            group.variable.context_query.weight.normal_(std=.1)
            group.variable.source_context_projection.weight.normal_(std=.1)
            group.temporal.source_context_projection.weight.normal_(std=.1)
    return inputs, provider, networks


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_grouped_h_v_t_match_independent_reference(dtype):
    from malr.models.e23.stacked_group_kernels import (prepare_grouped_variable_sources,
        grouped_historical, grouped_variable, grouped_temporal)
    inputs, provider, groups = parts(dtype=dtype)
    states = torch.stack([inputs.base_latents+.2*i for i in range(2)], -2).requires_grad_()
    embedding = torch.stack([g.condition_embedding.weight[0] for g in groups])
    hcache = provider.prepare_candidates(inputs)
    vcaches = prepare_grouped_variable_sources(inputs, [g.variable for g in groups], epoch=1, explore=False)
    context = torch.rand(inputs.damaged.shape) > .5
    results = [grouped_historical(inputs, states, hcache, [g.historical for g in groups], embedding),
               grouped_variable(inputs, states, [g.variable for g in groups], vcaches,
                                source_context_allowed=context, embeddings=embedding),
               grouped_temporal(inputs, states, [g.temporal for g in groups],
                                source_context_allowed=context, embeddings=embedding)]
    tolerance = 1e-10 if dtype == torch.float64 else 1e-6
    for index, result in enumerate(results):
        for i, group in enumerate(groups):
            current = replace(inputs, base_latents=states[..., i, :])
            if index == 0:
                expected = group.historical.from_cache(current, hcache, round_embedding=embedding[i])
            elif index == 1:
                expected = group.variable.from_cache(current, vcaches[i], round_embedding=embedding[i], source_context_allowed=context)
            else:
                expected = group.temporal.from_cache(current, group.temporal.prepare_sources(inputs), round_embedding=embedding[i], source_context_allowed=context)
            for field in ('message', 'quality', 'available'):
                torch.testing.assert_close(getattr(result, field)[..., i, :] if field != 'available' else result.available[..., i],
                                           getattr(expected, field), rtol=0, atol=tolerance)
            # Compare each group's parameter and state gradients with its independent module.
            parameters = list(getattr(group, ('historical', 'variable', 'temporal')[index]).parameters())
            actual_grads = torch.autograd.grad(result.message[..., i, :].square().mean()+result.quality[..., i, :].square().mean(),
                                               [states, *parameters], retain_graph=True, allow_unused=True)
            expected_grads = torch.autograd.grad(expected.message.square().mean()+expected.quality.square().mean(),
                                                 [states, *parameters], retain_graph=True, allow_unused=True)
            for actual, expected_grad in zip(actual_grads, expected_grads):
                assert (actual is None) == (expected_grad is None)
                if actual is not None:
                    torch.testing.assert_close(actual, expected_grad, rtol=0, atol=tolerance)


def test_grouped_v_uses_each_groups_source_embeddings():
    from malr.models.e23.stacked_group_kernels import prepare_grouped_variable_sources, grouped_variable
    inputs, _, groups = parts()
    with torch.no_grad():
        for i, g in enumerate(groups):
            g.variable.target_embedding.weight.fill_(1)
            g.variable.source_embedding.weight.zero_()
            g.variable.source_embedding.weight[i+1].fill_(20)
    caches = prepare_grouped_variable_sources(inputs, [g.variable for g in groups], epoch=1, explore=False)
    assert not torch.equal(caches[0].source_ix, caches[1].source_ix)
    result = grouped_variable(inputs, inputs.base_latents[..., None, :].expand(-1,-1,-1,2,-1),
        [g.variable for g in groups], caches, source_context_allowed=None,
        embeddings=torch.stack([g.condition_embedding.weight[0] for g in groups]))
    for i, g in enumerate(groups):
        torch.testing.assert_close(result.message[..., i, :], g.variable.from_cache(inputs, caches[i]).message, rtol=0, atol=1e-10)


@pytest.mark.parametrize('observed', [False, True])
def test_grouped_single_variable_and_empty_evidence_are_finite(observed):
    from malr.models.e23.stacked_group_kernels import prepare_grouped_variable_sources, grouped_tv_h
    inputs, provider, groups = parts(variables=1)
    if not observed:
        inputs = replace(inputs, patch_mask=torch.zeros_like(inputs.patch_mask))
        provider.index = None
    result = grouped_tv_h(inputs, groups, provider.prepare_candidates(inputs),
        prepare_grouped_variable_sources(inputs, [g.variable for g in groups], epoch=1, explore=False),
        torch.ones(*inputs.damaged.shape, 2, 3, dtype=torch.bool), None)
    assert torch.isfinite(result.latents).all()
    if not observed:
        torch.testing.assert_close(result.latents, inputs.base_latents[..., None, :].expand_as(result.latents), rtol=0, atol=0)
        assert result.weights[..., 0].eq(1).all()


def test_grouped_tv_h_matches_one_round_reference():
    from malr.models.e23.stacked_group_kernels import prepare_grouped_variable_sources, grouped_tv_h
    inputs, provider, groups = parts()
    result = grouped_tv_h(inputs, groups, provider.prepare_candidates(inputs),
        prepare_grouped_variable_sources(inputs, [g.variable for g in groups], epoch=1, explore=False),
        torch.ones(*inputs.damaged.shape, 2, 3, dtype=torch.bool), None)
    for i, group in enumerate(groups):
        h = HistoricalExpert(6,8,8,7,contextual=True,similarity_threshold=-1).double()
        h.load_state_dict({**provider.state_dict(), **group.historical.state_dict()})
        h.retrieval.load_state_dict(provider.retrieval.state_dict()); h.index = provider.index
        controller = IterativeTVHController(8,1).double()
        controller.round_embedding.load_state_dict(group.condition_embedding.state_dict())
        expected = controller(inputs, historical=h, variable=group.variable, temporal=group.temporal,
            router=group.router,epoch=1,explore_variable=False,specialist_exposure=0,specialist_seed=1)
        torch.testing.assert_close(result.latents[..., i, :], expected.router.latents, rtol=0, atol=1e-10)
        torch.testing.assert_close(result.weights[..., i, :], expected.router.weights, rtol=0, atol=1e-10)
        torch.testing.assert_close(result.preview_gates[..., i, :], expected.iterations[0].preview_gates, rtol=0, atol=1e-10)


def test_grouped_padding_and_hidden_values_cannot_change_output():
    from malr.models.e23.stacked_group_kernels import prepare_grouped_variable_sources, grouped_tv_h
    inputs, provider, groups = parts()
    validity = inputs.patch_validity.clone(); validity[..., -1, -2:] = False
    inputs = replace(inputs, patch_validity=validity)
    def run(x):
        return grouped_tv_h(x, groups, provider.prepare_candidates(x),
            prepare_grouped_variable_sources(x,[g.variable for g in groups],epoch=1,explore=False),
            torch.ones(*x.damaged.shape,2,3,dtype=torch.bool),None)
    before = run(inputs)
    changed = inputs.patch_values.clone(); changed[~(inputs.patch_mask & validity)] = 100000
    after = run(replace(inputs,patch_values=changed))
    torch.testing.assert_close(before.latents,after.latents,rtol=0,atol=0)
