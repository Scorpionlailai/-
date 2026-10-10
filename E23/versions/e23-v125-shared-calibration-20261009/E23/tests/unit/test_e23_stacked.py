from dataclasses import replace

import pytest
import torch

from tests.unit.test_e23_grouped_experts import parts


def fixture(layers=2,groups=2):
    from malr.models.e23.stacked import StackedTVHController
    inputs,provider,_=parts(dtype=torch.float32)
    controller=StackedTVHController(7,6,8,layers=layers,groups=groups)
    return inputs,provider,controller


def test_stack_parameters_are_distinct_across_layers_and_groups():
    _,provider,controller=fixture()
    parameters=[p for layer in controller.layers for group in layer.groups for p in group.parameters()]
    assert len({p.data_ptr() for p in parameters})==len(parameters)
    for layer in controller.layers:
        for group in layer.groups:
            assert not hasattr(group.historical,'key_encoder')
            assert group.variable.source_embedding.weight.shape==(7,8)
            assert group.temporal.source_encoder[0].in_features==12
    assert not any(p is q for p in provider.parameters() for q in controller.parameters())


def test_next_layer_reads_previous_fused_state_without_detach(monkeypatch):
    import malr.models.e23.stacked as module
    inputs,provider,controller=fixture()
    seen=[]; method=module.grouped_tv_h
    def capture(inputs,*args,**kwargs):
        seen.append(inputs.base_latents); return method(inputs,*args,**kwargs)
    monkeypatch.setattr(module,'grouped_tv_h',capture)
    output=controller(inputs,historical=provider)
    torch.testing.assert_close(seen[1],output.layers[0].latents,rtol=0,atol=0)
    assert seen[1].grad_fn is not None
    output.router.latents.square().mean().backward()
    for layer in controller.layers:
        for group in layer.groups:
            for expert in (group.historical,group.variable,group.temporal):
                assert expert.message_projector[0].weight.grad.abs().sum()>0
            assert group.variable.source_embedding.weight.grad.abs().sum()>0
            assert group.router.residual['historical'][0].weight.grad.abs().sum()>0


def test_stack_historical_lookup_occurs_once(monkeypatch):
    inputs,provider,controller=fixture()
    calls=[]; query=provider.index.query
    def counted(*args,**kwargs): calls.append(1); return query(*args,**kwargs)
    monkeypatch.setattr(provider.index,'query',counted)
    controller(inputs,historical=provider)
    assert calls==[1]


def test_stack_specialist_sample_occurs_once_across_all_groups_and_layers(monkeypatch):
    import malr.models.e23.stacked as module
    inputs,provider,controller=fixture()
    calls=[]; sample=module.sample_specialist_exposure
    def counted(damaged,availability,**kwargs):
        calls.append(availability); return sample(damaged,availability,**kwargs)
    monkeypatch.setattr(module,'sample_specialist_exposure',counted)
    output=controller(inputs,historical=provider,specialist_exposure=1,specialist_seed=9)
    assert len(calls)==1 and calls[0].shape==(*inputs.damaged.shape,3)
    kinds=None
    for layer in output.layers:
        allowed=layer.allowed[output.specialist_selected].any(dim=1)
        assert allowed.sum(-1).le(1).all()
        if kinds is not None: assert torch.equal(kinds,allowed)
        kinds=allowed


def test_singleton_variable_cannot_read_temporal_source_updates(monkeypatch):
    import malr.models.e23.stacked as module
    inputs,provider,controller=fixture()
    def controlled(damaged,availability,**kwargs):
        allowed=torch.zeros_like(availability)
        allowed[:,0,:,1]=True; allowed[:,1:,:,2]=True
        return allowed & availability,torch.ones_like(damaged)
    monkeypatch.setattr(module,'sample_specialist_exposure',controlled)
    for layer in controller.layers:
        for group in layer.groups:
            with torch.no_grad(): group.variable.source_context_projection.weight.copy_(torch.eye(8))
    before=controller(inputs,historical=provider,specialist_exposure=1)
    method=module.grouped_tv_h
    # Perturb T network strongly, including neighboring targets, in every group/layer.
    with torch.no_grad():
        for layer in controller.layers:
            for group in layer.groups:
                for parameter in group.temporal.parameters(): parameter.add_(.7)
    after=controller(inputs,historical=provider,specialist_exposure=1)
    for a,b in zip(before.layers,after.layers):
        torch.testing.assert_close(a.latents[:,0],b.latents[:,0],rtol=0,atol=0)


@pytest.mark.parametrize('disabled',['group','temporal','variable','historical'])
def test_disabled_group_or_expert_cannot_change_any_downstream_layer(disabled):
    inputs,provider,controller=fixture()
    kwargs={'disabled_groups':frozenset([(0,0)])} if disabled=='group' else {'disabled_experts':frozenset([disabled])}
    before=controller(inputs,historical=provider,**kwargs)
    with torch.no_grad():
        targets=[controller.layers[0].groups[0]] if disabled=='group' else [getattr(g,disabled) for layer in controller.layers for g in layer.groups]
        for target in targets:
            for parameter in target.parameters(): parameter.add_(.3)
    after=controller(inputs,historical=provider,**kwargs)
    for a,b in zip(before.layers,after.layers): torch.testing.assert_close(a.latents,b.latents,rtol=0,atol=0)


def test_all_unavailable_returns_original_base_exactly():
    inputs,provider,controller=fixture()
    inputs=replace(inputs,patch_mask=torch.zeros_like(inputs.patch_mask)); provider.index=None
    output=controller(inputs,historical=provider)
    for layer in output.layers:
        torch.testing.assert_close(layer.latents,inputs.base_latents,rtol=0,atol=0)
        assert layer.group_weights[...,0].eq(1).all()


def test_fully_observed_and_padding_patch_states_are_anchored():
    inputs,provider,controller=fixture()
    mask=inputs.patch_mask.clone(); validity=inputs.patch_validity.clone()
    mask[:,:,0]=True; validity[:,:,-1]=False
    inputs=replace(inputs,patch_mask=mask,patch_validity=validity,
                   rho=(mask & validity).float().mean(-1),damaged=inputs.damaged & validity.any(-1) & ~(mask & validity).all(-1))
    output=controller(inputs,historical=provider)
    for layer in output.layers:
        torch.testing.assert_close(layer.latents[:,:,0],inputs.base_latents[:,:,0],rtol=0,atol=0)
        torch.testing.assert_close(layer.latents[:,:,-1],inputs.base_latents[:,:,-1],rtol=0,atol=0)


def test_stack_group_batch_matches_module_reference():
    from malr.models.e23.stacked_group_kernels import prepare_grouped_variable_sources,grouped_tv_h
    inputs,provider,controller=fixture(layers=1)
    groups=controller.layers[0].groups
    expected=grouped_tv_h(inputs,groups,provider.prepare_candidates(inputs),
        prepare_grouped_variable_sources(inputs,[g.variable for g in groups],epoch=0,explore=False),
        torch.ones(*inputs.damaged.shape,2,3,dtype=torch.bool),None)
    fused=controller.layers[0].fusion(inputs.base_latents,expected.latents-inputs.base_latents[...,None,:],
        expected.weights,inputs.rho,expected.available[...,1:].any(-1))
    actual=controller(inputs,historical=provider)
    torch.testing.assert_close(actual.router.latents,fused.latents,rtol=0,atol=0)
