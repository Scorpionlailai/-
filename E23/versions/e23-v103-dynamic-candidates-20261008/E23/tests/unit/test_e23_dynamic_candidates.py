"""Verify confidence-dependent counts, strict budgets and observed-only ranking."""
from dataclasses import replace

import torch

from malr.models.e23.contracts import ExpertInputs
from malr.models.e23.variable import SparseVariableExpert

POLICY = 'window_context_nucleus32_p90_cap16_v1'


def fixture(variables=21):
    torch.manual_seed(11)
    values=torch.randn(2,variables,3,16)
    mask=torch.ones_like(values,dtype=torch.bool)
    mask[1]=False
    return ExpertInputs(torch.randn(2,variables,3,64), values.masked_fill(~mask,0),
                        mask,torch.ones_like(mask),mask.float().mean(-1),
                        torch.tensor([[0,8,16],[0,8,16]]),torch.zeros(2,variables),
                        torch.ones(2,variables),torch.ones(2,variables,3,dtype=torch.bool))


def test_dynamic_counts_concentrated_diffuse_and_no_sources():
    x=fixture()
    model=SparseVariableExpert(21,16,64,topk=16,selection=POLICY,contextual=True)
    with torch.no_grad():
        model.source_embedding.weight.zero_()
        model.target_embedding.weight.zero_()
        model.target_embedding.weight[0,0]=8
        model.source_embedding.weight[1,0]=20
    selected,prior,keep=model._contextual_sources(x,epoch=1,explore=False)
    assert keep[0,0,0].sum()==1
    assert keep[0,2,0].sum()==16  # diffuse mass is truncated by the budget
    assert not keep[1].any()
    assert selected.shape==(2,21,3,16)
    target=torch.arange(21)[None,:,None,None]
    assert not ((selected==target)&keep).any()
    changed=replace(x,patch_values=x.patch_values.masked_fill(~x.patch_mask,1e6))
    other=model._contextual_sources(changed,epoch=1,explore=False)
    assert torch.equal(keep,other[2]) and torch.equal(selected,other[0])
    evidence=model(x,epoch=1,explore=False)
    assert not evidence.available[1].any()
    assert torch.isfinite(evidence.message).all()
    evidence.message.square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_large_variable_search_pool_is_bounded_and_training_repeatable():
    x=fixture(64)
    model=SparseVariableExpert(64,16,64,topk=16,selection=POLICY,contextual=True)
    cache=model.prepare_sources(x,epoch=2,explore=True)
    assert cache.provenance['source_pool_size']==32
    assert cache.provenance['selected_source_count'].max()<=16
    repeated=model.prepare_sources(x,epoch=2,explore=True)
    assert torch.equal(cache.source_ix,repeated.source_ix)
    assert torch.equal(cache.valid,repeated.valid)


def test_hidden_nan_is_excluded_from_dynamic_evidence():
    x=fixture()
    model=SparseVariableExpert(21,16,64,topk=16,selection=POLICY,contextual=True)
    changed=replace(x,patch_values=x.patch_values.masked_fill(~x.patch_mask,float('nan')))
    a=model(x,epoch=0,explore=False)
    b=model(changed,epoch=0,explore=False)
    assert torch.equal(a.message,b.message)
    assert torch.isfinite(b.message).all()


def test_dynamic_parameter_geometry_matches_existing_v():
    a=SparseVariableExpert(21,16,64,topk=4,selection='window_context_topk_v1',contextual=True)
    b=SparseVariableExpert(21,16,64,topk=16,selection=POLICY,contextual=True)
    b.load_state_dict(a.state_dict(),strict=True)
    assert {k:tuple(v.shape) for k,v in a.state_dict().items()}=={k:tuple(v.shape) for k,v in b.state_dict().items()}
