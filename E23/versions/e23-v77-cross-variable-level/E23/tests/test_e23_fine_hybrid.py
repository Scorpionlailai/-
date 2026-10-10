from dataclasses import replace

import pytest
import torch

from malr.models.e23.contracts import ExpertInputs
from malr.models.e23.fine_history import FineHistoricalExpert
from malr.models.e23.historical_candidates import HistoricalCandidateCache, HistoricalCandidateProvider
from malr.models.e23.hybrid_point import DenseRecoveryBlock


def fine_inputs(p):
    values = torch.arange(96.).reshape(1,1,96//p,p)
    mask = torch.ones_like(values,dtype=torch.bool)
    mask.reshape(1,1,96)[...,4:8]=False
    values = values.masked_fill(~mask,0)
    return ExpertInputs(base_latents=torch.zeros(1,1,96//p,64),patch_values=values,
        patch_mask=mask,patch_validity=torch.ones_like(mask),rho=mask.float().mean(-1),
        patch_starts=torch.arange(100,196,p)[None],query_mean=torch.zeros(1,1),
        query_std=torch.ones(1,1),damaged=~mask.all(-1))


@pytest.mark.parametrize('p',[4,8,16])
def test_parent_observations_and_starts(p):
    h=FineHistoricalExpert(p,64,1)
    x=fine_inputs(p);parent=h.parent_inputs(x)
    assert parent.patch_values.shape==(1,1,6,16)
    assert torch.equal(parent.patch_values.reshape(96),x.patch_values.reshape(96))
    assert torch.equal(parent.patch_starts,torch.arange(100,196,16)[None])
    assert torch.equal(parent.patch_mask.reshape(96),x.patch_mask.reshape(96))


@pytest.mark.parametrize('p',[4,8,16])
def test_candidate_crop_and_missing_child_inherits_visible_parent(monkeypatch,p):
    def fake(provider,parent):
        raw=torch.arange(16.).reshape(1,1,1,1,16).expand(1,1,6,4,16).clone()
        valid=torch.ones(1,1,6,4,dtype=torch.bool)
        starts=torch.full((1,1,6,4),10,dtype=torch.long)
        return HistoricalCandidateCache(raw,torch.zeros(1,1,6,4,5),torch.zeros(1,1,6,4),
            valid,valid.any(-1),dict(candidate_raw=raw,candidate_starts=starts,
                candidate_valid=valid,candidate_count=valid.sum(-1)))
    monkeypatch.setattr(HistoricalCandidateProvider,'prepare_candidates',fake)
    h=FineHistoricalExpert(p,64,1);x=fine_inputs(p);cache=h.prepare_candidates(x)
    assert cache.normalized_raw.shape==(1,1,96//p,4,p)
    for index in range(16//p):
        assert torch.equal(cache.normalized_raw[0,0,index,0],torch.arange(index*p,(index+1)*p).float())
        assert cache.provenance['candidate_starts'][0,0,index,0]==10+index*p
    assert torch.equal(cache.available,x.damaged)
    assert (cache.provenance['parent_candidate_starts']+16<=cache.provenance['parent_query_starts'][:,None,:,None]).all()
    evidence=h.from_cache(x,cache)
    assert torch.isfinite(evidence.message).all()
    if p==4:
        assert not x.patch_mask[0,0,1].any()
        assert evidence.available[0,0,1]


def test_parent_query_does_not_see_hidden_values():
    h=FineHistoricalExpert(4,64,1);x=fine_inputs(4)
    contaminated=replace(x,patch_values=torch.where(x.patch_mask,x.patch_values,torch.full_like(x.patch_values,99999)))
    # Forward's patch construction masks hidden values; parent assembly also must mask.
    assert torch.equal(h.parent_inputs(x).patch_values,h.parent_inputs(contaminated).patch_values)


@pytest.mark.parametrize('attention',[False,True])
def test_dense_block_all_missing_and_gradients(attention):
    torch.manual_seed(2)
    block=DenseRecoveryBlock(64,attention)
    x=torch.randn(2,7,64,96,requires_grad=True)
    output=block(x,torch.zeros(2,7,96,dtype=torch.bool))
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in block.parameters() if p.grad is not None)


def test_cnn_attention_have_equal_parameters_and_initial_weights():
    torch.manual_seed(3);cnn=DenseRecoveryBlock(64,False)
    torch.manual_seed(3);hybrid=DenseRecoveryBlock(64,True)
    assert sum(p.numel() for p in cnn.parameters())==sum(p.numel() for p in hybrid.parameters())
    assert all(torch.equal(v,hybrid.state_dict()[k]) for k,v in cnn.state_dict().items())
