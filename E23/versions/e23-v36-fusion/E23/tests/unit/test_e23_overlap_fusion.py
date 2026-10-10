import pytest
import torch
from malr.models.explicit_repair import build_explicit_repair
from malr.models.e23.model import MultiExpertRepairEncoder


def test_weighted_overlap_recovers_coordinates_and_clamps_observations():
    candidates=torch.tensor([[[[0.,0.,0.,0.],[2.,2.,2.,2.],[4.,4.,4.,4.]]]],requires_grad=True)
    validity=torch.ones_like(candidates,dtype=torch.bool);validity[:,:,2,2:]=False
    mask=torch.zeros_like(validity);mask[:,:,0,3]=True;mask[:,:,1,1]=True
    observed=torch.zeros_like(candidates);observed[:,:,0,3]=3;observed[:,:,1,1]=3
    weights=torch.ones_like(candidates);weights[:,:,1]=3;weights=weights.requires_grad_()
    out=build_explicit_repair(candidates,observed,mask,validity,torch.zeros(1,1),torch.ones(1,1),
        window_len=6,patch_len=4,stride=2,candidate_weights=weights)
    torch.testing.assert_close(out.normalized_sequence[0,:,0],torch.tensor([0.,0.,1.5,3.,2.5,2.5]))
    assert out.sequence_observed[0,3,0]
    out.normalized_sequence.sum().backward()
    assert torch.isfinite(weights.grad).all() and weights.grad.abs().sum()>0
    assert candidates.grad[0,0,0,3]==0 and candidates.grad[0,0,1,1]==0


def test_weighted_overlap_rejects_nonpositive_valid_weights():
    shape=(1,1,2,4);values=torch.ones(shape);mask=torch.zeros(shape,dtype=torch.bool)
    with pytest.raises(ValueError,match='positive finite'):
        build_explicit_repair(values,values,mask,torch.ones_like(mask),torch.zeros(1,1),torch.ones(1,1),
            window_len=6,patch_len=4,stride=2,candidate_weights=torch.zeros(shape))


def build(mode):
    torch.manual_seed(2021)
    return MultiExpertRepairEncoder(3,96,16,8,64,64,variable_selection='window_context_topk_v1',
        recovery_schedule='iterative_tv_h_v1',overlap_fusion=mode)


def test_zero_initial_fusion_preserves_model_and_masked_inputs():
    a=build('uniform_v1');b=build('position_weighted_v1')
    for k,v in a.state_dict().items():assert torch.equal(v,b.state_dict()[k])
    torch.manual_seed(9);values=torch.randn(2,96,3);mask=torch.rand(2,96,3)>.55
    mean,std=torch.zeros(2,3),torch.ones(2,3);starts=torch.arange(12)[None].expand(2,-1)*8+1000
    out=a(values,mask,mean,std,starts);extended=b(values,mask,mean,std,starts)
    assert torch.equal(out.explicit.normalized_sequence,extended.explicit.normalized_sequence)
    torch.nn.init.normal_(b.overlap_fusion.score[-1].weight,std=.02)
    active=b(values,mask,mean,std,starts);changed=values.clone();changed[~mask]=12345
    other=b(changed,mask,mean,std,starts)
    assert torch.equal(active.explicit.normalized_sequence,other.explicit.normalized_sequence)
    assert torch.equal(active.explicit.sequence_observed,mask)
    active.explicit.normalized_sequence.square().mean().backward()
    assert b.overlap_fusion.score[0].weight.grad.norm()>0
    assert all(torch.isfinite(p.grad).all() for p in b.overlap_fusion.parameters() if p.grad is not None)
