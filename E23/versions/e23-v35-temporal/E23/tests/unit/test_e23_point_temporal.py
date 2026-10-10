import torch
from malr.models.e23.model import MultiExpertRepairEncoder


def build(mode):
    torch.manual_seed(2021)
    return MultiExpertRepairEncoder(3,96,16,8,64,64,
        variable_selection='window_context_topk_v1',recovery_schedule='iterative_tv_h_v1',
        temporal_selection=mode)


def batch():
    torch.manual_seed(9); values=torch.randn(2,96,3)
    mask=torch.rand(2,96,3)>.55
    return values,mask,torch.zeros(2,3),torch.ones(2,3),torch.arange(12)[None].expand(2,-1)*8+1000


def test_zero_initial_point_context_preserves_original_single_pass_recovery():
    original=build('banded_v1'); extended=build('point_convolution_v1')
    for k,v in original.state_dict().items(): assert torch.equal(v,extended.state_dict()[k])
    a=original(*batch()); b=extended(*batch())
    assert torch.equal(a.explicit.normalized_sequence,b.explicit.normalized_sequence)
    assert torch.equal(a.router.weights,b.router.weights)


def test_active_point_context_has_gradients_and_ignores_hidden_inputs():
    model=build('point_convolution_v1')
    torch.nn.init.normal_(model.temporal.point_context.projection.weight,std=.02)
    values,mask,mean,std,starts=batch()
    first=model(values,mask,mean,std,starts)
    changed=values.clone(); changed[~mask]=100000
    second=model(changed,mask,mean,std,starts)
    assert torch.equal(first.explicit.normalized_sequence,second.explicit.normalized_sequence)
    assert torch.equal(first.explicit.sequence_observed,mask)
    assert torch.equal(first.explicit.normalized_sequence[mask],values[mask])
    first.explicit.normalized_sequence.square().mean().backward()
    grads=[p.grad for p in model.temporal.point_context.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert model.temporal.point_context.stem[0].weight.grad.norm()>0
