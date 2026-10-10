import pytest
import torch


def _sample(width=64,groups=8):
    from malr.models.e23.stacked_fusion import StackedGroupFusion
    torch.manual_seed(2)
    base=torch.randn(2,3,4,width)
    deltas=torch.randn(2,3,4,groups,width)
    routes=torch.softmax(torch.randn(2,3,4,groups,4),-1)
    rho=torch.full((2,3,4),.5)
    available=torch.ones(2,3,4,groups,dtype=torch.bool)
    return StackedGroupFusion(width,groups),[base,deltas,routes,rho,available]


def test_fusion_initializes_as_uniform_valid_residual_sum():
    fusion,args=_sample()
    output=fusion(*args)
    torch.testing.assert_close(output.group_weights,torch.full_like(output.group_weights,1/9),rtol=0,atol=0)
    torch.testing.assert_close(output.latents,args[0]+args[1].sum(-2)/9)
    for block in fusion.projection.weight.split(64,dim=1):
        torch.testing.assert_close(block,torch.eye(64),rtol=0,atol=0)
    torch.testing.assert_close(output.projected_deltas.sum(-2)+args[0],output.latents)


def test_fusion_missing_groups_and_fully_observed_patches_are_exact():
    fusion,args=_sample()
    args[-1][...,1:]=False
    args[-1][:,:,0]=False
    args[-2][:,:,1]=1
    output=fusion(*args)
    assert output.group_weights[...,2:].eq(0).all()
    torch.testing.assert_close(output.latents[:,:,0],args[0][:,:,0],rtol=0,atol=0)
    torch.testing.assert_close(output.latents[:,:,1],args[0][:,:,1],rtol=0,atol=0)
    torch.testing.assert_close(output.group_weights[:,:,0,0],torch.ones_like(args[-2][:,:,0]),rtol=0,atol=0)


def test_single_group_is_exact_bypass_without_new_parameters():
    fusion,args=_sample(width=8,groups=1)
    assert list(fusion.parameters()) == []
    output=fusion(*args)
    torch.testing.assert_close(output.latents,args[0]+args[1].squeeze(-2),rtol=0,atol=0)
    assert output.group_weights[...,1].eq(1).all()


def test_fusion_gradients_reach_every_valid_group_and_projection():
    fusion,args=_sample()
    args[1].requires_grad_()
    output=fusion(*args)
    output.latents.square().mean().backward()
    assert args[1].grad.abs().sum((0,1,2,4)).gt(0).all()
    assert fusion.projection.weight.grad.abs().sum()>0
    assert fusion.group_score[-1].weight.grad.abs().sum()>0


@pytest.mark.parametrize('field',[0,1,2,3])
def test_fusion_rejects_nonfinite_values(field):
    fusion,args=_sample()
    args[field].flatten()[0]=float('nan')
    with pytest.raises(ValueError,match='finite'): fusion(*args)


def test_fusion_rejects_wrong_geometry():
    fusion,args=_sample(); args[1]=args[1][...,:7,:]
    with pytest.raises(ValueError,match='shape'): fusion(*args)
