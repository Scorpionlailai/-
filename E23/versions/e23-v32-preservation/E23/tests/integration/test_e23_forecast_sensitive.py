import importlib
import torch
import pytest


def api():
    try:
        return importlib.import_module('integrations.e23.forecast_sensitive')
    except ModuleNotFoundError:
        pytest.fail('forecast-sensitive training interfaces not implemented')


@pytest.mark.parametrize('gain',[-1.,0.,1e-9,1.])
def test_weights_have_mean_one_and_stop_gradient(gain):
    hidden=torch.ones(3,96,7,dtype=torch.bool)
    hidden[0,:16,0]=False
    gains=torch.full((2,42),gain,requires_grad=True)
    gains.data[0,-1]=max(gain,0.)*100
    weights=api().gains_to_weights(gains,hidden)
    assert not weights.requires_grad
    assert torch.isfinite(weights).all() and (weights>0).all()
    for i in range(3):
        torch.testing.assert_close(weights[i][hidden[i]].mean(),torch.tensor(1.))
    assert torch.equal(weights[2],torch.ones_like(weights[2]))
    if gain<=0: assert torch.equal(weights,torch.ones_like(weights))


def test_weights_map_original_time_regions_and_skip_empty_windows():
    hidden=torch.ones(2,96,7,dtype=torch.bool);hidden[1]=False
    gains=torch.zeros(2,42);gains[:,5*7+4]=1
    w=api().gains_to_weights(gains,hidden)
    assert (w[0,80:96,4]>w[0,:80,4].max()).all()
    assert torch.equal(w[1],torch.ones_like(w[1]))


def test_uniform_loss_matches_global_mse_and_has_repair_gradient():
    x=torch.randn(2,96,7,requires_grad=True);y=torch.randn_like(x)
    hidden=torch.rand_like(x)>.3
    std=torch.full((2,7),1e-5,requires_grad=True)
    weights=torch.ones_like(x,requires_grad=True)
    loss=api().weighted_repair_loss(x,y,hidden,std,weights)
    expected=(((x-y)*std.detach()[:,None])[hidden]**2).mean()
    torch.testing.assert_close(loss,expected)
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert std.grad is None and weights.grad is None


def test_empty_hidden_loss_is_differentiable_zero():
    x=torch.randn(2,96,7,requires_grad=True)
    loss=api().weighted_repair_loss(x,torch.zeros_like(x),torch.zeros_like(x,dtype=torch.bool),torch.ones(2,7),torch.ones_like(x))
    loss.backward();assert loss.item()==0 and torch.equal(x.grad,torch.zeros_like(x))


def test_region_oracle_replaces_all_patch_copies_without_changing_provenance():
    from types import SimpleNamespace
    from torch import nn
    from malr.data.patching import patchify_preserving_values
    from integrations.forecast_compare.protocol import P4Batch
    from malr.data.windows import WindowRef

    class Repair(nn.Module):
        def forward(self,values,mask,mean,std,starts):
            patches=patchify_preserving_values(values,mask,16,8)
            return SimpleNamespace(explicit=SimpleNamespace(repaired_patches=patches.values.masked_fill(~patches.validity,0),observation_patches=patches.effective_mask,validity_patches=patches.validity))
        stride=8
        patch_count=12

    class Predictor(nn.Module):
        def forward(self,values,mask,mean,std,*,patch_validity):
            assert torch.equal(mask,expected_mask.expand_as(mask))
            assert torch.equal(patch_validity,expected_valid.expand_as(patch_validity))
            assert torch.equal(mean,torch.zeros_like(mean))
            assert torch.equal(std,torch.ones_like(std))
            assert not values.requires_grad
            # Every valid copy of missing original coordinate 88 is restored together.
            copies=values[:,4,10,8],values[:,4,11,0]
            assert torch.equal(copies[0],copies[1])
            assert torch.equal(values[:,0,0,0],torch.full_like(values[:,0,0,0],3))
            return values[:,4,11,0,None,None].expand(-1,7,96)

    mask=torch.ones(1,96,7,dtype=torch.bool);mask[:,88,4]=False
    observed=torch.zeros(1,96,7);observed[:,0,0]=3
    target=observed.clone();target[:,88,4]=1
    p=patchify_preserving_values(observed,mask,16,8)
    expected_mask,expected_valid=p.effective_mask,p.validity
    batch=P4Batch(observed,target,mask,torch.zeros(1,7),torch.ones(1,7),torch.ones(1,7,96),refs=(WindowRef(0,96,96,192,"train"),))
    teacher=nn.Module();teacher.repair=Repair();teacher.predictor=Predictor();teacher.eval()
    gains=api().region_oracle_gains(teacher,batch,subset=16)
    assert gains.shape==(1,42) and not gains.requires_grad
    expected=torch.zeros_like(gains);expected[:,5*7+4]=1
    torch.testing.assert_close(gains,expected)
