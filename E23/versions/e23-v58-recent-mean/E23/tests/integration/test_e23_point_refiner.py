from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml

from integrations.e23 import train as training
from integrations.e23.config import load_e23_settings
from integrations.e23.forecast_sensitive import parameter_hash
from malr.models.e23.contracts import RouterOutput
from malr.models.e23.point_refiner import PointResidualRefiner
from malr.models.explicit_repair import build_explicit_repair
from malr.data.patching import patchify_with_state, patchify_preserving_values
from tests.integration.test_e23_stage1 import _protocol_settings, _batch

ROOT = Path(__file__).resolve().parents[2]


def settings(tmp_path, protocol, span):
    document = yaml.safe_load((ROOT/'configs/e23_multi_expert_p4_width64_iterative_tv_h_r1_v29.yaml').read_text())
    document['training']['seed'] = protocol.training_seed
    document['model']['point_refiner_span'] = span
    path = tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump(document,sort_keys=False))
    return load_e23_settings(path,protocol=protocol)


def inputs():
    torch.manual_seed(42)
    value=torch.randn(2,96,3)
    mask=torch.rand(2,96,3)>.4
    valid=torch.ones_like(mask);valid[:,-1]=False
    patches=patchify_with_state(value,mask,16,8,validity=valid)
    mean=torch.randn(2,3);std=torch.rand(2,3)+.3
    explicit=build_explicit_repair(torch.randn_like(patches.values),patches.values,
        patches.effective_mask,patches.validity,mean,std,window_len=96,patch_len=16,stride=8)
    weights=torch.randn(2,3,12,4).softmax(-1)
    router=RouterOutput(torch.randn(2,3,12,64),weights,{},torch.ones_like(weights,dtype=torch.bool))
    return explicit,router,mean,std


def test_scopes_share_exact_parameters_and_preserve_unselected_coordinates():
    explicit,router,mean,std=inputs()
    torch.manual_seed(123);tail=PointResidualRefiner(64,96,16,8,32)
    torch.manual_seed(123);full=PointResidualRefiner(64,96,16,8,96)
    assert parameter_hash(tail)==parameter_hash(full)
    assert sum(p.numel() for p in tail.parameters())==sum(p.numel() for p in full.parameters())
    with torch.no_grad():tail.output.bias.fill_(.25);full.output.bias.fill_(.25)
    a,b=tail(explicit,router,mean,std),full(explicit,router,mean,std)
    assert torch.equal(a.normalized_sequence[:,:64],explicit.normalized_sequence[:,:64])
    assert torch.equal(a.normalized_sequence[explicit.sequence_observed],explicit.normalized_sequence[explicit.sequence_observed])
    changed=explicit.sequence_missing.clone();changed[:,:64]=False
    torch.testing.assert_close(a.normalized_sequence[changed],explicit.normalized_sequence[changed]+.25)
    assert torch.equal(a.normalized_sequence[:,64:],b.normalized_sequence[:,64:])
    assert torch.equal(a.observation_patches,explicit.observation_patches)
    assert torch.equal(a.validity_patches,explicit.validity_patches)
    assert a.repaired_patches[~a.validity_patches].eq(0).all()
    patches=patchify_preserving_values(a.normalized_sequence,a.sequence_observed,16,8,validity=a.sequence_validity)
    torch.testing.assert_close(a.repaired_patches,patches.values.masked_fill(~patches.validity,0),rtol=0,atol=0)


def test_feature_fold_averages_valid_overlapping_copies_and_excludes_padding():
    explicit,router,_,_=inputs();refiner=PointResidualRefiner(64,96,16,8,32)
    values=router.latents.clone();values[:,:,11]=100
    result=refiner._points(values,explicit.validity_patches,explicit.coverage).reshape(2,3,64,96)
    torch.testing.assert_close(result[:,:,:,8],(values[:,:,0]+values[:,:,1])/2)
    assert result[:,:,:,-1].eq(0).all()


def test_zero_start_and_nonzero_branch_gradients_after_first_optimizer_update():
    explicit,router,mean,std=inputs();refiner=PointResidualRefiner(64,96,16,8,32)
    original=refiner(explicit,router,mean,std)
    torch.testing.assert_close(original.normalized_sequence,explicit.normalized_sequence,rtol=0,atol=0)
    optimizer=torch.optim.Adam(refiner.parameters(),lr=.001)
    for step in range(2):
        optimizer.zero_grad();out=refiner(explicit,router,mean,std)
        (out.normalized_sequence[explicit.sequence_missing]-1).square().mean().backward()
        assert refiner.output.weight.grad.abs().sum()>0
        if step:
            assert refiner.input[0].weight.grad.abs().sum()>0
            assert refiner.blocks[0].depthwise.weight.grad.abs().sum()>0
        assert all(torch.isfinite(p.grad).all() for p in refiner.parameters() if p.grad is not None)
        optimizer.step()


def test_added_capacity_does_not_change_original_head_or_original_repair_initialization(tmp_path):
    protocol,_=_protocol_settings();trial=settings(tmp_path,protocol,32)
    torch.manual_seed(502);base=training.build_forecast_model(protocol,replace(trial,point_refiner_span=0),device=torch.device('cpu'))
    torch.manual_seed(502);new=training.build_forecast_model(protocol,trial,device=torch.device('cpu'))
    assert parameter_hash(base.predictor)==parameter_hash(new.predictor)
    assert sum(p.numel() for p in new.predictor.parameters())==35427
    for name,value in base.repair.state_dict().items():
        torch.testing.assert_close(new.repair.state_dict()[name],value,rtol=0,atol=0)
    assert training._trial_identity(trial)['point_refiner_span']==32
    payload=training._trial_identity(trial)
    with pytest.raises(ValueError,match='point_refiner_span'):
        training._verify_trial_payload(payload,replace(trial,point_refiner_span=96))


def test_recovery_and_refinement_never_read_hidden_input_canary(tmp_path):
    protocol,_=_protocol_settings();trial=settings(tmp_path,protocol,32)
    model=training.build_forecast_model(protocol,trial,device=torch.device('cpu')).eval()
    with torch.no_grad():
        model.repair.point_refiner.output.weight.normal_(std=.05)
    batch=_batch('validation')
    mask=batch.observed_mask.clone();mask[:,80:88]=False
    batch=replace(batch,observed_mask=mask,normalized_observed=batch.normalized_target*mask)
    starts=training._starts(batch,stride=8,patches=12,device=torch.device('cpu'))
    changed=batch.normalized_observed.clone();changed[~batch.observed_mask]=1e6
    with torch.no_grad():
        a,_=model(batch.normalized_observed,batch.observed_mask,batch.query_mean,batch.query_std,starts)
        b,_=model(changed,batch.observed_mask,batch.query_mean,batch.query_std,starts)
    torch.testing.assert_close(a,b,rtol=0,atol=0)


@pytest.mark.parametrize('span',[True,16,64,-1])
def test_settings_reject_unsupported_point_scopes(tmp_path,span):
    protocol,_=_protocol_settings()
    with pytest.raises(ValueError):settings(tmp_path,protocol,span)
