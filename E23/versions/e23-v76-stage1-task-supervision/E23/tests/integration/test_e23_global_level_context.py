"""Observed global source levels can enter recovery without changing final head."""
from dataclasses import replace
import json

import pytest
import torch
import yaml

from integrations.e23 import train as tr
from integrations.e23.config import load_e23_settings
from integrations.e23.forecast_sensitive import parameter_hash
from tests.integration.test_e23_segment_attention import configuration as attention_configuration, long_batch


def configuration(tmp_path):
    protocol, _ = attention_configuration(tmp_path)
    path = tmp_path/'True-True-False.yaml';doc=yaml.safe_load(path.read_text())
    doc['model'].update(point_context_readout='none',front_variable_context='signature_plain_pos_v3',
                        front_level_context='global_level_v1')
    path.write_text(yaml.safe_dump(doc))
    return protocol,load_e23_settings(path,protocol=protocol)


def batch():
    b=long_batch();C=b.query_mean.shape[-1]
    means=torch.linspace(-.8,.8,C)[None].expand_as(b.query_mean).clone()
    return replace(b,query_mean=means)


def test_configuration_rejects_unknown_level_policy_and_incompatible_front(tmp_path):
    protocol,cfg=configuration(tmp_path)
    assert cfg.front_level_context=='global_level_v1'
    path=tmp_path/'True-True-False.yaml'; original=yaml.safe_load(path.read_text())
    for bad in ['unknown',False]:
        doc=yaml.safe_load(path.read_text());doc['model']['front_level_context']=bad
        path.write_text(yaml.safe_dump(doc))
        with pytest.raises(ValueError):load_e23_settings(path,protocol=protocol)
        path.write_text(yaml.safe_dump(original))
    doc=original;doc['model']['front_context']='none';path.write_text(yaml.safe_dump(doc))
    with pytest.raises(ValueError):load_e23_settings(path,protocol=protocol)


def test_zero_level_branch_preserves_old_operator_parameters_rng_and_forecast(tmp_path):
    protocol,cfg=configuration(tmp_path)
    torch.manual_seed(2023);old=tr.build_forecast_model(protocol,replace(cfg,front_level_context='none'),device=torch.device('cpu')).eval();rng=torch.get_rng_state()
    torch.manual_seed(2023);new=tr.build_forecast_model(protocol,cfg,device=torch.device('cpu')).eval()
    assert torch.equal(rng,torch.get_rng_state());assert parameter_hash(old.predictor)==parameter_hash(new.predictor)
    for key,value in old.state_dict().items():assert torch.equal(value,new.state_dict()[key]),key
    assert sum(p.numel() for p in new.repair.parameters())-sum(p.numel() for p in old.repair.parameters())==128
    assert not new.repair.front_context.level_projection.weight.count_nonzero()
    b=batch();starts=tr._starts(b,stride=8,patches=12,device=torch.device('cpu'))
    with torch.no_grad():
        a,ra=old(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts)
        z,rz=new(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts)
    assert torch.equal(a,z);assert torch.equal(ra.explicit.normalized_sequence,rz.explicit.normalized_sequence)


def test_exact_visible_only_global_values_mean_and_five_feature_evidence_interface(tmp_path):
    protocol,cfg=configuration(tmp_path);model=tr.build_forecast_model(protocol,cfg,device=torch.device('cpu')).eval();front=model.repair.front_context;b=batch()
    mask=b.observed_mask.clone();mask[:,:,1]=False;valid=torch.ones_like(mask);valid[:,90,0]=False
    values=b.normalized_observed.clone();values[~mask | ~valid]=float('nan');seen=mask & valid
    captured=[];handle=front.level_projection.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().clone()))
    with torch.no_grad():_,points,evidence=front(values,mask,b.query_mean,b.query_std,torch.ones(b.normalized_observed.shape[0],7,12,16,dtype=torch.bool),valid_mask=valid,return_points=True)
    handle.remove();assert evidence.shape==(b.normalized_observed.shape[0]*7,5,96);assert points.shape==(b.normalized_observed.shape[0]*7,64,96)
    global_values=torch.where(seen,torch.where(seen,values,0)*b.query_std[:,None]+b.query_mean[:,None],0)
    mean=global_values.sum(1)/seen.sum(1).clamp_min(1)
    expected=torch.stack((global_values.transpose(1,2),mean[:,:,None].expand(-1,-1,96)),2)
    expected=expected.reshape(b.normalized_observed.shape[0]*7,2,96).masked_fill(~valid.transpose(1,2).reshape(b.normalized_observed.shape[0]*7,1,96),0)
    assert torch.equal(captured[0],expected);assert not captured[0].reshape(b.normalized_observed.shape[0],7,2,96)[:,1].count_nonzero()
    assert torch.isfinite(points).all()


def test_activated_global_source_level_reaches_another_variable_signature(tmp_path):
    protocol,cfg=configuration(tmp_path);model=tr.build_forecast_model(protocol,cfg,device=torch.device('cpu')).eval();front=model.repair.front_context;b=batch();valid=torch.ones_like(b.observed_mask);patch_valid=torch.ones(b.normalized_observed.shape[0],7,12,16,dtype=torch.bool)
    with torch.no_grad():
        front.level_projection.weight[:,1,0].copy_(torch.linspace(-.3,.3,64))
        first=front.variable_context.blocks[0];first.project.weight.zero_();first.project.bias.zero_();first.project.weight[:,:,0].copy_(torch.eye(64));first.scale1.fill_(.1)
        _,a,_=front(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,patch_valid,return_points=True)
        means=b.query_mean.clone();means[:,1]+=1
        _,z,_=front(b.normalized_observed,b.observed_mask,means,b.query_std,patch_valid,return_points=True)
    assert (a.reshape(b.normalized_observed.shape[0],7,64,96)[:,0]-z.reshape(b.normalized_observed.shape[0],7,64,96)[:,0]).abs().max()>1e-7


def test_hidden_and_invalid_payloads_do_not_change_global_context(tmp_path):
    protocol,cfg=configuration(tmp_path);model=tr.build_forecast_model(protocol,cfg,device=torch.device('cpu')).eval();front=model.repair.front_context;b=batch();valid=torch.ones_like(b.observed_mask);valid[:,90,0]=False;patch_valid=torch.ones(b.normalized_observed.shape[0],7,12,16,dtype=torch.bool)
    with torch.no_grad():
        front.level_projection.weight.normal_(std=.1)
        _,a,e=front(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,patch_valid,valid_mask=valid,return_points=True)
        for poison in [float('nan'),float('inf'),-1e20]:
            values=torch.where(b.observed_mask & valid,b.normalized_observed,poison)
            _,z,f=front(values,b.observed_mask,b.query_mean,b.query_std,patch_valid,valid_mask=valid,return_points=True)
            assert torch.equal(a,z);assert torch.equal(e,f)
    assert torch.isfinite(a).all()


def test_global_level_projection_learns_from_hidden_reconstruction(tmp_path):
    protocol,cfg=configuration(tmp_path);model=tr.build_forecast_model(protocol,cfg,device=torch.device('cpu'));b=batch();starts=tr._starts(b,stride=8,patches=12,device=torch.device('cpu'));optimizer=torch.optim.Adam(model.repair.parameters(),lr=.001)
    for step in range(2):
        optimizer.zero_grad();out=model.repair(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts)
        loss,_=tr.stage1_objective(out.explicit.normalized_sequence,b.normalized_target,b.observed_mask,torch.tensor(0.),retrieval_weight=.1,loss_mode='global_mse_v1',query_std=b.query_std,structure_mode='recent_mean_v2',gap_structure_mode='long_gap_moments_v1')
        loss.backward()
        if step:assert model.repair.front_context.level_projection.weight.grad.abs().sum()>0
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step()


def test_checkpoint_level_identity_and_frozen_stage2_roundtrip(tmp_path,monkeypatch):
    protocol,cfg=configuration(tmp_path);monkeypatch.setattr(tr,'iter_p4_batches',lambda _p,split,**kwargs:iter((batch(),)))
    s1=tr.train_repair(protocol,cfg,output=tmp_path/'s1',device=torch.device('cpu'),epochs=1)
    s2=tr.train_forecast(protocol,cfg,stage1_manifest=s1,output=tmp_path/'s2',device=torch.device('cpu'),epochs=1,selection_metric='mae',forecast_loss='mse_plus_0p2_mae',repair_retention='uniform_v1',repair_update_policy='frozen_v1',extra_mse_checkpoint=True)
    state1=torch.load(json.loads(s1.read_text())['checkpoint'],weights_only=False)['model_state']
    for choice in [s2,s2.parent/'e23-stage2-mse.json']:
        payload=tr.load_stage2_selection(choice,protocol,cfg,device=torch.device('cpu'))
        assert payload['front_level_context']=='global_level_v1';assert payload['front_context_features']==7;assert payload['front_evidence_features']==5;assert payload['front_level_extra_parameters']==128
        assert all(torch.equal(v,payload['model_state']['repair.'+k]) for k,v in state1.items())
    record=json.loads(s2.read_text());record['front_level_context']='none';wrong=tmp_path/'wrong.json';wrong.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='front_level_context.*mismatch'):tr.load_stage2_selection(wrong,protocol,cfg,device=torch.device('cpu'))
