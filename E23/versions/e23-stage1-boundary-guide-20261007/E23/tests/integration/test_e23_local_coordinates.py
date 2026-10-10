from dataclasses import replace
import json
import pytest
import torch
import yaml
from integrations.e23 import train as tr
from integrations.e23.config import load_e23_settings
from integrations.e23.forecast_sensitive import parameter_hash
from malr.models.e23.coordinates import numeric_features, repair_scale
from malr.models.e23.window_context import WindowContextEncoder
from tests.integration.test_e23_point_candidates import setup
from tests.integration.test_e23_stage1 import _batch


def test_centered_feature_ignores_level_and_hidden_payload_with_scale_floor():
    values=torch.ones(1,96,1);mask=torch.ones_like(values,dtype=torch.bool);mask[:,40:60]=False
    std=torch.tensor([[1e-6]]);mean=torch.tensor([[500.]])
    front=WindowContextEncoder(64,96,16,8,coordinates='robust_local_v1')
    a,_=front.features(values,mask,mean,std)
    z,_=front.features(torch.where(mask,values,float('inf')),mask,mean+1000,std)
    assert torch.equal(a,z) and torch.isfinite(z).all()
    torch.testing.assert_close(a[0,0,0],std[0,0]/repair_scale(std)[0,0])
    assert a[0,0,40:60].eq(0).all()


def test_numeric_coordinate_conversion_restores_global_units():
    std=torch.tensor([[1e-6,.5,2.]]);mean=torch.tensor([[10.,20.,30.]])
    x=torch.randn(1,3,2,16,4)
    local=numeric_features(x,mean,std,'robust_local_v1')
    restored=local*repair_scale(std)[:,:,None,None,None]+mean[:,:,None,None,None]
    torch.testing.assert_close(restored,numeric_features(x,mean,std,'global_v1'))
    assert torch.equal(local,numeric_features(x,mean+99,std,'robust_local_v1'))


def test_local_body_and_rng_match_global_at_zero_output(tmp_path):
    p,cfg=setup(tmp_path);torch.manual_seed(502)
    old=tr.build_forecast_model(p,cfg,device=torch.device('cpu')).eval();rng=torch.get_rng_state()
    torch.manual_seed(502);m=tr.build_forecast_model(p,replace(cfg,repair_coordinates='robust_local_v1'),device=torch.device('cpu')).eval()
    assert torch.equal(rng,torch.get_rng_state()) and parameter_hash(old.predictor)==parameter_hash(m.predictor)
    for k,v in old.state_dict().items():assert torch.equal(v,m.state_dict()[k])
    b=_batch('validation');starts=tr._starts(b,stride=8,patches=12,device=torch.device('cpu'))
    with torch.no_grad():
        a,oa=old(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts)
        z,oz=m(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts)
    assert torch.equal(a,z) and torch.equal(oa.explicit.normalized_sequence,oz.explicit.normalized_sequence)


@pytest.mark.parametrize('tiny',[False,True])
def test_active_local_modules_finite_gradients_visible_and_hidden_isolation(tmp_path,tiny):
    p,cfg=setup(tmp_path);cfg=replace(cfg,repair_coordinates='robust_local_v1')
    m=tr.build_forecast_model(p,cfg,device=torch.device('cpu')).eval();b=_batch('train')
    mask=b.observed_mask.clone();mask[:,-16:]=False
    b=replace(b,observed_mask=mask,normalized_observed=b.normalized_observed.masked_fill(~mask,0))
    std=torch.full_like(b.query_std,1e-6) if tiny else b.query_std
    starts=tr._starts(b,stride=8,patches=12,device=torch.device('cpu'))
    with torch.no_grad():
        m.repair.front_context.output.weight.normal_(std=.02)
        m.repair.point_router.score[-1].weight.normal_(std=.02)
        m.repair.overlap_confidence.score[-1].weight.normal_(std=.02)
        m.repair.point_candidate_readout.output[-1].weight.normal_(std=.02)
        m.repair.point_refiner.output.weight.normal_(std=.02)
    out=m.repair(b.normalized_observed,b.observed_mask,b.query_mean,std,starts)
    canary=torch.where(b.observed_mask,b.normalized_observed,float('inf'))
    z=m.repair(canary,b.observed_mask,b.query_mean,std,starts)
    assert torch.equal(out.explicit.normalized_sequence,z.explicit.normalized_sequence)
    assert torch.equal(out.explicit.normalized_sequence[b.observed_mask],b.normalized_observed[b.observed_mask])
    loss,_=tr.stage1_objective(out.explicit.normalized_sequence,b.normalized_target,b.observed_mask,torch.tensor(0.),retrieval_weight=.1,loss_mode='global_mse_v1',query_std=std)
    loss.backward();assert torch.isfinite(loss)
    assert all(torch.isfinite(x.grad).all() for x in m.parameters() if x.grad is not None)
    assert m.repair.point_candidate_readout.output[-1].weight.grad.abs().sum()>0
    assert m.repair.point_refiner.output.weight.grad.abs().sum()>0


def test_local_checkpoint_roundtrip_and_wrong_identity_rejection(tmp_path,monkeypatch):
    p,_=setup(tmp_path);path=tmp_path/'True-True-False.yaml';doc=yaml.safe_load(path.read_text())
    doc['model'].update(variable_calibration='none',front_variable_context='none',repair_coordinates='robust_local_v1');path.write_text(yaml.safe_dump(doc))
    cfg=load_e23_settings(path,protocol=p)
    monkeypatch.setattr(tr,'iter_p4_batches',lambda _p,split,**kw:iter((_batch(split),)))
    s1=tr.train_repair(p,cfg,output=tmp_path/'s1',device=torch.device('cpu'),epochs=1)
    s2=tr.train_forecast(p,cfg,stage1_manifest=s1,output=tmp_path/'s2',device=torch.device('cpu'),epochs=1,selection_metric='mae',forecast_loss='mse_plus_0p2_mae',repair_retention='uniform_v1',repair_update_policy='freeze_front_v1')
    payload=tr.load_stage2_selection(s2,p,cfg,device=torch.device('cpu'))
    assert payload['repair_coordinates']=='robust_local_v1' and payload['repair_scale_floor']==.1
    record=json.loads(s2.read_text());record['repair_coordinates']='global_v1'
    wrong=tmp_path/'wrong.json';wrong.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='repair_coordinates.*mismatch'):tr.load_stage2_selection(wrong,p,cfg,device=torch.device('cpu'))


@pytest.mark.parametrize('value',['wrong',True,1])
def test_invalid_coordinates_rejected(tmp_path,value):
    p,_=setup(tmp_path);path=tmp_path/'True-True-False.yaml';doc=yaml.safe_load(path.read_text())
    doc['model'].update(variable_calibration='none',front_variable_context='none',repair_coordinates=value);path.write_text(yaml.safe_dump(doc))
    with pytest.raises(ValueError):load_e23_settings(path,protocol=p)
