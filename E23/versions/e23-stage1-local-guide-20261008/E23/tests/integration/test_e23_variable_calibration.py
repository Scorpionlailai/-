from dataclasses import replace
import json
import torch
import pytest
import yaml

from integrations.e23 import train as tr
from integrations.e23.config import load_e23_settings
from integrations.e23.forecast_sensitive import parameter_hash
from malr.models.e23.variable import SparseVariableExpert
from tests.integration.test_e23_point_candidates import setup
from tests.integration.test_e23_stage1 import _batch
from tests.unit.test_e23_variable import _inputs


@pytest.mark.parametrize('readout',['none','eight_head_v1'])
def test_calibration_zero_start_preserves_rng_head_function(tmp_path,readout):
    p,cfg=setup(tmp_path);cfg=replace(cfg,point_candidate_readout=readout)
    torch.manual_seed(502);a=tr.build_forecast_model(p,cfg,device=torch.device('cpu')).eval()
    rng=torch.get_rng_state();torch.manual_seed(502)
    z=tr.build_forecast_model(p,replace(cfg,variable_calibration='identity_scale_v1'),device=torch.device('cpu')).eval()
    assert torch.equal(rng,torch.get_rng_state()) and parameter_hash(a.predictor)==parameter_hash(z.predictor)
    for k,v in a.repair.state_dict().items():assert torch.equal(v,z.repair.state_dict()[k]),k
    b=_batch('validation');starts=tr._starts(b,stride=8,patches=12,device=torch.device('cpu'))
    with torch.no_grad():
        pred,oa=a(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts)
        new,oz=z(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts)
    assert torch.equal(pred,new) and torch.equal(oa.explicit.normalized_sequence,oz.explicit.normalized_sequence)


def test_candidate_features_use_original_source_global_scale_and_both_identities():
    inp=_inputs();inp=replace(inp,query_mean=torch.arange(7.).expand(2,7),query_std=torch.arange(1.,8.).expand(2,7))
    v=SparseVariableExpert(7,6,8,selection='window_context_topk_v1',contextual=True,calibration='identity_scale_v1')
    cache=v.prepare_sources(inp,epoch=0,explore=False);captured=[]
    handle=v.calibration[0].register_forward_pre_hook(lambda _,args:captured.append(args[0].detach().clone()))
    v.from_cache(inp,cache);handle.remove();features=captured[0]
    ix=(0,0,1,0,1);source=int(cache.source_ix[ix]);mask=cache.source_observed[ix]
    expected=torch.where(mask,cache.visible_values[ix]*inp.query_std[0,source]+inp.query_mean[0,source],0)
    torch.testing.assert_close(features[ix][16:22],expected)
    torch.testing.assert_close(features[ix][:8],v.source_embedding.weight[source])
    torch.testing.assert_close(features[ix][8:16],v.target_embedding.weight[0])
    torch.testing.assert_close(features[ix][-4:],torch.tensor([float(source),float(inp.query_std[0,source].log()),0.,0.]))


def test_nonzero_calibration_isolates_hidden_values_and_receives_gradient(tmp_path):
    p,cfg=setup(tmp_path);cfg=replace(cfg,variable_calibration='identity_scale_v1')
    m=tr.build_forecast_model(p,cfg,device=torch.device('cpu')).eval()
    with torch.no_grad():m.repair.variable.calibration[-1].weight.normal_(std=.03)
    b=_batch('train');starts=tr._starts(b,stride=8,patches=12,device=torch.device('cpu'))
    with torch.no_grad():
        a,_=m(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts)
        polluted=torch.where(b.observed_mask,b.normalized_observed,torch.full_like(b.normalized_observed,1e9))
        z,o=m(polluted,b.observed_mask,b.query_mean,b.query_std,starts)
    assert torch.equal(a,z)
    assert torch.equal(o.explicit.normalized_sequence[b.observed_mask],b.normalized_observed[b.observed_mask])
    out=m.repair(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts)
    loss,_=tr.stage1_objective(out.explicit.normalized_sequence,b.normalized_target,b.observed_mask,
        torch.tensor(0.),retrieval_weight=.1,loss_mode='global_mse_v1',query_std=b.query_std)
    loss.backward()
    assert m.repair.variable.calibration[0].weight.grad.abs().sum()>0
    assert m.repair.variable.calibration[-1].weight.grad.abs().sum()>0
    assert all(torch.isfinite(x.grad).all() for x in m.parameters() if x.grad is not None)


def test_calibration_settings_validate(tmp_path):
    p,_=setup(tmp_path);path=tmp_path/'True-True-False.yaml';doc=yaml.safe_load(path.read_text())
    doc['model']['variable_calibration']='identity_scale_v1';path.write_text(yaml.safe_dump(doc))
    assert load_e23_settings(path,protocol=p).variable_calibration=='identity_scale_v1'
    doc['model']['variable_calibration']=True;path.write_text(yaml.safe_dump(doc))
    with pytest.raises(ValueError):load_e23_settings(path,protocol=p)


def test_calibration_checkpoint_roundtrip_and_identity_rejection(tmp_path,monkeypatch):
    p,_=setup(tmp_path);path=tmp_path/'True-True-False.yaml';doc=yaml.safe_load(path.read_text())
    doc['model']['variable_calibration']='identity_scale_v1';path.write_text(yaml.safe_dump(doc))
    cfg=load_e23_settings(path,protocol=p)
    monkeypatch.setattr(tr,'iter_p4_batches',lambda _p,split,**kw:iter((_batch(split),)))
    s1=tr.train_repair(p,cfg,output=tmp_path/'s1',device=torch.device('cpu'),epochs=1)
    s2=tr.train_forecast(p,cfg,stage1_manifest=s1,output=tmp_path/'s2',device=torch.device('cpu'),epochs=1,
        selection_metric='mae',forecast_loss='mse_plus_0p2_mae',repair_retention='uniform_v1',repair_update_policy='freeze_front_v1')
    payload=tr.load_stage2_selection(s2,p,cfg,device=torch.device('cpu'))
    assert payload['variable_calibration']=='identity_scale_v1'
    rec=json.loads(s2.read_text());rec['variable_calibration']='wrong';wrong=tmp_path/'wrong.json';wrong.write_text(json.dumps(rec))
    with pytest.raises(ValueError,match='variable_calibration.*mismatch'):
        tr.load_stage2_selection(wrong,p,cfg,device=torch.device('cpu'))
