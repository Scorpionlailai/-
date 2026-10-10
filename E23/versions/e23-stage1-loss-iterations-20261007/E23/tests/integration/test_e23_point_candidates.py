from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch
import yaml

from integrations.e23 import train as tr
from integrations.e23.config import load_e23_settings
from integrations.e23.forecast_sensitive import parameter_hash
from malr.models.e23.point_candidates import PointCandidateReadout
from malr.models.e23.contracts import CandidateEvidence, ExpertEvidence
from tests.integration.test_e23_point_series import settings
from tests.integration.test_e23_stage1 import _protocol_settings, _batch


def setup(tmp_path):
    p,_=_protocol_settings();cfg=settings(tmp_path,p,True,True)
    path=tmp_path/'True-True-False.yaml';doc=yaml.safe_load(path.read_text())
    doc['model']['point_candidate_readout']='eight_head_v1'
    path.write_text(yaml.safe_dump(doc))
    return p,load_e23_settings(path,protocol=p)


def test_zero_readout_preserves_body_rng_head_and_full_function(tmp_path):
    p,cfg=setup(tmp_path)
    torch.manual_seed(502);old=tr.build_forecast_model(p,replace(cfg,point_candidate_readout='none'),device=torch.device('cpu')).eval()
    rng=torch.get_rng_state()
    torch.manual_seed(502);new=tr.build_forecast_model(p,cfg,device=torch.device('cpu')).eval()
    assert torch.equal(rng,torch.get_rng_state())
    assert parameter_hash(old.predictor)==parameter_hash(new.predictor)
    for k,v in old.repair.state_dict().items():assert torch.equal(v,new.repair.state_dict()[k]),k
    b=_batch('validation');starts=tr._starts(b,stride=8,patches=12,device=torch.device('cpu'))
    with torch.no_grad():
        a,oa=old(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts)
        z,oz=new(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts)
    assert torch.equal(a,z) and torch.equal(oa.explicit.normalized_sequence,oz.explicit.normalized_sequence)
    d=oz.point_candidate_attention
    assert d['candidate_counts']==(4,12,10)
    assert d['weights'].shape[-3:]==(8,16,27)
    torch.testing.assert_close(d['weights'].sum(-1),torch.ones_like(d['weights'][...,0]))


def test_nonzero_readout_preserves_observed_hidden_canary_and_disabled_experts(tmp_path):
    p,cfg=setup(tmp_path);m=tr.build_forecast_model(p,cfg,device=torch.device('cpu')).eval()
    with torch.no_grad():m.repair.point_candidate_readout.output[-1].weight.normal_(std=.1)
    b=_batch('validation');starts=tr._starts(b,stride=8,patches=12,device=torch.device('cpu'))
    with torch.no_grad():
        _,a=m(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts)
        canary=torch.where(b.observed_mask,b.normalized_observed,torch.full_like(b.normalized_observed,1e9))
        _,z=m(canary,b.observed_mask,b.query_mean,b.query_std,starts)
        none=m.repair(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts,
                     disabled_experts=frozenset({'historical','variable','temporal'}))
    assert torch.equal(a.explicit.normalized_sequence,z.explicit.normalized_sequence)
    assert torch.equal(a.explicit.normalized_sequence[b.observed_mask],b.normalized_observed[b.observed_mask])
    d=a.point_candidate_attention
    assert d['weights'][...,1:].masked_select(~d['valid'][...,None,None,:].expand_as(d['weights'][...,1:])).eq(0).all()
    assert none.point_candidate_attention['weights'][...,0].eq(1).all()
    assert none.point_candidate_attention['weights'][...,1:].eq(0).all()
    assert torch.isfinite(none.explicit.normalized_sequence).all()
    # A point query reads candidates differently at different positions of a patch.
    assert (d['weights'][...,0,:]-d['weights'][...,15,:]).abs().sum()>0


def test_readout_gradient_reaches_unpooled_messages_and_query_after_zero_start(tmp_path):
    p,cfg=setup(tmp_path);m=tr.build_forecast_model(p,cfg,device=torch.device('cpu'))
    b=_batch('train');starts=tr._starts(b,stride=8,patches=12,device=torch.device('cpu'))
    opt=torch.optim.Adam(m.repair.parameters(),lr=.001)
    for step in range(2):
        opt.zero_grad();o=m.repair(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts)
        loss,_=tr.stage1_objective(o.explicit.normalized_sequence,b.normalized_target,b.observed_mask,
            torch.tensor(0.),retrieval_weight=.1,loss_mode='global_mse_v1',query_std=b.query_std)
        loss.backward();module=m.repair.point_candidate_readout
        assert module.output[-1].weight.grad.abs().sum()>0
        if step:
            assert module.query[0].weight.grad.abs().sum()>0
            assert module.key.weight.grad.abs().sum()>0
            assert module.value.weight.grad.abs().sum()>0
        assert all(torch.isfinite(x.grad).all() for x in m.parameters() if x.grad is not None)
        opt.step()


def test_candidate_checkpoint_roundtrip_and_wrong_identity_rejected(tmp_path,monkeypatch):
    p,cfg=setup(tmp_path)
    monkeypatch.setattr(tr,'iter_p4_batches',lambda _p,split,**kw:iter((_batch(split),)))
    s1=tr.train_repair(p,cfg,output=tmp_path/'s1',device=torch.device('cpu'),epochs=1)
    s2=tr.train_forecast(p,cfg,stage1_manifest=s1,output=tmp_path/'s2',device=torch.device('cpu'),epochs=1,
        selection_metric='mae',forecast_loss='mse_plus_0p2_mae',repair_retention='uniform_v1',repair_update_policy='freeze_front_v1')
    payload=tr.load_stage2_selection(s2,p,cfg,device=torch.device('cpu'))
    assert payload['point_candidate_heads']==8 and payload['point_candidate_scope']==96
    record=json.loads(s2.read_text());record['point_candidate_readout']='wrong'
    wrong=tmp_path/'wrong.json';wrong.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='point_candidate_readout.*mismatch'):
        tr.load_stage2_selection(wrong,p,cfg,device=torch.device('cpu'))


@pytest.mark.parametrize('value',[True,'wrong',8])
def test_invalid_candidate_mode_rejected(tmp_path,value):
    p,_=setup(tmp_path);path=tmp_path/'True-True-False.yaml';doc=yaml.safe_load(path.read_text())
    doc['model']['point_candidate_readout']=value;path.write_text(yaml.safe_dump(doc))
    with pytest.raises(ValueError):load_e23_settings(path,protocol=p)
