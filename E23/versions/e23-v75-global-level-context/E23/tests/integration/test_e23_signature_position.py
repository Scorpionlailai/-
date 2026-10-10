from dataclasses import replace
import json
import pytest,torch,yaml
from integrations.e23 import train as tr
from integrations.e23.config import load_e23_settings
from integrations.e23.forecast_sensitive import parameter_hash
from integrations.e23.repair_updates import build_stage2_optimizer,stage2_training_mode
from malr.models.e23.window_signature import WindowSignatureContext
from tests.integration.test_e23_point_candidates import setup
from tests.integration.test_e23_stage1 import _batch


def test_position_zero_start_keeps_repair_forecast_and_head_rng(tmp_path):
    p,cfg=setup(tmp_path)
    cfg=replace(cfg,front_variable_context='signature_plain_v2')
    torch.manual_seed(502);old=tr.build_forecast_model(p,cfg,device=torch.device('cpu')).eval();rng=torch.get_rng_state()
    torch.manual_seed(502);new=tr.build_forecast_model(p,replace(cfg,front_variable_context='signature_plain_pos_v3'),device=torch.device('cpu')).eval()
    assert torch.equal(rng,torch.get_rng_state())
    assert parameter_hash(old.predictor)==parameter_hash(new.predictor)
    for k,v in old.state_dict().items():assert torch.equal(v,new.state_dict()[k]),k
    b=_batch('validation');s=tr._starts(b,stride=8,patches=12,device=torch.device('cpu'))
    with torch.no_grad():
        a,oa=old(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,s)
        z,oz=new(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,s)
    assert torch.equal(a,z) and torch.equal(oa.explicit.normalized_sequence,oz.explicit.normalized_sequence)


def test_learned_positions_change_variable_features_and_mask_invalid_points():
    torch.manual_seed(8)
    m=WindowSignatureContext(64,3,statistics_conditioning=False,learned_position=True).eval()
    points=torch.randn(6,64,96);observed=torch.rand(2,96,3)>.7
    validity=torch.ones(6,1,96,dtype=torch.bool);validity[:,:,-1]=False
    mean=torch.zeros(2,3);std=torch.ones(2,3)
    with torch.no_grad():
        a=m(points,observed,validity,mean,std)
        m.position[:,1].fill_(.1)
        z=m(points,observed,validity,mean,std)
    assert not torch.equal(a.reshape(2,3,64,96)[:,1],z.reshape(2,3,64,96)[:,1])
    assert torch.equal(a.reshape(2,3,64,96)[:,0],z.reshape(2,3,64,96)[:,0])
    assert z[:,:,-1].eq(0).all()


def test_position_front_ignores_hidden_payload_and_learns_from_repair_loss(tmp_path):
    p,cfg=setup(tmp_path);cfg=replace(cfg,front_variable_context='signature_plain_pos_v3')
    m=tr.build_forecast_model(p,cfg,device=torch.device('cpu'));b=_batch('train');s=tr._starts(b,stride=8,patches=12,device=torch.device('cpu'));optimizer=torch.optim.Adam(m.repair.parameters(),lr=.001)
    for _ in range(3):
        m.train();optimizer.zero_grad();o=m.repair(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,s)
        loss,_=tr.stage1_objective(o.explicit.normalized_sequence,b.normalized_target,b.observed_mask,torch.tensor(0.),retrieval_weight=.1,loss_mode='global_mse_v1',query_std=b.query_std)
        loss.backward();assert torch.isfinite(loss);assert all(torch.isfinite(v.grad).all() for v in m.repair.parameters() if v.grad is not None);optimizer.step()
    position=m.repair.front_context.variable_context.position
    assert position.grad.abs().sum()>0 and position.detach().abs().sum()>0
    m.eval()
    polluted=torch.where(b.observed_mask,b.normalized_observed,torch.full_like(b.normalized_observed,float('nan')))
    with torch.no_grad():
        a=m.repair(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,s)
        z=m.repair(polluted,b.observed_mask,b.query_mean,b.query_std,s)
    assert torch.equal(a.explicit.normalized_sequence,z.explicit.normalized_sequence)
    assert torch.equal(a.explicit.normalized_sequence[b.observed_mask],b.normalized_observed[b.observed_mask])
    optimizer=build_stage2_optimizer(m,mode='freeze_front_v1',learning_rate=1e-4,weight_decay=0);stage2_training_mode(m,'freeze_front_v1')
    assert not position.requires_grad
    assert all(id(position)!=id(v) for group in optimizer.param_groups for v in group['params'])


def test_position_configuration_checkpoint_roundtrip_and_tamper_rejection(tmp_path,monkeypatch):
    p,_=setup(tmp_path);path=tmp_path/'True-True-False.yaml';d=yaml.safe_load(path.read_text());d['model'].update(front_variable_context='signature_plain_pos_v3',variable_calibration='none');path.write_text(yaml.safe_dump(d));cfg=load_e23_settings(path,protocol=p)
    monkeypatch.setattr(tr,'iter_p4_batches',lambda _p,split,**kw:iter((_batch(split),)))
    s1=tr.train_repair(p,cfg,output=tmp_path/'s1',device=torch.device('cpu'),epochs=1)
    s2=tr.train_forecast(p,cfg,stage1_manifest=s1,output=tmp_path/'s2',device=torch.device('cpu'),epochs=1,selection_metric='mae',forecast_loss='mse_plus_0p2_mae',repair_retention='uniform_v1',repair_update_policy='frozen_v1',extra_mse_checkpoint=True)
    for selected in [s2,s2.parent/'e23-stage2-mse.json']:
        payload=tr.load_stage2_selection(selected,p,cfg,device=torch.device('cpu'))
        assert payload['front_signature_position']=='variable_time_learned_zero_v1'
        assert payload['front_signature_statistics']=='none'
        assert payload['front_signature_position_parameters_per_variable']==6144
    record=json.loads(s2.read_text());record['front_signature_position']='wrong';wrong=tmp_path/'wrong.json';wrong.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='front_signature_position.*mismatch'):tr.load_stage2_selection(wrong,p,cfg,device=torch.device('cpu'))
