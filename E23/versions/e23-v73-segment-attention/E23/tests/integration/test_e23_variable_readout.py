from dataclasses import replace
import json
import pytest
import torch
import yaml
from integrations.e23 import train as tr
from integrations.e23.config import load_e23_settings
from integrations.e23.forecast_sensitive import parameter_hash
from malr.models.e23.variable_readout import VariableReadoutConditioner
from tests.integration.test_e23_point_candidates import setup
from tests.integration.test_e23_stage1 import _batch


def test_conditioner_zero_identity_then_independent_variable_modulation_and_bounds():
    c=VariableReadoutConditioner(7,64);x=torch.ones(2,7,12,16,64)
    assert torch.equal(c(x),x)
    with torch.no_grad():c.scale[4].fill_(100);c.bias[4].fill_(100)
    z=c(x);assert z[:,4].eq(2).all()
    assert torch.equal(z[:,[0,1,2,3,5,6]],x[:,[0,1,2,3,5,6]])
    with pytest.raises(ValueError):c(torch.ones(2,6,12,16,64))


@pytest.mark.parametrize('coords',['global_v1','robust_local_v1'])
def test_readout_preserves_zero_function_rng_head_and_adds_only_1792_parameters(tmp_path,coords):
    p,cfg=setup(tmp_path);cfg=replace(cfg,repair_coordinates=coords);torch.manual_seed(502)
    old=tr.build_forecast_model(p,cfg,device=torch.device('cpu')).eval();rng=torch.get_rng_state()
    torch.manual_seed(502);new=tr.build_forecast_model(p,replace(cfg,point_variable_readout='variable_conditioned_v1'),device=torch.device('cpu')).eval()
    assert torch.equal(rng,torch.get_rng_state()) and parameter_hash(old.predictor)==parameter_hash(new.predictor)
    assert sum(x.numel() for x in new.parameters())-sum(x.numel() for x in old.parameters())==1792
    for k,v in old.state_dict().items():assert torch.equal(v,new.state_dict()[k])
    b=_batch('validation');starts=tr._starts(b,stride=8,patches=12,device=torch.device('cpu'))
    with torch.no_grad():
        a,oa=old(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts)
        z,oz=new(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts)
    assert torch.equal(a,z) and torch.equal(oa.explicit.normalized_sequence,oz.explicit.normalized_sequence)


def test_active_variable_readout_excludes_hidden_input_and_preserves_other_variables(tmp_path):
    p,cfg=setup(tmp_path);cfg=replace(cfg,point_variable_readout='variable_conditioned_v1')
    m=tr.build_forecast_model(p,cfg,device=torch.device('cpu')).eval();b=_batch('validation')
    mask=b.observed_mask.clone();mask[:,-16:]=False;b=replace(b,observed_mask=mask,normalized_observed=b.normalized_observed.masked_fill(~mask,0))
    starts=tr._starts(b,stride=8,patches=12,device=torch.device('cpu'))
    with torch.no_grad():
        m.repair.point_refiner.output.weight.normal_(std=.1);m.repair.point_candidate_readout.output[-1].weight.normal_(std=.1)
        a=m.repair(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts)
        for module in [m.repair.point_refiner,m.repair.point_candidate_readout]:module.variable_readout.bias[4].fill_(1)
        z=m.repair(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts)
        other=torch.tensor([0,1,2,3,5,6]);assert torch.equal(a.explicit.normalized_sequence[:,:,other],z.explicit.normalized_sequence[:,:,other])
        assert not torch.equal(a.explicit.normalized_sequence[:,:,4],z.explicit.normalized_sequence[:,:,4])
        canary=torch.where(b.observed_mask,b.normalized_observed,float('inf'));q=m.repair(canary,b.observed_mask,b.query_mean,b.query_std,starts)
    assert torch.equal(z.explicit.normalized_sequence,q.explicit.normalized_sequence)
    assert torch.equal(z.explicit.normalized_sequence[b.observed_mask],b.normalized_observed[b.observed_mask])


def test_modulation_receives_global_repair_gradients_after_zero_start(tmp_path):
    p,cfg=setup(tmp_path);cfg=replace(cfg,point_variable_readout='variable_conditioned_v1')
    m=tr.build_forecast_model(p,cfg,device=torch.device('cpu'));b=_batch('train');mask=b.observed_mask.clone();mask[:,-16:]=False;b=replace(b,observed_mask=mask,normalized_observed=b.normalized_observed.masked_fill(~mask,0));starts=tr._starts(b,stride=8,patches=12,device=torch.device('cpu'));opt=torch.optim.Adam(m.repair.parameters(),lr=.001)
    for step in range(2):
        opt.zero_grad();o=m.repair(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts)
        loss,_=tr.stage1_objective(o.explicit.normalized_sequence,b.normalized_target,b.observed_mask,torch.tensor(0.),retrieval_weight=.1,loss_mode='global_mse_v1',query_std=b.query_std)
        loss.backward();assert torch.isfinite(loss)
        if step:
            for module in [m.repair.point_refiner,m.repair.point_candidate_readout]:assert module.variable_readout.scale.grad.abs().sum()>0 and module.variable_readout.bias.grad.abs().sum()>0
        opt.step()


def test_checkpoint_roundtrip_and_wrong_modulation_identity_rejected(tmp_path,monkeypatch):
    p,_=setup(tmp_path);path=tmp_path/'True-True-False.yaml';doc=yaml.safe_load(path.read_text());doc['model'].update(variable_calibration='none',front_variable_context='none',repair_coordinates='global_v1',point_variable_readout='variable_conditioned_v1');path.write_text(yaml.safe_dump(doc));cfg=load_e23_settings(path,protocol=p)
    monkeypatch.setattr(tr,'iter_p4_batches',lambda _p,split,**kw:iter((_batch(split),)))
    s1=tr.train_repair(p,cfg,output=tmp_path/'s1',device=torch.device('cpu'),epochs=1);s2=tr.train_forecast(p,cfg,stage1_manifest=s1,output=tmp_path/'s2',device=torch.device('cpu'),epochs=1,selection_metric='mae',forecast_loss='mse_plus_0p2_mae',repair_retention='uniform_v1',repair_update_policy='freeze_front_v1')
    payload=tr.load_stage2_selection(s2,p,cfg,device=torch.device('cpu'));assert payload['repair_readout_extra_parameters']==1792
    rec=json.loads(s2.read_text());rec['point_variable_readout']='wrong';wrong=tmp_path/'wrong.json';wrong.write_text(json.dumps(rec))
    with pytest.raises(ValueError,match='point_variable_readout.*mismatch'):tr.load_stage2_selection(wrong,p,cfg,device=torch.device('cpu'))


@pytest.mark.parametrize('value',['wrong',True,1])
def test_invalid_variable_readout_rejected(tmp_path,value):
    p,_=setup(tmp_path);path=tmp_path/'True-True-False.yaml';doc=yaml.safe_load(path.read_text());doc['model'].update(variable_calibration='none',front_variable_context='none',repair_coordinates='global_v1',point_variable_readout=value);path.write_text(yaml.safe_dump(doc))
    with pytest.raises(ValueError):load_e23_settings(path,protocol=p)
