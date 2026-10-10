from dataclasses import replace
import json
import pytest
import torch
import yaml
from integrations.e23 import train as tr
from integrations.e23.config import load_e23_settings
from integrations.e23.forecast_sensitive import parameter_hash
from tests.integration.test_e23_point_candidates import setup
from tests.integration.test_e23_stage1 import _batch


def trial(tmp_path):
    p,cfg=setup(tmp_path);return p,replace(cfg,repair_coordinates='robust_local_v1',recent_mean_projection='mean_v1',stage1_structure='recent_mean_v2')


def batch():
    b=_batch('train');mask=b.observed_mask.clone();mask[:,64:]=False;mask[:,64::4]=True
    return replace(b,observed_mask=mask,normalized_observed=b.normalized_observed.masked_fill(~mask,0))


def test_zero_projector_preserves_all_existing_weights_rng_function_and_head(tmp_path):
    p,cfg=trial(tmp_path);torch.manual_seed(502);old=tr.build_forecast_model(p,replace(cfg,recent_mean_projection='none',stage1_structure='none'),device=torch.device('cpu')).eval();rng=torch.get_rng_state()
    torch.manual_seed(502);m=tr.build_forecast_model(p,cfg,device=torch.device('cpu')).eval()
    assert torch.equal(rng,torch.get_rng_state()) and parameter_hash(old.predictor)==parameter_hash(m.predictor)
    assert sum(v.numel() for v in m.parameters())-sum(v.numel() for v in old.parameters())==13249
    for k,v in old.state_dict().items():assert torch.equal(v,m.state_dict()[k]),k
    b=batch();starts=tr._starts(b,stride=8,patches=12,device=torch.device('cpu'))
    with torch.no_grad():
        a,oa=old(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts)
        z,oz=m(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts)
    assert torch.equal(a,z) and torch.equal(oa.explicit.normalized_sequence,oz.explicit.normalized_sequence)


def test_nonzero_mean_update_has_correct_global_units_preserves_observation_and_before64(tmp_path):
    p,cfg=trial(tmp_path);m=tr.build_forecast_model(p,cfg,device=torch.device('cpu')).eval();b=batch();starts=tr._starts(b,stride=8,patches=12,device=torch.device('cpu'))
    with torch.no_grad():
        _,a=m(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts)
        m.repair.recent_mean_projector.network[-1].bias.fill_(.2)
        _,z=m(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts)
        canary=torch.where(b.observed_mask,b.normalized_observed,float('inf'));_,q=m(canary,b.observed_mask,b.query_mean,b.query_std,starts)
    assert torch.equal(z.explicit.normalized_sequence,q.explicit.normalized_sequence)
    assert torch.equal(z.explicit.normalized_sequence[b.observed_mask],b.normalized_observed[b.observed_mask])
    assert torch.equal(a.explicit.normalized_sequence[:,:64],z.explicit.normalized_sequence[:,:64])
    change=(z.explicit.standardized_sequence-a.explicit.standardized_sequence)[:,64:].mean(1)
    torch.testing.assert_close(change,z.recent_mean_delta,rtol=1e-4,atol=1e-6)
    assert torch.equal(z.explicit.coverage,a.explicit.coverage)


def test_small_std_empty_and_fully_visible_cases_finite_and_projection_does_not_create_observation(tmp_path):
    p,cfg=trial(tmp_path);m=tr.build_forecast_model(p,cfg,device=torch.device('cpu')).eval();b=batch();starts=tr._starts(b,stride=8,patches=12,device=torch.device('cpu'))
    with torch.no_grad():m.repair.recent_mean_projector.network[-1].bias.fill_(.5)
    for mask in [torch.zeros_like(b.observed_mask),torch.ones_like(b.observed_mask),b.observed_mask]:
        values=torch.where(mask,b.normalized_observed,float('nan'));std=torch.full_like(b.query_std,1e-6)
        with torch.no_grad():out=m.repair(values,mask,b.query_mean,std,starts)
        assert torch.isfinite(out.explicit.normalized_sequence).all() and torch.isfinite(out.explicit.standardized_sequence).all()
        assert torch.equal(out.explicit.sequence_observed,mask)
        assert torch.equal(out.explicit.normalized_sequence[mask],values[mask])
        if mask.all():assert out.recent_mean_delta.eq(0).all()


def test_level_supervision_uses_only_hidden_lookback_global_units_and_has_projection_gradient(tmp_path):
    p,cfg=trial(tmp_path);m=tr.build_forecast_model(p,cfg,device=torch.device('cpu'));b=batch();starts=tr._starts(b,stride=8,patches=12,device=torch.device('cpu'));out=m.repair(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts)
    loss,parts=tr.stage1_objective(out.explicit.normalized_sequence,b.normalized_target,b.observed_mask,torch.tensor(0.),retrieval_weight=.1,loss_mode='global_mse_v1',query_std=b.query_std,structure_mode='recent_mean_v2')
    error=torch.where(~b.observed_mask,(out.explicit.normalized_sequence-b.normalized_target)*b.query_std[:,None],0)
    expected=error[:,64:].mean(1).square().mean();torch.testing.assert_close(parts['recent_structure_weighted'],expected)
    loss.backward();assert m.repair.recent_mean_projector.network[-1].weight.grad.abs().sum()>0
    assert all(torch.isfinite(v.grad).all() for v in m.parameters() if v.grad is not None)


def test_mean_checkpoint_roundtrip_and_wrong_identity_rejected(tmp_path,monkeypatch):
    p,_=setup(tmp_path);path=tmp_path/'True-True-False.yaml';d=yaml.safe_load(path.read_text());d['model'].update(variable_calibration='none',front_variable_context='none',repair_coordinates='robust_local_v1',point_variable_readout='none',historical_key_policy='legacy_v1',recent_mean_projection='mean_v1');d['training']['stage1_structure']='recent_mean_v2';path.write_text(yaml.safe_dump(d));cfg=load_e23_settings(path,protocol=p)
    monkeypatch.setattr(tr,'iter_p4_batches',lambda _p,split,**kw:iter((batch(),)))
    s1=tr.train_repair(p,cfg,output=tmp_path/'s1',device=torch.device('cpu'),epochs=1);s2=tr.train_forecast(p,cfg,stage1_manifest=s1,output=tmp_path/'s2',device=torch.device('cpu'),epochs=1,selection_metric='mae',forecast_loss='mse_plus_0p2_mae',repair_retention='uniform_v1',repair_update_policy='freeze_front_v1')
    payload=tr.load_stage2_selection(s2,p,cfg,device=torch.device('cpu'));assert payload['recent_level_weight']==1 and payload['recent_trend_weight']==0
    d=json.loads(s2.read_text());d['recent_mean_projection']='wrong';wrong=tmp_path/'wrong.json';wrong.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='recent_mean_projection.*mismatch'):tr.load_stage2_selection(wrong,p,cfg,device=torch.device('cpu'))


@pytest.mark.parametrize('value',['wrong',True,1])
def test_invalid_mean_projection_rejected(tmp_path,value):
    p,_=setup(tmp_path);path=tmp_path/'True-True-False.yaml';d=yaml.safe_load(path.read_text());d['model'].update(variable_calibration='none',front_variable_context='none',repair_coordinates='robust_local_v1',point_variable_readout='none',historical_key_policy='legacy_v1',recent_mean_projection=value);d['training']['stage1_structure']='recent_mean_v2';path.write_text(yaml.safe_dump(d))
    with pytest.raises(ValueError):load_e23_settings(path,protocol=p)
