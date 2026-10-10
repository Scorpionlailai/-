from dataclasses import replace
import json
import pytest
import torch
import yaml
from integrations.e23 import train as tr
from integrations.e23.config import load_e23_settings
from integrations.e23.forecast_sensitive import parameter_hash
from tests.integration.test_e23_point_candidates import setup
from tests.integration.test_e23_recent_mean import batch


def trial(tmp_path):
    p,cfg=setup(tmp_path);return p,replace(cfg,repair_coordinates='robust_local_v1',temporal_evidence_readout='visible_cross_attention_v1')


def test_zero_time_readout_preserves_old_parameters_rng_head_and_function(tmp_path):
    p,cfg=trial(tmp_path);torch.manual_seed(502);old=tr.build_forecast_model(p,replace(cfg,temporal_evidence_readout='none'),device=torch.device('cpu')).eval();rng=torch.get_rng_state();torch.manual_seed(502);m=tr.build_forecast_model(p,cfg,device=torch.device('cpu')).eval()
    assert torch.equal(rng,torch.get_rng_state()) and parameter_hash(old.predictor)==parameter_hash(m.predictor)
    for k,v in old.state_dict().items():assert torch.equal(v,m.state_dict()[k]),k
    extra=sum(v.numel() for v in m.repair.temporal_evidence_refiner.parameters());assert extra==83969
    b=batch();starts=tr._starts(b,stride=8,patches=12,device=torch.device('cpu'))
    with torch.no_grad():a,oa=old(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts);z,oz=m(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts)
    assert torch.equal(a,z) and torch.equal(oa.explicit.normalized_sequence,oz.explicit.normalized_sequence)


def test_attention_reads_true_visible_only_and_empty_source_is_finite(tmp_path):
    p,cfg=trial(tmp_path);m=tr.build_forecast_model(p,cfg,device=torch.device('cpu')).eval();b=batch();starts=tr._starts(b,stride=8,patches=12,device=torch.device('cpu'))
    for mask in [b.observed_mask,torch.zeros_like(b.observed_mask)]:
        with torch.no_grad():out=m.repair(torch.where(mask,b.normalized_observed,float('inf')),mask,b.query_mean,b.query_std,starts)
        info=out.temporal_evidence_attention;w=info['weights'];assert w.shape==(14,2,8,32,97)
        forbidden=~mask.transpose(1,2).reshape(14,96)[:,None,None,None].expand_as(w[...,1:]);assert w[...,1:][forbidden].eq(0).all()
        torch.testing.assert_close(w.sum(-1),torch.ones_like(w[...,0]));assert torch.isfinite(out.explicit.standardized_sequence).all()
        if not mask.any():assert w[...,0].eq(1).all()


def test_nonzero_time_update_changes_recent_hidden_only_and_masks_canaries(tmp_path):
    p,cfg=trial(tmp_path);m=tr.build_forecast_model(p,cfg,device=torch.device('cpu')).eval();b=batch();starts=tr._starts(b,stride=8,patches=12,device=torch.device('cpu'))
    with torch.no_grad():
        a=m.repair(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts);m.repair.temporal_evidence_refiner.output[-1].bias.fill_(.2);z=m.repair(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts);q=m.repair(torch.where(b.observed_mask,b.normalized_observed,float('nan')),b.observed_mask,b.query_mean,b.query_std,starts)
    assert torch.equal(z.explicit.normalized_sequence,q.explicit.normalized_sequence)
    assert torch.equal(z.explicit.normalized_sequence[b.observed_mask],b.normalized_observed[b.observed_mask]) and torch.equal(a.explicit.normalized_sequence[:,:64],z.explicit.normalized_sequence[:,:64])
    assert torch.equal(z.explicit.coverage,a.explicit.coverage)
    assert (z.explicit.normalized_sequence[:,64:]-a.explicit.normalized_sequence[:,64:]).abs().sum()>0
    change=(z.explicit.standardized_sequence-a.explicit.standardized_sequence).abs();assert (change<=2*(b.query_std.square()+.01).sqrt()[:,None]+1e-5).all()


def test_time_projection_global_loss_has_gradient_and_small_std_stays_finite(tmp_path):
    p,cfg=trial(tmp_path);m=tr.build_forecast_model(p,cfg,device=torch.device('cpu'));b=batch();starts=tr._starts(b,stride=8,patches=12,device=torch.device('cpu'));out=m.repair(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts)
    loss,_=tr.stage1_objective(out.explicit.normalized_sequence,b.normalized_target,b.observed_mask,torch.tensor(0.),retrieval_weight=.1,loss_mode='global_mse_v1',query_std=b.query_std);loss.backward();assert m.repair.temporal_evidence_refiner.output[-1].weight.grad.abs().sum()>0
    with torch.no_grad():m.repair.temporal_evidence_refiner.output[-1].bias.fill_(.2);o=m.repair(b.normalized_observed,b.observed_mask,b.query_mean,torch.full_like(b.query_std,1e-6),starts)
    assert torch.isfinite(o.explicit.standardized_sequence).all()


def test_time_checkpoint_roundtrip_identity_and_mean_composition(tmp_path,monkeypatch):
    p,_=setup(tmp_path);path=tmp_path/'True-True-False.yaml';d=yaml.safe_load(path.read_text());d['model'].update(variable_calibration='none',front_variable_context='none',repair_coordinates='robust_local_v1',point_variable_readout='none',historical_key_policy='legacy_v1',recent_mean_projection='mean_v1',temporal_evidence_readout='visible_cross_attention_v1');d['training']['stage1_structure']='recent_mean_v2';path.write_text(yaml.safe_dump(d));cfg=load_e23_settings(path,protocol=p)
    monkeypatch.setattr(tr,'iter_p4_batches',lambda _p,split,**kw:iter((batch(),)))
    s1=tr.train_repair(p,cfg,output=tmp_path/'s1',device=torch.device('cpu'),epochs=1);s2=tr.train_forecast(p,cfg,stage1_manifest=s1,output=tmp_path/'s2',device=torch.device('cpu'),epochs=1,selection_metric='mae',forecast_loss='mse_plus_0p2_mae',repair_retention='uniform_v1',repair_update_policy='freeze_front_v1')
    payload=tr.load_stage2_selection(s2,p,cfg,device=torch.device('cpu'));assert payload['temporal_evidence_sources']=='true_visible_same_variable_all96' and payload['recent_mean_projection']=='mean_v1'
    d=json.loads(s2.read_text());d['temporal_evidence_readout']='wrong';wrong=tmp_path/'wrong.json';wrong.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='temporal_evidence_readout.*mismatch'):tr.load_stage2_selection(wrong,p,cfg,device=torch.device('cpu'))


@pytest.mark.parametrize('value',['wrong',True,1])
def test_bad_time_policy_rejected(tmp_path,value):
    p,_=setup(tmp_path);path=tmp_path/'True-True-False.yaml';d=yaml.safe_load(path.read_text());d['model'].update(variable_calibration='none',front_variable_context='none',repair_coordinates='robust_local_v1',point_variable_readout='none',historical_key_policy='legacy_v1',recent_mean_projection='none',temporal_evidence_readout=value);path.write_text(yaml.safe_dump(d))
    with pytest.raises(ValueError):load_e23_settings(path,protocol=p)
