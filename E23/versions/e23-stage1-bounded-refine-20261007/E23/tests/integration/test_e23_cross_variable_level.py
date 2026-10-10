"""Global source levels reach recent repair while the formal head stays original."""
from dataclasses import replace
import json
import pytest
import torch
import yaml
from integrations.e23 import train as tr
from integrations.e23.config import load_e23_settings
from integrations.e23.forecast_sensitive import parameter_hash
from tests.integration.test_e23_global_level_context import configuration as global_configuration, batch


def configuration(tmp_path):
    protocol,_=global_configuration(tmp_path)
    path=tmp_path/'True-True-False.yaml';doc=yaml.safe_load(path.read_text())
    doc['model'].update(front_level_context='none',recent_mean_projection='global_level_mean_v2')
    path.write_text(yaml.safe_dump(doc))
    return protocol,load_e23_settings(path,protocol=protocol)


def prepare(tmp_path):
    protocol,cfg=configuration(tmp_path)
    model=tr.build_forecast_model(protocol,cfg,device=torch.device('cpu')).eval()
    b=batch();starts=tr._starts(b,stride=8,patches=12,device=torch.device('cpu'))
    return protocol,cfg,model,b,starts


def activate(module):
    with torch.no_grad():
        net=module.global_level_network
        net[0].weight.zero_();net[0].bias.zero_();net[-1].weight.zero_();net[-1].bias.zero_()
        # Whole96 mean of source variable1. GELU is deliberately not constant.
        net[0].weight[0,6]=.3;net[0].bias[0]=1
        net[-1].weight[0,0]=.2


def test_zero_start_preserves_function_existing_weights_rng_and_predictor(tmp_path):
    protocol,cfg=configuration(tmp_path)
    torch.manual_seed(2023);old=tr.build_forecast_model(protocol,replace(cfg,recent_mean_projection='mean_v1'),device=torch.device('cpu')).eval();rng=torch.get_rng_state()
    torch.manual_seed(2023);new=tr.build_forecast_model(protocol,cfg,device=torch.device('cpu')).eval()
    assert torch.equal(rng,torch.get_rng_state());assert parameter_hash(old.predictor)==parameter_hash(new.predictor)
    for k,v in old.state_dict().items():assert torch.equal(v,new.state_dict()[k]),k
    assert sum(p.numel() for p in new.repair.parameters())-sum(p.numel() for p in old.repair.parameters())==3265
    b=batch();s=tr._starts(b,stride=8,patches=12,device=torch.device('cpu'))
    with torch.no_grad():
        a,ra=old(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,s)
        z,rz=new(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,s)
    assert torch.equal(a,z);assert torch.equal(ra.explicit.normalized_sequence,rz.explicit.normalized_sequence)


def test_other_variable_global_level_has_direct_target_repair_access(tmp_path):
    _,_,m,b,s=prepare(tmp_path);activate(m.repair.recent_mean_projector)
    with torch.no_grad():
        a=m.repair(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,s)
        means=b.query_mean.clone();means[:,1]+=1
        z=m.repair(b.normalized_observed,b.observed_mask,means,b.query_std,s)
    hidden=~b.observed_mask[:,64:,0]
    assert (a.explicit.standardized_sequence[:,64:,0]-z.explicit.standardized_sequence[:,64:,0])[hidden].abs().max()>1e-4
    assert torch.equal(a.explicit.normalized_sequence[:,:64,0],z.explicit.normalized_sequence[:,:64,0])


def test_global_units_not_scaled_by_target_std_and_observations_scope_coverage_kept(tmp_path):
    _,_,m,b,s=prepare(tmp_path)
    with torch.no_grad():
        base=m.repair(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,s)
        m.repair.recent_mean_projector.global_level_network[-1].bias.fill_(.2)
        changed=m.repair(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,s)
        tiny=replace(b,query_std=torch.full_like(b.query_std,1e-6))
        out=m.repair(tiny.normalized_observed,tiny.observed_mask,tiny.query_mean,tiny.query_std,s)
    expected=torch.full_like(b.query_std,2*torch.tanh(torch.tensor(.1)))
    torch.testing.assert_close(changed.recent_mean_delta-base.recent_mean_delta,expected)
    torch.testing.assert_close(out.recent_mean_delta,expected)
    assert torch.equal(base.explicit.normalized_sequence[:,:64],changed.explicit.normalized_sequence[:,:64])
    assert torch.equal(changed.explicit.normalized_sequence[b.observed_mask],b.normalized_observed[b.observed_mask])
    assert torch.equal(base.explicit.coverage,changed.explicit.coverage)
    assert torch.equal(changed.explicit.sequence_observed,b.observed_mask)


def test_hidden_and_invalid_payloads_cannot_enter_global_summary_or_output(tmp_path):
    _,_,m,b,s=prepare(tmp_path);activate(m.repair.recent_mean_projector)
    valid=torch.ones_like(b.observed_mask);valid[:,90,0]=False
    with torch.no_grad():
        clean=torch.where(b.observed_mask & valid,b.normalized_observed,0)
        a=m.repair(clean,b.observed_mask,b.query_mean,b.query_std,s,valid_mask=valid)
        for poison in [float('nan'),float('inf'),-float('inf')]:
            values=torch.where(b.observed_mask & valid,b.normalized_observed,poison)
            z=m.repair(values,b.observed_mask,b.query_mean,b.query_std,s,valid_mask=valid)
            assert torch.equal(a.explicit.normalized_sequence,z.explicit.normalized_sequence)
            assert torch.equal(a.recent_mean_delta,z.recent_mean_delta)
    assert torch.isfinite(a.explicit.normalized_sequence).all()


def test_no_observations_have_zero_new_delta_and_source_summary_is_unavailable(tmp_path):
    _,_,m,b,s=prepare(tmp_path)
    with torch.no_grad():m.repair.recent_mean_projector.global_level_network[-1].bias.fill_(.5)
    mask=torch.zeros_like(b.observed_mask)
    with torch.no_grad():out=m.repair(torch.full_like(b.normalized_observed,float('nan')),mask,b.query_mean,torch.full_like(b.query_std,1e-6),s)
    assert torch.isfinite(out.explicit.normalized_sequence).all();assert not out.recent_mean_delta.count_nonzero()
    assert torch.equal(out.explicit.sequence_observed,mask)


def test_original_repair_losses_train_new_branch_with_finite_gradients(tmp_path):
    _,_,m,b,s=prepare(tmp_path);m.train();opt=torch.optim.Adam(m.repair.parameters(),lr=.001)
    for step in range(2):
        opt.zero_grad();out=m.repair(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,s)
        loss,_=tr.stage1_objective(out.explicit.normalized_sequence,b.normalized_target,b.observed_mask,torch.tensor(0.),retrieval_weight=.1,loss_mode='global_mse_v1',query_std=b.query_std,structure_mode='recent_mean_v2',gap_structure_mode='long_gap_moments_v1')
        loss.backward();assert m.repair.recent_mean_projector.global_level_network[-1].weight.grad.abs().sum()>0
        if step:assert m.repair.recent_mean_projector.global_level_network[0].weight.grad.abs().sum()>0
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
        opt.step()


def test_checkpoint_roundtrip_frozen_repair_identity_and_rejection(tmp_path,monkeypatch):
    protocol,cfg=configuration(tmp_path);monkeypatch.setattr(tr,'iter_p4_batches',lambda _p,split,**kwargs:iter((batch(),)))
    s1=tr.train_repair(protocol,cfg,output=tmp_path/'s1',device=torch.device('cpu'),epochs=1)
    s2=tr.train_forecast(protocol,cfg,stage1_manifest=s1,output=tmp_path/'s2',device=torch.device('cpu'),epochs=1,selection_metric='mae',forecast_loss='mse_plus_0p2_mae',repair_retention='uniform_v1',repair_update_policy='frozen_v1',extra_mse_checkpoint=True)
    state1=torch.load(json.loads(s1.read_text())['checkpoint'],weights_only=False)['model_state']
    for choice in [s2,s2.parent/'e23-stage2-mse.json']:
        payload=tr.load_stage2_selection(choice,protocol,cfg,device=torch.device('cpu'))
        assert payload['recent_mean_global_features']==49 and payload['recent_mean_global_extra_parameters']==3265
        assert payload['recent_mean_global_delta_units']=='train_standardized'
        assert all(torch.equal(v,payload['model_state']['repair.'+k]) for k,v in state1.items())
    rec=json.loads(s2.read_text());rec['recent_mean_global_delta_units']='window_normalized';wrong=tmp_path/'wrong.json';wrong.write_text(json.dumps(rec))
    with pytest.raises(ValueError,match='recent_mean_global_delta_units.*mismatch'):tr.load_stage2_selection(wrong,protocol,cfg,device=torch.device('cpu'))
