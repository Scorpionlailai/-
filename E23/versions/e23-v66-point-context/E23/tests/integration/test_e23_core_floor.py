from dataclasses import replace
import json
import pytest,torch,yaml
from integrations.e23 import train as tr
from integrations.e23.config import load_e23_settings
from integrations.e23.forecast_sensitive import parameter_hash
from malr.models.e23.coordinates import repair_scale
from tests.integration.test_e23_point_candidates import setup
from tests.integration.test_e23_recent_mean import batch


def trial(tmp_path):
 p,cfg=setup(tmp_path);return p,replace(cfg,repair_coordinates='robust_local_v1',repair_input_scaling='all_floor_v1')


def test_full_floor_adds_no_parameters_changes_no_initialization_or_head(tmp_path):
 p,cfg=trial(tmp_path);torch.manual_seed(502);old=tr.build_forecast_model(p,replace(cfg,repair_input_scaling='legacy_v1'),device=torch.device('cpu'));rng=torch.get_rng_state();torch.manual_seed(502);m=tr.build_forecast_model(p,cfg,device=torch.device('cpu'));assert torch.equal(rng,torch.get_rng_state()) and parameter_hash(old.predictor)==parameter_hash(m.predictor);assert old.state_dict().keys()==m.state_dict().keys()
 for k,v in old.state_dict().items():assert torch.equal(v,m.state_dict()[k]),k


@pytest.mark.parametrize('tiny',[False,True])
def test_entire_repair_uses_floor_and_returns_original_units_masks_and_observations(tmp_path,tiny):
 p,cfg=trial(tmp_path);m=tr.build_forecast_model(p,cfg,device=torch.device('cpu')).eval();b=batch();std=torch.full_like(b.query_std,1e-6) if tiny else b.query_std;starts=tr._starts(b,stride=8,patches=12,device=torch.device('cpu'));seen={}
 def hook(_m,args):seen['values']=args[0].clone();seen['std']=args[3].clone()
 h=m.repair.front_context.register_forward_pre_hook(hook)
 out=m.repair(b.normalized_observed,b.observed_mask,b.query_mean,std,starts);h.remove();torch.testing.assert_close(seen['std'],repair_scale(std));assert seen['std'].ge(.1).all();torch.testing.assert_close(seen['values'],torch.where(b.observed_mask,b.normalized_observed,0)*(std/repair_scale(std))[:,None])
 e=out.explicit;assert torch.equal(e.normalized_sequence[b.observed_mask],b.normalized_observed[b.observed_mask]) and torch.equal(e.observed_sequence[b.observed_mask],b.normalized_observed[b.observed_mask]);assert torch.equal(e.sequence_observed,b.observed_mask);torch.testing.assert_close(e.standardized_sequence,e.normalized_sequence*std[:,None]+b.query_mean[:,None]);assert torch.isfinite(e.normalized_sequence).all()
 loss,_=tr.stage1_objective(e.normalized_sequence,b.normalized_target,b.observed_mask,torch.tensor(0.),retrieval_weight=.1,loss_mode='global_mse_v1',query_std=std);loss.backward();assert all(torch.isfinite(v.grad).all() for v in m.parameters() if v.grad is not None)


def test_hidden_canaries_all_empty_all_visible_and_validity_are_preserved(tmp_path):
 p,cfg=trial(tmp_path);m=tr.build_forecast_model(p,cfg,device=torch.device('cpu')).eval();b=batch();starts=tr._starts(b,stride=8,patches=12,device=torch.device('cpu'))
 for mask in [b.observed_mask,torch.ones_like(b.observed_mask),torch.zeros_like(b.observed_mask)]:
  values=torch.where(mask,b.normalized_observed,0);canary=torch.where(mask,b.normalized_observed,float('inf'))
  with torch.no_grad():a=m.repair(values,mask,b.query_mean,b.query_std,starts);z=m.repair(canary,mask,b.query_mean,b.query_std,starts)
  assert torch.equal(a.explicit.normalized_sequence,z.explicit.normalized_sequence);assert torch.equal(z.explicit.normalized_sequence[mask],values[mask]);assert torch.equal(z.explicit.sequence_observed,mask);assert torch.isfinite(z.explicit.standardized_sequence).all()
 valid=torch.ones_like(mask);valid[:,-4:]=False
 with torch.no_grad():o=m.repair(values,mask,b.query_mean,b.query_std,starts,valid_mask=valid)
 assert torch.equal(o.explicit.sequence_validity,valid) and o.explicit.normalized_sequence[~valid].eq(0).all()


def test_key_pair_training_uses_same_floor_and_no_forecast_labels(tmp_path):
 p,cfg=trial(tmp_path);torch.manual_seed(502);m=tr.build_forecast_model(p,cfg,device=torch.device('cpu'));torch.manual_seed(502);old=tr.build_forecast_model(p,replace(cfg,repair_input_scaling='legacy_v1'),device=torch.device('cpu'));b=batch();scale=repair_scale(b.query_std);ratio=(b.query_std/scale)[:,None];converted=replace(b,normalized_observed=torch.where(b.observed_mask,b.normalized_observed,0)*ratio,normalized_target=b.normalized_target*ratio,query_std=scale)
 a=tr._retrieval_loss(m.repair,b);z=tr._retrieval_loss(old.repair,converted);torch.testing.assert_close(a,z);canary=replace(b,forecast_target=torch.full_like(b.forecast_target,float('inf')));torch.testing.assert_close(a,tr._retrieval_loss(m.repair,canary))


def test_floor_checkpoint_composes_with_mean_and_padding_identity(tmp_path,monkeypatch):
 p,_=setup(tmp_path);path=tmp_path/'True-True-False.yaml';d=yaml.safe_load(path.read_text());d['model'].update(variable_calibration='none',front_variable_context='none',repair_coordinates='robust_local_v1',point_variable_readout='none',historical_key_policy='legacy_v1',recent_mean_projection='mean_v1',temporal_evidence_readout='none',repair_input_scaling='all_floor_v1');d['training']['stage1_structure']='recent_mean_v2';path.write_text(yaml.safe_dump(d));cfg=load_e23_settings(path,protocol=p);monkeypatch.setattr(tr,'iter_p4_batches',lambda _p,split,**kw:iter((batch(),)))
 s1=tr.train_repair(p,cfg,output=tmp_path/'s1',device=torch.device('cpu'),epochs=1);s2=tr.train_forecast(p,cfg,stage1_manifest=s1,output=tmp_path/'s2',device=torch.device('cpu'),epochs=1,selection_metric='mae',forecast_loss='mse_plus_0p2_mae',repair_retention='uniform_v1',repair_update_policy='freeze_front_v1',forecast_padding='replicate_last_value_v1');payload=tr.load_stage2_selection(s2,p,cfg,device=torch.device('cpu'));assert payload['repair_input_scaling']=='all_floor_v1' and payload['repair_output_statistics']=='original_visible_stats'
 d=json.loads(s2.read_text());d['repair_input_scaling']='wrong';wrong=tmp_path/'wrong.json';wrong.write_text(json.dumps(d))
 with pytest.raises(ValueError,match='repair_input_scaling.*mismatch'):tr.load_stage2_selection(wrong,p,cfg,device=torch.device('cpu'))


@pytest.mark.parametrize('value',['wrong',True,1])
def test_invalid_input_scale_rejected(tmp_path,value):
 p,_=setup(tmp_path);path=tmp_path/'True-True-False.yaml';d=yaml.safe_load(path.read_text());d['model'].update(variable_calibration='none',front_variable_context='none',repair_coordinates='robust_local_v1',point_variable_readout='none',historical_key_policy='legacy_v1',recent_mean_projection='none',temporal_evidence_readout='none',repair_input_scaling=value);path.write_text(yaml.safe_dump(d))
 with pytest.raises(ValueError):load_e23_settings(path,protocol=p)
