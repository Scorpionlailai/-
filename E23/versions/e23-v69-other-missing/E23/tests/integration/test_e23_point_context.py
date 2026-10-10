from dataclasses import replace
import json
import pytest,torch,yaml
from integrations.e23 import train as tr
from integrations.e23.config import load_e23_settings
from integrations.e23.forecast_sensitive import parameter_hash
from tests.integration.test_e23_point_candidates import setup
from tests.integration.test_e23_recent_mean import batch


def configuration(tmp_path):
 p,cfg=setup(tmp_path)
 return p,replace(cfg,repair_coordinates='robust_local_v1',point_context_readout='direct_point_v1')


def test_zero_point_context_preserves_body_rng_head_and_original_function(tmp_path):
 p,cfg=configuration(tmp_path);torch.manual_seed(502);old=tr.build_forecast_model(p,replace(cfg,point_context_readout='none'),device=torch.device('cpu')).eval();rng=torch.get_rng_state();torch.manual_seed(502);new=tr.build_forecast_model(p,cfg,device=torch.device('cpu')).eval()
 assert torch.equal(rng,torch.get_rng_state()) and parameter_hash(old.predictor)==parameter_hash(new.predictor)
 for k,v in old.repair.state_dict().items():assert torch.equal(v,new.repair.state_dict()[k]),k
 assert sum(v.numel() for v in new.repair.point_context_readout.parameters())==9153
 b=batch();starts=tr._starts(b,stride=8,patches=12,device=torch.device('cpu'))
 with torch.no_grad():
  a,oa=old(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts)
  z,oz=new(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts)
 assert torch.equal(a,z) and torch.equal(oa.explicit.normalized_sequence,oz.explicit.normalized_sequence)


def test_point_context_changes_only_recent_missing_and_excludes_hidden_payload(tmp_path):
 p,cfg=configuration(tmp_path);m=tr.build_forecast_model(p,cfg,device=torch.device('cpu')).eval();b=batch();mask=b.observed_mask.clone();mask[:,72:80]=False;values=torch.where(mask,b.normalized_observed,0);starts=tr._starts(b,stride=8,patches=12,device=torch.device('cpu'))
 with torch.no_grad():
  before=m.repair(values,mask,b.query_mean,b.query_std,starts).explicit
  m.repair.point_context_readout.network[-1].weight.normal_(std=.1)
  after=m.repair(values,mask,b.query_mean,b.query_std,starts).explicit
  polluted=torch.where(mask,values,torch.full_like(values,float('inf')));canary=m.repair(polluted,mask,b.query_mean,b.query_std,starts).explicit
 assert torch.equal(after.normalized_sequence[:,:64],before.normalized_sequence[:,:64])
 assert torch.equal(after.normalized_sequence[mask],values[mask]) and torch.equal(after.normalized_sequence,canary.normalized_sequence)
 assert torch.equal(after.sequence_observed,mask) and torch.equal(after.coverage,before.coverage)
 assert not torch.equal(after.normalized_sequence[:,72:80],before.normalized_sequence[:,72:80])
 assert torch.isfinite(after.normalized_sequence).all()


def test_point_context_learns_from_hidden_repair_loss(tmp_path):
 p,cfg=configuration(tmp_path);m=tr.build_forecast_model(p,cfg,device=torch.device('cpu'));b=batch();mask=b.observed_mask.clone();mask[:,72:80]=False;starts=tr._starts(b,stride=8,patches=12,device=torch.device('cpu'));opt=torch.optim.Adam(m.repair.parameters(),lr=.001)
 for step in range(3):
  opt.zero_grad();o=m.repair(b.normalized_observed,mask,b.query_mean,b.query_std,starts);loss,_=tr.stage1_objective(o.explicit.normalized_sequence,b.normalized_target,mask,torch.tensor(0.),retrieval_weight=.1,loss_mode='global_mse_v1',query_std=b.query_std);loss.backward()
  assert m.repair.point_context_readout.network[-1].weight.grad.abs().sum()>0
  if step:assert m.repair.point_context_readout.network[0].weight.grad.abs().sum()>0
  assert all(torch.isfinite(v.grad).all() for v in m.parameters() if v.grad is not None);opt.step()


def test_point_context_config_and_dual_checkpoint_roundtrip(tmp_path,monkeypatch):
 p,_=setup(tmp_path);path=tmp_path/'True-True-False.yaml';doc=yaml.safe_load(path.read_text());doc['model'].update(variable_calibration='none',front_variable_context='none',repair_coordinates='robust_local_v1',point_variable_readout='none',historical_key_policy='legacy_v1',recent_mean_projection='none',temporal_evidence_readout='none',repair_input_scaling='legacy_v1',point_context_readout='direct_point_v1');path.write_text(yaml.safe_dump(doc));cfg=load_e23_settings(path,protocol=p);monkeypatch.setattr(tr,'iter_p4_batches',lambda _p,split,**kw:iter((batch(),)))
 s1=tr.train_repair(p,cfg,output=tmp_path/'s1',device=torch.device('cpu'),epochs=1)
 s2=tr.train_forecast(p,cfg,stage1_manifest=s1,output=tmp_path/'s2',device=torch.device('cpu'),epochs=1,selection_metric='mae',forecast_loss='mse_plus_0p2_mae',repair_retention='uniform_v1',repair_update_policy='freeze_front_v1',extra_mse_checkpoint=True)
 for choice in [s2,s2.parent/'e23-stage2-mse.json']:
  payload=tr.load_stage2_selection(choice,p,cfg,device=torch.device('cpu'));assert payload['point_context_readout']=='direct_point_v1' and payload['point_context_extra_parameters']==9153
 rec=json.loads(s2.read_text());rec['point_context_scope']=96;wrong=tmp_path/'wrong.json';wrong.write_text(json.dumps(rec))
 with pytest.raises(ValueError,match='point_context_scope.*mismatch'):tr.load_stage2_selection(wrong,p,cfg,device=torch.device('cpu'))
