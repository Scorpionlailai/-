from dataclasses import replace
import json
from types import SimpleNamespace
import pytest
import torch
import yaml
from integrations.e23 import train as tr
from integrations.e23.config import load_e23_settings
from integrations.e23.diverse_keys import clean_key_regularization
from integrations.e23.forecast_sensitive import parameter_hash
from malr.data.patching import patchify_with_state
from malr.prototypes.retrieval import RetrievalEncoderEMA
from tests.integration.test_e23_point_candidates import setup
from tests.integration.test_e23_stage1 import _batch


def test_clean_content_variance_penalty_discourages_near_collapse_and_has_finite_gradients():
    torch.manual_seed(6);near=(torch.randn(4,7,11,16)*1e-3+.25).requires_grad_()
    diverse=torch.nn.functional.normalize(torch.randn(4,7,11,16),dim=-1)
    va,ca=clean_key_regularization(near);vb,cb=clean_key_regularization(diverse)
    assert va>.5 and vb<1e-3 and torch.isfinite(ca+cb)
    (va+.05*ca).backward();assert near.grad.abs().sum()>0 and torch.isfinite(near.grad).all()


def test_new_policy_preserves_all_initial_parameters_rng_and_predictor(tmp_path):
    p,cfg=setup(tmp_path);torch.manual_seed(502);old=tr.build_forecast_model(p,cfg,device=torch.device('cpu'));rng=torch.get_rng_state()
    torch.manual_seed(502);m=tr.build_forecast_model(p,replace(cfg,historical_key_policy='global_diverse_v1'),device=torch.device('cpu'))
    assert torch.equal(rng,torch.get_rng_state()) and parameter_hash(old.predictor)==parameter_hash(m.predictor)
    for k,v in old.state_dict().items():assert torch.equal(v,m.state_dict()[k]),k


def test_history_inference_queries_use_global_visible_values_and_mask_before_arithmetic(tmp_path,monkeypatch):
    p,cfg=setup(tmp_path);m=tr.build_forecast_model(p,replace(cfg,historical_key_policy='global_diverse_v1'),device=torch.device('cpu'));b=_batch('validation');starts=tr._starts(b,stride=8,patches=12,device=torch.device('cpu'));seen=[]
    def encode(values,mask):seen.append((values.detach().clone(),mask.detach().clone()));return torch.ones(len(values),16)
    def stop(*args,**kwargs):raise RuntimeError('captured query')
    m.repair.historical.index=SimpleNamespace(member_keys=torch.ones(1,16),query=stop)
    monkeypatch.setattr(m.repair.historical.retrieval,'encode_frozen',encode)
    with pytest.raises(RuntimeError,match='captured query'):m.repair(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts)
    patches=patchify_with_state(b.normalized_observed,b.observed_mask,16,8)
    expected=torch.where(patches.effective_mask,patches.values*b.query_std[:,:,None,None]+b.query_mean[:,:,None,None],0).reshape(-1,16)
    assert torch.equal(seen[0][0],expected)
    assert seen[0][0][~seen[0][1]].eq(0).all()


def test_key_objective_uses_past_targets_and_rejects_empty_sources_with_finite_gradients(tmp_path):
    p,cfg=setup(tmp_path);m=tr.build_repair_model(p,replace(cfg,historical_key_policy='global_diverse_v1'),device=torch.device('cpu'));b=_batch('train');a=tr._retrieval_loss(m,b)
    z=tr._retrieval_loss(m,replace(b,forecast_target=b.forecast_target+1e9));assert torch.equal(a,z)
    a.backward();assert torch.isfinite(a) and all(torch.isfinite(v.grad).all() for v in m.historical.key_encoder.parameters() if v.grad is not None)
    assert m.historical.key_encoder.network[0].weight.grad.abs().sum()>0
    empty=replace(b,observed_mask=torch.zeros_like(b.observed_mask),normalized_observed=torch.zeros_like(b.normalized_observed));q=tr._retrieval_loss(m,empty);assert q==0 and torch.isfinite(q)


def test_ema_updates_each_step_snapshot_publishes_per_epoch_and_checkpoint_reloads(tmp_path,monkeypatch):
    p,_=setup(tmp_path);path=tmp_path/'True-True-False.yaml';doc=yaml.safe_load(path.read_text());doc['model'].update(variable_calibration='none',front_variable_context='none',repair_coordinates='robust_local_v1',point_variable_readout='none',historical_key_policy='global_diverse_v1');path.write_text(yaml.safe_dump(doc));cfg=load_e23_settings(path,protocol=p)
    calls=[];old=RetrievalEncoderEMA.update_ema
    def spy(self):calls.append(self.frozen_epoch);return old(self)
    monkeypatch.setattr(RetrievalEncoderEMA,'update_ema',spy)
    monkeypatch.setattr(tr,'iter_p4_batches',lambda _p,split,**kw:iter((_batch(split),_batch(split))) if split=='train' else iter((_batch(split),)))
    s1=tr.train_repair(p,cfg,output=tmp_path/'s1',device=torch.device('cpu'),epochs=2);assert calls==[1,1,2,2]
    saved=tr.load_stage1_selection(s1,p,cfg,device=torch.device('cpu'));assert saved['historical_key_ema_update']=='optimizer_step'
    s2=tr.train_forecast(p,cfg,stage1_manifest=s1,output=tmp_path/'s2',device=torch.device('cpu'),epochs=1,selection_metric='mae',forecast_loss='mse',repair_retention='none',repair_update_policy='freeze_front_v1');tr.load_stage2_selection(s2,p,cfg,device=torch.device('cpu'))
    d=json.loads(s1.read_text());d['historical_key_policy']='legacy_v1';wrong=tmp_path/'wrong.json';wrong.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='historical_key_policy.*mismatch'):tr.load_stage1_selection(wrong,p,cfg,device=torch.device('cpu'))


@pytest.mark.parametrize('value',['wrong',True,1])
def test_unknown_history_key_policy_rejected(tmp_path,value):
    p,_=setup(tmp_path);path=tmp_path/'True-True-False.yaml';doc=yaml.safe_load(path.read_text());doc['model'].update(variable_calibration='none',front_variable_context='none',repair_coordinates='robust_local_v1',point_variable_readout='none',historical_key_policy=value);path.write_text(yaml.safe_dump(doc))
    with pytest.raises(ValueError):load_e23_settings(path,protocol=p)
