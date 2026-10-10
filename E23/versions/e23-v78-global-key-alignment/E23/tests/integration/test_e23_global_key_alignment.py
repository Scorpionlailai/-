"""History query/teacher/library share global coordinates, preserving old recipes."""
from dataclasses import replace
import json
from types import SimpleNamespace
import pytest
import torch
import yaml
from integrations.e23 import train as tr
from integrations.e23.config import load_e23_settings
from integrations.e23.forecast_sensitive import parameter_hash
from malr.data.patching import patchify_with_state,patchify_preserving_values
from malr.prototypes.retrieval import RetrievalEncoderEMA
from tests.integration.test_e23_cross_variable_level import configuration as level_configuration
from tests.integration.test_e23_global_level_context import batch


def configuration(tmp_path):
    p,_=level_configuration(tmp_path);path=tmp_path/'True-True-False.yaml';doc=yaml.safe_load(path.read_text())
    doc['model']['historical_key_policy']='global_aligned_v2';path.write_text(yaml.safe_dump(doc))
    return p,load_e23_settings(path,protocol=p)


def prepare(tmp_path):
    p,cfg=configuration(tmp_path);m=tr.build_forecast_model(p,cfg,device=torch.device('cpu')).eval()
    b=batch();b=replace(b,query_std=torch.linspace(.2,1.4,7)[None].expand_as(b.query_std).clone())
    return p,cfg,m,b


def capture_query(m,b,monkeypatch):
    captured=[]
    original=m.repair.historical.retrieval.encode_frozen
    def encode(values,mask):
        captured.append((values.detach().clone(),mask.detach().clone(),original(values,mask).detach().clone()))
        return captured[-1][2]
    def stop(*args,**kwargs):raise RuntimeError('captured historical query')
    m.repair.historical.index=SimpleNamespace(member_keys=torch.ones(1,16),query=stop)
    monkeypatch.setattr(m.repair.historical.retrieval,'encode_frozen',encode)
    starts=tr._starts(b,stride=8,patches=12,device=torch.device('cpu'))
    with pytest.raises(RuntimeError,match='captured historical query'):
        m.repair(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts)
    monkeypatch.setattr(m.repair.historical.retrieval,'encode_frozen',original)
    return captured[0]


def test_new_policy_keeps_parameters_rng_formal_head_and_empty_bank_function(tmp_path):
    p,cfg=configuration(tmp_path)
    torch.manual_seed(2023);old=tr.build_forecast_model(p,replace(cfg,historical_key_policy='legacy_v1'),device=torch.device('cpu')).eval();rng=torch.get_rng_state()
    torch.manual_seed(2023);new=tr.build_forecast_model(p,cfg,device=torch.device('cpu')).eval()
    assert torch.equal(rng,torch.get_rng_state());assert parameter_hash(old.predictor)==parameter_hash(new.predictor)
    assert list(old.state_dict())==list(new.state_dict())
    for k,v in old.state_dict().items():assert torch.equal(v,new.state_dict()[k]),k
    b=batch();s=tr._starts(b,stride=8,patches=12,device=torch.device('cpu'))
    with torch.no_grad():
        a,ra=old(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,s)
        z,rz=new(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,s)
    assert torch.equal(a,z) and torch.equal(ra.explicit.normalized_sequence,rz.explicit.normalized_sequence)


def test_training_query_and_clean_encoder_inputs_use_global_library_coordinates(tmp_path):
    _,_,m,b=prepare(tmp_path);captured=[]
    handle=m.repair.historical.key_encoder.register_forward_pre_hook(lambda module,args:captured.append(tuple(v.detach().clone() for v in args)))
    loss=tr._retrieval_loss(m.repair,b);handle.remove()
    safe=torch.where(b.observed_mask,b.normalized_observed,0)
    qglobal=torch.where(b.observed_mask,safe*b.query_std[:,None]+b.query_mean[:,None],0)
    q=patchify_with_state(qglobal,b.observed_mask,16,8)
    tglobal=b.normalized_target*b.query_std[:,None]+b.query_mean[:,None]
    t=patchify_preserving_values(tglobal,torch.ones_like(b.observed_mask),16,8)
    damaged=q.rho.lt(1)
    torch.testing.assert_close(captured[0][0],q.values[damaged].masked_fill(~q.effective_mask[damaged],0))
    assert torch.equal(captured[0][1],q.effective_mask[damaged])
    torch.testing.assert_close(captured[1][0],t.values[damaged].masked_fill(~t.validity[damaged],0))
    assert torch.equal(captured[1][1],t.validity[damaged]);assert torch.isfinite(loss)


def test_inference_query_coordinates_equal_visible_global_values(tmp_path,monkeypatch):
    _,_,m,b=prepare(tmp_path);values,mask,keys=capture_query(m,b,monkeypatch)
    q=patchify_with_state(b.normalized_observed,b.observed_mask,16,8)
    raw=torch.where(q.effective_mask,q.values*b.query_std[:,:,None,None]+b.query_mean[:,:,None,None],0)
    torch.testing.assert_close(values,raw.reshape(-1,16));assert torch.equal(mask,q.effective_mask.reshape(-1,16))
    assert torch.isfinite(keys).all()


def test_same_global_signal_in_different_local_frames_has_same_history_query_keys(tmp_path,monkeypatch):
    _,_,m,b=prepare(tmp_path);a=capture_query(m,b,monkeypatch)
    global_values=b.normalized_observed*b.query_std[:,None]+b.query_mean[:,None]
    mean=b.query_mean+.125;std=b.query_std*1.5
    values=torch.where(b.observed_mask,(global_values-mean[:,None])/std[:,None],0)
    changed=replace(b,normalized_observed=values,query_mean=mean,query_std=std)
    z=capture_query(m,changed,monkeypatch)
    torch.testing.assert_close(a[0],z[0],atol=2e-7,rtol=2e-7);torch.testing.assert_close(a[2],z[2],atol=2e-7,rtol=2e-7)


def test_hidden_nonfinite_payloads_never_enter_frozen_query(tmp_path,monkeypatch):
    _,_,m,b=prepare(tmp_path);a=capture_query(m,b,monkeypatch)
    for poison in [float('nan'),float('inf'),-float('inf')]:
        changed=replace(b,normalized_observed=torch.where(b.observed_mask,b.normalized_observed,poison))
        z=capture_query(m,changed,monkeypatch);assert torch.equal(a[0],z[0]) and torch.equal(a[2],z[2])
        assert z[0][~z[1]].eq(0).all()


def test_future_is_excluded_from_key_loss_and_global_query_has_finite_gradient(tmp_path):
    _,_,m,b=prepare(tmp_path);a=tr._retrieval_loss(m.repair,b)
    z=tr._retrieval_loss(m.repair,replace(b,forecast_target=b.forecast_target+1e9));assert torch.equal(a,z)
    a.backward();assert torch.isfinite(a)
    assert m.repair.historical.key_encoder.network[0].weight.grad.abs().sum()>0
    assert all(torch.isfinite(p.grad).all() for p in m.repair.historical.key_encoder.parameters() if p.grad is not None)


def test_epoch_ema_observed_library_and_checkpoint_identity_are_preserved(tmp_path,monkeypatch):
    p,cfg,m,b=prepare(tmp_path);calls=[];original=RetrievalEncoderEMA.update_ema
    def spy(self):calls.append(self.frozen_epoch);return original(self)
    monkeypatch.setattr(RetrievalEncoderEMA,'update_ema',spy)
    monkeypatch.setattr(tr,'iter_p4_batches',lambda _p,split,**kw:iter((b,b)) if split=='train' else iter((b,)))
    staged=[];oldstage=tr._stage_observed
    def stage(memory,model,batch,starts,known_identities=None):
        raw_values=batch.normalized_target*batch.query_std[:,None]+batch.query_mean[:,None]
        patches=patchify_with_state(raw_values,batch.observed_mask,16,8)
        eligible=patches.rho.eq(1) & patches.validity.all(-1)
        expected={(int(v),int(starts[int(sample),int(n)])):patches.values[int(sample),int(v),int(n)] for sample,v,n in eligible.nonzero()}
        original_method=memory.stage_patch
        def check(variable,start,raw,key,*,observed):
            assert observed is True and (variable,start) in expected
            torch.testing.assert_close(torch.as_tensor(raw),expected[(variable,start)])
            staged.append((variable,start));return original_method(variable,start,raw,key,observed=observed)
        memory.stage_patch=check
        try:return oldstage(memory,model,batch,starts,known_identities)
        finally:memory.stage_patch=original_method
    monkeypatch.setattr(tr,'_stage_observed',stage)
    s1=tr.train_repair(p,cfg,output=tmp_path/'s1',device=torch.device('cpu'),epochs=2)
    assert calls==[1,2] and staged
    saved=tr.load_stage1_selection(s1,p,cfg,device=torch.device('cpu'))
    assert saved['historical_key_coordinates']=='train_standardized' and saved['historical_key_ema_update']=='epoch'
    assert saved['historical_key_loss']=='legacy_pair_and_contrastive'
    assert 'historical_key_variance_weight' not in saved
    s2=tr.train_forecast(p,cfg,stage1_manifest=s1,output=tmp_path/'s2',device=torch.device('cpu'),epochs=1,selection_metric='mae',forecast_loss='mse_plus_0p2_mae',repair_retention='uniform_v1',repair_update_policy='frozen_v1',extra_mse_checkpoint=True)
    state1=torch.load(json.loads(s1.read_text())['checkpoint'],weights_only=False)['model_state']
    for choice in [s2,s2.parent/'e23-stage2-mse.json']:
        payload=tr.load_stage2_selection(choice,p,cfg,device=torch.device('cpu'))
        assert all(torch.equal(v,payload['model_state']['repair.'+k]) for k,v in state1.items())
    rec=json.loads(s2.read_text());rec['historical_key_coordinates']='window_normalized';wrong=tmp_path/'wrong.json';wrong.write_text(json.dumps(rec))
    with pytest.raises(ValueError,match='historical_key_coordinates.*mismatch'):tr.load_stage2_selection(wrong,p,cfg,device=torch.device('cpu'))
