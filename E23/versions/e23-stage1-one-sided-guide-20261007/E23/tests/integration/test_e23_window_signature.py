from dataclasses import replace
import json
import pytest
import torch
import yaml
from integrations.e23 import train as tr
from integrations.e23.config import load_e23_settings
from integrations.e23.forecast_sensitive import parameter_hash
from malr.data.patching import patchify_with_state
from malr.models.e23.window_context import WindowContextEncoder
from tests.integration.test_e23_point_candidates import setup
from tests.integration.test_e23_stage1 import _batch


def test_signature_zero_start_preserves_front_features_body_rng_and_head(tmp_path):
    p,cfg=setup(tmp_path);torch.manual_seed(502)
    old=tr.build_forecast_model(p,cfg,device=torch.device('cpu')).eval();rng=torch.get_rng_state()
    torch.manual_seed(502);new=tr.build_forecast_model(p,replace(cfg,front_variable_context='signature_attention_v1'),device=torch.device('cpu')).eval()
    assert torch.equal(rng,torch.get_rng_state()) and parameter_hash(old.predictor)==parameter_hash(new.predictor)
    for k,v in old.repair.state_dict().items():assert torch.equal(v,new.repair.state_dict()[k]),k
    b=_batch('validation');starts=tr._starts(b,stride=8,patches=12,device=torch.device('cpu'))
    with torch.no_grad():
        a,oa=old(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts)
        z,oz=new(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts)
    assert torch.equal(a,z) and torch.equal(oa.explicit.normalized_sequence,oz.explicit.normalized_sequence)


def test_nonzero_signature_reads_other_visible_variables_and_excludes_hidden_and_invalid_payloads():
    torch.manual_seed(9);x=torch.randn(2,96,3);mask=torch.rand_like(x)>.73
    valid=torch.ones_like(mask);valid[:,-1]=False
    pv=patchify_with_state(x,mask,16,8,validity=valid).validity
    mean=torch.zeros(2,3);std=torch.ones(2,3)
    front=WindowContextEncoder(64,96,16,8,variable_context='signature_attention_v1',variable_count=3).eval()
    with torch.no_grad():
        front.output.weight.normal_(std=.05)
        for block in front.variable_context.blocks:block.project.weight.normal_(std=.03)
    a=front(x,mask,mean,std,pv,valid_mask=valid)
    polluted=torch.where(mask&valid,x,torch.full_like(x,float('inf')))
    z=front(polluted,mask,mean,std,pv,valid_mask=valid)
    assert torch.equal(a,z)
    changed=x.clone();changed[:,:,1]+=5
    q=front(changed,mask,mean,std,pv,valid_mask=valid)
    assert not torch.equal(a[:,0],q[:,0])
    mask[:, :, 1:]=False
    pv=patchify_with_state(x,mask,16,8,validity=valid).validity
    a=front(x,mask,mean,std,pv,valid_mask=valid)
    z=front(changed,mask,mean,std,pv,valid_mask=valid)
    assert torch.equal(a,z)
    empty=torch.zeros_like(mask)
    assert torch.isfinite(front(polluted,empty,mean,std,pv,valid_mask=valid)).all()


def test_signature_learns_from_hidden_repair_and_is_entirely_frozen_by_stage2(tmp_path):
    p,cfg=setup(tmp_path);cfg=replace(cfg,front_variable_context='signature_attention_v1')
    m=tr.build_forecast_model(p,cfg,device=torch.device('cpu'));b=_batch('train')
    starts=tr._starts(b,stride=8,patches=12,device=torch.device('cpu'));opt=torch.optim.Adam(m.repair.parameters(),lr=.001)
    for step in range(4):
        opt.zero_grad();o=m.repair(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts)
        loss,_=tr.stage1_objective(o.explicit.normalized_sequence,b.normalized_target,b.observed_mask,
            torch.tensor(0.),retrieval_weight=.1,loss_mode='global_mse_v1',query_std=b.query_std)
        loss.backward();block=m.repair.front_context.variable_context.blocks[0]
        if step>=1:assert block.project.weight.grad.abs().sum()>0
        if step>=2:
            assert block.query.large.weight.grad.abs().sum()>0
            assert block.statistics[-1].weight.grad.abs().sum()>0
        assert all(torch.isfinite(x.grad).all() for x in m.parameters() if x.grad is not None)
        opt.step()
    from integrations.e23.repair_updates import build_stage2_optimizer,stage2_training_mode
    opt=build_stage2_optimizer(m,mode='freeze_front_v1',learning_rate=1e-4,weight_decay=0)
    stage2_training_mode(m,'freeze_front_v1')
    assert not m.repair.front_context.variable_context.training
    assert not any(x.requires_grad for x in m.repair.front_context.variable_context.parameters())
    ids={id(x) for g in opt.param_groups for x in g['params']}
    assert not any(id(x) in ids for x in m.repair.front_context.variable_context.parameters())


def test_signature_checkpoint_roundtrip_and_identity_rejection(tmp_path,monkeypatch):
    p,_=setup(tmp_path);path=tmp_path/'True-True-False.yaml';doc=yaml.safe_load(path.read_text())
    doc['model'].update(variable_calibration='none',front_variable_context='signature_attention_v1');path.write_text(yaml.safe_dump(doc))
    cfg=load_e23_settings(path,protocol=p)
    monkeypatch.setattr(tr,'iter_p4_batches',lambda _p,split,**kw:iter((_batch(split),)))
    s1=tr.train_repair(p,cfg,output=tmp_path/'s1',device=torch.device('cpu'),epochs=1)
    s2=tr.train_forecast(p,cfg,stage1_manifest=s1,output=tmp_path/'s2',device=torch.device('cpu'),epochs=1,
        selection_metric='mae',forecast_loss='mse_plus_0p2_mae',repair_retention='uniform_v1',repair_update_policy='freeze_front_v1')
    payload=tr.load_stage2_selection(s2,p,cfg,device=torch.device('cpu'))
    assert payload['front_context_cross_variable'] is True and payload['front_signature_head_dim']==96
    rec=json.loads(s2.read_text());rec['front_variable_context']='wrong';wrong=tmp_path/'wrong.json';wrong.write_text(json.dumps(rec))
    with pytest.raises(ValueError,match='front_variable_context.*mismatch'):tr.load_stage2_selection(wrong,p,cfg,device=torch.device('cpu'))
