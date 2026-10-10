from dataclasses import replace
import json
import pytest
import torch
from integrations.e23 import train as tr
from integrations.e23.forecast_sensitive import parameter_hash
from tests.integration.test_e23_point_candidates import setup
from tests.integration.test_e23_stage1 import _batch


def test_same_repair_and_head_stage1_handoff_pure_mse_without_padding_or_rng_change(tmp_path,monkeypatch):
    p,cfg=setup(tmp_path);cfg=replace(cfg,repair_coordinates='robust_local_v1')
    monkeypatch.setattr(tr,'iter_p4_batches',lambda _p,split,**kw:iter((_batch(split),)))
    s1=tr.train_repair(p,cfg,output=tmp_path/'s1',device=torch.device('cpu'),epochs=1)
    old=tr.train_forecast(p,cfg,stage1_manifest=s1,output=tmp_path/'old',device=torch.device('cpu'),epochs=1,selection_metric='mae',forecast_loss='mse_plus_0p2_mae',repair_retention='uniform_v1',repair_update_policy='freeze_front_v1')
    new=tr.train_forecast(p,cfg,stage1_manifest=s1,output=tmp_path/'new',device=torch.device('cpu'),epochs=1,selection_metric='mae',forecast_loss='mse',repair_retention='none',repair_update_policy='freeze_front_v1')
    a=json.loads(old.read_text());b=json.loads(new.read_text());assert a['predictor_initial_sha256']==b['predictor_initial_sha256'] and a['repair_initial_sha256']==b['repair_initial_sha256']
    assert b['loss']=='forecast_mse_only' and b.get('repair_retention','none')=='none' and b.get('downstream_protocol','legacy')=='legacy'
    assert b.get('forecast_padding','zero_invalid')!='replicate_last_value_v1'
    payload=tr.load_stage2_selection(new,p,cfg,device=torch.device('cpu'));original=tr.load_stage1_selection(s1,p,cfg,device=torch.device('cpu'))
    assert all(torch.equal(v,payload['model_state']['repair.'+k]) for k,v in original['model_state'].items() if k.startswith('front_context.'))
    dual=new.parent/'e23-stage2-mse.json';assert dual.exists();tr.load_stage2_selection(dual,p,cfg,device=torch.device('cpu'))


@pytest.mark.parametrize('mode',['frozen_v1','slow_lr_0p1_v1'])
def test_pure_legacy_mode_rejected_without_explicit_supported_front_recipe(tmp_path,mode):
    p,cfg=setup(tmp_path)
    with pytest.raises(ValueError,match='repair update controls'):tr.train_forecast(p,cfg,stage1_manifest=tmp_path/'absent',output=tmp_path/'out',device=torch.device('cpu'),epochs=1,selection_metric='mae',forecast_loss='mse',repair_retention='none',repair_update_policy=mode)
    assert not (tmp_path/'out').exists()
