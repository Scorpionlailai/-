from dataclasses import replace
import json
import torch
from integrations.e23 import train as tr
from tests.integration.test_e23_point_candidates import setup
from tests.integration.test_e23_recent_mean import batch


def test_pure_mse_frozen_legacy_keeps_head_init_padding_and_entire_repair(tmp_path,monkeypatch):
 p,cfg=setup(tmp_path);cfg=replace(cfg,repair_coordinates='robust_local_v1',recent_mean_projection='mean_v1',stage1_structure='recent_mean_v2',front_variable_context='signature_plain_v2');monkeypatch.setattr(tr,'iter_p4_batches',lambda _p,split,**kw:iter((batch(),)))
 s1=tr.train_repair(p,cfg,output=tmp_path/'s1',device=torch.device('cpu'),epochs=1);first=tr.load_stage1_selection(s1,p,cfg,device=torch.device('cpu'))
 s2=tr.train_forecast(p,cfg,stage1_manifest=s1,output=tmp_path/'pure',device=torch.device('cpu'),epochs=2,selection_metric='mae',forecast_loss='mse',repair_retention='none',repair_update_policy='frozen_v1',extra_mse_checkpoint=True)
 mixed=tr.train_forecast(p,cfg,stage1_manifest=s1,output=tmp_path/'mixed',device=torch.device('cpu'),epochs=2,selection_metric='mae',forecast_loss='mse_plus_0p2_mae',repair_retention='uniform_v1',repair_update_policy='frozen_v1',extra_mse_checkpoint=True)
 assert json.loads(s2.read_text())['predictor_initial_sha256']==json.loads(mixed.read_text())['predictor_initial_sha256']
 for selected in [s2,s2.parent/'e23-stage2-mse.json']:
  r=json.loads(selected.read_text());payload=tr.load_stage2_selection(selected,p,cfg,device=torch.device('cpu'));assert r['repair_update_policy']=='frozen_v1' and r['repair_trainable_parameters']==0 and r['loss']=='forecast_mse_only'
  assert r.get('repair_retention','none')=='none' and r.get('downstream_protocol','legacy')=='legacy' and r.get('forecast_padding','zero_invalid_v1')=='zero_invalid_v1'
  assert all(torch.equal(v,payload['model_state']['repair.'+k]) for k,v in first['model_state'].items())
  assert payload['memory_sha256']==first['memory_sha256'] and payload['key_sha256']==first['key_sha256']
