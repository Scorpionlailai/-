from dataclasses import replace
import json
import pytest,torch
from integrations.e23 import train as tr
from tests.integration.test_e23_point_candidates import setup
from tests.integration.test_e23_recent_mean import batch


def test_legacy_mixed_loss_extra_mse_selection_same_run_roundtrip(tmp_path,monkeypatch):
 p,cfg=setup(tmp_path);cfg=replace(cfg,repair_coordinates='robust_local_v1');monkeypatch.setattr(tr,'iter_p4_batches',lambda _p,split,**kw:iter((batch(),)))
 s1=tr.train_repair(p,cfg,output=tmp_path/'s1',device=torch.device('cpu'),epochs=1,training_batch_size=32);r1=json.loads(s1.read_text());assert r1['stage1_training_batch_size']==32 and r1['validation_batch_size']==128
 s2=tr.train_forecast(p,cfg,stage1_manifest=s1,output=tmp_path/'s2',device=torch.device('cpu'),epochs=1,selection_metric='mae',forecast_loss='mse_plus_0p2_mae',repair_retention='uniform_v1',repair_update_policy='freeze_front_v1',extra_mse_checkpoint=True)
 secondary=s2.parent/'e23-stage2-mse.json';assert secondary.exists()
 for path,metric in [(s2,'mae'),(secondary,'mse')]:
  r=json.loads(path.read_text());payload=tr.load_stage2_selection(path,p,cfg,device=torch.device('cpu'));assert r['selection_metric']==payload['selection_metric']==metric;assert r['loss']=='forecast_mse_plus_0p2_mae' and r['repair_retention']=='uniform_v1';assert payload['checkpoint_tracking']=='mae_and_mse_v1'
 assert json.loads(secondary.read_text())['training_resource_parent_manifest']==str(s2.resolve())
 d=json.loads(s2.read_text());d['checkpoint_tracking']='wrong';wrong=tmp_path/'wrong.json';wrong.write_text(json.dumps(d))
 with pytest.raises(ValueError,match='checkpoint tracking'):tr.load_stage2_selection(wrong,p,cfg,device=torch.device('cpu'))


@pytest.mark.parametrize('flag',[1,'yes',None])
def test_bad_tracking_flag_rejected(tmp_path,flag):
 p,cfg=setup(tmp_path)
 with pytest.raises(ValueError,match='checkpoint'):tr.train_forecast(p,cfg,stage1_manifest=tmp_path/'absent',output=tmp_path/'output',device=torch.device('cpu'),extra_mse_checkpoint=flag)
