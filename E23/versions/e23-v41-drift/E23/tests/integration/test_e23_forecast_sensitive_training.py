import torch
from torch import nn
from tests.integration.test_e23_forecast_sensitive import api


def test_teacher_is_independent_frozen_snapshot_and_refreshes_next_epoch():
    student=nn.Sequential(nn.Linear(3,4),nn.Dropout(.5));student.train()
    teacher=api().snapshot_teacher(student)
    assert not teacher.training and student.training
    for a,b in zip(student.parameters(),teacher.parameters()):
        assert a.data_ptr()!=b.data_ptr() and not b.requires_grad
        torch.testing.assert_close(a,b)
    old=[p.clone() for p in teacher.parameters()]
    with torch.no_grad():
        for p in student.parameters():p.add_(1)
    for before,after in zip(old,teacher.parameters()):torch.testing.assert_close(before,after)
    refreshed=api().snapshot_teacher(student)
    for a,b in zip(student.parameters(),refreshed.parameters()):torch.testing.assert_close(a,b)


def test_retention_identity_and_warmup_are_explicit():
    assert api().retention_identity('none')=={}
    identity=api().retention_identity('uniform_v1')
    assert identity['repair_retention_warmup']==10
    assert identity['repair_retention_weight']==.05
    c=api().retention_identity('forecast_gain_v1')
    assert c['repair_retention_teacher']=='previous_epoch_final'
    assert c['repair_retention_subset']==16 and c['repair_retention_regions']==42
    for epoch in (1,10):assert api().retention_active('uniform_v1',epoch) is False
    for mode in ('uniform_v1','forecast_gain_v1'):assert api().retention_active(mode,11) is True
    assert api().retention_active('none',100) is False


def test_stage2_retention_records_identity_and_rejects_changed_manifest(monkeypatch,tmp_path):
    import json
    import pytest
    from dataclasses import replace
    from tests.integration.test_e23_stage2 import _selected_stage1
    from integrations.e23.train import train_forecast,load_stage2_selection
    protocol,settings,stage1=_selected_stage1(monkeypatch,tmp_path)
    settings=replace(settings,stage2_epochs=11)
    selected=train_forecast(protocol,settings,stage1_manifest=stage1,output=tmp_path/'uniform',device=torch.device('cpu'),epochs=11,forecast_loss='mse_plus_0p2_mae',selection_metric='mae',repair_retention='uniform_v1')
    record=json.loads(selected.read_text());assert record['repair_retention']=='uniform_v1'
    assert record['repair_retention_active_batches']==1
    assert record['repair_retention_teacher_seconds']==0
    load_stage2_selection(selected,protocol,settings,device=torch.device('cpu'))
    record['repair_retention_weight']=.1;selected.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='retention'):
        load_stage2_selection(selected,protocol,settings,device=torch.device('cpu'))


def test_gain_mode_runs_teacher_only_after_warmup_and_keeps_bank_frozen(monkeypatch,tmp_path):
    import json
    from dataclasses import replace
    from tests.integration.test_e23_stage2 import _selected_stage1
    from integrations.e23.train import train_forecast,load_stage2_selection
    protocol,settings,stage1=_selected_stage1(monkeypatch,tmp_path)
    settings=replace(settings,stage2_epochs=11)
    selected=train_forecast(protocol,settings,stage1_manifest=stage1,output=tmp_path/'gain',device=torch.device('cpu'),epochs=11,forecast_loss='mse_plus_0p2_mae',selection_metric='mae',repair_retention='forecast_gain_v1')
    record=json.loads(selected.read_text());parent=json.loads(stage1.read_text())
    assert record['repair_retention']=='forecast_gain_v1'
    assert record['repair_retention_active_batches']==1
    assert record['repair_retention_weighted_windows']>0
    assert record['repair_retention_teacher_seconds']>0
    assert record['memory_sha256']==parent['memory_sha256'] and record['key_sha256']==parent['key_sha256']
    load_stage2_selection(selected,protocol,settings,device=torch.device('cpu'))


def test_predictor_initialization_hash_is_same_across_retention_conditions(monkeypatch,tmp_path):
    import json
    from tests.integration.test_e23_stage2 import _selected_stage1
    from integrations.e23.train import train_forecast
    protocol,settings,stage1=_selected_stage1(monkeypatch,tmp_path)
    hashes=[]
    for mode in ('none','uniform_v1','forecast_gain_v1'):
        selection=train_forecast(protocol,settings,stage1_manifest=stage1,output=tmp_path/mode,device=torch.device('cpu'),epochs=1,forecast_loss='mse_plus_0p2_mae',selection_metric='mae',repair_retention=mode)
        hashes.append(json.loads(selection.read_text())['predictor_initial_sha256'])
    assert len(set(hashes))==1


def test_region_training_diagnostics_keep_signed_gains_and_errors():
    from integrations.forecast_compare.protocol import P4Batch
    x=torch.zeros(1,96,7);target=torch.ones_like(x);hidden=torch.ones_like(x,dtype=torch.bool)
    batch=P4Batch(x,target,~hidden,torch.zeros(1,7),torch.ones(1,7),torch.zeros(1,7,96))
    diagnostics=api().RegionTrainingDiagnostics()
    gains=torch.arange(42,dtype=torch.float32)[None]-20
    diagnostics.update(gains,x,batch)
    result=diagnostics.result()
    assert result['mean_signed_gain'][0]==-20
    assert result['mean_signed_gain'][-1]==21
    assert result['region_repair_mse']==[1.]*42
    assert result['scored_windows']==1 and result['scored_batches']==1
