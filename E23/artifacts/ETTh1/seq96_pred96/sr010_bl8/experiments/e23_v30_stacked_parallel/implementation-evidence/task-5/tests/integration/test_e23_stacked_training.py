from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch
import yaml

from integrations.e23.config import load_e23_settings
from integrations.e23 import train as training
from tests.integration.test_e23_stage1 import _protocol_settings,_batch

ROOT=Path(__file__).resolve().parents[2]


def settings(protocol,layers=1,groups=8):
    return load_e23_settings(ROOT/f'configs/e23_multi_expert_p4_width64_stack_l{layers}_g{groups}_v30.yaml',protocol=protocol)


@pytest.mark.parametrize('layers,groups',[(1,8),(4,1),(4,8)])
def test_stacked_configs_preserve_training_budget_and_original_predictor(layers,groups):
    protocol,_=_protocol_settings(); s=settings(protocol,layers,groups)
    assert s.stage1_epochs==s.stage2_epochs==100
    model=training.build_forecast_model(protocol,s,device=torch.device('cpu'))
    assert len(model.repair.stacked.layers)==layers
    assert len(model.repair.stacked.layers[0].groups)==groups
    assert model.predictor.core.d_model==16


@pytest.mark.parametrize('change',[{'recovery_layers':True},{'recovery_layers':2},{'expert_groups':4},
    {'recovery_rounds':3},{'activation_checkpointing':'false'},{'multihead_router':'eight_full_width_gated_v1'},
    {'post_fusion_refiner':'evidence_factorized_v1'},{'unknown':2}])
def test_stacked_settings_reject_wrong_combinations(tmp_path,change):
    protocol,_=_protocol_settings()
    document=yaml.safe_load((ROOT/'configs/e23_multi_expert_p4_width64_global_stage1_v28.yaml').read_text())
    document['model'].update(recovery_schedule='stacked_parallel_tv_h_v1',recovery_rounds=1,
        recovery_layers=4,expert_groups=8,activation_checkpointing=False)
    document['model'].update(change)
    path=tmp_path/'wrong.yaml'; path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError): load_e23_settings(path,protocol=protocol)


def test_stacked_two_stage_training_preserves_identity_global_scale_and_frozen_history(monkeypatch,tmp_path):
    protocol,_=_protocol_settings(); s=settings(protocol,4,1)
    def batches(_protocol,split,**kwargs):
        batch=_batch(split); return iter([replace(batch,query_std=torch.full_like(batch.query_std,.02))])
    monkeypatch.setattr(training,'iter_p4_batches',batches)
    calls=[]; publish=training._publish_epoch
    def counted(*args,**kwargs): calls.append(1); return publish(*args,**kwargs)
    monkeypatch.setattr(training,'_publish_epoch',counted)
    manifest=training.train_repair(protocol,s,output=tmp_path/'stage1',device=torch.device('cpu'),epochs=2)
    assert len(calls)==2
    record=json.loads(manifest.read_text())
    payload=training.load_stage1_selection(manifest,protocol,s,device=torch.device('cpu'))
    assert record['recovery_schedule']=='stacked_parallel_tv_h_v1'
    assert record['recovery_layers']==s.recovery_layers and record['expert_groups']==s.expert_groups
    assert record['stage1_selection_metric_space']=='train_standardized'
    assert payload['key_sha256']==record['key_sha256']
    for field,value in [('recovery_layers',1),('expert_groups',8),('activation_checkpointing',True)]:
        with pytest.raises(ValueError,match=field+'.*mismatch'):
            training.load_stage1_selection(manifest,protocol,replace(s,**{field:value}),device=torch.device('cpu'))
    forecast=training.train_forecast(protocol,s,stage1_manifest=manifest,output=tmp_path/'stage2',
        device=torch.device('cpu'),epochs=1,selection_metric='mae',forecast_loss='mse_plus_0p2_mae')
    selected=training.load_stage2_selection(forecast,protocol,s,device=torch.device('cpu'))
    assert selected['key_sha256']==payload['key_sha256']
    assert len(calls)==2
    wrong=dict(selected,group_fusion='wrong')
    with pytest.raises(ValueError,match='group_fusion.*mismatch'): training._verify_trial_payload(wrong,s)


@pytest.mark.parametrize('options',[{'forecast_input':'implicit_latent_v1'},{'forecast_mixer':'patch_attention_v1'},
    {'history_adapter':'aligned_v1'}])
def test_stacked_rejects_changed_predictor_interface(options):
    protocol,_=_protocol_settings(); s=settings(protocol)
    with pytest.raises(ValueError,match='stacked'):
        training.build_forecast_model(protocol,s,device=torch.device('cpu'),**options)


def test_stacked_rejects_different_stage2_objective_before_loading(tmp_path):
    protocol,_=_protocol_settings(); s=settings(protocol)
    with pytest.raises(ValueError,match='stacked'):
        training.train_forecast(protocol,s,stage1_manifest=tmp_path/'absent',output=tmp_path/'unused',
            device=torch.device('cpu'),epochs=1,selection_metric='mse',forecast_loss='mse')


def test_stacked_loader_rejects_wrong_objective_before_checkpoint_access(tmp_path):
    protocol,_=_protocol_settings(); s=settings(protocol)
    path=tmp_path/'wrong-selection.json'
    path.write_text(json.dumps({'stage':'forecast','best_epoch':1,'loss':'forecast_mse_only',
        'selection_metric':'mae','forecast_mixer':'none','forecast_input':'explicit_v1'}))
    with pytest.raises(ValueError,match='stacked'):
        training.load_stage2_selection(path,protocol,s,device=torch.device('cpu'))
