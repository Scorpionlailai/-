import json
from dataclasses import replace
from pathlib import Path
import pytest
import torch
from malr.data.patching import patchify_preserving_values
from integrations.e23 import train as training
from integrations.e23.forecast_sensitive import parameter_hash
from tests.integration.test_e23_stage1 import _batch,_protocol_settings
from tests.integration.test_e23_iterative_training import _settings


def test_common_padding_matches_t1_patchifier_without_changing_repair_or_head(tmp_path):
    protocol,_=_protocol_settings();settings=_settings(tmp_path,protocol,rounds=1)
    torch.manual_seed(502);model=training.build_forecast_model(protocol,settings,device=torch.device('cpu')).eval()
    b=_batch('train');starts=training._starts(b,stride=8,patches=12,device=torch.device('cpu'))
    head_before=parameter_hash(model.predictor);repair_before=parameter_hash(model.repair)
    captured=[]
    hook=model.predictor.register_forward_pre_hook(lambda _m,args: captured.append(args[0].detach().clone()))
    with torch.no_grad():
        _,original=model(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts)
        model.forecast_padding='replicate_last_value_v1'
        _,common=model(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts)
    hook.remove()
    patches=patchify_preserving_values(common.explicit.normalized_sequence,b.observed_mask,16,8)
    torch.testing.assert_close(captured[-1],patches.values,rtol=0,atol=0)
    assert torch.equal(original.explicit.normalized_sequence,common.explicit.normalized_sequence)
    assert torch.equal(captured[0][patches.validity],captured[1][patches.validity])
    assert head_before==parameter_hash(model.predictor) and repair_before==parameter_hash(model.repair)


def test_common_stage2_saves_pure_mse_protocol_and_rejects_relabelling(monkeypatch,tmp_path):
    protocol,_=_protocol_settings();settings=_settings(tmp_path,protocol,rounds=1)
    monkeypatch.setattr(training,'iter_p4_batches',lambda _p,split,**kw:iter((_batch(split),)))
    first=training.train_repair(protocol,settings,output=tmp_path/'stage1',device=torch.device('cpu'),epochs=1)
    selected=training.train_forecast(protocol,settings,stage1_manifest=first,output=tmp_path/'stage2',
        device=torch.device('cpu'),epochs=2,common_stage2=True,selection_metric='mae')
    record=json.loads(selected.read_text())
    payload=training.load_stage2_selection(selected,protocol,settings,device=torch.device('cpu'))
    assert record['loss']=='forecast_mse_only'
    assert record['downstream_protocol']==payload['downstream_protocol']=='t1_mse_v1'
    assert record.get('repair_retention','none')=='none'
    alternate=selected.parent/'e23-stage2-mse.json'
    mse_record=json.loads(alternate.read_text())
    assert mse_record['selection_metric']=='mse'
    assert training.load_stage2_selection(alternate,protocol,settings,device=torch.device('cpu'))['selection_metric']=='mse'
    wrong=tmp_path/'wrong.json';wrong.write_text(json.dumps({k:v for k,v in record.items() if k!='downstream_protocol'}))
    with pytest.raises(ValueError,match='common Stage2 checkpoint identity mismatch'):
        training.load_stage2_selection(wrong,protocol,settings,device=torch.device('cpu'))
    with pytest.raises(ValueError,match='common Stage2 requires'):
        training.train_forecast(protocol,settings,stage1_manifest=first,output=tmp_path/'bad',
            device=torch.device('cpu'),epochs=1,common_stage2=True,selection_metric='mae',forecast_loss='mse_plus_0p2_mae')


def test_two_checkpoint_rules_select_different_epochs_from_one_trajectory(monkeypatch,tmp_path):
    protocol,_=_protocol_settings();settings=_settings(tmp_path,protocol,rounds=1)
    monkeypatch.setattr(training,'iter_p4_batches',lambda _p,split,**kw:iter((_batch(split),)))
    first=training.train_repair(protocol,settings,output=tmp_path/'stage1',device=torch.device('cpu'),epochs=1)
    rows=iter([dict(mse=.1,mae=.4),dict(mse=.2,mae=.3)])
    monkeypatch.setattr(training,'_validation_forecast_metrics',lambda *a,**kw:next(rows))
    selected=training.train_forecast(protocol,settings,stage1_manifest=first,output=tmp_path/'stage2',
        device=torch.device('cpu'),epochs=2,common_stage2=True,selection_metric='mae')
    assert json.loads(selected.read_text())['best_epoch']==2
    assert json.loads((selected.parent/'e23-stage2-mse.json').read_text())['best_epoch']==1


def test_increased_stage1_budget_settings_preserve_head_geometry(tmp_path):
    protocol,_=_protocol_settings()
    settings=_settings(tmp_path,protocol,rounds=1,changes={'training':{'stage1_epochs':200}})
    assert settings.stage1_epochs==200 and settings.stage2_epochs==100 and settings.batch_size==128
    model=training.build_forecast_model(protocol,settings,device=torch.device('cpu'))
    assert model.predictor.core.d_model==16 and model.repair.d_model==64
