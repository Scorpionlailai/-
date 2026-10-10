import importlib
import pytest
from pathlib import Path


def module(name):
    try:return importlib.import_module('scripts.'+name)
    except ModuleNotFoundError:pytest.fail(name+' not implemented')


def test_matrix_uses_shared_stage1_and_fixed_modes():
    runner=module('run_e23_v31_trial')
    common,rows=runner.build_training_commands(Path('/tmp/v31'),'cuda:1')
    assert 'train-repair' in common
    assert [row['condition'] for row in rows]==['A','B','C']
    for row,mode in zip(rows,('none','uniform_v1','forecast_gain_v1')):
        cmd=row['stage2'];assert cmd[cmd.index('--repair-retention')+1]==mode
        assert cmd[cmd.index('--stage1-manifest')+1]=='/tmp/v31/common_stage1/runs/e23-stage1.json'
        assert cmd[cmd.index('--selection-metric')+1]=='mae'
        assert cmd[cmd.index('--forecast-loss')+1]=='mse_plus_0p2_mae'


def test_gate_uses_validation_only_and_strict_mse_boundary():
    report=module('report_e23_v31_trial')
    row=dict(condition='B',best_val_mse=.7499365438950035,best_val_mae=.5840456761470457,extra_training_seconds=0.)
    assert report.choose_candidate([row]) is None
    row['best_val_mse']-=.001
    assert report.choose_candidate([row])['condition']=='B'
    other={**row,'condition':'C','test_mse':0.,'extra_training_seconds':1.}
    assert report.choose_candidate([row,other])['condition']=='B'
    with pytest.raises(ValueError):report.choose_candidate([{**row,'best_val_mse':float('nan')}])


def test_test_summary_rejects_nan():
    import json,copy
    from scripts.report_e23_v30_trial import MASK_BANK,MASK_BANK_SHA
    report=module('report_e23_v31_trial')
    bank=json.loads(MASK_BANK.read_text())
    record=dict(checkpoint_sha256='a',selection_sha256='b',e23_config_sha256='c',training_resources={})
    groups=[dict(g,windows=2688,target_elements=1806336,mse=1.,mae=1.) for g in bank['groups']]
    result=dict(record,test_mask_bank_sha256=MASK_BANK_SHA,h_state_unchanged=True,test_groups=groups,test_mean={'mse':float('nan'),'mae':1.},test_std={'mse':0.,'mae':0.})
    with pytest.raises(ValueError,match='summary'):report.validate_tests(record,result)


def test_repair_diagnostic_rejects_unbound_or_nonfinite_values():
    report=module('report_e23_v31_trial')
    record=dict(checkpoint_sha256='a',selection_sha256='b',e23_config_sha256='c')
    valid=dict(record,split='original_fixed_validation',metric_space='train_standardized',windows=2688,h_state_unchanged=True,global_mse=.2,global_mae=.3,hidden_coordinates=100,repair_parameters=135321,predictor_parameters=35427,diagnostic_seconds=1.)
    report.validate_repair_diagnostic(record,valid)
    for change in ({'checkpoint_sha256':'wrong'},{'global_mse':float('nan')},{'windows':1}):
        with pytest.raises(ValueError):report.validate_repair_diagnostic(record,{**valid,**change})
