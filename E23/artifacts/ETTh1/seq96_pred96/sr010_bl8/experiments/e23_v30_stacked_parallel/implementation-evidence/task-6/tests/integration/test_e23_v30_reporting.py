import copy
import json
from pathlib import Path

import pytest
import torch


def record(layers=1,groups=8,mse=.70,mae=.57,parameters=100):
    sha='a'*64
    identity=dict(recovery_schedule='stacked_parallel_tv_h_v1',recovery_rounds=1,recovery_layers=layers,
        expert_groups=groups,group_width=64,group_fusion='gated_concat_v1',activation_checkpointing=False)
    replay=dict(split='validation',windows=2688,coordinates=1806336,mse=mse,mae=mae,
        stage2_checkpoint_sha256=sha,**identity)
    for kind in ('key','bank'): replay[f'h_{kind}_sha256_before']=replay[f'h_{kind}_sha256_after']=sha
    return dict(model=f'E23_v30_l{layers}_g{groups}',stage='forecast',epochs_completed=100,
        stage1_epochs_completed=100,selection_metric='mae',loss='forecast_mse_plus_0p2_mae',
        forecast_input='explicit_v1',forecast_mixer='none',history_adapter='none',stage1_objective='global_mse_v1',
        stage1_selection_metric_space='train_standardized',checkpoint_sha256=sha,e23_config_sha256=sha,
        stage1_checkpoint_sha256=sha,best_val_mse=mse,best_val_mae=mae,repair_parameters=parameters,
        validation_replay=replay,**identity)


def test_v30_selection_uses_only_strict_validation_mse_and_mae_boundary():
    from scripts.report_e23_v30_trial import choose_v30_candidate
    assert choose_v30_candidate([record(mse=.7499365438950035,mae=.5840456761470457)]) is None
    assert choose_v30_candidate([record(mse=.7499,mae=.5840456761470457)]) is not None
    assert choose_v30_candidate([record(mse=.70,mae=.584046)]) is None
    a=record(); b=record(4,1,mse=.69,mae=.58)
    a['test_mse']=999; b['test_mse']=0
    assert choose_v30_candidate([a,b]) is a


def test_v30_ties_use_mae_mse_parameter_count_then_fixed_order():
    from scripts.report_e23_v30_trial import choose_v30_candidate
    a=record(1,8); b=record(4,1); c=record(4,8)
    assert choose_v30_candidate([c,b,a]) is a
    b['repair_parameters']=99
    assert choose_v30_candidate([a,b,c]) is b


@pytest.mark.parametrize('field,value',[('best_val_mse',float('nan')),('epochs_completed',99),
    ('stage1_epochs_completed',99),('checkpoint_sha256','bad'),('expert_groups',4),('recovery_layers',True)])
def test_v30_rejects_invalid_record(field,value):
    from scripts.report_e23_v30_trial import choose_v30_candidate
    bad=record();bad[field]=value
    with pytest.raises(ValueError): choose_v30_candidate([bad])


def diagnostic(r):
    result=dict(split='original_fixed_validation',metric_space='train_standardized',diagnostic_only=True,
        model=r['model'],e23_config_sha256=r['e23_config_sha256'],recovery_layers=r['recovery_layers'],expert_groups=r['expert_groups'])
    for stage,key in [('stage1','stage1_checkpoint_sha256'),('stage2','checkpoint_sha256')]:
        result[stage]=dict(checkpoint_sha256=r[key],windows=2688,layers=[dict(layer_index=i+1,
            global_mse=.3,global_mae=.4,mean_group_weights=[1/(r['expert_groups']+1)]*(r['expert_groups']+1))
            for i in range(r['recovery_layers'])])
        for kind in ('key','bank'): result[stage][f'h_{kind}_sha256_before']=result[stage][f'h_{kind}_sha256_after']='a'*64
    return result


@pytest.mark.parametrize('mutation',['checkpoint','config','layers','groups','history'])
def test_diagnostics_reject_stale_or_incompatible_evidence(mutation):
    from scripts.report_e23_v30_trial import validate_v30_diagnostics
    r=record();d=diagnostic(r);validate_v30_diagnostics(r,d)
    if mutation=='checkpoint': d['stage2']['checkpoint_sha256']='b'*64
    elif mutation=='config': d['e23_config_sha256']='b'*64
    elif mutation=='layers': d['stage2']['layers']=[]
    elif mutation=='groups': d['expert_groups']=1
    else: d['stage1']['h_bank_sha256_after']='b'*64
    with pytest.raises(ValueError):validate_v30_diagnostics(r,d)


def test_runner_matrix_preserves_order_budget_protocol_and_stops_before_test():
    from scripts.run_e23_v30_trial import build_training_commands
    root=Path('/tmp/e23_trial');commands=build_training_commands(root,'cuda:1')
    assert [row['cell'] for row in commands]==['l1_g8','l4_g1','l4_g8']
    for row in commands:
        assert row['stage1_epochs']==row['stage2_epochs']==100
        assert 'train-repair' in row['stage1'] and 'train-forecast' in row['stage2']
        assert 'evaluate' not in row['stage1']+row['stage2']
        assert row['stage2'][row['stage2'].index('--selection-metric')+1]=='mae'
        assert row['stage2'][row['stage2'].index('--forecast-loss')+1]=='mse_plus_0p2_mae'
        assert any('E22/artifacts' in arg for arg in row['stage1'])


def test_evaluate_requires_all_locked_records_and_refuses_overwrite(tmp_path):
    from scripts.run_e23_v30_trial import evaluation_preconditions
    with pytest.raises(FileNotFoundError):evaluation_preconditions(tmp_path)
    (tmp_path/'validation_selection.json').write_text('{}')
    (tmp_path/'l1_g8').mkdir();(tmp_path/'l1_g8'/'three-test-draws.json').write_text('{}')
    with pytest.raises(FileExistsError):evaluation_preconditions(tmp_path)


def test_final_layer_decode_and_forecast_match_normal_forward():
    from scripts.diagnose_e23_v30_stack import decode_layers
    from tests.integration.test_e23_stacked_model import model,arguments
    from malr.models.e23.model import E23ForecastModel
    from tests.integration.test_e23_model import _model
    repair=model(); forecast=E23ForecastModel(repair,_model().predictor).eval()
    args,kwargs=arguments(); prediction,output=forecast(*args,**kwargs)
    decoded=decode_layers(repair,output,*args[:4],valid_mask=kwargs['valid_mask'])
    torch.testing.assert_close(decoded[-1].normalized_sequence,output.explicit.normalized_sequence,rtol=0,atol=0)
    last=decoded[-1]
    expected=forecast.predictor(last.repaired_patches,last.observation_patches,args[2],args[3],patch_validity=last.validity_patches)
    torch.testing.assert_close(prediction,expected,rtol=0,atol=0)


def test_test_summary_rejects_resources_from_another_model():
    from scripts.report_e23_v30_trial import validate_test_result
    r=record();r['training_resources']={'resource_record_paths':['own-stage1','own-stage2']}
    result=dict(checkpoint_sha256=r['checkpoint_sha256'],e23_config_sha256=r['e23_config_sha256'],h_state_unchanged=True,
        test_groups=[dict(seed=seed,mse=.4,mae=.42,windows=2688,target_elements=1806336) for seed in (2021,2026100201,2026100202)],
        test_mean=dict(mse=.4,mae=.42),test_std=dict(mse=0,mae=0),training_resources=r['training_resources'])
    validate_test_result(r,result)
    result['training_resources']={'resource_record_paths':['other']}
    with pytest.raises(ValueError,match='resource'):validate_test_result(r,result)


def test_reporting_script_clis_are_runnable():
    import subprocess,sys
    root=Path(__file__).resolve().parents[2]
    for script in ('run_e23_v30_trial.py','report_e23_v30_trial.py','diagnose_e23_v30_stack.py'):
        result=subprocess.run([sys.executable,str(root/'scripts'/script),'--help'],capture_output=True,text=True)
        assert result.returncode==0,result.stderr
