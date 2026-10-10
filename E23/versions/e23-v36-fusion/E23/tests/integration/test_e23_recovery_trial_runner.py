from pathlib import Path
from scripts.run_e23_recovery_trial import build_commands
from scripts.report_e23_recovery_trial import improves


def test_new_recovery_module_uses_original_stage2_protocol():
    trial=dict(condition='T',config='/tmp/locked-t.yaml')
    commands=build_commands(Path('/tmp/single-recovery-test'),trial,'cuda:1')
    cmd=commands['stage2']
    assert cmd[cmd.index('--forecast-loss')+1]=='mse'
    assert cmd[cmd.index('--forecast-input')+1]=='explicit_v1'
    assert cmd[cmd.index('--selection-metric')+1]=='mae'
    assert '--repair-retention' not in cmd
    assert 'train-repair' in commands['stage1']


def test_only_fixed_validation_metrics_determine_recovery_promotion():
    reference=dict(validation_mse=.745,validation_mae=.581)
    candidate=dict(best_val_mse=.744,best_val_mae=.581,test_mean=dict(mse=100))
    assert improves(candidate,reference)
    candidate['best_val_mae']=.582
    assert not improves(candidate,reference)


def test_point_temporal_config_accepts_original_global_repair_objective():
    from scripts.diagnose_e23_v30_stack import load_protocol
    from integrations.e23.config import load_e23_settings
    root=Path(__file__).resolve().parents[2]
    settings=load_e23_settings(root/'configs/e23_multi_expert_p4_point_temporal_v35.yaml',
                              protocol=load_protocol())
    assert settings.temporal_selection=='point_convolution_v1'
    assert settings.stage1_objective=='global_mse_v1'
    assert settings.historical_query_mode=='window_patch_v1'
