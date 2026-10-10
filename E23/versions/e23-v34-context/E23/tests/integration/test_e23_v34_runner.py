from pathlib import Path
from scripts.run_e23_v34_trial import build_commands
from scripts.report_e23_v34_trial import improves


def test_both_context_trials_use_original_forecast_objective_and_fresh_stages():
    rows=build_commands(Path('/tmp/v34-unit-commands'),'cuda:1')
    assert [r['condition'] for r in rows]==['A','B']
    for row in rows:
        cmd=row['stage2']
        assert cmd[cmd.index('--forecast-loss')+1]=='mse'
        assert cmd[cmd.index('--selection-metric')+1]=='mae'
        assert cmd[cmd.index('--forecast-input')+1]=='explicit_v1'
        assert '--repair-retention' not in cmd
        assert 'train-repair' in row['stage1']


def test_promotion_requires_validation_mse_improvement_and_no_mae_regression():
    reference=dict(validation_mse=.75,validation_mae=.58)
    assert improves(dict(best_val_mse=.74,best_val_mae=.58),reference)
    assert not improves(dict(best_val_mse=.75,best_val_mae=.57),reference)
    assert not improves(dict(best_val_mse=.74,best_val_mae=.59),reference)
