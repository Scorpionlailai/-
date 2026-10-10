import importlib
from pathlib import Path

import pytest


def api(name):
    try: return importlib.import_module('scripts.'+name)
    except ModuleNotFoundError: pytest.fail('v32 trial implementation missing')


def test_v32_matrix_uses_independent_stage1_and_pure_mse_stage2():
    rows=api('run_e23_v32_trial').build_training_commands(Path('/tmp/v32'), 'cuda:1')
    assert [r['condition'] for r in rows] == ['A','B']
    for row,mode in zip(rows,('none','clean_forecast_v1')):
        first,second=row['stage1'],row['stage2']
        assert first[first.index('--stage1-preservation')+1] == mode
        assert second[second.index('--forecast-loss')+1] == 'mse'
        assert second[second.index('--selection-metric')+1] == 'mae'
        assert second[second.index('--stage1-manifest')+1] == f"/tmp/v32/{row['condition']}/stage1/runs/e23-stage1.json"


def test_validation_gate_rejects_tradeoff_tie_and_nonfinite_values():
    passes=api('report_e23_v32_trial').improves
    reference={'best_val_mse':.75,'best_val_mae':.58}
    assert passes({'best_val_mse':.74,'best_val_mae':.58},reference)
    assert not passes({'best_val_mse':.75,'best_val_mae':.57},reference)
    assert not passes({'best_val_mse':.74,'best_val_mae':.59},reference)
    with pytest.raises(ValueError):passes({'best_val_mse':float('nan'),'best_val_mae':.57},reference)


def test_v32_rejects_incorrect_preservation_and_mixed_stage2_loss():
    check=api('report_e23_v32_trial').check_trial_identity
    base=dict(loss='forecast_mse_only',selection_metric='mae',forecast_input='explicit_v1',
              forecast_mixer='none',epochs_completed=100,optimizer_state_restored=False,rng_reset_seed=2021)
    check('A',base)
    with pytest.raises(ValueError):check('A',{**base,'loss':'forecast_mse_plus_0p2_mae'})
    with pytest.raises(ValueError):check('B',base)
