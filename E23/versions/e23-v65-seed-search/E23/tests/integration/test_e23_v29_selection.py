"""The locked validation gate must prevent test-driven trial selection."""
from copy import deepcopy
import pytest


def _record(rounds, mse=0.73, mae=0.58):
    return {
        'model': f'E23_v29_r{rounds}', 'stage': 'forecast',
        'recovery_schedule': 'iterative_tv_h_v1', 'recovery_rounds': rounds,
        'epochs_completed': 100, 'stage1_epochs_completed': 100,
        'checkpoint_sha256': 'a' * 64, 'e23_config_sha256': 'b' * 64,
        'stage1_checkpoint_sha256': 'c' * 64,
        'best_val_mse': mse, 'best_val_mae': mae,
        'selection_metric': 'mae', 'loss': 'forecast_mse_plus_0p2_mae',
        'forecast_input': 'explicit_v1', 'stage1_objective': 'global_mse_v1',
        'validation_replay': {
            'split': 'validation', 'mse': mse, 'mae': mae,
            'windows': 2688, 'coordinates': 1806336,
            'stage2_checkpoint_sha256': 'a' * 64,
            'recovery_schedule': 'iterative_tv_h_v1', 'recovery_rounds': rounds,
            'h_key_sha256_before': 'd' * 64, 'h_key_sha256_after': 'd' * 64,
            'h_bank_sha256_before': 'e' * 64, 'h_bank_sha256_after': 'e' * 64,
        },
    }


def test_v29_selection_uses_only_validation_and_both_gates():
    from scripts.report_e23_v29_trial import choose_v29_candidate
    bad_mse = _record(1, mse=0.75, mae=0.55)
    bad_mae = _record(3, mse=0.70, mae=0.59)
    one = _record(1, mae=0.581)
    three = _record(3, mae=0.579)
    one['test_mse'] = 0.001
    three['test_mse'] = 100.0
    assert choose_v29_candidate([bad_mse, bad_mae]) is None
    assert choose_v29_candidate([bad_mse, three, one, bad_mae]) is three
    equality = _record(1, mse=0.7402293330687785, mae=0.5855096656989619)
    assert choose_v29_candidate([equality]) is None
    equality['best_val_mae'] -= 1e-8
    equality['validation_replay']['mae'] = equality['best_val_mae']
    assert choose_v29_candidate([equality]) is equality


def test_v29_selection_ties_prefer_fewer_rounds():
    from scripts.report_e23_v29_trial import choose_v29_candidate
    one, three = _record(1), _record(3)
    assert choose_v29_candidate([three, one]) is one
    three['best_val_mse'] = three['validation_replay']['mse'] = 0.72
    assert choose_v29_candidate([one, three]) is three


@pytest.mark.parametrize('change', [
    {'epochs_completed': 99}, {'stage1_epochs_completed': 99},
    {'best_val_mse': float('nan')}, {'best_val_mae': float('inf')},
    {'recovery_rounds': True}, {'recovery_schedule': 'parallel_v1'},
    {'checkpoint_sha256': None}, {'e23_config_sha256': ''},
    {'validation_replay': None},
])
def test_v29_selection_rejects_incomplete_or_nonfinite_records(change):
    from scripts.report_e23_v29_trial import choose_v29_candidate
    record = _record(1)
    record.update(change)
    with pytest.raises(ValueError):
        choose_v29_candidate([record])


@pytest.mark.parametrize('field,value', [
    ('mae', 0.1), ('windows', 1), ('coordinates', 100),
    ('stage2_checkpoint_sha256', 'f' * 64), ('recovery_rounds', 3),
    ('h_key_sha256_after', 'f' * 64), ('h_bank_sha256_after', 'f' * 64),
])
def test_v29_selection_rejects_unverified_replays(field, value):
    from scripts.report_e23_v29_trial import choose_v29_candidate
    record = deepcopy(_record(1))
    record['validation_replay'][field] = value
    with pytest.raises(ValueError):
        choose_v29_candidate([record])


def _diagnostic():
    return {
        'split': 'original_fixed_validation', 'metric_space': 'train_standardized',
        'recovery_rounds': 1, 'diagnostic_only': True,
        **{stage: {'checkpoint_sha256': sha * 64, 'windows': 2688,
                   'h_key_sha256_before': 'd'*64, 'h_key_sha256_after': 'd'*64,
                   'h_bank_sha256_before': 'e'*64, 'h_bank_sha256_after': 'e'*64,
                   'rounds': [{'round_index': 0, 'global_mse': .4, 'global_mae': .3}]}
           for stage, sha in [('stage1', 'c'), ('stage2', 'a')]},
    }


@pytest.mark.parametrize('change', [
    'stage1_sha', 'stage2_sha', 'declared_rounds', 'actual_rounds',
    'split', 'metric_space', 'windows', 'h_hash', 'missing_identity',
])
def test_report_rejects_stale_or_misidentified_round_diagnostics(change):
    from scripts.report_e23_v29_trial import validate_round_diagnostics
    record, diagnostic = _record(1), _diagnostic()
    validate_round_diagnostics(record, diagnostic)
    if change == 'stage1_sha': diagnostic['stage1']['checkpoint_sha256'] = 'f'*64
    elif change == 'stage2_sha': diagnostic['stage2']['checkpoint_sha256'] = 'f'*64
    elif change == 'declared_rounds': diagnostic['recovery_rounds'] = 3
    elif change == 'actual_rounds': diagnostic['stage2']['rounds'] *= 3
    elif change == 'split': diagnostic['split'] = 'test'
    elif change == 'metric_space': diagnostic['metric_space'] = 'window_normalized'
    elif change == 'windows': diagnostic['stage1']['windows'] = 1
    elif change == 'h_hash': diagnostic['stage2']['h_bank_sha256_after'] = 'f'*64
    elif change == 'missing_identity': del diagnostic['stage1']['checkpoint_sha256']
    with pytest.raises(ValueError):
        validate_round_diagnostics(record, diagnostic)
