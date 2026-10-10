import copy
import importlib.util
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent/'e23_v69_signature_position_sr015_bl8_20261005'

def controller():
    path = ROOT/'execute.py'
    if not path.exists():
        pytest.fail('scenario execution/identity checks have not been implemented')
    spec = importlib.util.spec_from_file_location('scenario_execution', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

def record():
    value = json.loads((BASE/'seed_2023_frozen_mae/stage2/runs/e23-stage2.json').read_text())
    plan = json.loads((ROOT/'plan.json').read_text())
    value.update(historical_tv_query='fully_hidden_v1',tv_query_alignment_weight=.1,
                 tv_query_alignment_target='frozen_global_history_key_v1',
                 e23_config_sha256=controller().sha(plan['config']))
    return value

def setup_lock():
    plan = json.loads((ROOT/'plan.json').read_text())
    scene = {**plan['scenarios'][0], 'validation_mask_sha256': record()['validation_mask_sha256']}
    return plan, scene

def test_stage2_lock_refuses_seed_or_partial_training_or_repair_updates():
    m = controller(); plan, scene = setup_lock()
    for key, bad in [('seed',502), ('epochs_completed',99), ('repair_trainable_parameters',1),
                     ('repair_update_policy','freeze_front_v1'), ('selection_metric','mse'),
                     ('historical_tv_query','none'), ('tv_query_alignment_weight',.2),
                     ('e23_config_sha256','wrong')]:
        r = record(); r[key] = bad
        with pytest.raises(ValueError):
            m.stage2_lock(r, plan, scene, 'mae')
    assert m.stage2_lock(record(), plan, scene, 'mae')['test_metrics_seen'] is False

def test_pairing_refuses_different_or_reordered_mask_positions():
    m = controller()
    p = BASE/'seed_2023_frozen_mae/evaluation/twenty-test-draws.json'
    a = json.loads(p.read_text()); b = copy.deepcopy(a)
    assert len(m.paired_draws(a,b)) == 20
    b['test_draws'][0],b['test_draws'][1] = b['test_draws'][1],b['test_draws'][0]
    with pytest.raises(ValueError):
        m.paired_draws(a,b)
    b = copy.deepcopy(a); b['test_mask_bank_sha256']='changed'
    with pytest.raises(ValueError):
        m.paired_draws(a,b)

def test_pairing_refuses_nonfinite_scores():
    m = controller()
    a = json.loads((BASE/'seed_2023_frozen_mae/evaluation/twenty-test-draws.json').read_text())
    b = copy.deepcopy(a); b['test_draws'][0]['mse'] = float('nan')
    with pytest.raises(ValueError):
        m.paired_draws(a,b)

def test_commands_keep_seed_frozen_repair_and_original_recipe_in_each_scene():
    m = controller(); plan = json.loads((ROOT/'plan.json').read_text())
    for scene in plan['scenarios']:
        command = m.training_command(plan, scene, 'forecast')
        for flag, value in [('--training-seed','2023'), ('--repair-update-policy','frozen_v1'),
                            ('--forecast-loss','mse_plus_0p2_mae'), ('--selection-metric','mae'),
                            ('--forecast-padding','zero_invalid_v1')]:
            assert command[command.index(flag)+1] == value
        assert '--extra-mse-checkpoint' in command
        assert command[command.index('--e23-config')+1] == plan['config']
