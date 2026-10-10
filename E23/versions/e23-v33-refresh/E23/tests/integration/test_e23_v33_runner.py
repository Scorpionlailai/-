from pathlib import Path
from scripts.run_e23_v33_trial import build_commands


def test_matrix_reuses_only_saved_control_stage1_and_keeps_stage2_protocol(tmp_path):
    commands = build_commands(tmp_path,'cuda:1')
    assert [r['condition'] for r in commands] == ['C0','A','B']
    assert commands[0]['stage1'] is None
    assert 'e23_v29_iterative_tv_h_r3' in commands[0]['stage1_manifest']
    for row in commands:
        command = row['stage2']
        assert command[command.index('--forecast-loss')+1] == 'mse'
        assert command[command.index('--selection-metric')+1] == 'mae'
        assert command[command.index('--forecast-input')+1] == 'explicit_v1'
        assert command[command.index('--forecast-mixer')+1] == 'none'
        assert '--repair-retention' not in command
        assert '--stage1-preservation' not in command
