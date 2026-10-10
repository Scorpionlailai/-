from pathlib import Path
import json

import numpy as np
import pytest

from integrations.forecast_compare.protocol import load_p4_protocol
from integrations.forecast_compare.test_mask_bank import create_test_mask_bank
from integrations.forecast_compare.three_draw_evaluation import evaluate_three_draws
from malr.evaluation.report import file_sha256

ROOT = Path(__file__).resolve().parents[2]
MASKS = ROOT.parent / 'E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context'


def test_one_locked_checkpoint_three_test_draws_and_population_sd(tmp_path):
    base = load_p4_protocol(ROOT / 'configs/stage1_seq96_dynamic_temporal_repair.yaml', MASKS)
    bank = create_test_mask_bank(base, tmp_path / 'masks')
    checkpoint = tmp_path / 'selected.pt'
    checkpoint.write_bytes(b'locked model')
    selection = tmp_path / 'selection.json'
    selection.write_text(json.dumps({'checkpoint': str(checkpoint),
                                    'checkpoint_sha256': file_sha256(checkpoint)}))
    seen = []

    def score(group):
        assert group.fixed_masks['validation'] is base.fixed_masks['validation']
        seen.append(group.mask_sha256['test'])
        return {'mse': float(len(seen)), 'mae': float(len(seen)) / 2,
                'windows': 2688, 'target_elements': 2688 * 96 * 7}

    result = evaluate_three_draws(base, bank, selection, score)
    assert len(set(seen)) == 3
    assert result['test_mean']['mse'] == 2.0
    assert result['test_std']['mse'] == pytest.approx(np.std([1., 2., 3.], ddof=0))
    assert result['checkpoint_sha256'] == file_sha256(checkpoint)
    assert result['training_runs'] == 1


def test_rejects_checkpoint_mutation_during_test(tmp_path):
    base = load_p4_protocol(ROOT / 'configs/stage1_seq96_dynamic_temporal_repair.yaml', MASKS)
    bank = create_test_mask_bank(base, tmp_path / 'masks')
    checkpoint = tmp_path / 'selected.pt'
    checkpoint.write_bytes(b'locked model')
    selection = tmp_path / 'selection.json'
    selection.write_text(json.dumps({'checkpoint': str(checkpoint),
                                    'checkpoint_sha256': file_sha256(checkpoint)}))

    def bad_score(group):
        checkpoint.write_bytes(b'changed model')
        return {'mse': 1., 'mae': 1., 'windows': 2688, 'target_elements': 1806336}

    with pytest.raises(ValueError, match='checkpoint'):
        evaluate_three_draws(base, bank, selection, bad_score)


def test_formal_evaluation_rejects_unannounced_test_seeds_before_scoring(tmp_path):
    base = load_p4_protocol(ROOT / 'configs/stage1_seq96_dynamic_temporal_repair.yaml', MASKS)
    bank = create_test_mask_bank(base, tmp_path / 'masks', seeds=(13, 14))
    checkpoint = tmp_path / 'selected.pt'
    checkpoint.write_bytes(b'locked model')
    selection = tmp_path / 'selection.json'
    selection.write_text(json.dumps({'checkpoint': str(checkpoint),
                                    'checkpoint_sha256': file_sha256(checkpoint)}))
    with pytest.raises(ValueError, match='seed'):
        evaluate_three_draws(base, bank, selection,
                            lambda group: pytest.fail('unannounced masks were scored'))
