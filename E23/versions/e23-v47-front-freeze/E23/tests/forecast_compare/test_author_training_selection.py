from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from integrations.forecast_compare.author_profiles import load_author_profiles, build_author_model
from integrations.forecast_compare.protocol import load_p4_protocol
from integrations.forecast_compare.coifnet_p4 import load_timestamp_features
from scripts.run_author_aligned_p4 import _train_stage, score

ROOT = Path(__file__).resolve().parents[2]
MASKS = ROOT.parent / 'E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context'


def test_real_author_training_saves_and_replays_one_validation_selection(tmp_path):
    torch.set_num_threads(4)
    base = load_p4_protocol(ROOT / 'configs/stage1_seq96_dynamic_temporal_repair.yaml', MASKS)
    refs = dict(base.refs); refs['train'] = refs['train'][:512]; refs['validation'] = refs['validation'][:128]
    protocol = replace(base, refs=refs)
    cfg = load_author_profiles()['models']['CoIFNet']
    training = {**cfg['training'], 'epochs': 2, 'patience': None}
    model = build_author_model('CoIFNet', cfg, ROOT / cfg['root'], torch.device('cpu'))
    stamps = load_timestamp_features(Path(protocol.config.data.datasets['ETTh1'].path))
    selection = _train_stage('CoIFNet', model, protocol, stamps, cfg, training,
                             'forecast', tmp_path, torch.device('cpu'), {})
    record = json.loads(selection.read_text())
    assert record['epochs_completed'] == 2 and record['best_epoch'] in (1, 2)
    assert record['status'] == 'validation_finalized' and not record['test_metrics_seen']
    replay = score('CoIFNet', model, protocol, 'validation', torch.device('cpu'), stamps, training)
    assert replay == record['best_validation']
    with pytest.raises(FileExistsError):
        _train_stage('CoIFNet', model, protocol, stamps, cfg, training,
                     'forecast', tmp_path, torch.device('cpu'), {})
