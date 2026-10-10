import json
from types import SimpleNamespace

import pytest
import torch

from integrations.forecast_compare.resources import selected_training_resources
from malr.evaluation.report import file_sha256
from scripts import run_author_aligned_p4 as author
from tests.integration.test_e23_stage1 import _batch


def test_author_dual_selection_keeps_two_epochs_with_two_updates(monkeypatch, tmp_path):
    model = torch.nn.Linear(1, 1, bias=False)
    calls = []
    monkeypatch.setattr(author, 'iter_p4_batches', lambda *a, **kw: iter((_batch('train'),)))
    def loss(*args):
        calls.append(1)
        return model.weight.square().mean()
    monkeypatch.setattr(author, '_loss', loss)
    rows = iter([dict(mse=.1, mae=.4), dict(mse=.2, mae=.3)])
    monkeypatch.setattr(author, 'score', lambda *a, **kw: next(rows))
    training = dict(epochs=2, patience=None, precision='float32', learning_rate=.01,
                    weight_decay=0, batch_size=128, selection_metric='mae', lradj='constant')
    path = author._train_stage('T1', model, SimpleNamespace(training_seed=502), None,
                               dict(version='test'), training, 'forecast', tmp_path,
                               torch.device('cpu'), {}, dual_selection=True)
    mae = json.loads(path.read_text())
    mse = json.loads((tmp_path/'forecast-mse-selection.json').read_text())
    assert len(calls) == 2
    assert (mae['best_epoch'], mse['best_epoch']) == (2, 1)
    assert mse['training_resource_parent_sha256'] == file_sha256(path)
    assert mse['training']['selection_metric'] == 'mse'
    a = torch.load(mae['checkpoint'], weights_only=False)['model_state']['weight']
    b = torch.load(mse['checkpoint'], weights_only=False)['model_state']['weight']
    assert not torch.equal(a, b)


def test_cached_stage1_and_paired_selector_count_training_once(tmp_path):
    def put(name, data):
        p = tmp_path/name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data))
        return p
    base = dict(status='completed', mode='training', device={'name':'GPU'}, batch_size=128,
                full_workflow_seconds=10., training_stages_seconds=9.,
                peak_allocated_bytes=1024**3, peak_reserved_bytes=2*1024**3)
    original = put('old/resource.json', base)
    first = put('old/repair.json', dict(training_seconds=7.))
    cached = put('new/cached.json', {**base, 'selection_manifest_sha256':file_sha256(first),
        'source_resource_record':str(original), 'source_resource_record_sha256':file_sha256(original),
        'legacy_training_seconds':7.})
    identity = dict(seed=502, epochs_completed=100, predictor_initial_sha256='same',
        repair_selection_manifest=str(first), repair_selection_sha256=file_sha256(first),
        cached_stage1_resource_record=str(cached), cached_stage1_resource_sha256=file_sha256(cached),
        training_seconds=17.)
    main = put('new/mae.json', identity)
    put('new/resources/measured.json', {**base, 'full_workflow_seconds':20.,
        'training_stages_seconds':18., 'selection_manifest_sha256':file_sha256(main)})
    alternate = put('new/mse.json', {**identity,
        'training_resource_parent_manifest':str(main), 'training_resource_parent_sha256':file_sha256(main)})
    a, b = selected_training_resources([main]), selected_training_resources([alternate])
    assert a['full_workflow_seconds'] == b['full_workflow_seconds'] == 30.
    assert a['resource_record_paths'] == b['resource_record_paths']
    assert b['legacy_training_loop_seconds'] == 24.
    assert b['stage1_legacy_loop_seconds'] == 7.
    assert b['stage2_legacy_loop_seconds'] == 17.
    alternate.write_text(json.dumps({**json.loads(alternate.read_text()), 'seed':102}))
    with pytest.raises(ValueError, match='training identity differs'):
        selected_training_resources([alternate])
