"""Resource reporting must preserve the ability to verify saved baseline runs."""

import importlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from malr.evaluation.report import file_sha256


# Source identities saved with the completed 2026-10-02 experiments.
@pytest.mark.parametrize('name,source_identity', [
    ('crib', '9b05859b8004ab2609f0042fe96a51f570bd0bcef1eafa787626baa60a076357'),
    ('coifnet', '95beb033f9a23ffd353071e7a1c390b37db83ace1090ba68209dd61eab64be9b'),
    ('bitgraph', '9c6b85428bccc1a1c5b23a9549d7d4470b11b6c5331d9a51c25500fdcd6fdf99'),
    ('ctf', '508bcdf2ec2f25d09497963a7c226a6c8d9b40981c97cac12a37d78b45cc5847'),
])
def test_legacy_selection_passes_source_verification_before_checkpoint_check(
    tmp_path, monkeypatch, name, source_identity,
):
    runner = importlib.import_module(f'scripts.run_{name}_p4')
    # Isolate upstream repository discovery; retain actual adapter/checkpoint checks.
    monkeypatch.setattr(runner, 'upstream_revision', lambda root: 'upstream-revision')
    monkeypatch.setattr(runner, 'official_source_hash', lambda root: 'upstream-source')
    checkpoint = tmp_path / 'damaged-checkpoint.pt'
    checkpoint.write_bytes(b'damaged checkpoint')
    selection = tmp_path / f'{name}-selection.json'
    selection.write_text(json.dumps({
        'source_csv_sha256': 'source',
        'validation_mask_sha256': 'validation',
        'test_mask_sha256': 'test',
        'upstream_revision': 'upstream-revision',
        'official_source_sha256': 'upstream-source',
        'adapter_sha256': source_identity,
        'epochs_completed': 100,
        'best_epoch': 50,
        'checkpoint': str(checkpoint),
        'checkpoint_sha256': 'original-checkpoint',
    }))
    protocol = SimpleNamespace(
        source_sha256='source',
        mask_sha256={'validation': 'validation', 'test': 'test'},
    )
    with pytest.raises(ValueError, match='checkpoint SHA-256 mismatch'):
        runner.evaluate_selected(protocol, tmp_path, tmp_path, torch.device('cpu'))
    assert not (tmp_path / f'{name}-test.json').exists()


def test_measured_wrapper_preserves_training_source_and_binds_saved_selection(tmp_path):
    source = tmp_path / 'legacy_training.py'
    selection = tmp_path / 'selected.json'
    source.write_text('''from pathlib import Path
import json
import sys
import torch
model = torch.nn.Linear(2, 1)
optimizer = torch.optim.Adam(model.parameters())
model(torch.ones(2, 2)).square().mean().backward()
optimizer.step()
Path(sys.argv[1]).write_text(json.dumps({'epochs_completed': 1, 'training_seconds': 0.1}))
''')
    original_source_sha256 = file_sha256(source)
    wrapper = Path(__file__).resolve().parents[2] / 'scripts' / 'run_measured_training.py'
    result = subprocess.run([
        sys.executable, str(wrapper), '--script', str(source),
        '--model', 'legacy-probe', '--stage', 'forecast', '--device', 'cpu',
        '--batch-size', '2', '--output', str(tmp_path),
        '--selection-manifest', str(selection), '--', str(selection),
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert file_sha256(source) == original_source_sha256
    assert json.loads(selection.read_text()) == {'epochs_completed': 1, 'training_seconds': 0.1}
    reports = list((tmp_path / 'resources').glob('*.json'))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text())
    assert report['status'] == 'completed'
    assert report['selection_manifest_sha256'] == file_sha256(selection)
    assert report['full_workflow_seconds'] >= report['training_stages_seconds'] > 0
    assert report['peak_allocated_bytes'] is None
