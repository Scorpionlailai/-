"""Freeze, train once, lock validation, then evaluate twenty existing test masks."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from datetime import datetime, timezone

WORK = Path(__file__).resolve().parents[1]
ROOT = Path('/home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v46_front_context_sr015_bl8_seed502_20261005')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path, value):
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:json.dump(value, f, indent=2);f.write('\n')
def sources():
    paths = [WORK/'pyproject.toml']
    for folder in ('src', 'integrations', 'scripts', 'configs', 'tests'):
        paths.extend(p for p in (WORK/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc')
    return {str(p.relative_to(WORK)):digest(p) for p in sorted(paths)}


def main():
    assert read(ROOT/'preflight.json')['status'] == 'passed'
    assert read(ROOT/'test-results.json')['status'] == 'passed'
    plan = read(ROOT/'plan.json');protected = read(ROOT/'protected-files.json')
    definition = dict(created_utc=datetime.now(timezone.utc).isoformat(), sources=sources(),
        plan_sha256=digest(ROOT/'plan.json'), test_bank_sha256=digest(plan['test_bank']),
        scenario_sha256=digest(plan['scenario_manifest']), protected=protected,
        config_sha256=digest(plan['config']))
    write(ROOT/'frozen-definition.json', definition)
    with tarfile.open(ROOT/'frozen-source.tar.gz', 'w:gz', dereference=True) as archive:
        for relative in definition['sources']:archive.add(WORK/relative, arcname=relative, recursive=False)
    env = {**os.environ, 'PYTHONPATH':'src:.', 'OMP_NUM_THREADS':'4', 'MKL_NUM_THREADS':'4',
           'MALR_MISSING_SCENARIO':plan['scenario_manifest']}
    def verify():
        assert sources() == definition['sources'], 'Source changed'
        assert digest(ROOT/'plan.json') == definition['plan_sha256']
        assert digest(plan['test_bank']) == definition['test_bank_sha256']
        assert digest(plan['scenario_manifest']) == definition['scenario_sha256']
        assert digest(plan['config']) == definition['config_sha256']
        for path, sha in protected.items():assert digest(path) == sha, path
    def run(command, name):
        verify();write(ROOT/(name+'-command.json'), dict(command=command, environment={'MALR_MISSING_SCENARIO':plan['scenario_manifest']}))
        print(json.dumps(dict(event='start', stage=name, utc=datetime.now(timezone.utc).isoformat())), flush=True)
        with (ROOT/(name+'.log')).open('x') as log:
            subprocess.run(command, cwd=WORK, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        verify()
    common = ['--protocol-config', str(WORK/'configs/stage1_seq96_dynamic_temporal_repair.yaml'),
        '--migration-catalog', str(WORK.parent/'E22/migration/paper_catalog.jsonl'),
        '--mask-artifacts', str(WORK.parent/'E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context'),
        '--e23-config', plan['config'], '--training-seed', '502', '--device', 'cuda:1']
    stage1 = ROOT/'stage1/runs/e23-stage1.json';stage2 = ROOT/'stage2/runs/e23-stage2.json'
    run([sys.executable, '-m', 'integrations.e23.run', 'train-repair', *common, '--output', str(ROOT/'stage1')], 'stage1')
    assert read(stage1)['epochs_completed'] == 100
    run([sys.executable, '-m', 'integrations.e23.run', 'train-forecast', *common,
        '--stage1-manifest', str(stage1), '--selection-metric', 'mae', '--forecast-loss', 'mse_plus_0p2_mae',
        '--repair-retention', 'uniform_v1', '--repair-update-policy', 'slow_lr_0p1_v1', '--output', str(ROOT/'stage2')], 'stage2')
    r = read(stage2)
    assert r['epochs_completed'] == 100 and r['selection_metric'] == 'mae'
    assert r['loss'] == 'forecast_mse_plus_0p2_mae' and r['repair_retention'] == 'uniform_v1'
    assert r['repair_update_policy'] == 'slow_lr_0p1_v1'
    assert r['predictor_initial_sha256'] == plan['predictor_initial_sha256']
    assert r['front_context'] == 'window_cnn_v1'
    gate = r['best_val_mse'] < plan['reference_validation']['mse'] and r['best_val_mae'] <= plan['reference_validation']['mae']
    lock = dict(validation={k:r[k] for k in ('best_val_mse', 'best_val_mae', 'best_epoch', 'checkpoint_sha256')},
        matched_v41_validation_gate=gate, criterion='MSE lower and MAE no worse than matched v41', test_metrics_seen=False)
    write(ROOT/'validation-lock-before-tests.json', lock)
    print(json.dumps(dict(event='validation_locked', **lock)), flush=True)
    run([sys.executable, 'scripts/evaluate_e23_t1_replicate.py', '--model', 'E23-v31-B',
        '--model-label', plan['label'], '--training-seed', '502', '--e23-config', plan['config'],
        '--selection', str(stage2), '--test-mask-bank', plan['test_bank'], '--output', str(ROOT/'evaluation'), '--device', 'cuda:1'], 'test')
    run([sys.executable, 'scripts/diagnose_e23_front_context.py', '--cell', str(ROOT)], 'diagnostics')
    result = read(ROOT/'evaluation/twenty-test-draws.json')
    assert len(result['test_draws']) == 20
    verify()
    write(ROOT/'completed-audit.json', dict(status='complete', validation=lock,
        test_mean=result['test_mean'], test_std=result['test_mask_std_ddof1'], test_used_for_selection=False,
        new_T1_training=False, source_unchanged=True, protected_files_unchanged=True))
    print(json.dumps(read(ROOT/'completed-audit.json')), flush=True)


if __name__ == '__main__':main()
