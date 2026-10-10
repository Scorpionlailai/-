"""Lock a completed shared candidate, then perform one four-scene confirmation."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
BASELINE = Path('/home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_t1_pure_mse_frozen_four_scenarios_20261007')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2); f.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate-root', type=Path, required=True)
    parser.add_argument('--label', required=True)
    args = parser.parse_args()
    candidate_root = args.candidate_root.resolve()
    result = read(candidate_root / 'verification.json')
    assert result['status'] == 'verified' and result['all_four_better']
    plan = read(candidate_root / 'plan.json')
    assert read(candidate_root / 'complete.json')['status'] == 'completed'
    reserve = read(ROOT / 'reserved-lock.json')
    assert reserve['status'] == 'sealed_not_evaluated'
    assert sha(ROOT / 'create_bank.py') == reserve['generator_sha256']
    frozen = read(candidate_root / 'frozen-definition.json')
    protected = {str(ROOT / 'reserved-lock.json'): sha(ROOT / 'reserved-lock.json'),
        str(candidate_root / 'plan.json'): frozen['plan_sha256'],
        str(candidate_root / 'verification.json'): sha(candidate_root / 'verification.json'),
        str(ROOT / 'evaluate_locked.py'): sha(ROOT / 'evaluate_locked.py'),
        str(Path(plan['e23_config'])): sha(plan['e23_config'])}
    t1_plan = read(BASELINE / 'plan.json')
    protected[str(BASELINE / 't1-plan.json')] = sha(BASELINE / 't1-plan.json')
    protected[t1_plan['e23_config']] = sha(t1_plan['e23_config'])
    for work in [Path(plan['work']), Path(read(BASELINE / 'plan.json')['work'])]:
        for directory in ['src', 'integrations', 'scripts', 'configs']:
            for path in (work / directory).rglob('*'):
                if path.is_file() and path.suffix in ['.py', '.json', '.yaml']:
                    protected[str(path)] = sha(path)
    cells = []
    for scene, bank in zip(plan['scenarios'], reserve['scenarios']):
        assert scene['name'] == bank['scene'] and sha(bank['test_bank']) == bank['bank_sha256']
        assert set(bank['seeds']).isdisjoint(read(scene['test_bank'])['seeds'])
        protected[bank['test_bank']] = bank['bank_sha256']
        e23_selection = candidate_root / scene['name'] / args.label / 'stage2/runs/e23-stage2.json'
        t1_selection = BASELINE / 'T1' / scene['name'] / 'training/forecast-mae-selection.json'
        for path in [e23_selection, t1_selection]:
            record = read(path)
            assert record['loss'] == 'forecast_mse_only' and record['repair_update_policy'] == 'frozen_v1'
            assert record['selection_metric'] == 'mae' and record['epochs_completed'] == 100
            assert record['predictor_initial_sha256'] == plan['predictor_initial_sha256']
            assert sha(record['checkpoint']) == record['checkpoint_sha256']
            protected[str(path)] = sha(path)
            protected[record['checkpoint']] = record['checkpoint_sha256']
        cells.append(dict(**bank, scenario_manifest=scene['scenario_manifest'],
            validation_mask_sha256=scene['validation_mask_sha256'], e23_selection=str(e23_selection), t1_selection=str(t1_selection)))
    assert all(sha(path) == digest for path, digest in protected.items())
    lock = dict(created_utc=datetime.now(timezone.utc).isoformat(), status='locked_before_reserved_scores',
        candidate_label=args.label, candidate_root=str(candidate_root), candidate_work=plan['work'],
        candidate_config=plan['e23_config'], shared_config_sha256=sha(plan['e23_config']),
        selection_metric='mae', forecast_loss='pure_mse', repair_update_policy='frozen_v1',
        test_used_for_checkpoint_selection=False, one_confirmation_candidate_only=True,
        protected=protected, scenarios=cells, controller_sha256=sha(__file__))
    write(ROOT / 'candidate-lock-before-confirmation.json', lock)
    env = dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    env.pop('MALR_MISSING_SCENARIO', None)
    for cell in cells:
        for model in ['T1', 'E23']:
            work = plan['work'] if model == 'E23' else read(BASELINE / 'plan.json')['work']
            job_env = dict(env, PYTHONPATH=f'{work}/src:{work}')
            log = ROOT / model / cell['scene'] / 'confirmation.log'
            log.parent.mkdir(parents=True, exist_ok=True)
            print(json.dumps(dict(event='started', model=model, scene=cell['scene'])), flush=True)
            with log.open('x') as f:
                subprocess.run([sys.executable, str(ROOT / 'evaluate_locked.py'), '--model', model, '--scene', cell['scene']],
                    cwd=work, env=job_env, stdout=f, stderr=subprocess.STDOUT, check=True)
            assert all(sha(path) == digest for path, digest in protected.items())
        a = read(ROOT / 'E23' / cell['scene'] / 'evaluation_mae/twenty-test-draws.json')
        b = read(ROOT / 'T1' / cell['scene'] / 'evaluation_mae/twenty-test-draws.json')
        assert len(a['test_draws']) == len(b['test_draws']) == 20
        for x, y in zip(a['test_draws'], b['test_draws']):
            assert all(x[k] == y[k] for k in ['group_id', 'mask_seed', 'mask_sha256', 'windows', 'target_elements'])
        print(json.dumps(dict(event='scene_complete', scene=cell['scene'], e23=a['test_mean'], t1=b['test_mean'])), flush=True)
    write(ROOT / 'complete.json', dict(status='completed', candidate_label=args.label, scenarios=4, groups_per_scene=20,
        no_training=True, no_checkpoint_selection=True, reserved_evaluation_count=1))


if __name__ == '__main__':
    main()
