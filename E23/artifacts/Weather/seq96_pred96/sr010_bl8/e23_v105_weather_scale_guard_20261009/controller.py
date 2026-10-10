"""Queue the v105 repair-output experiment after the v97 Weather control."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
PLAN = json.loads((ROOT/'plan.json').read_text())
BASE = Path(PLAN['parent_experiment'])


def status(state, **extra):
    record = dict(status=state, updated_at=datetime.now(timezone.utc).isoformat(), **extra)
    temporary = ROOT/'status.tmp'
    temporary.write_text(json.dumps(record, indent=2))
    temporary.replace(ROOT/'status.json')
    print(json.dumps(record), flush=True)


def freeze_check():
    for path, expected in json.loads((ROOT/'source_freeze.json').read_text())['files'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Sealed source/input changed: '+path)


def run(script):
    freeze_check()
    log = ROOT/(Path(script).stem+'.log')
    with log.open('xb') as stream:
        process = subprocess.Popen([sys.executable, '-u', str(ROOT/script)], cwd=ROOT,
                                   stdout=stream, stderr=subprocess.STDOUT)
        status('running', phase=script, pid=process.pid, log=str(log))
        checked = time.monotonic()
        while process.poll() is None:
            time.sleep(30)
            if time.monotonic()-checked >= 300:
                lines = log.read_text(errors='replace').splitlines()
                status('running', phase=script, pid=process.pid, log=str(log),
                       last_progress=lines[-1] if lines else None)
                checked = time.monotonic()
        if process.returncode:
            raise RuntimeError(f'{script} failed with exit {process.returncode}; inspect {log}')


def wait_for(stage1=False):
    checked = -float('inf')
    selection = Path(PLAN['parent_selection'])
    while True:
        parent = json.loads((BASE/'status.json').read_text())
        if parent['status'] == 'failed':
            raise RuntimeError('v97 dependency failed: '+parent.get('error', 'unknown'))
        if stage1 and selection.exists():
            selected = json.loads(selection.read_text())
            if selected['epochs_completed'] == 100:
                return
        elif not stage1 and parent['status'] == 'completed':
            gpu = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-compute-apps=pid',
                                           '--format=csv,noheader'], text=True).strip()
            if not gpu:
                return
        if time.monotonic()-checked >= 300:
            status('queued', waiting_for='v97 Stage1' if stage1 else 'v97 full control and idle GPU0',
                   parent_status=parent, next_stage='CPU output projection' if stage1 else 'frozen Stage2')
            checked = time.monotonic()
        time.sleep(30)


def report():
    directories = [Path(PLAN['coarse_parent_selection']).parents[1],
                   Path(PLAN['parent_selection']).parents[1],
                   ROOT/'E23-v105/sr010_bl8/stage1', ROOT/'E23-v105/sr010_bl8/stage2']
    rows = []
    for directory in directories:
        records = [json.loads(p.read_text()) for p in (directory/'resources').glob('*.json')]
        records = [r for r in records if r['status']=='completed']
        if len(records) != 1:
            raise RuntimeError('Expected one completed resource record: '+str(directory))
        record = records[0]
        rows.append(dict(directory=str(directory), seconds=record['full_workflow_seconds'],
                         allocated_gib=record['peak_allocated_bytes']/2**30,
                         reserved_gib=record['peak_reserved_bytes']/2**30, device=record['device']))
    result = dict(stages=rows, full_training_seconds=sum(r['seconds'] for r in rows),
                  peak_allocated_gib=max(r['allocated_gib'] for r in rows),
                  peak_reserved_gib=max(r['reserved_gib'] for r in rows),
                  scope='sum of reused coarse/refinement, new CPU projection validation and frozen Stage2; '
                        'no additional Stage1 optimizer updates; excludes final test and search costs; '
                        'not an uninterrupted cold run', training_seed=2023,
                  baseline_control_stage2_is_search_cost_not_production_cost=True)
    (ROOT/'complete_training_cost.json').write_text(json.dumps(result, indent=2))
    # The original evaluation collector sees only projection + new Stage2.
    # Provide a separate table containing all reused parent training costs.
    import csv
    comparison = []
    for model, base in [('E23-v97', BASE), ('E23-v105', ROOT)]:
        for criterion in ('mae', 'mse'):
            evaluation = json.loads((base/f'{model}/sr010_bl8/evaluation_{criterion}/twenty-test-draws.json').read_text())
            cost = json.loads((base/'complete_training_cost.json').read_text())
            row = dict(model=model, stage2_selection=criterion, test_groups=20,
                       full_training_seconds=cost['full_training_seconds'],
                       peak_allocated_gib=cost['peak_allocated_gib'],
                       peak_reserved_gib=cost['peak_reserved_gib'],
                       repair_frozen_verified=evaluation['repair_frozen_verified'])
            for key in ('forecast_mse', 'forecast_mae', 'repair96_mse', 'repair96_mae', 'repair32_mse', 'repair32_mae'):
                row[key] = evaluation['summary'][key]['mean']
                row[key+'_mask_std_ddof1'] = evaluation['summary'][key]['mask_std_ddof1']
            comparison.append(row)
    with (ROOT/'v97_v105_comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)


def main():
    freeze_check()
    wait_for(stage1=True)
    run('project_stage1_cpu.py')
    wait_for(stage1=False)
    run('train_forecast_and_evaluate.py')
    report()
    status('completed', comparison=str(ROOT/'v97_v105_comparison.csv'))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status('failed', error=repr(error))
        raise
