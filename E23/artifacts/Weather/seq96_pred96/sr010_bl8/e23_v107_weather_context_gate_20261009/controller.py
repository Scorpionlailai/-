"""Run the v97 Weather repair and frozen forecast stages on GPU 0."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


def write_status(state, **extra):
    record = dict(status=state, updated_at=datetime.now(timezone.utc).isoformat(), **extra)
    temporary = ROOT / 'status.tmp'
    temporary.write_text(json.dumps(record, indent=2))
    temporary.replace(ROOT / 'status.json')
    print(json.dumps(record), flush=True)


def freeze_check():
    for path, expected in json.loads((ROOT/'source_freeze.json').read_text())['files'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Sealed input changed: '+path)


def run(script, *args):
    freeze_check()
    log = ROOT / (Path(script).stem+'.log')
    with log.open('xb') as stream:
        process = subprocess.Popen([PYTHON, '-u', str(ROOT/script), *args], cwd=ROOT,
                                   stdout=stream, stderr=subprocess.STDOUT)
        write_status('running', phase=script, pid=process.pid, log=str(log))
        checked = time.monotonic()
        while process.poll() is None:
            time.sleep(30)
            if time.monotonic()-checked >= 300:
                lines = log.read_text(errors='replace').splitlines()
                write_status('running', phase=script, pid=process.pid, log=str(log),
                             last_progress=lines[-1] if lines else None)
                checked = time.monotonic()
        if process.returncode:
            raise RuntimeError(f'{script}: exit {process.returncode}; inspect {log}')


def cost_report():
    plan = json.loads((ROOT/'plan.json').read_text())
    directories = [Path(plan['parent_selection']).parents[1],
                   ROOT/'E23-v107/sr010_bl8/stage1', ROOT/'E23-v107/sr010_bl8/stage2']
    rows = []
    for directory in directories:
        records = [json.loads(p.read_text()) for p in (directory/'resources').glob('*.json')]
        records = [r for r in records if r['status']=='completed']
        if len(records) != 1:
            raise RuntimeError('Expected one completed training resource record: '+str(directory))
        record = records[0]
        rows.append(dict(directory=str(directory), seconds=record['full_workflow_seconds'],
                         allocated_gib=record['peak_allocated_bytes']/2**30,
                         reserved_gib=record['peak_reserved_bytes']/2**30,
                         device=record['device']))
    result = dict(stages=rows, full_training_seconds=sum(r['seconds'] for r in rows),
                  peak_allocated_gib=max(r['allocated_gib'] for r in rows),
                  peak_reserved_gib=max(r['reserved_gib'] for r in rows),
                  scope='sum of reused coarse repair and new refinement/Stage2 commands; '
                        'includes cache construction and validation; excludes test evaluation; '
                        'not one uninterrupted cold run', training_seed=2023)
    (ROOT/'complete_training_cost.json').write_text(json.dumps(result, indent=2))


def comparison_report():
    import csv
    parent = ROOT.parent
    definitions = [
        ('E23-v97', parent/'e23_v97_weather_cached_20261008', 'E23-v97/sr010_bl8/evaluation_{criterion}'),
        ('E23-v105', parent/'e23_v105_weather_scale_guard_20261009', 'E23-v105/sr010_bl8/evaluation_{criterion}'),
        ('E23-v107', ROOT, 'E23-v107/sr010_bl8/evaluation_{criterion}'),
        ('T1', Path('/home/xyw/paper/E24/artifacts/Weather/seq96_pred96/sr010_bl8/weather_author_comparison_20261008/T1'), 'evaluation/{criterion}'),
        ('SAITS', Path('/home/xyw/paper/E24/artifacts/Weather/seq96_pred96/sr010_bl8/weather_author_comparison_20261008/SAITS'), 'evaluation/{criterion}'),
    ]
    rows = []
    keys = ('forecast_mse','forecast_mae','repair96_mse','repair96_mae','repair32_mse','repair32_mae')
    for model, base, layout in definitions:
        cost_path = base/'complete_training_cost.json'
        cost = json.loads(cost_path.read_text()) if cost_path.exists() else None
        if cost is None:
            resources = [json.loads(p.read_text()) for stage in ('stage1','stage2')
                         for p in (base/stage/'resources').glob('*.json')]
            resources = [r for r in resources if r['status']=='completed']
            assert len(resources)==2, model
            cost = dict(full_training_seconds=sum(r['full_workflow_seconds'] for r in resources),
                        peak_allocated_gib=max(r['peak_allocated_bytes'] for r in resources)/2**30,
                        peak_reserved_gib=max(r['peak_reserved_bytes'] for r in resources)/2**30)
        for criterion in ('mae','mse'):
            path = base/layout.format(criterion=criterion)/'twenty-test-draws.json'
            evaluation = json.loads(path.read_text())
            row = dict(model=model, stage2_selection=criterion, test_groups=20,
                       full_training_seconds=cost['full_training_seconds'],
                       peak_allocated_gib=cost['peak_allocated_gib'], peak_reserved_gib=cost['peak_reserved_gib'],
                       evaluation_source=str(path))
            for key in keys:
                source_key = key
                if key.startswith('forecast_') and key not in evaluation['summary']:
                    source_key = key.removeprefix('forecast_')
                row[key] = evaluation['summary'][source_key]['mean']
                row[key+'_mask_std_ddof1'] = evaluation['summary'][source_key]['mask_std_ddof1']
            rows.append(row)
    with (ROOT/'v97_v105_v107_T1_SAITS_comparison.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    freeze_check()
    plan = json.loads((ROOT/'plan.json').read_text())
    dependency = Path(plan['queue_after'])
    checked = -float('inf')
    while True:
        parent = json.loads((dependency/'status.json').read_text())
        if parent['status'] == 'failed':
            raise RuntimeError('Dependency failed: '+parent.get('error','unknown'))
        gpu = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-compute-apps=pid',
                                       '--format=csv,noheader'], text=True).strip()
        if parent['status'] == 'completed' and not gpu:
            break
        if time.monotonic()-checked >= 300:
            write_status('queued', waiting_for=str(dependency), parent_status=parent,
                         next_stage='fresh full Stage1 refinement, 100 epochs')
            checked = time.monotonic()
        time.sleep(30)
    # CPU preflight completed before enqueueing; verify it belongs to sealed source.
    preflight = json.loads((ROOT/'preflight.json').read_text())
    assert preflight['zero_residual_reproduces_parent_exactly']
    run('train_refinement.py', 'sr010_bl8')
    run('train_forecast_and_evaluate.py')
    cost_report()
    comparison_report()
    write_status('completed', exit_code=0,
                 results=str(ROOT/'E23-v107/sr010_bl8/evaluation_mae/twenty-test-draws.json'))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        write_status('failed', exit_code=1, error=repr(error))
        raise
