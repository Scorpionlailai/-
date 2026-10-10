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
                   ROOT/'E23-v97/sr010_bl8/stage1', ROOT/'E23-v97/sr010_bl8/stage2']
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


def main():
    freeze_check()
    gpu = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-compute-apps=pid',
                                   '--format=csv,noheader'], text=True).strip()
    if gpu:
        raise RuntimeError('GPU0 has an existing compute process: '+gpu)
    run('preflight.py')
    run('train_refinement.py', 'sr010_bl8')
    run('train_forecast_and_evaluate.py')
    cost_report()
    write_status('completed', results=str(ROOT/'E23-v97/sr010_bl8/evaluation_mae/twenty-test-draws.json'))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        write_status('failed', error=repr(error))
        raise
