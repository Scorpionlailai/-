"""Run controlled Stage1 changes continuously, with fixed frozen Stage2 comparisons."""
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def inspect(process, entry):
    stages = []
    root = Path(entry['root'])
    for scene in json.loads((root / 'plan.json').read_text())['scenarios']:
        cell = root / scene['name'] / entry['label']
        for stage, manifest in [('stage1', 'e23-stage1.json'), ('stage2', 'e23-stage2.json')]:
            log = cell / (stage + '.log')
            if not log.exists():
                continue
            with log.open('rb') as handle:
                handle.seek(max(0, log.stat().st_size - 6000))
                lines = handle.read().decode(errors='replace').splitlines()
            validation = [s for s in lines if ' val_' in s]
            stages.append(dict(scene=scene['name'], stage=stage,
                completed=(cell / stage / 'runs' / manifest).exists(),
                last_line=lines[-1] if lines else '',
                last_validation=validation[-1] if validation else None))
    row = dict(checked_utc=datetime.now(timezone.utc).isoformat(), model=entry['label'],
               pid=process.pid, returncode=process.poll(), stages=stages)
    for directory in [ROOT, root]:
        with (directory / 'monitor-checks.jsonl').open('a') as handle:
            handle.write(json.dumps(row) + '\n')
    print(json.dumps(row), flush=True)


if __name__ == '__main__':
    plan = json.loads((ROOT / 'queue-plan.json').read_text())
    assert not (ROOT / 'queue-state.json').exists(), 'Refuse to duplicate an existing queue'
    finished = []
    for entry in plan['models']:
        root = Path(entry['root'])
        with (root / 'controller.log').open('x') as handle:
            process = subprocess.Popen([sys.executable, str(root / 'execute.py')],
                stdout=handle, stderr=subprocess.STDOUT, cwd=entry['work'])
            write(ROOT / 'queue-state.json', dict(status='running', active=entry,
                controller_pid=process.pid, completed=finished,
                started_utc=datetime.now(timezone.utc).isoformat(), monitor_interval_seconds=300))
            print(json.dumps(dict(event='candidate_started', model=entry['label'], pid=process.pid)), flush=True)
            while True:
                try:
                    result = process.wait(timeout=300)
                except subprocess.TimeoutExpired:
                    inspect(process, entry)
                    continue
                inspect(process, entry)
                break
        if result != 0:
            write(ROOT / 'queue-state.json', dict(status='failed', active=entry,
                returncode=result, completed=finished, error_log=str(root / 'controller.log')))
            sys.exit(result)
        with (root / 'report.log').open('x') as handle:
            report = subprocess.run([sys.executable, str(root / 'summarize.py')],
                stdout=handle, stderr=subprocess.STDOUT)
        if report.returncode:
            write(ROOT / 'queue-state.json', dict(status='verification_failed', active=entry,
                completed=finished, error_log=str(root / 'report.log')))
            sys.exit(report.returncode)
        value = json.loads((root / 'verification.json').read_text())
        passed = (value['repair_beats_t1_all_metrics_all_scenes'] and
                  value['forecast_better_than_v79_all_metrics_all_scenes'])
        finished.append(dict(model=entry['label'], verification=str(root / 'verification.json'),
                             report=value['report'], passed_development_criteria=passed))
        print(json.dumps(dict(event='candidate_verified', **finished[-1])), flush=True)
        if passed:
            write(ROOT / 'queue-state.json', dict(status='candidate_passed_development',
                completed=finished, candidate=entry['label'],
                formal_confirmation='still requires unexposed shared-mask confirmation'))
            sys.exit(0)
    write(ROOT / 'queue-state.json', dict(status='preset_trials_completed_need_next_design',
        completed=finished, objective_achieved=False))
