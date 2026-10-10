"""Execute once and inspect training logs every five minutes, including final exit."""
import json
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent


def inspect(process):
    active = []
    for scene in json.loads((ROOT / 'plan.json').read_text())['scenarios']:
        cell = ROOT / scene['name'] / 'E23-v80'
        for stage, manifest in [('stage1', 'e23-stage1.json'), ('stage2', 'e23-stage2.json')]:
            log = cell / (stage + '.log')
            if not log.exists():
                continue
            with log.open('rb') as handle:
                handle.seek(max(0, log.stat().st_size - 4000))
                lines = handle.read().decode(errors='replace').splitlines()
            active.append(dict(scene=scene['name'], stage=stage,
                               completed=(cell / stage / 'runs' / manifest).exists(),
                               last_lines=lines[-2:]))
    item = dict(checked_utc=datetime.now(timezone.utc).isoformat(),
                pid=process.pid, returncode=process.poll(), stages=active)
    with (ROOT / 'monitor-checks.jsonl').open('a') as handle:
        handle.write(json.dumps(item) + '\n')
    print(json.dumps(item), flush=True)


if __name__ == '__main__':
    with (ROOT / 'controller.log').open('x') as handle:
        process = subprocess.Popen([sys.executable, str(ROOT / 'execute.py')],
                                   stdout=handle, stderr=subprocess.STDOUT)
        (ROOT / 'monitor-start.json').write_text(json.dumps(dict(
            started_utc=datetime.now(timezone.utc).isoformat(),
            controller_pid=process.pid, interval_seconds=300), indent=2) + '\n')
        while True:
            try:
                result = process.wait(timeout=300)
            except subprocess.TimeoutExpired:
                inspect(process)
                continue
            inspect(process)
            sys.exit(result)
