"""Generate the verified report after the queued experiment completes."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent

if __name__ == '__main__':
    while not (ROOT / 'complete.json').exists():
        checks = ROOT / 'monitor-checks.jsonl'
        if checks.exists():
            last = json.loads(checks.read_text().splitlines()[-1])
            if last['returncode'] is not None:
                print(json.dumps(dict(status='training_stopped', returncode=last['returncode'],
                                      controller_log=str(ROOT / 'controller.log'))), flush=True)
                sys.exit(last['returncode'] or 1)
        time.sleep(300)
    subprocess.run([sys.executable, str(ROOT / 'summarize.py')], check=True)
