import json, os, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

root = Path('/home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v74_pre_tail_attention_seed2023_20261006')
started = datetime.now(timezone.utc).isoformat()
record = dict(status='running', launcher_pid=os.getpid(), started_utc=started)
with (root/'console.log').open('x') as log:
    child = subprocess.Popen([sys.executable,str(root/'execute.py')],stdout=log,stderr=subprocess.STDOUT)
    record['controller_pid'] = child.pid
    (root/'status.json').write_text(json.dumps(record,indent=2)+'\n')
    code = child.wait()
record.update(status='complete' if code == 0 else 'failed',exit_code=code,
              finished_utc=datetime.now(timezone.utc).isoformat())
(root/'status.json').write_text(json.dumps(record,indent=2)+'\n')
