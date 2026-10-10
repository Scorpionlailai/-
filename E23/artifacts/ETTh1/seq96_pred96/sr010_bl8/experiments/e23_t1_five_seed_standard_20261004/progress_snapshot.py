"""Read formal comparison progress and append its five-minute audit trail."""
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
from zoneinfo import ZoneInfo

r = Path(__file__).resolve().parent
s = json.loads((r/'comparison_statistics.json').read_text())
p = max((r/'logs').glob('*.log'), key=lambda p: p.stat().st_mtime)
rows = [{k: x[k] for k in ['model', 'training_seed', 'mse', 'mae',
                         'full_training_seconds', 'peak_allocated_gib', 'peak_reserved_gib']}
        for x in csv.DictReader((r/'per_training_seed.csv').open())]
now = datetime.now(timezone.utc)
x = {'utc': now.isoformat(), 'client_time': now.astimezone(ZoneInfo('Asia/Shanghai')).strftime('%H:%M:%S'),
     'status': s['status'], 'completed_runs': s['completed_runs'],
     'paired_completed_seeds': s['paired_completed_seeds'], 'latest_log': p.name,
     'last_lines': p.read_text().splitlines()[-2:], 'results': rows,
     'coordinator_error': 'Traceback' in (r/'coordinator.log').read_text(),
     'publisher': json.loads((r/'publish-status.json').read_text())}
with (r/'progress_history.jsonl').open('a') as stream:
    stream.write(json.dumps(x)+'\n')
print(json.dumps(x))
