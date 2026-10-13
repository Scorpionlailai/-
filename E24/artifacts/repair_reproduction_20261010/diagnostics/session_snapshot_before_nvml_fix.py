import datetime
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent
queue = json.loads((root / 'queue_status.json').read_text())
review = json.loads((root / 'review_status.json').read_text())
snapshot = dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    completed_repairs=review['completed_repairs'], completed_forecasts=review['completed_forecasts'],
    queue_state=queue['state'], failed_runs=queue.get('failed_runs', []), active=[])
for job in queue.get('active', []):
    path = root / job['dataset'] / job['scene'] / job['model'] / 'seed_2023/status.json'
    status = json.loads(path.read_text()) if path.exists() else {}
    snapshot['active'].append(dict(gpu=job['gpu'], model=job['model'], dataset=job['dataset'],
        scene=job['scene'], phase=status.get('phase'), epoch=status.get('epoch'),
        completed_masks=status.get('completed_masks'), elapsed_seconds=status.get('elapsed_seconds')))
snapshot['gpu'] = subprocess.check_output(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used', '--format=csv,noheader,nounits'], text=True).strip().splitlines()
with (root / 'session_monitoring.jsonl').open('a') as stream:
    stream.write(json.dumps(snapshot) + '\n')
print(json.dumps(snapshot))
