import json
from pathlib import Path
import time
from datetime import datetime,timezone

root=Path(__file__).resolve().parent
while True:
    lines=(root/'matrix.log').read_text().splitlines()
    starts=[]
    for line in lines:
        if line.startswith('{'):
            try:
                d=json.loads(line)
                if d.get('event')=='start':starts.append(d)
            except json.JSONDecodeError:pass
    active=Path(starts[-1]['log']) if starts else None
    tail=active.read_text().splitlines()[-1:] if active and active.exists() else []
    print(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),active=str(active),last_line=tail,
        tests_completed=len(list(root.glob('*/evaluation_*/twenty-test-draws.json'))),
        complete=(root/'matrix-completed.json').exists())),flush=True)
    if (root/'matrix-completed.json').exists() or any('Traceback' in line for line in lines):break
    time.sleep(30)
