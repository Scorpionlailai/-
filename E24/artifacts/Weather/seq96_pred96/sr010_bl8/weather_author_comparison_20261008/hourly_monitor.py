import json,os,time
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent
INTERVAL=3600
def live(pid):
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False
def report():
    state=json.loads((ROOT/'status.json').read_text());active=state.get('active');progress=None;validation=None
    if active:
        for line in Path(active['log']).read_text(errors='replace').splitlines():
            try:item=json.loads(line)
            except ValueError:continue
            if 'batch' in item:progress=item
            if 'validation' in item:validation=item
    value=dict(time_utc=datetime.now(timezone.utc).isoformat(),queue_status=state['status'],queue_pid=state['pid'],queue_alive=live(state['pid']),active=active,progress=progress,last_validation=validation,models=state['models'])
    temporary=ROOT/'hourly_status.json.tmp';temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temporary.replace(ROOT/'hourly_status.json')
    with (ROOT/'hourly_reports.jsonl').open('a') as stream:stream.write(json.dumps(value,ensure_ascii=False)+'\n')
    print(json.dumps(value,ensure_ascii=False),flush=True)
    return value
(ROOT/'hourly_monitor.pid').write_text(str(os.getpid())+'\n')
value=report();next_report=time.monotonic()+INTERVAL
while value['queue_status'] not in ('completed','completed_with_failures'):
    time.sleep(60)
    current=json.loads((ROOT/'status.json').read_text())
    if time.monotonic()>=next_report or current['status'] in ('completed','completed_with_failures'):
        value=report();next_report=time.monotonic()+INTERVAL
