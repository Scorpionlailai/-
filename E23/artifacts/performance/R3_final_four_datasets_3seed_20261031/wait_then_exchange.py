"""Start Exchange only after BOTH original queues finish all 36 models."""
from pathlib import Path
import argparse,json,time,datetime,subprocess,sys
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);a=p.parse_args()
target=ROOT/f'exchange_start_gate_gpu{a.gpu}.json'
def save(obj):
 obj['checked_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(target)
while True:
 queues=[json.loads((ROOT/f'queue_gpu{gpu}.json').read_text()) for gpu in [0,1]]
 done=sum(len(q['completed']) for q in queues)
 if any(q['failed'] or q['status']=='finished_with_failures' for q in queues):
  save(dict(status='blocked_by_original_queue_failure',original_completed=done));raise RuntimeError('Resolve original three-dataset queue failures before Exchange')
 if all(q['status']=='complete' for q in queues):
  if done!=36:raise RuntimeError('Original queue completion count is not36')
  for ds in ['ETTh1','BeijingAirQuality','PhysioNet2012']:
   records=[json.loads(x.read_text()) for x in (ROOT/'completed').glob(f'{ds}_*.json')]
   if len(records)!=12:raise RuntimeError('Original dataset incomplete: '+ds)
  audit=json.loads((ROOT/'audit/Exchange-Rate_mask_alignment.json').read_text());assert audit['status']=='passed'
  save(dict(status='gate_passed_starting_exchange',original_completed=done))
  subprocess.run([sys.executable,str(ROOT/'run_exchange_queue.py'),'--gpu',str(a.gpu)],check=True,cwd=ROOT)
  q=json.loads((ROOT/f'exchange_queue_gpu{a.gpu}.json').read_text())
  save(dict(status=q['status'],original_completed=done,exchange_completed=len(q['completed']),failed=q['failed']))
  break
 save(dict(status='waiting_for_all_original_36_models',gpu=a.gpu,original_completed=done,current=[q.get('current') for q in queues]))
 time.sleep(60)
