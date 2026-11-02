from pathlib import Path
import argparse,os,sys,json,subprocess,time,datetime,hashlib,fcntl
from summarize import build,group
R=Path(__file__).resolve().parent;P=json.loads((R/'plan.json').read_text());L=json.loads((R/'source_lock.json').read_text())
p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);a=p.parse_args()
lock=(R/f'queue_gpu{a.gpu}.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def verify():
 for name,h in L['files'].items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==h,name
 for v in P['variants']:
  for mode in ['initial','export']:assert json.loads((R/'preflight'/f'{v}_{mode}.json').read_text())['status']=='passed'
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);t=path.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2)+'\n');t.replace(path)
verify()
env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
tasks=[dict(variant=v,seed=s) for s in P['seeds'] for v in P['allocation'][str(a.gpu)]]
status=dict(status='running',gpu=a.gpu,started_utc=stamp(),current=None,pending=tasks.copy(),completed=[],failed=[]);sp=R/f'queue_gpu{a.gpu}.json';write(sp,status)
for task in tasks:
 v=task['variant'];s=task['seed'];G=R/group(v);tag=f'{v}_seed{s}';status['current']=task;write(sp,status);verify();start=time.perf_counter();started=stamp();log=R/'logs'/f'{tag}.log';log.parent.mkdir(exist_ok=True)
 try:
  common=['--dataset',P['dataset'],'--scene',P['scene'],'--seed',str(s),'--variant',v,'--data-root',P['data_root'],'--output-root',str(G/'outputs'),'--device','cuda:0']
  with log.open('a') as f:
   subprocess.run([sys.executable,str(G/'implementation/train.py'),*common,'--stage','both'],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
   for stage in ['main','calibration']:
    new=G/'outputs/runs'/P['dataset']/P['scene']/f'seed_{s}'/v/stage
    base=Path(P['reuse_E0'])/'outputs/runs'/P['dataset']/P['scene']/f'seed_{s}'/'H_GLOBAL'/stage
    done=json.loads((new/'complete.json').read_text());assert done['epochs']==100 and done['updates']==19000
    if stage=='calibration':assert done['frozen_before']==done['frozen_after']
    for epoch in range(1,101):
     x=json.loads((new/'epochs'/f'{epoch:03}.json').read_text());b=json.loads((base/'epochs'/f'{epoch:03}.json').read_text());assert x['batch_fingerprint']==b['batch_fingerprint'],(tag,stage,epoch)
   write(R/'audit'/f'{tag}.json',dict(status='passed',paired_epochs=200,updates_each_stage=19000,coarse_frozen=True))
   subprocess.run([sys.executable,str(G/'implementation/evaluate.py'),*common],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
   subprocess.run([sys.executable,str(G/'evaluate_details.py'),'--variant',v,'--scene',P['scene'],'--seed',str(s)],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
  row={**task,'gpu':a.gpu,'started_utc':started,'finished_utc':stamp(),'wallclock_seconds':time.perf_counter()-start,'scope':'imports+preparation+100main+100frozenCalibration+validation+save+20mask_test+export+coverage+latency'}
  write(R/'completed'/f'{tag}.json',row);status['completed'].append(row);build(f'gpu{a.gpu}')
 except BaseException as exc:
  error={**task,'error':repr(exc),'log':str(log),'finished_utc':stamp()};write(R/'failures'/f'{tag}.json',error);status['failed'].append(error)
  # Fail closed: a broken protocol must be inspected before further training.
  status['status']='failed';write(sp,status);raise
 status['pending']=[x for x in status['pending'] if x!=task];write(sp,status)
status['current']=None;status['status']='complete';status['finished_utc']=stamp();write(sp,status);build(f'gpu{a.gpu}')
