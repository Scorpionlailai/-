"""Durable two-slot queue. Each service owns one physical GPU, runs sequential jobs."""
from pathlib import Path
import os,sys,json,hashlib,subprocess,time,datetime,traceback,argparse,csv,statistics
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'implementation'))
from support import write_json

def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def summary():
 rows=[]
 for file in (ROOT/'outputs/evaluation').glob('*/*/seed_*/H_GLOBAL/results.json'):
  r=json.loads(file.read_text())
  for stage,field in [('coarse','precal_test'),('final','test')]:
   metric=r[field]['full96'];rows.append(dict(dataset=r['dataset'],scene=r['scene'],seed=r['seed'],stage=stage,mse=metric['mse_mean'],mae=metric['mae_mean'],test_masks=r['test_masks']))
 # Per-slot summaries avoid races; final status can be reconstructed from per-model results.
 out=ROOT/f'summary_exchange_gpu{ARGS.gpu}.csv'
 if rows:
  with out.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 aggregate=[]
 for key in sorted(set((r['dataset'],r['scene'],r['stage']) for r in rows)):
  rr=[r for r in rows if (r['dataset'],r['scene'],r['stage'])==key]
  aggregate.append(dict(dataset=key[0],scene=key[1],stage=key[2],training_seeds=sorted(x['seed'] for x in rr),n_training_seeds=len(rr),mse_mean=statistics.mean(x['mse'] for x in rr),mse_std_ddof1=statistics.stdev(x['mse'] for x in rr) if len(rr)>1 else None,mae_mean=statistics.mean(x['mae'] for x in rr),mae_std_ddof1=statistics.stdev(x['mae'] for x in rr) if len(rr)>1 else None,formal_three_seed_complete=len(rr)==3))
 write_json(ROOT/f'multiseed_summary_exchange_gpu{ARGS.gpu}.json',aggregate)

p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);ARGS=p.parse_args()
env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(ARGS.gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
plan=json.loads((ROOT/'plan.json').read_text());plan['datasets']=['Exchange-Rate'];tasks=[]
for ds in plan['datasets']:
 for seed in plan['training_seeds']:
  for scene in ['sr015_bl8','sr020_bl8','sr010_bl8','sr010_bl16']:tasks.append(dict(dataset=ds,seed=seed,scene=scene))
my=[r for i,r in enumerate(tasks) if i%2==ARGS.gpu];status={'gpu':ARGS.gpu,'started_utc':stamp(),'status':'running','pending':my,'completed':[],'failed':[]}
path=ROOT/f'exchange_queue_gpu{ARGS.gpu}.json';write_json(path,status)
for task in my:
 ds=task['dataset'];scene=task['scene'];seed=task['seed'];tag=f'{ds}_{scene}_seed{seed}'
 status['current']=task;write_json(path,status)
 audit=ROOT/'audit'/f'{ds}_mask_alignment.json'
 # Never run a dataset until all baseline masks and supporting assets pass audit.
 if not audit.exists() or json.loads(audit.read_text())['status']!='passed':
  status['failed'].append({**task,'error':'mask alignment gate not passed'});write_json(path,status);continue
 for name,h in plan['model_files'].items():
  if hashlib.sha256((ROOT/'implementation'/name).read_bytes()).hexdigest()!=h:raise RuntimeError('final R3 model file changed: '+name)
 log=ROOT/'logs'/f'{tag}.log';log.parent.mkdir(exist_ok=True);started=stamp();t=time.perf_counter()
 common=['--dataset',ds,'--scene',scene,'--seed',str(seed),'--variant','R3','--device','cuda:0','--data-root',str(ROOT/'data_root'),'--output-root',str(ROOT/'outputs')]
 try:
  with log.open('a') as f:
   f.write('\nJOB_START '+started+'\n');f.flush()
   subprocess.run([sys.executable,str(ROOT/'implementation/train.py'),*common,'--stage','both'],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
   subprocess.run([sys.executable,str(ROOT/'implementation/evaluate.py'),*common],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
   subprocess.run([sys.executable,str(ROOT/'benchmark.py'),'--dataset',ds,'--scene',scene,'--seed',str(seed)],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
  record={**task,'gpu':ARGS.gpu,'started_utc':started,'finished_utc':stamp(),'wallclock_seconds':time.perf_counter()-t,'scope':'imports+data_preparation+main100+frozen_calibration100+validation+save+20test_masks+export+benchmark'}
  write_json(ROOT/'completed'/f'{tag}.json',record);status['completed'].append(record);summary()
 except BaseException as exc:
  error={**task,'error':repr(exc),'log':str(log),'finished_utc':stamp()};status['failed'].append(error)
  write_json(ROOT/'failures'/f'{tag}.json',error)
 status['pending']=[x for x in status['pending'] if x!=task];write_json(path,status)
status['current']=None;status['status']='complete' if not status['failed'] else 'finished_with_failures';status['finished_utc']=stamp();write_json(path,status);summary()
