from pathlib import Path
import json,os,sys,argparse,subprocess,datetime
R=Path(__file__).resolve().parent;P=json.loads((R/'plan.json').read_text());p=argparse.ArgumentParser();p.add_argument('--gpu',required=True,type=int);a=p.parse_args()
env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
def write(s):
 t=R/f'queue_gpu{a.gpu}.tmp';t.write_text(json.dumps(s,indent=2)+'\n');t.replace(R/f'queue_gpu{a.gpu}.json')
s=dict(status='running',gpu=a.gpu,current=None,completed=[],pending=P['allocations'][str(a.gpu)].copy(),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());write(s)
for job in P['allocations'][str(a.gpu)]:
 s['current']=job;write(s)
 try:
  for extra in [['--smoke'],[]]:subprocess.run([sys.executable,str(R/'run_experiment.py'),'--job',job,*extra],env=env,check=True)
  subprocess.run([sys.executable,str(R/'compare.py'),'--suffix',job],env=env,check=True)
 except BaseException as e:
  s.update(status='failed',error=repr(e));write(s);raise
 s['completed'].append(job);s['pending'].remove(job);write(s)
s.update(status='complete',current=None,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());write(s)
