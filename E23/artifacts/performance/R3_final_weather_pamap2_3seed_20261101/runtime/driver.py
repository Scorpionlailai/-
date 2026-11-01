"""Two-GPU persistent bounded queue; Exchange waits for exact baseline masks."""
import subprocess,sys,os,json,time,fcntl,traceback
from pathlib import Path
from report import report
ROOT=Path(__file__).resolve().parent

def write(path,obj):
 p=ROOT/path;tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(p)
def done(j):return (ROOT/'jobs'/f"{j['dataset']}_{j['scene']}_{j['seed']}_run.json").exists()
def ready(j):
 if j['dataset']!='Exchange-Rate':return True
 return (ROOT/'exchange_baseline_ready.json').exists() and (ROOT/'jobs/Exchange-Rate_sr015_bl8_2023_smoke.json').exists()
def launch(j,gpu):
 log=(ROOT/'logs'/f"{j['dataset']}_{j['scene']}_seed{j['seed']}_gpu{gpu}.log").open('a')
 args=[sys.executable,'-u',str(ROOT/'worker.py'),'--kind','run','--dataset',j['dataset'],'--scene',j['scene'],'--seed',str(j['seed']),'--gpu',str(gpu)]
 return subprocess.Popen(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT),log,j

def main():
 lock=(ROOT/'driver.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for d in ['Weather','PAMAP2']:
  if not (ROOT/'jobs'/f'{d}_sr015_bl8_2023_smoke.json').exists():raise RuntimeError('missing dataset smoke: '+d)
 queue=[{'dataset':d,'scene':s,'seed':seed} for s in ['sr015_bl8','sr010_bl8','sr020_bl8','sr010_bl16'] for seed in [2023,2024,2025] for d in ['Weather','PAMAP2','Exchange-Rate']];running={};failed=[]
 (ROOT/'logs').mkdir(exist_ok=True)
 while queue or running:
  for gpu,(p,log,j) in list(running.items()):
   code=p.poll()
   if code is not None:
    log.close();del running[gpu]
    if code!=0:failed.append({**j,'exit':code,'gpu':gpu})
    elif not done(j):raise RuntimeError('worker exited without result')
    report()
  queue=[j for j in queue if not done(j)]
  for gpu in [0,1]:
   if gpu in running:continue
   job=next((j for j in queue if ready(j)),None)
   if job is not None:queue.remove(job);running[gpu]=launch(job,gpu)
  completed=len(list((ROOT/'jobs').glob('*_run.json')))
  status='running' if running else ('waiting_Exchange_baseline_masks' if queue else ('complete_with_failures' if failed else 'complete'))
  write(Path('status.json'),{'status':status,'updated_unix':time.time(),'completed_models':completed,'planned_models':36,'active':{str(g):{'pid':p.pid,**j} for g,(p,l,j) in running.items()},'pending_ready':sum(ready(j) for j in queue),'pending_blocked':sum(not ready(j) for j in queue),'failures':failed})
  if queue or running:time.sleep(5)
 report()
if __name__=='__main__':
 try:main()
 except BaseException as e:write(Path('driver_failure.json'),{'error':repr(e),'traceback':traceback.format_exc()});raise
