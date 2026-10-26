"""Independent seed workers on two or more GPUs; no DDP or seed changes."""
import argparse,os,subprocess,sys,time,json
from pathlib import Path
from bootstrap import ROOT

def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',default='ETTh1');p.add_argument('--scene',default='sr015_bl8');p.add_argument('--variant',choices=['R2','FULL','CTX','SUM'],default='R2');p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);p.add_argument('--seeds',type=int,nargs='+',default=[2023,2024,2025]);p.add_argument('--gpus',type=int,nargs='+',default=[0,1]);a=p.parse_args()
 if len(set(a.gpus))!=len(a.gpus) or len(set(a.seeds))!=len(a.seeds):p.error('GPUs and seeds must be unique')
 queue=list(a.seeds);running={};a.output_root.mkdir(parents=True,exist_ok=True)
 while queue or running:
  for gpu,(proc,log,seed) in list(running.items()):
   code=proc.poll()
   if code is not None:
    log.close();del running[gpu]
    if code:raise RuntimeError(f'seed{seed} GPU{gpu} failed; see log; completed checkpoints retained')
  for gpu in a.gpus:
   if gpu in running or not queue:continue
   seed=queue.pop(0);log=(a.output_root/f'{a.dataset}_{a.scene}_{a.variant}_seed{seed}.log').open('a')
   common=['--dataset',a.dataset,'--scene',a.scene,'--variant',a.variant,'--seed',str(seed),'--device',f'cuda:{gpu}','--data-root',str(a.data_root.resolve()),'--output-root',str(a.output_root.resolve())]
   # A tiny child orchestrates its dependent stages; training CLI owns resume/locks.
   code='import subprocess,sys; subprocess.run([sys.executable,sys.argv[1],*sys.argv[3:]],check=True); subprocess.run([sys.executable,sys.argv[2],*sys.argv[3:]],check=True)'
   proc=subprocess.Popen([sys.executable,'-u','-c',code,str(ROOT/'train.py'),str(ROOT/'evaluate.py'),*common],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
   running[gpu]=(proc,log,seed);print(json.dumps({'seed':seed,'gpu':gpu,'pid':proc.pid}),flush=True)
  if running:time.sleep(3)
if __name__=='__main__':main()
