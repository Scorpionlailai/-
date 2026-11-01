"""One seed/scene on one GPU; immutable R3 training and evaluation entrypoints."""
import argparse,json,os,sys,time,traceback
from pathlib import Path
from adapter import ROOT,SOURCE,install

def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--scene',required=True);p.add_argument('--seed',type=int,default=2023);p.add_argument('--gpu',type=int,default=0);p.add_argument('--kind',choices=['selfcheck','smoke','run'],default='run');a=p.parse_args()
 data_root=Path('/home/xyw/paper');output=ROOT/'outputs'
 os.environ.update(A64_DATASET=a.dataset,R2_DATA_ROOT=str(data_root),R2_OUTPUT_ROOT=str(output))
 install()
 from support import write_json,sha
 for name,h in json.loads((ROOT/'execution_lock.json').read_text())['files'].items():
  if sha(ROOT/name)!=h:raise RuntimeError('execution adapter changed: '+name)
 if a.kind=='selfcheck':
  import selfcheck
  sys.argv=['selfcheck','--dataset',a.dataset,'--scene',a.scene,'--data-root',str(data_root),'--assets',str(ROOT/'preflight_assets'/a.dataset/a.scene)]
  selfcheck.main();return
 start=time.perf_counter()
 common=['--dataset',a.dataset,'--scene',a.scene,'--variant','R3','--seed',str(a.seed),'--device',f'cuda:{a.gpu}','--data-root',str(data_root),'--output-root',str(output)]
 if a.kind=='smoke':common+=['--smoke-batches','1']
 try:
  import train
  sys.argv=['train',*common,'--stage','both'];train.main()
  import evaluate
  sys.argv=['evaluate',*common];evaluate.main()
  write_json(ROOT/'jobs'/f'{a.dataset}_{a.scene}_{a.seed}_{a.kind}.json',{'status':'complete','dataset':a.dataset,'scene':a.scene,'seed':a.seed,'gpu':a.gpu,'wallclock_seconds':time.perf_counter()-start,'smoke':a.kind=='smoke'})
 except BaseException as e:
  write_json(ROOT/'jobs'/f'{a.dataset}_{a.scene}_{a.seed}_{a.kind}_failure.json',{'error':repr(e),'traceback':traceback.format_exc(),'wallclock_seconds':time.perf_counter()-start});raise
if __name__=='__main__':main()
