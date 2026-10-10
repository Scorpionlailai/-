"""Wait for v103, then execute Weather v97 sequentially on GPU0."""
from pathlib import Path
import json,os,time,subprocess,sys,hashlib,datetime
ROOT=Path(__file__).resolve().parent
PYTHON='/home/xyw/miniforge3/envs/malr/bin/python'
DEP=Path('/home/xyw/paper/E23/artifacts/performance/e23_v103_dynamic_candidates_20261008/training_job.json')

def status(value,**extra):
 record=dict(status=value,checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra)
 (ROOT/'queue_status.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)

def freeze_check():
 freeze=json.loads((ROOT/'source_freeze.json').read_text())
 for path,sha in freeze['files'].items():
  if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=sha:raise RuntimeError('Changed sealed input: '+path)

def run(script,args=()):
 freeze_check();log=ROOT/(Path(script).stem+'.log')
 with log.open('xb') as f:
  process=subprocess.Popen([PYTHON,'-u',str(ROOT/script),*args],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
  status('running',script=script,pid=process.pid,log=str(log))
  while process.poll() is None:time.sleep(30)
  if process.returncode:raise RuntimeError(f'{script} failed with {process.returncode}; inspect {log}')

def cost_report():
 plan=json.loads((ROOT/'plan.json').read_text());paths=[Path(plan['parent_selection']).parents[1],ROOT/'E23-v97/sr010_bl8/stage1',ROOT/'E23-v97/sr010_bl8/stage2'];rows=[]
 for directory in paths:
  resources=[json.loads(p.read_text()) for p in (directory/'resources').glob('*.json')];resources=[r for r in resources if r['status']=='completed'];assert len(resources)==1
  r=resources[0];rows.append(dict(directory=str(directory),seconds=r['full_workflow_seconds'],allocated_gib=r['peak_allocated_bytes']/2**30,reserved_gib=r['peak_reserved_bytes']/2**30,device=r['device']))
 record=dict(stages=rows,full_training_seconds=sum(r['seconds'] for r in rows),peak_allocated_gib=max(r['allocated_gib'] for r in rows),peak_reserved_gib=max(r['reserved_gib'] for r in rows),scope='sum of reused parent coarse and measured refinement/Stage2 commands; not a single uninterrupted cold run; excludes test evaluation',training_seed=2023)
 (ROOT/'complete_training_cost.json').write_text(json.dumps(record,indent=2))

def main():
 freeze_check();job=json.loads(DEP.read_text());status('queued',waiting_for_pid=job['pid'],device='cuda:0',scenario='sr010_bl8')
 while True:
  proc=Path('/proc')/str(job['pid'])
  alive=proc.exists() and (proc/'stat').read_text().split()[2]!='Z'
  if not alive:break
  time.sleep(30)
 output=Path(job['output'])
 assert (output/'runs/e23-stage1.json').exists(),'Dependency did not complete Stage1'
 assert (output.parent/'stage1_repair_20_masks.json').exists(),'Dependency did not complete repair evaluation'
 assert json.loads((output/'runs/e23-stage1.json').read_text())['epochs_completed']==100
 run('train_refinement.py',['sr010_bl8'])
 run('train_forecast_and_evaluate.py')
 cost_report();status('completed',results=str(ROOT/'E23-v97/sr010_bl8/evaluation_mae/twenty-test-draws.json'))
if __name__=='__main__':
 try:main()
 except Exception as error:
  status('failed',error=repr(error));raise
