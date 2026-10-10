"""Wait for the existing GPU0 trials, then run Stage1 refinement only."""
from pathlib import Path
import os,sys,json,subprocess,time,hashlib
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent
P=json.loads((ROOT/'plan.json').read_text());W=Path(P['work'])
ENV={**os.environ,'PYTHONPATH':f'{W}/src:{W}','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','PYTHONUNBUFFERED':'1'}
ENV.pop('MALR_MISSING_SCENARIO',None)
STATE={'status':'waiting_for_gpu0_queue','pid':os.getpid(),'monitor_interval_seconds':300,'user_report_interval_seconds':600,'active':None,'completed':[]}
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
SOURCES={str(x):sha(x) for folder in ['src','integrations','configs'] for x in (W/folder).rglob('*') if x.is_file() and x.suffix in ['.py','.yaml','.json']}
SOURCES.update({str(ROOT/x):sha(ROOT/x) for x in ['train_refinement.py','evaluate_repair.py','plan.json','e23_v95_stage1_tail_selection.yaml']})
def check():assert all(sha(Path(k))==v for k,v in SOURCES.items())
def publish():
 STATE['last_checked_utc']=datetime.now(timezone.utc).isoformat();write(ROOT/'queue-state.json',STATE);print(json.dumps(STATE),flush=True)
def run(args,log,scene,stage):
 check();STATE['status']='running';STATE['active']={'scene':scene['name'],'stage':stage,'log':str(log)}
 env=dict(ENV)
 if scene['scenario_manifest']:env['MALR_MISSING_SCENARIO']=scene['scenario_manifest']
 with log.open('x') as stream:
  child=subprocess.Popen(args,cwd=W,env=env,stdout=stream,stderr=subprocess.STDOUT);STATE['active']['child_pid']=child.pid;publish()
  while True:
   try:code=child.wait(timeout=300);break
   except subprocess.TimeoutExpired:
    STATE['active']['recent_log']=''.join(log.read_text().splitlines(keepends=True)[-3:]);publish();check()
  assert code==0,f'child failed exit={code}: {log}'
 check();publish()
def main():
 write(ROOT/'frozen-definition.json',{'sources':SOURCES,'stage2_training':False,'frozen_parent_v79':True})
 publish()
 for index,scene in enumerate(P['scenarios']):
  cell=ROOT/'E23-v95'/scene['name'];cell.mkdir(parents=True,exist_ok=True)
  run([sys.executable,str(ROOT/'train_refinement.py'),scene['name']],cell/'stage1.log',scene,'stage1_refinement')
  run([sys.executable,str(ROOT/'evaluate_repair.py'),'E23-v95',scene['name']],cell/'repair-audit.log',scene,'repair_validation_and_test20')
  result=json.loads((cell/'repair-audit.json').read_text());STATE['completed'].append({'model':'E23-v95','scene':scene['name'],'validation':result['split_validation'],'test_mean':result['test_mean'],'beats_t1_test':result['beats_t1_test_all_four_metrics']});STATE['active']=None;publish()
  if index==0 and not result['screening_passed_v79_validation']:
   STATE['status']='validation_screen_failed_need_next_design';publish();return
 STATE['status']='four_scenes_completed_review_required';publish()
if __name__=='__main__':
 try:main()
 except Exception as e:STATE['status']='error';STATE['error']=repr(e);publish();raise
