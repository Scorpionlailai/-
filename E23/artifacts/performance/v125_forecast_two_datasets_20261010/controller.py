"""One persistent sequential forecast queue per GPU/dataset."""
from pathlib import Path
from datetime import datetime,timezone
import sys,os,json,subprocess,time,hashlib,csv
D=Path(sys.argv[1]).resolve();Q=D.parent;p=json.loads((D/'plan.json').read_text())
def write(path,x):
 path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(path)
def freeze():
 for file,h in json.loads((D/'source_freeze.json').read_text())['files'].items():assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==h,file
ENV=dict(os.environ,CUDA_VISIBLE_DEVICES=p['gpu_uuid'],OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
for key in ['E24_DATASET_MANIFEST','MALR_MISSING_SCENARIO','MALR_E23_CODE_FREEZE']:ENV.pop(key,None)
def tail(log):
 with log.open('rb') as f:f.seek(max(0,log.stat().st_size-2200));lines=f.read().decode(errors='replace').splitlines()
 return lines[-1] if lines else ''
def run(case,action,metric=None):
 freeze();log=case/(action+('_'+metric if metric else '')+'.log');cmd=[sys.executable,'-u',p['worker'],action,str(case)]
 if metric:cmd+=['--selection',metric]
 with log.open('xb') as f:
  child=subprocess.Popen(cmd,cwd=case,env=ENV,stdout=f,stderr=subprocess.STDOUT)
  write(case/(action+('_'+metric if metric else '')+'_launch.json'),dict(pid=child.pid,command=cmd,gpu_uuid=p['gpu_uuid'],started_utc=datetime.now(timezone.utc).isoformat()))
  while child.poll() is None:
   s=dict(status='running',dataset=p['dataset'],scene=case.name,phase=action,selection=metric,pid=child.pid,last_line=tail(log),updated_utc=datetime.now(timezone.utc).isoformat())
   write(case/'status.json',s);write(D/'status.json',s);time.sleep(30)
  assert child.returncode==0,f'{action} exit {child.returncode}: {log}'
 freeze()
def main():
 rows=[]
 for root in p['case_roots']:
  case=Path(root);assert json.loads((case/'preflight.json').read_text())['status']=='passed'
  run(case,'train')
  for metric in ['mae','mse']:
   run(case,'evaluate',metric)
   x=json.loads((case/('evaluation_'+metric)/'twenty-test-draws.json').read_text());assert x['repair_frozen_verified'] and x['model_memory_keys_unchanged']
   row=dict(dataset=p['dataset'],scene=case.name,selection_metric=metric,best_epoch=x['best_epoch'],test_masks=20)
   for k,v in x['summary'].items():row[k]=v['mean'];row[k+'_mask_std_ddof1']=v['mask_std_ddof1']
   for label,key in [('stage2','stage2_training_resources'),('stage1_and_2','training_resources')]:
    cost=x[key];row[label+'_full_minutes']=cost['full_workflow_seconds']/60;row[label+'_peak_allocated_gib']=cost['peak_allocated_gib'];row[label+'_peak_reserved_gib']=cost['peak_reserved_gib']
   row['source']=str(case/('evaluation_'+metric)/'twenty-test-draws.json');rows.append(row)
  write(case/'status.json',dict(status='completed',dataset=p['dataset'],scene=case.name,forecast_loss='pure_mse',repair_update_policy='frozen_v1',updated_utc=datetime.now(timezone.utc).isoformat()))
  with (D/'forecast_results.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 freeze();write(D/'status.json',dict(status='completed',dataset=p['dataset'],cases=4,selected_reports=8,updated_utc=datetime.now(timezone.utc).isoformat()))
try:main()
except BaseException as error:
 write(D/'status.json',dict(status='failed',dataset=p['dataset'],error=repr(error),updated_utc=datetime.now(timezone.utc).isoformat()));raise
