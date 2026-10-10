from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,subprocess,os,sys,time,traceback,csv,statistics
Q=Path(__file__).resolve().parent
plan=json.loads((Q/'plan.json').read_text())
GPU=plan['gpu_uuid'];SOURCE=plan['source']
def write(path,value):
 p=Path(path);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)
def state(root,phase,**kw):
 value=dict(status='running',phase=phase,time_utc=datetime.now(timezone.utc).isoformat(),**kw)
 write(root/'status.json',value);write(Q/'status.json',dict(scene_root=str(root),**value));print(json.dumps(dict(scene_root=str(root),**value)),flush=True)
def freeze(root):
 for path,digest in json.loads((root/'source_freeze.json').read_text())['files'].items():
  assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
 for path,digest in json.loads((root/'acquired_parent_freeze.json').read_text()).items() if (root/'acquired_parent_freeze.json').exists() else []:
  assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
ENV=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONPATH=SOURCE+':'+SOURCE+'/src')
for key in ['E24_DATASET_MANIFEST','MALR_MISSING_SCENARIO','MALR_E23_CODE_FREEZE']:ENV.pop(key,None)
def tail(log):
 if not log.exists():return ''
 with log.open('rb') as f:f.seek(max(0,log.stat().st_size-4000));lines=f.read().decode(errors='replace').splitlines()
 return lines[-1] if lines else ''
def job(root,phase,args,cwd,log,env=ENV):
 freeze(root)
 with log.open('xb') as f:
  p=subprocess.Popen(args,cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT)
  while p.poll() is None:
   state(root,phase,pid=p.pid,last_line=tail(log));time.sleep(30)
  assert p.returncode==0,f'{phase} exit {p.returncode}: {log}'
 freeze(root)
def report(root,w):
 path=Path(w['ensemble_root'])/'E23-v125'/w['scenario']['name']/'repair_evaluation_mae/twenty-test-draws.json'
 result=json.loads(path.read_text());assert len(result['draws'])==20
 assert result['start_rate']==w['scenario']['start_rate'] and result['block_len']==w['scenario']['block_len']
 assert result['visible_values_exact'] and result['parameters_memory_and_keys_unchanged']
 keys=['repair96_mse','repair96_mae','repair32_mse','repair32_mae']
 row=dict(model='E23-v125',scenario=w['scenario']['name'],test_masks=20)
 for k in keys:row[k]=result['summary'][k]['mean'];row[k+'_mask_std_ddof1']=statistics.stdev(d[k] for d in result['draws'])
 r=result['training_resources'];assert r['resource_status']=='measured_all_stages'
 row.update(stage1_full_seconds=r['full_workflow_seconds'],stage1_peak_allocated_gib=r['peak_allocated_gib'],stage1_peak_reserved_gib=r['peak_reserved_gib'],source=str(path))
 with (root/'repair_results.csv').open('x',newline='') as f:
  wr=csv.DictWriter(f,fieldnames=list(row));wr.writeheader();wr.writerow(row)
 (root/'result.md').write_text('# Weather '+w['scenario']['name']+' v125\n\n'+ '\n'.join(f'- {k}: {row[k]:.8f}' for k in keys)+'\n\n100 coarse + 100 C64 + 100 C256; fixed0.5/0.5; 20 sealed development masks; seed2023; no Stage2. Full Stage1 source cost '+str(r['full_workflow_seconds']/60)+' minutes.\n\nSame20-mask T1 repair comparison: matched_repair_comparison.csv; no Stage2.\n')
 return {k:row[k] for k in keys}
def main():
 while True:
  dependency=json.loads(Path(plan['weather_dependency']).read_text())
  if dependency['status']=='completed':break
  assert dependency['status']!='failed', 'Weather dependency failed; ETTh1 not launched'
  write(Q/'status.json',dict(status='waiting_for_weather',weather_dependency=plan['weather_dependency'],weather_progress=dependency,time_utc=datetime.now(timezone.utc).isoformat()))
  time.sleep(60)
 for rootname in plan['scene_roots']:
  root=Path(rootname);w=json.loads((root/'workflow.json').read_text());scene=w['scenario']['name']
  coarse_env=dict(ENV)
  if w['scenario'].get('scenario_manifest'):coarse_env['MALR_MISSING_SCENARIO']=w['scenario']['scenario_manifest']
  job(root,'coarse100',w['coarse_command'],w['coarse_cwd'],root/'coarse_train.log',coarse_env)
  parent=root/'coarse/stage1/runs/e23-stage1.json';record=json.loads(parent.read_text());assert record['epochs_completed']==100
  write(root/'acquired_parent_freeze.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [parent,Path(record['checkpoint'])]})
  processes=[];streams=[]
  for b in w['branches']:
   directory=Path(b['root']);job(root,'preflight'+str(b['width']),[sys.executable,'-u',str(directory/'preflight.py')],directory,directory/'preflight.log')
  for b in w['branches']:
   directory=Path(b['root']);f=(directory/'train_refinement.log').open('xb');streams.append(f)
   p=subprocess.Popen([sys.executable,'-u',str(directory/'train_refinement.py'),scene],cwd=directory,env=ENV,stdout=f,stderr=subprocess.STDOUT);processes.append((b,p))
   write(directory/'launch.json',dict(pid=p.pid,gpu_uuid=GPU,time_utc=datetime.now(timezone.utc).isoformat()))
  while any(p.poll() is None for _,p in processes):
   state(root,'parallel_calibrations',progress=[dict(width=b['width'],pid=p.pid,exit_code=p.poll(),last_line=tail(Path(b['root'])/'train_refinement.log')) for b,p in processes]);time.sleep(30)
  for f in streams:f.close()
  assert all(p.returncode==0 for _,p in processes),'calibration failed'
  for b,_ in processes:
   rec=json.loads(Path(b['selection']).read_text());assert rec['epochs_completed']==100 and rec['optimizer_updates']==6600 and rec['selection_metric']=='mae'
  directory=Path(w['ensemble_root'])
  job(root,'materialize',[sys.executable,'-u',str(directory/'materialize.py')],directory,directory/'materialize.log')
  job(root,'twenty_mask_repair_evaluation',[sys.executable,'-u',str(directory/'evaluate_repair.py'),'--selection','mae'],directory,directory/'evaluate_repair.log')
  metrics=report(root,w);freeze(root)
  write(root/'status.json',dict(status='completed',forecast_deferred=True,metrics=metrics,time_utc=datetime.now(timezone.utc).isoformat()))
 write(Q/'status.json',dict(status='completed',forecast_deferred=True,scene_roots=plan['scene_roots'],time_utc=datetime.now(timezone.utc).isoformat()))
if __name__=='__main__':
 try:main()
 except BaseException as error:
  write(Q/'status.json',dict(status='failed',error=repr(error),traceback=traceback.format_exc(),time_utc=datetime.now(timezone.utc).isoformat()));raise
