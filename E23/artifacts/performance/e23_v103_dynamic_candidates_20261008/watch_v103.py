"""Read-only experiment monitor; never changes training code or model state."""
from pathlib import Path
import json,csv,time,datetime,os,tempfile
R=Path(__file__).resolve().parent
job=json.loads((R/'training_job.json').read_text())
stage1=Path(job['output']);baseline=Path(job['baseline']);fixed_eval=baseline.parents[2]/'evaluation_mae/twenty-test-draws.json'
T1=Path('/home/xyw/paper/E24/artifacts/Weather/seq96_pred96/sr010_bl8/weather_author_comparison_20261008/T1')
fixed=json.loads(baseline.read_text());fixed_test=json.loads(fixed_eval.read_text());t1=json.loads((T1/'stage1/selection.json').read_text());t1_test=json.loads((T1/'evaluation/mae/twenty-test-draws.json').read_text())
assert fixed_test['repair_frozen_verified'] and t1_test['repair_frozen_verified']
assert fixed_test['dataset_manifest_sha256']==t1_test['dataset_manifest_sha256']
assert [r['mask_sha256'] for r in fixed_test['draws']]==[r['mask_sha256'] for r in t1_test['draws']]

def read(path):
 return json.loads(path.read_text()) if path.exists() else None

def atomic(path,value):
 fd,name=tempfile.mkstemp(dir=path.parent,prefix=path.name+'.tmp-')
 with os.fdopen(fd,'w') as f:json.dump(value,f,indent=2)
 os.replace(name,path)

def resource(directory):
 reports=[read(p) for p in (directory/'resources').glob('*.json')]
 reports=[r for r in reports if r and r.get('status')=='completed']
 if not reports:return {}
 assert len(reports)==1
 r=reports[0]
 return dict(stage1_full_seconds=r['full_workflow_seconds'],stage1_peak_allocated_gib=r['peak_allocated_bytes']/2**30,stage1_peak_reserved_gib=r['peak_reserved_bytes']/2**30,stage1_device=r['device']['requested'])

def row(label,selection,test,seed,batch,status):
 metrics=selection.get('validation_repair_metrics',selection.get('best_validation',{}))
 result=dict(model=label,status=status,train_seed=seed,batch_size=batch,stage1_epochs=selection.get('epochs_completed'),best_epoch=selection.get('best_epoch'),validation_repair96_mse=metrics.get('global_mse',metrics.get('repair96_mse')),validation_repair96_mae=metrics.get('global_mae',metrics.get('repair96_mae')),test_masks=20 if test else '',forecast_evaluated=False if label=='E23-v103-dynamic-V' else 'existing separate Stage2',test_scope='existing development masks; not blind test')
 if test:
  for key in ('repair96_mse','repair96_mae','repair32_mse','repair32_mae'):
   result[key+'_mean']=test['summary'][key]['mean'];result[key+'_mask_std_ddof1']=test['summary'][key]['mask_std_ddof1']
 return result

def update():
 history=read(stage1/'runs/e23-stage1-history.json');epochs=history['epochs'] if history else []
 best=min(epochs,key=lambda e:e['metrics']['global_mse']) if epochs else None
 selection=read(stage1/'runs/e23-stage1.json');test=read(stage1.parent/'stage1_repair_20_masks.json')
 proc=Path('/proc')/str(job['pid']);alive=proc.exists() and (proc/'stat').read_text().split()[2]!='Z'
 if test:status='completed'
 elif alive:status='evaluating' if selection else 'training'
 else:status='interrupted_or_failed'
 if test:
  assert [r['mask_identity']['sha256'] for r in test['rows']]==[r['mask_sha256'] for r in fixed_test['draws']]
  for n,b,t in zip(test['rows'],fixed_test['draws'],t1_test['draws']):
   assert n['windows']==b['windows']==t['windows']
   assert n['hidden96_coordinates']==b['hidden96_coordinates']==t['hidden96_coordinates']
   assert n['hidden32_coordinates']==b['hidden32_coordinates']==t['hidden32_coordinates']
  assert test['repair_state_unchanged']
 snapshot=dict(checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=job['pid'],status=status,completed_epochs=len(epochs),best_validation_epoch=best['epoch'] if best else None,best_validation_metrics=best['metrics'] if best else None,selection=str(stage1/'runs/e23-stage1.json') if selection else None,repair_evaluation=str(stage1.parent/'stage1_repair_20_masks.json') if test else None)
 atomic(R/'monitor_status.json',snapshot)
 b=row('Weather-v79-fixed4',fixed,fixed_test,2023,128,'completed');b.update(resource(baseline.parents[1]))
 t=row('T1',t1,t1_test,t1['profile']['seed'],t1['profile']['training']['batch_size'],'completed');t.update(resource(T1/'stage1'))
 sel=selection or dict(epochs_completed=len(epochs),best_epoch=best['epoch'] if best else None,validation_repair_metrics=best['metrics'] if best else {})
 n=row('E23-v103-dynamic-V',sel,test,2023,128,status);n.update(resource(stage1))
 rows=[b,t,n];keys=list(dict.fromkeys(k for r in rows for k in r));temporary=R/'stage1_comparison.csv.tmp'
 with temporary.open('w') as f:
  writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
 os.replace(temporary,R/'stage1_comparison.csv')
 if epochs:
  old=read(baseline.parent/'e23-stage1-history.json')['epochs'];lookup={e['epoch']:e['metrics'] for e in old};matched=[]
  for e in epochs:
   a=e['metrics'];c=lookup[e['epoch']];matched.append(dict(epoch=e['epoch'],fixed4_validation_mse=c['global_mse'],dynamic_validation_mse=a['global_mse'],dynamic_minus_fixed4=a['global_mse']-c['global_mse'],dynamic_last32_mse=a['last32_global_mse'],dynamic_source_count_mean=a['variable_source_count_mean']))
  temporary=R/'matched_epoch_validation.csv.tmp'
  with temporary.open('w') as f:
   writer=csv.DictWriter(f,fieldnames=list(matched[0]));writer.writeheader();writer.writerows(matched)
  os.replace(temporary,R/'matched_epoch_validation.csv')
 print(json.dumps(snapshot),flush=True)
 return status

if __name__=='__main__':
 import sys
 while True:
  status=update()
  if '--once' in sys.argv or status in ('completed','interrupted_or_failed'):break
  time.sleep(300)
