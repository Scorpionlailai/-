"""Sequential GPU0 repair trials; inspect active training every 300 seconds."""
from pathlib import Path
import ast,csv,hashlib,json,os,subprocess,sys,tarfile,time
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent
P=json.loads((ROOT/'plan.json').read_text());W=Path(P['work'])
ENV={**os.environ,'PYTHONPATH':f'{W}/src:{W}','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','PYTHONUNBUFFERED':'1'}
ENV.pop('MALR_MISSING_SCENARIO',None)
STATE={'status':'running','pid':os.getpid(),'device':P['device'],'goal':P['goal'],'active':None,'completed':[],'started_utc':datetime.now(timezone.utc).isoformat(),'monitor_interval_seconds':300}
DOCS=Path('/home/xyw/paper/E23/docs/experiments')

def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
SOURCES={str(p.relative_to(W)):sha(p) for folder in ['src','integrations','configs'] for p in (W/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.json','.yaml')}
def check():assert all(sha(W/k)==v for k,v in SOURCES.items()),'source changed during training'
def publish():
 STATE['last_checked_utc']=datetime.now(timezone.utc).isoformat();write(ROOT/'queue-state.json',STATE)
 rows=[]
 for entry in P['models']:
  for scene in P['scenarios']:
   cell=ROOT/entry['label']/scene['name'];sel=cell/'stage1/runs/e23-stage1.json';audit=cell/'repair-audit.json'
   row=dict(model=entry['label'],scene=scene['name'],status='queued',val_full_mse='',val_full_mae='',val_last32_mse='',val_last32_mae='',test_full_mse='',test_full_mae='',test_last32_mse='',test_last32_mae='',training_seconds='',peak_allocated_gib='',peak_reserved_gib='')
   if sel.exists():
    r=json.loads(sel.read_text());row.update(status='stage1_complete',val_full_mse=r['validation_repair_metrics']['global_mse'],val_full_mae=r['validation_repair_metrics']['global_mae'])
   if audit.exists():
    r=json.loads(audit.read_text());v=r['split_validation']['metrics'];t=r['test_mean'];row.update(status='repair_audited',val_last32_mse=v['last32_hidden']['mse'],val_last32_mae=v['last32_hidden']['mae'],test_full_mse=t['all_hidden']['mse'],test_full_mae=t['all_hidden']['mae'],test_last32_mse=t['last32_hidden']['mse'],test_last32_mae=t['last32_hidden']['mae'])
   if STATE['active'] and STATE['active'].get('model')==entry['label'] and STATE['active'].get('scene')==scene['name']:row['status']=STATE['active']['stage']
   for resource in (cell/'stage1/resources').glob('*.json'):
    r=json.loads(resource.read_text())
    if r.get('status')=='completed':row.update(training_seconds=r.get('full_workflow_seconds',''),peak_allocated_gib=r.get('peak_allocated_gib',''),peak_reserved_gib=r.get('peak_reserved_gib',''))
   rows.append(row)
 with (DOCS/'E23_stage1_repair_focus_gpu0_20261007.csv').open('w',newline='') as f:
  wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
 def f(x):return '—' if x=='' else f'{x:.6f}'
 lines=['# E23 Stage1修复迭代（GPU0）','',f'更新时间UTC：{STATE["last_checked_utc"]}；状态：{STATE["status"]}。Weather使用GPU1，本队列不修改E24。','',P['screening_policy'],'','Stage1固定100轮/seed2023/batch128。只按验证全局修复MSE选权重；20组旧测试仅作开发评估，未执行Stage2。','', '| 模型 | 场景 | 状态 | 验证全窗口 MSE/MAE | 验证末32 MSE/MAE | 测试20组全窗口 MSE/MAE | 测试20组末32 MSE/MAE |','|---|---|---|---:|---:|---:|---:|']
 for r in rows:lines.append(f'| {r["model"]} | {r["scene"]} | {r["status"]} | {f(r["val_full_mse"])} / {f(r["val_full_mae"])} | {f(r["val_last32_mse"])} / {f(r["val_last32_mae"])} | {f(r["test_full_mse"])} / {f(r["test_full_mae"])} | {f(r["test_last32_mse"])} / {f(r["test_last32_mae"])} |')
 lines+=['','资源详见同名CSV与stage1/resources；两张卡同时运行，CPU/磁盘可能共享，耗时不作同卡独占效率结论。']
 (DOCS/'E23_stage1_repair_focus_gpu0_20261007.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps({'time':STATE['last_checked_utc'],'status':STATE['status'],'active':STATE['active']}),flush=True)

def run(args,log,scene=None,entry=None,stage='preflight'):
 check();log=Path(log);log.parent.mkdir(parents=True,exist_ok=True);env=dict(ENV)
 if scene and scene['scenario_manifest']:env['MALR_MISSING_SCENARIO']=scene['scenario_manifest']
 STATE['active']=dict(model=entry['label'] if entry else None,scene=scene['name'] if scene else None,stage=stage,log=str(log));publish()
 if log.exists():
  log=log.with_name(log.stem+'.resume-'+str(os.getpid())+log.suffix)
  STATE['active']['log']=str(log)
 with log.open('x') as stream:
  child=subprocess.Popen(args,cwd=W,env=env,stdout=stream,stderr=subprocess.STDOUT);STATE['active']['child_pid']=child.pid;publish()
  while True:
   try:code=child.wait(timeout=300);break
   except subprocess.TimeoutExpired:
    with log.open() as f:tail=f.readlines()[-3:]
    STATE['active']['recent_log']=''.join(tail).strip();publish();check()
  if code:raise RuntimeError(f'failed exit={code}: {log}')
 check();publish()

def main():
 # Exact Stage2 training and head construction remain the v79 implementation.
 parent=Path('/home/xyw/paper/E23/versions/e23-v79-global-level-four/E23')
 for file,names in [('integrations/e23/train.py',['train_forecast','build_forecast_model','stage2_objective']),('src/malr/models/patchtst.py',[])]:
  if names:
   def functions(path):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
   a,b=functions(parent/file),functions(W/file);assert all(a[n]==b[n] for n in names)
  else:assert sha(parent/file)==sha(W/file)
 definition={'sources':SOURCES,'plan_sha256':sha(ROOT/'plan.json'),'stage2_functions_unchanged':True,'head_source_unchanged':True}
 if (ROOT/'frozen-definition.json').exists():
  assert json.loads((ROOT/'frozen-definition.json').read_text())==definition
 else:
  write(ROOT/'frozen-definition.json',definition)
  with tarfile.open(ROOT/'frozen-source.tar.gz','x:gz') as t:
   for rel in SOURCES:t.add(W/rel,arcname=rel,recursive=False)
 run([sys.executable,'-m','pytest','tests/integration/test_e23_global_dense.py','tests/integration/test_e23_stage1_mean_weight.py','-q'],ROOT/'preflight-tests-resumed.log')
 for entry in P['models']:
  results=[]
  for index,scene in enumerate(P['scenarios']):
   cell=ROOT/entry['label']/scene['name'];args=[sys.executable,'-m','integrations.e23.run','train-repair','--protocol-config',P['protocol_config'],'--migration-catalog','/home/xyw/paper/E22/migration/paper_catalog.jsonl','--mask-artifacts',P['mask_artifacts'],'--e23-config',entry['config'],'--training-seed','2023','--device',P['device'],'--auxiliary-forecast','none','--repair-batch-size','128','--output',str(cell/'stage1')]
   if not (cell/'stage1/runs/e23-stage1.json').exists():
    write(cell/'stage1.command.json',{'command':args,'device':P['device'],'concurrent_weather_gpu1':True});run(args,cell/'stage1.log',scene,entry,'stage1_training')
   if not (cell/'repair-audit.json').exists():
    run([sys.executable,str(ROOT/'evaluate_repair.py'),entry['label'],scene['name']],cell/'repair-audit.log',scene,entry,'repair_validation_and_test20')
   result=json.loads((cell/'repair-audit.json').read_text());results.append(result)
   STATE['completed'].append({'model':entry['label'],'scene':scene['name'],'beats_t1_validation':result['beats_t1_validation_all_four_metrics'],'beats_t1_test':result['beats_t1_test_all_four_metrics']});STATE['active']=None;publish()
   if index==0 and not result['screening_passed_v79_validation']:
    write(ROOT/entry['label']/'screening-decision.json',{'status':'screen_failed','rule':P['screening_policy'],'basis':'validation_only','later_scenes_not_run':True});break
  if len(results)==4 and all(r['beats_t1_validation_all_four_metrics'] and r['beats_t1_test_all_four_metrics'] for r in results):
   STATE['status']='repair_target_met_development_requires_fresh_confirmation';STATE['candidate']=entry['label'];publish();return
 STATE['status']='preset_trials_completed_need_next_design';STATE['active']=None;publish()

if __name__=='__main__':
 try:main()
 except Exception as e:STATE['status']='error';STATE['error']=repr(e);publish();raise
