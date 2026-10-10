import json,os,sys,subprocess,hashlib,time
from pathlib import Path
R=Path(__file__).resolve().parent
p=json.loads((R/'plan.json').read_text());w=Path(p['work'])
env={**os.environ,'PYTHONPATH':f'{w}/src:{w}','PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}
def write(x): (R/'status.json').write_text(json.dumps(x,indent=2))
def run(args,log):
 with log.open('x') as f:
  child=subprocess.Popen(args,cwd=w,env=env,stdout=f,stderr=subprocess.STDOUT)
  while True:
   try: rc=child.wait(timeout=300);break
   except subprocess.TimeoutExpired: print(json.dumps({'event':'five_minute_check','pid':child.pid,'log':str(log),'tail':log.read_text().splitlines()[-2:]}),flush=True)
  if rc: raise RuntimeError(f'{log} exit {rc}')
try:
 for s in p['scenarios']:
  cell=R/s['name'];cell.mkdir(exist_ok=True);env.pop('MALR_MISSING_SCENARIO',None)
  if s['scenario_manifest']:env['MALR_MISSING_SCENARIO']=s['scenario_manifest']
  write({'scene':s['name'],'stage':'training','status':'running'})
  args=[sys.executable,'-m','integrations.e23.run','train-forecast','--protocol-config',p['protocol_config'],'--migration-catalog','/home/xyw/paper/E22/migration/paper_catalog.jsonl','--mask-artifacts',p['mask_artifacts'],'--e23-config',p['e23_config'],'--training-seed','2023','--device','cuda:0','--stage1-manifest',s['stage1_manifest'],'--selection-metric','mae','--forecast-loss','mse','--repair-retention','none','--repair-update-policy','frozen_v1','--forecast-padding','zero_invalid_v1','--extra-mse-checkpoint','--output',str(cell/'stage2')]
  (cell/'command.json').write_text(json.dumps(args,indent=2));
  if not (cell/'stage2/runs/e23-stage2.json').exists(): run(args,cell/'stage2.log')
  for metric in ['mae','mse']:
   selection=cell/'stage2/runs'/('e23-stage2.json' if metric=='mae' else 'e23-stage2-mse.json');record=json.loads(selection.read_text())
   assert record['repair_trainable_parameters']==0 and record['repair_update_policy']=='frozen_v1' and record['loss']=='forecast_mse_only' and record['epochs_completed']==100
   assert record['predictor_initial_sha256']==p['predictor_initial_sha256']
   write({'scene':s['name'],'stage':'evaluation_'+metric,'status':'running'})
   run([sys.executable,str(w/'scripts/evaluate_e23_t1_replicate.py'),'--model','E23-v31-B','--model-label','E23-v97-frozen-'+metric,'--training-seed','2023','--e23-config',p['e23_config'],'--selection',str(selection),'--test-mask-bank',s['test_bank'],'--output',str(cell/('evaluation_'+metric)),'--device','cuda:0'],cell/('evaluation_'+metric+'.log'))
  print(json.dumps({'scene':s['name'],'result':json.loads((cell/'evaluation_mae/twenty-test-draws.json').read_text())['test_mean']}),flush=True)
 write({'status':'completed','scenes':4})
except Exception as e:
 write({'status':'failed','error':repr(e)});raise
