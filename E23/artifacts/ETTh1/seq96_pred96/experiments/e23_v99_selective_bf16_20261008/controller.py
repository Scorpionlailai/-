import os,sys,json,subprocess,hashlib,tarfile,time
from pathlib import Path
R=Path(__file__).resolve().parent;p=json.loads((R/'plan.json').read_text());w=Path(p['work']);s=next(s for s in p['scenarios'] if s['name']==p['first_scene']);cell=R/'E23-v99'/s['name'];cell.mkdir(parents=True);env={**os.environ,'PYTHONPATH':f'{w}/src:{w}','PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','TORCHINDUCTOR_CACHE_DIR':str(R/'inductor_cache')};env['MALR_MISSING_SCENARIO']=s['scenario_manifest']
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
sources={str(f):sha(f) for folder in ['src','integrations','scripts','configs'] for f in (w/folder).rglob('*') if f.is_file() and f.suffix in ['.py','.yaml','.json']}
for f in ['plan.json','controller.py','preflight.py','train_refinement.py','evaluate_repair.py','coarse_config.yaml','refinement_config.yaml']:sources[str(R/f)]=sha(R/f)
(R/'frozen-definition.json').write_text(json.dumps(dict(sources=sources),indent=2))
with tarfile.open(R/'frozen-source-and-runner.tar.gz','w:gz') as archive:
 for f in sources:archive.add(f,arcname=str(Path(f)).lstrip('/'),recursive=False)
def verify():assert all(sha(f)==h for f,h in sources.items())
def run(args,log,stage):
 verify();log.parent.mkdir(parents=True,exist_ok=True);(R/'status.json').write_text(json.dumps(dict(status='running',stage=stage,scene=s['name']),indent=2));log.with_suffix('.command.json').write_text(json.dumps(args,indent=2));started=time.perf_counter()
 with log.open('x') as f:
  child=subprocess.Popen(args,cwd=w,env=env,stdout=f,stderr=subprocess.STDOUT)
  while True:
   try:rc=child.wait(timeout=300);break
   except subprocess.TimeoutExpired:print(json.dumps(dict(event='five_minute_check',stage=stage,tail=log.read_text().splitlines()[-2:])),flush=True);verify()
 if rc:raise RuntimeError(f'{stage} failed exit {rc}: {log}')
 verify();print(json.dumps(dict(stage=stage,command_seconds=time.perf_counter()-started)),flush=True)
base=[sys.executable,'-m','integrations.e23.run']
shared=['--protocol-config',p['protocol_config'],'--migration-catalog','/home/xyw/paper/E22/migration/paper_catalog.jsonl','--mask-artifacts',p['mask_artifacts'],'--training-seed','2023','--device','cuda:0']
try:
 coarse=Path(p['parent_root'])/s['name']/'E23-v99-coarse'
 run(base+['train-repair']+shared+['--e23-config',p['parent_settings'],'--auxiliary-forecast','none','--repair-compute-precision','selective_bf16_v1','--compile-repair-readout','--repair-batch-size','128','--output',str(coarse/'stage1')],coarse/'stage1.log','coarse_stage1')
 run([sys.executable,str(R/'train_refinement.py'),s['name']],cell/'stage1.log','refinement_stage1')
 # Same official T1 checkpoint/masks; reuse its verified read-only repair audit.
 original=Path('/home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_stage1_mean_and_dense_20261008')
 cached=R/'T1'/s['name']/'repair-audit.json';cached.parent.mkdir(parents=True);cached.write_bytes((original/'T1'/s['name']/'repair-audit.json').read_bytes())
 run([sys.executable,str(R/'evaluate_repair.py'),'E23-v99',s['name']],cell/'repair-audit.log','repair_evaluation')
 run(base+['train-forecast']+shared+['--e23-config',p['models'][0]['config'],'--stage1-manifest',str(cell/'stage1/runs/e23-stage1.json'),'--selection-metric','mae','--forecast-loss','mse','--repair-retention','none','--repair-update-policy','frozen_v1','--forecast-padding','zero_invalid_v1','--extra-mse-checkpoint','--cache-fixed-validation','--output',str(cell/'stage2')],cell/'stage2.log','frozen_stage2')
 for metric in ['mae','mse']:
  selection=cell/'stage2/runs'/('e23-stage2.json' if metric=='mae' else 'e23-stage2-mse.json');d=json.loads(selection.read_text());assert d['repair_trainable_parameters']==0 and d['epochs_completed']==100 and d['loss']=='forecast_mse_only' and d['predictor_initial_sha256']==p['predictor_initial_sha256']
  run([sys.executable,str(w/'scripts/evaluate_e23_t1_replicate.py'),'--model','E23-v31-B','--model-label','E23-v99-selective-bf16-'+metric,'--training-seed','2023','--e23-config',p['models'][0]['config'],'--selection',str(selection),'--test-mask-bank',s['test_bank'],'--output',str(cell/('evaluation_'+metric)),'--device','cuda:0'],cell/('evaluation_'+metric+'.log'),'forecast_evaluation_'+metric)
 (R/'status.json').write_text(json.dumps(dict(status='pilot_completed',scene=s['name']),indent=2))
except Exception as e:
 (R/'status.json').write_text(json.dumps(dict(status='failed',error=repr(e)),indent=2));raise
