import os,sys,json,subprocess,hashlib,tarfile,time
from pathlib import Path
R=Path(__file__).resolve().parent;p=json.loads((R/'plan.json').read_text());w=Path(p['work']);scene=next(s for s in p['scenarios'] if s['name']=='sr010_bl8');cell=R/scene['name'];cell.mkdir()
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
sources={str(f):sha(f) for folder in ['src','integrations','scripts','configs'] for f in (w/folder).rglob('*') if f.is_file() and f.suffix in ['.py','.yaml','.json']}
protected={scene['stage1_manifest']:sha(scene['stage1_manifest'])};(R/'frozen-definition.json').write_text(json.dumps(dict(sources=sources,protected=protected),indent=2))
with tarfile.open(R/'frozen-source.tar.gz','w:gz') as archive:
 for f in sources:archive.add(f,arcname='source/'+str(Path(f).relative_to(w)),recursive=False)
 for f in ['plan.json','controller.py','preflight.py']:archive.add(R/f,arcname='experiment/'+f)
args=[sys.executable,'-m','integrations.e23.run','train-forecast','--protocol-config',p['protocol_config'],'--migration-catalog','/home/xyw/paper/E22/migration/paper_catalog.jsonl','--mask-artifacts',p['mask_artifacts'],'--e23-config',p['e23_config'],'--training-seed','2023','--device','cuda:0','--stage1-manifest',scene['stage1_manifest'],'--selection-metric','mae','--forecast-loss','mse','--repair-retention','none','--repair-update-policy','frozen_v1','--forecast-padding','zero_invalid_v1','--extra-mse-checkpoint','--cache-fixed-validation','--output',str(cell/'stage2')]
(cell/'command.json').write_text(json.dumps(args,indent=2));env={**os.environ,'PYTHONPATH':f'{w}/src:{w}','PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'};env.pop('MALR_MISSING_SCENARIO',None)
(R/'status.json').write_text(json.dumps({'status':'running','scene':scene['name']}));started=time.perf_counter()
with (cell/'stage2.log').open('x') as log:
 child=subprocess.Popen(args,cwd=w,env=env,stdout=log,stderr=subprocess.STDOUT)
 while True:
  try:rc=child.wait(timeout=300);break
  except subprocess.TimeoutExpired:print(json.dumps({'event':'five_minute_check','tail':(cell/'stage2.log').read_text().splitlines()[-2:]}),flush=True)
 if rc:(R/'status.json').write_text(json.dumps({'status':'failed','exit_code':rc}));raise SystemExit(rc)
assert all(sha(f)==h for f,h in sources.items()) and all(sha(f)==h for f,h in protected.items())
# Checkpoint contents are compared, not serialized bytes with changed metadata.
import torch
old=Path(p['paired_original_root'])/scene['name'];choices={}
for metric in ['mae','mse']:
 name='e23-stage2.json' if metric=='mae' else 'e23-stage2-mse.json';a=json.loads((cell/'stage2/runs'/name).read_text());b=json.loads((old/'stage2/runs'/name).read_text());x=torch.load(a['checkpoint'],map_location='cpu',weights_only=False);y=torch.load(b['checkpoint'],map_location='cpu',weights_only=False)
 assert a['repair_trainable_parameters']==0 and a['loss']=='forecast_mse_only' and a['epochs_completed']==100
 assert a['predictor_initial_sha256']==b['predictor_initial_sha256']==p['predictor_initial_sha256']
 assert x['model_state'].keys()==y['model_state'].keys() and all(torch.equal(v,y['model_state'][k]) for k,v in x['model_state'].items())
 assert all(a[k]==b[k] for k in ['best_epoch','best_val_mse','best_val_mae','memory_sha256','key_sha256'])
 choices[metric]=dict(best_epoch=a['best_epoch'],mse=a['best_val_mse'],mae=a['best_val_mae'],entire_model_state_exact_equal=True,reused_test_result=str(old/('evaluation_'+metric)/'twenty-test-draws.json'))
a=json.loads(next((cell/'stage2/resources').glob('*.json')).read_text());b=json.loads(next((old/'stage2/resources').glob('*.json')).read_text());result=dict(status='passed',scene=scene['name'],choices=choices,cached_full_workflow_seconds=a['full_workflow_seconds'],original_full_workflow_seconds=b['full_workflow_seconds'],reduction_percent=100*(1-a['full_workflow_seconds']/b['full_workflow_seconds']),cache=a['selection_manifest'],controller_command_wall_seconds=time.perf_counter()-started,formal_test_rerun=False,test_reused_after_exact_state_verification=True)
(R/'acceptance.json').write_text(json.dumps(result,indent=2));(R/'status.json').write_text(json.dumps({'status':'completed','acceptance':'passed'}));print(json.dumps(result,indent=2))
