"""Frozen v103 Stage2 plus both locked validation selections on 20 masks."""
from time import perf_counter
ENTRY_STARTED=perf_counter()
from pathlib import Path
import sys,json,os,importlib.util,hashlib
R=Path(__file__).resolve().parent
W=Path('/home/xyw/.cache/codex-worktrees/e23-v103-dynamic-candidates-20261008')
sys.path[:0]=[str(W/'src'),str(W)]
os.environ['MALR_E23_CODE_FREEZE']=str(R/'frozen_source.json')
import torch
from integrations.e23.config import load_e23_settings
from integrations.e23.train import train_forecast,load_stage1_selection,build_forecast_model
from integrations.e23.forecast_sensitive import parameter_hash
from integrations.forecast_compare.resources import TrainingResources
spec=importlib.util.spec_from_file_location('readonly_weather_data','/home/xyw/paper/E24/integrations/e24/data.py')
data=importlib.util.module_from_spec(spec);sys.modules[spec.name]=data;spec.loader.exec_module(data)
from evaluate_forecast import evaluate
MANIFEST=Path('/home/xyw/paper/E24/artifacts/Weather/seq96_pred96/sr010_bl8/shared_protocol_v1/dataset_manifest.json')
CONFIG=R/'weather_v103_v79_dynamic.yaml'
BASE=Path(json.loads((R/'training_job.json').read_text())['output']).parent

def freeze_check():
 d=json.loads((R/'frozen_source.json').read_text())
 for name,sha in d['files'].items():
  if name.startswith('experiment/'):path=R/name[len('experiment/'):]
  elif name=='readonly_external/e24_data.py':path=Path('/home/xyw/paper/E24/integrations/e24/data.py')
  else:path=W/name
  assert hashlib.sha256(path.read_bytes()).hexdigest()==sha,str(path)

def protocol_settings():
 identity=json.loads(MANIFEST.read_text());p=data.load_dataset_protocol(MANIFEST,config_path=Path(identity['config_path']),artifacts=MANIFEST.parent)
 return p,load_e23_settings(CONFIG,protocol=p,allow_variable_scaling=True)

def cost_report():
 rows=[]
 for stage in ('stage1','stage2'):
  records=[(p,json.loads(p.read_text())) for p in (BASE/stage/'resources').glob('*.json')]
  records=[(p,r) for p,r in records if r['status']=='completed'];assert len(records)==1
  p,r=records[0];rows.append(dict(stage=stage,resource_record=str(p),full_workflow_seconds=r['full_workflow_seconds'],peak_allocated_gib=r['peak_allocated_bytes']/2**30,peak_reserved_gib=r['peak_reserved_bytes']/2**30))
 result=dict(stages=rows,complete_training_seconds=sum(r['full_workflow_seconds'] for r in rows),peak_allocated_gib=max(r['peak_allocated_gib'] for r in rows),peak_reserved_gib=max(r['peak_reserved_gib'] for r in rows),scope='sum of two complete commands; not one uninterrupted cold run; final tests excluded')
 (BASE/'complete_training_cost.json').write_text(json.dumps(result,indent=2))

def main():
 freeze_check();torch.set_num_threads(4)
 if '--preflight' in sys.argv:
  p,s=protocol_settings();payload=load_stage1_selection(BASE/'stage1/runs/e23-stage1.json',p,s,device=torch.device('cpu'))
  torch.manual_seed(2023);model=build_forecast_model(p,s,device=torch.device('cpu'))
  assert parameter_hash(model.predictor)=='fbbe8cde92c6417f1b5f67db888e20437dae434709b9e2adc834ce91118f6561'
  model.repair.load_state_dict(payload['model_state'],strict=True)
  print('Stage1 identity/checkpoint verified; original predictor initialization verified',flush=True);return
 device=torch.device('cuda:0');output=BASE/'stage2'
 with TrainingResources(output,device=device,batch_size=128,model='E23-v103-Weather',started_at=ENTRY_STARTED) as resources:
  with resources.stage('data_preparation'):
   protocol,settings=protocol_settings()
  with resources.stage('forecast_training'):
   selection=train_forecast(protocol,settings,stage1_manifest=BASE/'stage1/runs/e23-stage1.json',output=output,device=device,epochs=100,selection_metric='mae',forecast_loss='mse',repair_update_policy='frozen_v1',extra_mse_checkpoint=True,cache_fixed_validation=True)
   resources.selection_manifest=selection
 cost_report()
 evaluate(MANIFEST,CONFIG,selection,BASE/'evaluation_mae',device)
 evaluate(MANIFEST,CONFIG,output/'runs/e23-stage2-mse.json',BASE/'evaluation_mse',device)
 (R/'prediction_completed.json').write_text(json.dumps(dict(status='completed',primary=str(BASE/'evaluation_mae/twenty-test-draws.json'),secondary=str(BASE/'evaluation_mse/twenty-test-draws.json')),indent=2))
if __name__=='__main__':main()
