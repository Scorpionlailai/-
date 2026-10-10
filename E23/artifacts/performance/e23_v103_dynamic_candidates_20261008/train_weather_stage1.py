"""Isolated Weather Stage1 dynamic-V trial; does not import E24 model code."""
from time import perf_counter
ENTRY_STARTED=perf_counter()
import sys,json,os,importlib.util
from pathlib import Path
W=Path('/home/xyw/.cache/codex-worktrees/e23-v103-dynamic-candidates-20261008')
R=Path(__file__).resolve().parent
sys.path[:0]=[str(W/'src'),str(W)]
import torch
from integrations.e23.config import load_e23_settings
from integrations.e23.train import train_repair
from integrations.forecast_compare.resources import TrainingResources

def main():
 torch.set_num_threads(4);device=torch.device('cuda:0')
 output=Path('/home/xyw/paper/E23/artifacts/Weather/seq96_pred96/sr010_bl8/e23_v103_dynamic_candidates_20261008/seed_2023/E23-v103/stage1')
 os.environ['MALR_E23_CODE_FREEZE']=str(R/'frozen_source.json')
 with TrainingResources(output,device=device,batch_size=128,model='E23-v103-dynamic-V',started_at=ENTRY_STARTED) as resources:
  with resources.stage('data_preparation'):
   spec=importlib.util.spec_from_file_location('readonly_e24_data','/home/xyw/paper/E24/integrations/e24/data.py')
   module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
   manifest=Path('/home/xyw/paper/E24/artifacts/Weather/seq96_pred96/sr010_bl8/shared_protocol_v1/dataset_manifest.json')
   identity=json.loads(manifest.read_text())
   protocol=module.load_dataset_protocol(manifest,config_path=Path(identity['config_path']),artifacts=manifest.parent)
   settings=load_e23_settings(R/'weather_v103_v79_dynamic.yaml',protocol=protocol,allow_variable_scaling=True)
   assert settings.variable_count==21 and settings.variable_topk==16
  with resources.stage('repair_training'):
   selection=train_repair(protocol,settings,output=output,device=device,epochs=100)
   resources.selection_manifest=selection
 print(json.dumps({'selection':str(selection),'resources':str(resources.report_path)}),flush=True)
 from evaluate_weather_stage1 import evaluate
 evaluate(protocol,settings,selection,output.parent/'stage1_repair_20_masks.json',device)
if __name__=='__main__':main()
