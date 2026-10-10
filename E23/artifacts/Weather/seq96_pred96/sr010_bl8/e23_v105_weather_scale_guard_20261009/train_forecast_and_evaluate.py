from time import perf_counter
ENTRY_STARTED=perf_counter()
from weather_common import ROOT,protocol_for
import json,torch
from integrations.e23.config import load_e23_settings
from integrations.e23.train import train_forecast
from integrations.forecast_compare.resources import TrainingResources
from evaluate_forecast import evaluate

def main():
 torch.set_num_threads(4);plan=json.loads((ROOT/'plan.json').read_text());protocol=protocol_for(plan,plan['scenarios'][0]);device=torch.device('cuda:0')
 settings=load_e23_settings(ROOT/'v105_weather.yaml',protocol=protocol,allow_variable_scaling=True)
 base=ROOT/'E23-v105/sr010_bl8';output=base/'stage2'
 with TrainingResources(output,device=device,batch_size=128,model='E23-v105-Weather',started_at=ENTRY_STARTED) as resources:
  with resources.stage('forecast_training'):
   selection=train_forecast(protocol,settings,stage1_manifest=base/'stage1/runs/e23-stage1.json',output=output,device=device,epochs=100,selection_metric='mae',forecast_loss='mse',repair_update_policy='frozen_v1',extra_mse_checkpoint=True,cache_fixed_validation=True)
   resources.selection_manifest=selection
 evaluate(plan['dataset_manifest'],ROOT/'v105_weather.yaml',selection,base/'evaluation_mae',device)
 evaluate(plan['dataset_manifest'],ROOT/'v105_weather.yaml',output/'runs/e23-stage2-mse.json',base/'evaluation_mse',device)
if __name__=='__main__':main()
