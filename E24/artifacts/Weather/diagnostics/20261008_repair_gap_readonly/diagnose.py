import json, sys, csv
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/home/xyw/paper/E24')
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from integrations.e24_sr015.data import read_manifest,load_dataset_protocol,with_test_mask
from integrations.e23.config import load_e23_settings
from integrations.e23.train import build_repair_model,load_stage1_selection,_new_memory,_starts
from integrations.forecast_compare.protocol import iter_p4_batches
from integrations.forecast_compare.t1_p4 import t1_inputs,t1_clean_history
from integrations.weather_baselines_sr015.models import build_author

torch.set_num_threads(4)
device=torch.device('cuda:1')
base=ROOT/'artifacts/Weather/seq96_pred96/sr015_bl8'
manifest=base/'shared_protocol_v1/dataset_manifest.json'
record=read_manifest(manifest)
protocol=load_dataset_protocol(manifest,config_path=Path(record['config_path']),artifacts=manifest.parent)
settings=load_e23_settings(ROOT/'configs/e24_v79.yaml',protocol=protocol)
selection=next(base.glob('e24_v79*/seed_2023/E24-v79/stage1/runs/e23-stage1.json'))
payload=load_stage1_selection(selection,protocol,settings,device=device)
ours=build_repair_model(protocol,settings,device=device)
ours.load_state_dict(payload['model_state'],strict=True)
ours.historical.retrieval.load_state_dict(payload['retrieval_state'])
memory=_new_memory(protocol);memory.load_state_dict(payload['memory_state'])
ours.historical.refresh_snapshot(memory.snapshot());ours.eval();del payload
sel=json.loads((base/'weather_author_comparison_sr015_20261008/SAITS/stage1/selection.json').read_text())
saits=build_author('SAITS',sel['profile'],protocol,device)
payload=torch.load(sel['checkpoint'],map_location=device,weights_only=False)
saits.load_state_dict(payload['model_state'],strict=True);saits.eval();del payload
item=next(x for x in record['masks'] if x['split']=='test')
protocol=with_test_mask(protocol,record,item)
acc={name:{length:{'sq':np.zeros(21),'abs':np.zeros(21),'n':np.zeros(21,dtype=np.int64)} for length in (96,32)} for name in ('ours','SAITS')}
abs_samples={name:[] for name in acc}
with torch.inference_mode():
 for i,cpu in enumerate(iter_p4_batches(protocol,'test',batch_size=64)):
  batch=cpu.to(device);values,mask=t1_inputs(batch);target=t1_clean_history(batch)
  out=ours(batch.normalized_observed,batch.observed_mask,batch.query_mean,batch.query_std,_starts(batch,stride=ours.stride,patches=ours.patch_count,device=device))
  candidates={'ours':out.explicit.normalized_sequence*batch.query_std[:,None]+batch.query_mean[:,None],'SAITS':saits(values,mask=mask)}
  for name,cand in candidates.items():
   error=(cand-target).double();hidden=~batch.observed_mask
   abs_samples[name].append(error.abs()[hidden].float().cpu().numpy())
   for length in (96,32):
    e=error[:,-length:];h=hidden[:,-length:]
    a=acc[name][length];a['sq']+=(e.square()*h).sum((0,1)).cpu().numpy();a['abs']+=(e.abs()*h).sum((0,1)).cpu().numpy();a['n']+=h.sum((0,1)).cpu().numpy()
  if i%40==0:print('batch',i,flush=True)
result={'scope':'Read-only fixed selected Stage1 checkpoints; first predeclared test mask only, no updates. Not a replacement for the formal twenty-mask results.','seed':item['seed'],'channels':record['columns'],'models':{}}
for name,groups in acc.items():
 rows={}
 for length,a in groups.items():
  rows[str(length)]={'mse':(a['sq']/a['n']).tolist(),'mae':(a['abs']/a['n']).tolist(),'counts':a['n'].tolist(),'mse_total':float(a['sq'].sum()/a['n'].sum()),'mae_total':float(a['abs'].sum()/a['n'].sum()),'mse_contribution':(a['sq']/a['sq'].sum()).tolist()}
 samples=np.concatenate(abs_samples[name]);rows['absolute_error_quantiles']=dict(zip(['p50','p90','p95','p99','p999'],np.quantile(samples,[.5,.9,.95,.99,.999]).tolist()))
 result['models'][name]=rows
Path(__file__).with_name('single_mask_result.json').write_text(json.dumps(result,indent=2))
for j,c in enumerate(record['columns']):
 print(c,*(round(result['models'][n]['96'][k][j],6) for n in ('ours','SAITS') for k in ('mse','mae','mse_contribution')),flush=True)
for n in result['models']:print(n,result['models'][n]['absolute_error_quantiles'],result['models'][n]['96']['mse_total'],result['models'][n]['96']['mae_total'])
