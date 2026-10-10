import json
from pathlib import Path
import torch
from integrations.e23.config import load_e23_p4_protocol,load_e23_settings
from integrations.e23.train import build_forecast_model,load_stage2_selection,_new_memory,_starts
from integrations.forecast_compare.protocol import iter_p4_batches
from malr.migration import MigrationCatalog
r=Path.cwd();p=load_e23_p4_protocol(catalog=MigrationCatalog.load(r.parent/'E22/migration/paper_catalog.jsonl',project='E23'),protocol_config=r/'configs/stage1_seq96_dynamic_temporal_repair.yaml',mask_artifacts=r.parent/'E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context');s=load_e23_settings(r/'configs/e23_multi_expert_p4_width64_window_v12.yaml',protocol=p)
device=torch.device('cuda:0');m=r/'artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/e23_v12_late_horizon_v14/stage2/runs/e23-stage2.json'
payload=load_stage2_selection(m,p,s,device=device);model=build_forecast_model(p,s,device=device);model.load_state_dict(payload['model_state'],strict=True);model.repair.historical.retrieval.load_state_dict(payload['retrieval_state']);memory=_new_memory(p);memory.load_state_dict(payload['memory_state']);model.repair.historical.refresh_snapshot(memory.snapshot());model.eval()
channel_abs=torch.zeros(7,device=device);channel_sq=torch.zeros(7,device=device)
horizon_abs=torch.zeros(4,device=device);horizon_sq=torch.zeros(4,device=device)
count=0;bucket_abs=torch.zeros(4,device=device);bucket_sq=torch.zeros(4,device=device);bucket_count=torch.zeros(4,device=device)
with torch.inference_mode():
 for cpu in iter_p4_batches(p,'validation',batch_size=s.batch_size):
  b=cpu.to(device);starts=_starts(b,stride=model.repair.stride,patches=model.repair.patch_count,device=device)
  y,_=model(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts)
  e=(y-b.forecast_target).abs();channel_abs+=e.sum((0,2));channel_sq+=e.square().sum((0,2));count+=y.shape[0]
  for i in range(4):horizon_abs[i]+=e[:,:,i*24:(i+1)*24].sum();horizon_sq[i]+=e[:,:,i*24:(i+1)*24].square().sum()
  frac=b.observed_mask.float().mean(1);bucket=torch.clamp((frac*4).long(),max=3)
  for i in range(4):
   selected=bucket==i;bucket_abs[i]+=e.sum(-1)[selected].sum();bucket_sq[i]+=e.square().sum(-1)[selected].sum();bucket_count[i]+=selected.sum()*96
out={'split':'validation','windows':count,'channel_mae':(channel_abs/(count*96)).cpu().tolist(),'channel_mse':(channel_sq/(count*96)).cpu().tolist(),'horizon_24_mae':(horizon_abs/(count*7*24)).cpu().tolist(),'horizon_24_mse':(horizon_sq/(count*7*24)).cpu().tolist(),'observation_fraction_buckets':['[0,.25)','[.25,.5)','[.5,.75)','[.75,1]'],'bucket_count':bucket_count.cpu().tolist(),'bucket_mae':(bucket_abs/bucket_count.clamp_min(1)).cpu().tolist(),'bucket_mse':(bucket_sq/bucket_count.clamp_min(1)).cpu().tolist()}
path=r/'artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/e23_v12_late_horizon_v14/benchmarks/v14_validation_error_map.json';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
