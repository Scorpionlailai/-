"""Fixed r3 checkpoint: validation-only forecasting from each repair prefix."""
import json,hashlib
from pathlib import Path
import torch
from integrations.e23.config import load_e23_p4_protocol,load_e23_settings
from integrations.e23.train import _new_memory,_key_sha256,_starts,build_forecast_model,load_stage2_selection
from integrations.forecast_compare.protocol import iter_p4_batches
from malr.migration import MigrationCatalog
from malr.data.patching import patchify_with_state
from malr.models.explicit_repair import build_explicit_repair

torch.set_num_threads(4)
root=Path('artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/e23_v29_iterative_tv_h_r3')
manifest=root/'stage2/runs/e23-stage2.json';record=json.loads(manifest.read_text());device=torch.device('cuda:1')
protocol=load_e23_p4_protocol(catalog=MigrationCatalog.load('../E22/migration/paper_catalog.jsonl',project='E23'),protocol_config='configs/stage1_seq96_dynamic_temporal_repair.yaml',mask_artifacts='../E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context')
settings=load_e23_settings('configs/e23_multi_expert_p4_width64_iterative_tv_h_r3_v29.yaml',protocol=protocol)
payload=load_stage2_selection(manifest,protocol,settings,device=device)
model=build_forecast_model(protocol,settings,device=device);model.load_state_dict(payload['model_state'],strict=True)
model.repair.historical.retrieval.load_state_dict(payload['retrieval_state']);memory=_new_memory(protocol);memory.load_state_dict(payload['memory_state']);model.repair.historical.refresh_snapshot(memory.snapshot());model.eval()
key_before=_key_sha256(model.repair.historical.retrieval.state_dict());bank_before=memory.snapshot_hash()
squared=[0.]*3;absolute=[0.]*3;count=windows=0
with torch.inference_mode():
 for cpu in iter_p4_batches(protocol,'validation',batch_size=settings.batch_size):
  batch=cpu.to(device);repair=model.repair;starts=_starts(batch,stride=repair.stride,patches=repair.patch_count,device=device)
  full,out=model(batch.normalized_observed,batch.observed_mask,batch.query_mean,batch.query_std,starts)
  patches=patchify_with_state(batch.normalized_observed,batch.observed_mask,repair.patch_length,repair.stride)
  original=patches.values.masked_fill(~patches.effective_mask,0)
  for i,item in enumerate(out.iterations):
   explicit=build_explicit_repair(repair.decoder(item.latents),original,patches.effective_mask,patches.validity,batch.query_mean,batch.query_std,window_len=repair.window_length,patch_len=repair.patch_length,stride=repair.stride)
   forecast=model.predictor(explicit.repaired_patches.masked_fill(~explicit.validity_patches,0),explicit.observation_patches,batch.query_mean,batch.query_std,patch_validity=explicit.validity_patches)
   if i==2:torch.testing.assert_close(forecast,full,rtol=0,atol=0)
   error=forecast-batch.forecast_target;squared[i]+=error.square().sum().item();absolute[i]+=error.abs().sum().item()
  count+=full.numel();windows+=full.shape[0]
key_after=_key_sha256(model.repair.historical.retrieval.state_dict());bank_after=memory.snapshot_hash();assert key_before==key_after and bank_before==bank_after
results=[dict(prefix_rounds=i+1,forecast_mse=squared[i]/count,forecast_mae=absolute[i]/count) for i in range(3)]
assert abs(results[-1]['forecast_mse']-record['best_val_mse'])<1e-6 and abs(results[-1]['forecast_mae']-record['best_val_mae'])<1e-6
result=dict(kind='fixed_checkpoint_validation_prefix_diagnostic',split='original_fixed_validation',metric_space='train_standardized',weights_trained_for_rounds=3,weights_selected_epoch=record['best_epoch'],epochs_completed=record['epochs_completed'],checkpoint_sha256=record['checkpoint_sha256'],config_sha256=record['e23_config_sha256'],retraining=False,windows=windows,forecast_coordinates=count,h_key_sha256_before=key_before,h_key_sha256_after=key_after,h_bank_sha256_before=bank_before,h_bank_sha256_after=bank_after,results=results,diagnostic_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
with (root/'prefix_forecast_validation.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result))
