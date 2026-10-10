from pathlib import Path
from time import perf_counter
import json,torch
from scripts.diagnose_e23_v30_stack import load_protocol
from scripts.report_e23_v31_trial import CONFIG
from integrations.e23.config import load_e23_settings
from integrations.e23.train import build_forecast_model,_starts,_key_sha256,load_stage1_selection,_new_memory
from integrations.forecast_compare.protocol import iter_p4_batches
from integrations.e23.forecast_sensitive import snapshot_teacher,region_oracle_gains,gains_to_weights,weighted_repair_loss
root=Path(__file__).resolve().parent;torch.set_num_threads(4);device=torch.device('cuda:1')
protocol=load_protocol();settings=load_e23_settings(CONFIG,protocol=protocol)
batch=next(iter_p4_batches(protocol,'train',batch_size=128,epoch=0)).to(device)
model=build_forecast_model(protocol,settings,device=device)
manifest=root.parent/'e23_v29_iterative_tv_h_r1/stage1/runs/e23-stage1.json'
payload=load_stage1_selection(manifest,protocol,settings,device=device)
model.repair.load_state_dict(payload['model_state']);model.repair.historical.retrieval.load_state_dict(payload['retrieval_state'])
memory=_new_memory(protocol);memory.load_state_dict(payload['memory_state']);model.repair.historical.refresh_snapshot(memory.snapshot())
model.train()
# Real input and teacher computation; no preflight weights used for formal initialization.
results=[]
for mode in ('uniform_v1','forecast_gain_v1'):
 model.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats(device);torch.cuda.synchronize(device);start=perf_counter()
 teacher=snapshot_teacher(model);before=_key_sha256(teacher.repair.historical.retrieval.state_dict());bank_before=memory.snapshot_hash()
 starts=_starts(batch,stride=8,patches=12,device=device)
 pred,out=model(batch.normalized_observed,batch.observed_mask,batch.query_mean,batch.query_std,starts)
 weights=torch.ones_like(batch.normalized_observed)
 if mode=='forecast_gain_v1':
  gain=region_oracle_gains(teacher,batch);weights=gains_to_weights(gain,~batch.observed_mask)
 loss=(pred-batch.forecast_target).square().mean()+.2*(pred-batch.forecast_target).abs().mean()+.05*weighted_repair_loss(out.explicit.normalized_sequence,batch.normalized_target,~batch.observed_mask,batch.query_std,weights)
 loss.backward();torch.cuda.synchronize(device)
 assert torch.isfinite(loss) and all(p.grad is None for p in teacher.parameters())
 assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.repair.parameters())
 assert before==_key_sha256(teacher.repair.historical.retrieval.state_dict());assert bank_before==memory.snapshot_hash()
 results.append(dict(mode=mode,seconds=perf_counter()-start,loss=loss.item(),batch=128,precision='float32',peak_allocated_gib=torch.cuda.max_memory_allocated(device)/1024**3,peak_reserved_gib=torch.cuda.max_memory_reserved(device)/1024**3))
 del teacher
(root/'preflight.json').write_text(json.dumps(dict(results=results,formal_initialization_reused=False,real_historical_bank=True,bank_members=sum(len(g.members) for vg in memory.snapshot().groups for g in vg),device=str(device),gpu=torch.cuda.get_device_name(device)),indent=2)+'\n')
print(json.dumps(results))
