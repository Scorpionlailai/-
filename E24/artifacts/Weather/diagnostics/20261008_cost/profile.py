import json,time,gc,os
from contrastive_rows import full_contrastive_rows
import integrations.e23.train as train_module
ROW_MODE=os.environ.get("WEATHER_CONTRASTIVE_ROWS")=="1"
if ROW_MODE:train_module.masked_contrastive_loss=full_contrastive_rows
from pathlib import Path
import torch
from integrations.e24.data import read_manifest,load_dataset_protocol
from integrations.e23.config import load_e23_settings
from integrations.e23.train import build_repair_model,_new_memory,_starts,_retrieval_loss,stage1_objective
from integrations.forecast_compare.protocol import iter_p4_batches
from malr.data.patching import patchify_with_state

root=Path(__file__).resolve().parent
manifest=Path('artifacts/Weather/seq96_pred96/sr010_bl8/shared_protocol_v1/dataset_manifest.json')
r=read_manifest(manifest);protocol=load_dataset_protocol(manifest,config_path=Path(r['config_path']),artifacts=manifest.parent)
settings=load_e23_settings(Path('configs/e24_v79.yaml'),protocol=protocol)
device=torch.device('cuda:0');torch.set_num_threads(4);torch.manual_seed(2023)
model=build_repair_model(protocol,settings,device=device)
selection=Path('artifacts/Weather/seq96_pred96/sr010_bl8/e24_v79_weather_20261007/seed_2023/E24-v79/stage1/runs/e23-stage1.json')
selected=json.loads(selection.read_text());payload=torch.load(selected['checkpoint'],map_location='cpu',weights_only=False)
model.load_state_dict(payload['model_state']);model.historical.retrieval.load_state_dict(payload['retrieval_state']);memory=_new_memory(protocol);memory.load_state_dict(payload['memory_state']);model.historical.refresh_snapshot(memory.snapshot());del payload
batch=next(iter_p4_batches(protocol,'train',batch_size=128,epoch=0)).to(device)
starts=_starts(batch,stride=model.stride,patches=model.patch_count,device=device)
p=patchify_with_state(batch.normalized_observed,batch.observed_mask,model.patch_length,model.stride);count=int((p.rho<1).sum())
index=model.historical.index
summary=dict(contrastive_implementation='row_blocks' if ROW_MODE else 'original_dense',batch_size=128,variables=21,patches=model.patch_count,damaged_patches=count,contrastive_matrix_elements=count*count,contrastive_one_matrix_GiB=count*count*4/2**30,
 gpu=torch.cuda.get_device_name(),phase_measurements=[],bank_group_shape=list(index.group_member_keys.shape),bank_members=int(index.member_keys.shape[0]))
def sync():torch.cuda.synchronize()
def measure(label,fn):
 sync();torch.cuda.reset_peak_memory_stats();base=torch.cuda.memory_allocated();start=time.perf_counter();out=fn();sync()
 row=dict(name=label,seconds=time.perf_counter()-start,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,additional_peak_GiB=(torch.cuda.max_memory_allocated()-base)/2**30);summary['phase_measurements'].append(row);print(json.dumps(row),flush=True);return out
model.train()
module_times=[];handles=[];active={}
for label,module in model.named_children():
 def before(m,args,label=label):
  event=torch.cuda.Event(enable_timing=True);event.record();active[label]=event
 def after(m,args,out,label=label):
  event=torch.cuda.Event(enable_timing=True);event.record();module_times.append((label,active.pop(label),event))
 handles.extend([module.register_forward_pre_hook(before),module.register_forward_hook(after)])
for warm in range(3):
 out=model(batch.normalized_observed,batch.observed_mask,batch.query_mean,batch.query_std,starts,epoch=80,explore_variable=True,specialist_exposure=.2,specialist_seed=2023)
 auxiliary=_retrieval_loss(model,batch)
 warm_loss,_=stage1_objective(out.explicit.normalized_sequence,batch.normalized_target,batch.observed_mask,auxiliary,retrieval_weight=settings.retrieval_loss_weight,loss_mode=settings.stage1_objective,query_std=batch.query_std,structure_mode=settings.stage1_structure)
 warm_loss.backward();model.zero_grad(set_to_none=True);del out,auxiliary,warm_loss
sync();module_times.clear()
output=measure('repair_forward_train',lambda:model(batch.normalized_observed,batch.observed_mask,batch.query_mean,batch.query_std,starts,epoch=80,explore_variable=True,specialist_exposure=.2,specialist_seed=2023))
retrieval=measure('retrieval_loss_forward',lambda:_retrieval_loss(model,batch))
loss,parts=stage1_objective(output.explicit.normalized_sequence,batch.normalized_target,batch.observed_mask,retrieval,retrieval_weight=settings.retrieval_loss_weight,loss_mode=settings.stage1_objective,query_std=batch.query_std,structure_mode=settings.stage1_structure)
measure('combined_backward',lambda:loss.backward())
sync()
summary['repair_forward_module_cuda_ms']={name:sum(start.elapsed_time(stop) for label,start,stop in module_times if label==name) for name in sorted({x[0] for x in module_times})}
for handle in handles:handle.remove()
model.zero_grad(set_to_none=True);del output,retrieval,loss,parts;gc.collect();torch.cuda.empty_cache()
model.eval()
with torch.no_grad():
 for i in range(4):
  output=measure('repair_frozen_forward_'+str(i),lambda:model(batch.normalized_observed,batch.observed_mask,batch.query_mean,batch.query_std,starts));del output
summary['checkpoint_read_only']=True;summary['not_a_formal_training_or_test_run']=True
(root/('cost_profile_rows.json' if ROW_MODE else 'cost_profile.json')).write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
