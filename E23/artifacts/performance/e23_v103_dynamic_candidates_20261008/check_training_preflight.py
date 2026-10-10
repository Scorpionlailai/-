import runpy,sys,json,dataclasses
from pathlib import Path
R=Path(__file__).resolve().parent
namespace=runpy.run_path(str(R/'check_weather_protocol.py'))
protocol=namespace['protocol'];device=namespace['torch'].device('cuda:0');torch=namespace['torch']
from integrations.e23.train import build_repair_model,_starts,stage1_objective,_validation_repair_metrics
from integrations.e23.config import load_e23_settings
from integrations.forecast_compare.protocol import iter_p4_batches
settings=load_e23_settings(R/'weather_v103_v79_dynamic.yaml',protocol=protocol,allow_variable_scaling=True)
model=build_repair_model(protocol,settings,device=device)
torch.cuda.reset_peak_memory_stats(device)
batch=next(iter_p4_batches(protocol,'train',batch_size=128,epoch=0)).to(device)
starts=_starts(batch,stride=model.stride,patches=model.patch_count,device=device)
model.train();out=model(batch.normalized_observed,batch.observed_mask,batch.query_mean,batch.query_std,starts,epoch=1,explore_variable=True,specialist_exposure=.2,specialist_seed=2023)
loss,_=stage1_objective(out.explicit.normalized_sequence,batch.normalized_target,batch.observed_mask,out.explicit.normalized_sequence.sum()*0,retrieval_weight=.1,loss_mode='global_mse_v1',query_std=batch.query_std,structure_mode=settings.stage1_structure)
assert torch.isfinite(loss);loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
peak=torch.cuda.max_memory_allocated(device)/2**30
model.zero_grad(set_to_none=True);del out,loss
small=dataclasses.replace(protocol,refs={**protocol.refs,'validation':protocol.refs['validation'][:4]})
metrics=_validation_repair_metrics(model,small,settings,device=device)
assert len(metrics['variable_source_count_histogram'])==17
assert sum(metrics['variable_source_count_histogram'])>0
assert metrics['last32_hidden_coordinates']>0
result=dict(batch_size=128,finite_forward_and_gradients=True,peak_allocated_gib=peak,empty_H_bank=True,validation_count_metrics_verified=True,last32_metrics_verified=True,stage2_frozen_verified=True,scope='preflight only; random model; no optimizer step for Stage1; not training accuracy')
(R/'training_preflight.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
