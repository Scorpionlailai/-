from pathlib import Path
import json,torch
from train_refinement import ROOT,protocol_for,coarse_hash,bank_content
from integrations.e23.config import load_e23_settings
from integrations.e23.train import build_repair_model,load_stage1_selection,_new_memory,_starts,_validation_repair_metrics
from integrations.forecast_compare.protocol import iter_p4_batches
p=json.loads((ROOT/'plan.json').read_text());scene=p['scenarios'][0];entry=p['models'][0];torch.set_num_threads(4);torch.manual_seed(2023);device=torch.device('cpu');protocol=protocol_for(p,scene)
settings=load_e23_settings(Path(entry['config']),protocol=protocol);parent=load_e23_settings(Path(p['parent_settings']),protocol=protocol)
selection=Path(p['parent_root'])/scene['name']/'E23-v79/stage1/runs/e23-stage1.json';record=json.loads(selection.read_text());payload=load_stage1_selection(selection,protocol,parent,device=device)
model=build_repair_model(protocol,settings,device=device);missing=model.load_state_dict(payload['model_state'],strict=False);assert not missing.unexpected_keys and all('.global_dense_refiner.' in k for k in missing.missing_keys)
model.historical.retrieval.load_state_dict(payload['retrieval_state']);memory=_new_memory(protocol);memory.load_state_dict(payload['memory_state']);model.historical.refresh_snapshot(memory.snapshot());model.eval().requires_grad_(False);refiner=model.recent_mean_projector.global_dense_refiner;refiner.requires_grad_(True)
initial=_validation_repair_metrics(model,protocol,settings,device=device);print({'initial_cpu_mse':initial['global_mse'],'saved_gpu_mse':record['validation_repair_metrics']['global_mse']},flush=True);dense=model.recent_mean_projector.global_dense_refiner;model.recent_mean_projector.global_dense_refiner=None;parent_cpu=_validation_repair_metrics(model,protocol,settings,device=device);model.recent_mean_projector.global_dense_refiner=dense;assert initial['global_mse']==parent_cpu['global_mse']
before=coarse_hash(model);bank=bank_content(memory);epoch=payload['epoch']+1;r=model.historical.retrieval;r.ema_encoder.load_state_dict(r.frozen_encoder.state_dict());r.begin_epoch(epoch)
b=next(iter(iter_p4_batches(protocol,'train',batch_size=128,epoch=0)));starts=_starts(b,stride=model.stride,patches=model.patch_count,device=device);refiner.train()
out=model(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts,epoch=epoch,explore_variable=False,specialist_exposure=0).explicit
hidden=~b.observed_mask & out.sequence_validity;e=(out.normalized_sequence-b.normalized_target)*b.query_std[:,None];loss=e[hidden].square().mean();loss.backward()
assert torch.isfinite(loss) and all(v.grad is None for name,v in model.named_parameters() if '.global_dense_refiner.' not in name)
assert refiner.output.weight.grad is not None and refiner.output.weight.grad.abs().sum()>0
optimizer=torch.optim.Adam(refiner.parameters(),lr=.0001);optimizer.step();memory.publish_epoch(epoch);model.historical.refresh_snapshot(memory.snapshot());assert coarse_hash(model)==before and bank_content(memory)==bank
result={'initial_parent_mse':parent_cpu['global_mse'],'saved_gpu_parent_mse':record['validation_repair_metrics']['global_mse'],'initial_candidate_mse':initial['global_mse'],'zero_residual_reproduces_parent':True,'frozen_parameter_gradients_none':True,'nonzero_refiner_output_gradient':True,'coarse_weights_unchanged_after_update':True,'history_contents_unchanged_after_paired_epoch_advance':True,'trainable_parameters':sum(v.numel() for v in model.parameters() if v.requires_grad)}
(ROOT/'preflight-frozen-refinement.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
