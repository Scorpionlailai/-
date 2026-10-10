import json,torch
from pathlib import Path
from integrations.e23.config import load_e23_p4_protocol,load_e23_settings
from integrations.e23.train import build_forecast_model,_starts,stage1_objective,_retrieval_loss
from integrations.e23.forecast_sensitive import parameter_hash
from integrations.e23.repair_updates import build_stage2_optimizer,stage2_training_mode
from integrations.forecast_compare.protocol import iter_p4_batches
from malr.migration import MigrationCatalog
W=Path(__file__).resolve().parents[1]
R=Path('/home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v69_signature_position_sr015_bl8_20261005')
p=json.loads((R/'plan.json').read_text());torch.set_num_threads(4);dev=torch.device('cuda:1')
proto=load_e23_p4_protocol(catalog=MigrationCatalog.load(W.parent/'E22/migration/paper_catalog.jsonl',project='E23'),protocol_config=W/'configs/stage1_seq96_dynamic_temporal_repair.yaml',mask_artifacts=W.parent/'E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context',training_seed=p["seeds"][0])
cfg=load_e23_settings(Path(next(iter(p['configs'].values()))),protocol=proto);torch.manual_seed(p["seeds"][0]);m=build_forecast_model(proto,cfg,device=dev);assert parameter_hash(m.predictor)==p['predictor_initial_sha256'];b=next(iter_p4_batches(proto,'train',batch_size=4,epoch=0)).to(dev);starts=_starts(b,stride=8,patches=12,device=dev);opt=torch.optim.Adam(m.repair.parameters(),lr=.001)
for i in range(2):
 m.train();opt.zero_grad();out=m.repair(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts);keyloss=_retrieval_loss(m.repair,b);loss,parts=stage1_objective(out.explicit.normalized_sequence,b.normalized_target,b.observed_mask,keyloss,retrieval_weight=.1,loss_mode='global_mse_v1',query_std=b.query_std,structure_mode='recent_mean_v2');loss.backward();assert torch.isfinite(loss);assert all(torch.isfinite(v.grad).all() for v in m.repair.parameters() if v.grad is not None);print('position_gradient_step',i,float(m.repair.front_context.variable_context.position.grad.abs().sum()),flush=True);assert i==0 or m.repair.front_context.variable_context.position.grad.abs().sum()>0;opt.step()
for v in m.repair.historical.key_encoder.parameters():v.requires_grad_(False)
opt=build_stage2_optimizer(m,mode='freeze_front_v1',learning_rate=1e-4,weight_decay=0);stage2_training_mode(m,'freeze_front_v1');assert not any(v.requires_grad for v in m.repair.front_context.parameters())
r=dict(status='passed',configurations=[dict(training_seed=p['seeds'][0],config_key=next(iter(p['configs'])),repair_parameters=sum(v.numel() for v in m.repair.parameters()),front_context_parameters=sum(v.numel() for v in m.repair.front_context.parameters()),stage2_trainable_repair_parameters=sum(v.numel() for v in m.repair.parameters() if v.requires_grad))],predictor_initial_sha256=p['predictor_initial_sha256'],finite_global_repair_mean_objective=float(loss.detach()),recent_level_error=float(parts['recent_structure_weighted'].detach()),stage2_front_and_history_key_frozen=True,new_parameters=43008,preflight_gradient_batch_size=4,stage1_training_batch_size=p["stage1_training_batch_size"],stage1_optimizer_updates=p["stage1_optimizer_updates"],no_test_evaluation=True)
(R/'preflight.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
