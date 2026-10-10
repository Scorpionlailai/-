import json,torch
from pathlib import Path
from integrations.e23.config import load_e23_p4_protocol,load_e23_settings
from integrations.e23.train import build_forecast_model,load_stage1_selection,_starts,_new_memory,stage2_objective
from integrations.e23.forecast_sensitive import parameter_hash
from integrations.e23.repair_updates import build_stage2_optimizer,stage2_training_mode
from integrations.forecast_compare.protocol import iter_p4_batches
from malr.migration import MigrationCatalog
W=Path(__file__).resolve().parents[1]
R=Path("/home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v60_padding_sr015_bl8_seed502_20261005")
p=json.loads((R/'plan.json').read_text());torch.set_num_threads(4);dev=torch.device('cuda:1')
proto=load_e23_p4_protocol(catalog=MigrationCatalog.load(W.parent/'E22/migration/paper_catalog.jsonl',project='E23'),protocol_config=W/'configs/stage1_seq96_dynamic_temporal_repair.yaml',mask_artifacts=W.parent/'E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context',training_seed=502)
cfg=load_e23_settings(Path(p['base_config']),protocol=proto);payload=load_stage1_selection(Path(p['initial_parent']['root'])/'stage1/runs/e23-stage1.json',proto,cfg,device=dev)
torch.manual_seed(502);m=build_forecast_model(proto,cfg,device=dev);assert parameter_hash(m.predictor)==p['predictor_initial_sha256'];m.forecast_padding='replicate_last_value_v1';m.repair.load_state_dict(payload['model_state']);mem=_new_memory(proto);mem.load_state_dict(payload['memory_state']);m.repair.historical.retrieval.load_state_dict(payload['retrieval_state']);m.repair.historical.refresh_snapshot(mem.snapshot())
for v in m.repair.historical.key_encoder.parameters():v.requires_grad_(False)
opt=build_stage2_optimizer(m,mode='freeze_front_v1',learning_rate=1e-4,weight_decay=0);front=parameter_hash(m.repair.front_context);head=parameter_hash(m.predictor)
b=next(iter_p4_batches(proto,'train',batch_size=4,epoch=0)).to(dev);starts=_starts(b,stride=8,patches=12,device=dev)
for i in range(2):
 stage2_training_mode(m,'freeze_front_v1');opt.zero_grad();pred,out=m(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts);loss=stage2_objective(pred,b.forecast_target,loss_mode=p['forecast_loss']);loss.backward();assert torch.isfinite(loss);assert all(torch.isfinite(v.grad).all() for v in m.parameters() if v.grad is not None);opt.step();assert front==parameter_hash(m.repair.front_context)
assert head!=parameter_hash(m.predictor)
r=dict(status='passed',configurations=[dict(config_key='parent_architecture',repair_parameters=sum(v.numel() for v in m.repair.parameters()),front_context_parameters=sum(v.numel() for v in m.repair.front_context.parameters()),stage2_trainable_repair_parameters=sum(v.numel() for v in m.repair.parameters() if v.requires_grad))],predictor_initial_sha256=p['predictor_initial_sha256'],finite_unchanged_forecast_objective=True,forecast_padding="replicate_last_value_v1",front_frozen=True,no_test_evaluation=True)
(R/'preflight.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
