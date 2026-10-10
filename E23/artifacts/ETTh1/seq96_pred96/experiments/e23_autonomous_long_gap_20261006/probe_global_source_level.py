"""Small CPU counterfactual probe of source-level information access."""
from pathlib import Path
import hashlib,json,os
import torch
from integrations.e23.config import load_e23_p4_protocol,load_e23_settings
from integrations.e23.train import build_forecast_model,load_stage2_selection,_new_memory,_starts
from integrations.forecast_compare.protocol import iter_p4_batches
from malr.migration import MigrationCatalog
ROOT=Path(__file__).resolve().parent;WORK=Path('/home/xyw/paper/.e23-v74-pre-tail-attention/E23');cell=ROOT.parent/'e23_v72_long_gap_moments_seed2023_20261005/sr010_bl16/E23-v72-mae'
plan=json.loads((cell/'plan.json').read_text());os.environ['MALR_MISSING_SCENARIO']=json.loads((ROOT/'series.json').read_text())['scene']['scenario_manifest'];device=torch.device('cpu');torch.set_num_threads(4)
protocol=load_e23_p4_protocol(catalog=MigrationCatalog.load(WORK.parent/'E22/migration/paper_catalog.jsonl',project='E23'),protocol_config=WORK/'configs/stage1_seq96_dynamic_temporal_repair.yaml',mask_artifacts=WORK.parent/'E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context',training_seed=2023)
settings=load_e23_settings(Path(plan['config']),protocol=protocol);selection=cell/'stage2/runs/e23-stage2.json';payload=load_stage2_selection(selection,protocol,settings,device=device)
model=build_forecast_model(protocol,settings,device=device).eval();model.load_state_dict(payload['model_state'],strict=True);model.repair.historical.retrieval.load_state_dict(payload['retrieval_state']);memory=_new_memory(protocol);memory.load_state_dict(payload['memory_state']);model.repair.historical.refresh_snapshot(memory.snapshot());bankhash=memory.snapshot_hash()
batch=next(iter_p4_batches(protocol,'validation',batch_size=4));starts=_starts(batch,stride=8,patches=12,device=device)
shifted=batch.query_mean.clone();shifted[:,1]+=1
with torch.inference_mode():
    a,ra=model(batch.normalized_observed,batch.observed_mask,batch.query_mean,batch.query_std,starts)
    b,rb=model(batch.normalized_observed,batch.observed_mask,shifted,batch.query_std,starts)
result=dict(model='E23-v72',model_seed=2023,probe_windows=4,split='first_fixed_validation_batch',source_variable=1,target_variable=0,source_global_level_shift=1.0,target_repair_max_absolute_change=float((ra.explicit.normalized_sequence[:,:,0]-rb.explicit.normalized_sequence[:,:,0]).abs().max()),target_forecast_max_absolute_change=float((a[:,0]-b[:,0]).abs().max()),forecast_shape=list(a.shape),global_only_shift=True,no_training=True,no_test_scoring=True,bank_unchanged=bankhash==memory.snapshot_hash(),selection=str(selection),selection_sha256=hashlib.sha256(selection.read_bytes()).hexdigest(),note='Normalized visible patterns/masks/std and target mean held fixed; shifting source mean changes source global visible values. This is information-access probe, not causal evidence on real data.')
with (ROOT/'source-level-access-probe.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result))
