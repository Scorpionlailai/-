"""Read-only module swaps on fixed validation, with the original v46 Stage2 head."""
import hashlib,json,sys
from pathlib import Path
import torch
from integrations.e23.config import load_e23_p4_protocol,load_e23_settings
from integrations.e23.train import build_forecast_model,load_stage1_selection,load_stage2_selection,_new_memory,_starts,_key_sha256
from integrations.forecast_compare.protocol import iter_p4_batches
from malr.migration import MigrationCatalog
from integrations.e23.forecast_sensitive import parameter_hash
WORK=Path('/home/xyw/paper/.e23-v46-front/E23')
ROOT=Path('/home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v46_front_context_sr015_bl8_seed502_20261005')
OUT=ROOT/'validation_component_interventions_20261005'
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 torch.set_num_threads(4);device=torch.device('cuda:1');plan=json.loads((ROOT/'plan.json').read_text())
 p=load_e23_p4_protocol(catalog=MigrationCatalog.load(WORK.parent/'E22/migration/paper_catalog.jsonl',project='E23'),protocol_config=WORK/'configs/stage1_seq96_dynamic_temporal_repair.yaml',mask_artifacts=WORK.parent/'E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context',training_seed=502)
 cfg=load_e23_settings(Path(plan['config']),protocol=p)
 s1=load_stage1_selection(ROOT/'stage1/runs/e23-stage1.json',p,cfg,device=device)
 s2=load_stage2_selection(ROOT/'stage2/runs/e23-stage2.json',p,cfg,device=device)
 m=build_forecast_model(p,cfg,device=device);m.load_state_dict(s2['model_state']);head_sha=parameter_hash(m.predictor)
 m.repair.historical.retrieval.load_state_dict(s2['retrieval_state']);bank=_new_memory(p);bank.load_state_dict(s2['memory_state']);m.repair.historical.refresh_snapshot(bank.snapshot())
 bank_identity=bank.snapshot_hash(),_key_sha256(m.repair.historical.retrieval.state_dict());state={k:v.detach().clone() for k,v in m.repair.state_dict().items()}
 checkpoints=[Path(json.loads((ROOT/stage/'runs'/name).read_text())['checkpoint']) for stage,name in [('stage1','e23-stage1.json'),('stage2','e23-stage2.json')]]
 hashes={str(x):digest(x) for x in checkpoints}
 cases=[('original_stage2',None),('stage1_front_only','front_context.'),('stage1_tail_refiner_only','point_refiner.'),('stage1_decoder_only','decoder.'),('stage1_complete_repair','')]
 rows=[]
 for label,prefix in cases:
  m.repair.load_state_dict(state)
  if prefix is not None:
   replacement={k:(s1['model_state'][k] if k.startswith(prefix) else v) for k,v in state.items()}
   m.repair.load_state_dict(replacement)
  m.eval();sq=ab=rsq=rab=tail_sq=level=0.;count=hidden_count=tail_count=level_count=0
  with torch.inference_mode():
   for cpu in iter_p4_batches(p,'validation',batch_size=128):
    b=cpu.to(device);starts=_starts(b,stride=8,patches=12,device=device)
    prediction,o=m(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts)
    e=prediction-b.forecast_target;sq+=e.square().sum().item();ab+=e.abs().sum().item();count+=e.numel()
    re=(o.explicit.normalized_sequence-b.normalized_target)*b.query_std[:,None];hidden=~b.observed_mask
    rsq+=re[hidden].double().square().sum().item();rab+=re[hidden].double().abs().sum().item();hidden_count+=hidden.sum().item()
    tail=re[:,64:];tail_hidden=hidden[:,64:];tail_sq+=tail[tail_hidden].double().square().sum().item();tail_count+=tail_hidden.sum().item()
    te=tail.masked_fill(~tail_hidden,0).double();level+=te.mean(1).square().sum().item();level_count+=te.shape[0]*te.shape[2]
    assert torch.equal(o.explicit.normalized_sequence[b.observed_mask],b.normalized_observed[b.observed_mask])
  assert parameter_hash(m.predictor)==head_sha
  assert bank_identity==(bank.snapshot_hash(),_key_sha256(m.repair.historical.retrieval.state_dict()))
  row=dict(condition=label,prediction_mse=sq/count,prediction_mae=ab/count,repair_mse=rsq/hidden_count,repair_mae=rab/hidden_count,last32_repair_mse=tail_sq/tail_count,last32_level_mse=level/level_count,windows=count//(96*7))
  rows.append(row);print(json.dumps(row),flush=True)
 for path,sha in hashes.items():assert digest(path)==sha
 result=dict(scope='fixed_validation_only',predictor_held_fixed_sha256=head_sha,checkpoint_hashes=hashes,conditions=rows,no_training=True,no_test=True,limitations='Module swaps are diagnostic interventions and do not predict the result of retraining with a frozen module.')
 with (OUT/'results.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
if __name__=='__main__':main()
