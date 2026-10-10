from weather_common import ROOT,protocol_for
import json,torch,dataclasses,py_compile
from pathlib import Path
from integrations.e23.config import load_e23_settings
from integrations.e23.train import load_stage1_selection,build_repair_model,_starts
from integrations.forecast_compare.protocol import iter_p4_batches
from integrations.e23.forecast_sensitive import parameter_hash

def main():
 torch.set_num_threads(4);plan=json.loads((ROOT/'plan.json').read_text());protocol=protocol_for(plan,plan['scenarios'][0]);parent=load_e23_settings(Path(plan['parent_settings']),protocol=protocol,allow_variable_scaling=True);settings=load_e23_settings(ROOT/'v107_weather.yaml',protocol=protocol,allow_variable_scaling=True)
 device=torch.device('cpu');payload=load_stage1_selection(Path(plan['parent_selection']),protocol,parent,device=device)
 torch.manual_seed(2023);model=build_repair_model(protocol,settings,device=device);keys=model.load_state_dict(payload['model_state'],strict=False)
 assert not keys.unexpected_keys and keys.missing_keys and all('.global_dense_refiner.' in k for k in keys.missing_keys)
 assert sum(p.numel() for p in model.recent_mean_projector.parameters())==plan['models'][0]['trainable_parameters']
 old=build_repair_model(protocol,parent,device=device);old.load_state_dict(payload['model_state'],strict=True)
 for m in (model,old):
  m.eval();m.historical.retrieval.load_state_dict(payload['retrieval_state'])
 from integrations.e23.train import _new_memory
 memory=_new_memory(protocol);memory.load_state_dict(payload['memory_state'])
 for m in (model,old):m.historical.refresh_snapshot(memory.snapshot())
 full=next(iter_p4_batches(protocol,'validation',batch_size=128));batch=dataclasses.replace(full,**{f.name:getattr(full,f.name)[:2] for f in dataclasses.fields(full) if isinstance(getattr(full,f.name),torch.Tensor)},refs=full.refs[:2])
 starts=_starts(batch,stride=model.stride,patches=model.patch_count,device=device)
 with torch.no_grad():
  a=model(batch.normalized_observed,batch.observed_mask,batch.query_mean,batch.query_std,starts).explicit.normalized_sequence
  b=old(batch.normalized_observed,batch.observed_mask,batch.query_mean,batch.query_std,starts).explicit.normalized_sequence
 assert torch.equal(a,b)
 for name in ('train_refinement.py','train_forecast_and_evaluate.py','evaluate_forecast.py'):py_compile.compile(str(ROOT/name),doraise=True)
 result=dict(parent_identity_and_checkpoint_verified=True,missing_keys_only_new_dense=True,zero_residual_reproduces_parent_exactly=True,validation_windows_checked=2,trainable_refinement_parameters=111084,source='v97 architecture with zero-initialized contextual guide gate',device='cpu',training_not_started=True)
 (ROOT/'preflight.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
