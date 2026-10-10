import sys,json,time,hashlib
from pathlib import Path
import torch
R=Path(__file__).resolve().parent;p=json.loads((R/'plan.json').read_text());w=Path(p['work']);sys.path[:0]=[str(w/'src'),str(w)]
from integrations.e23.config import load_e23_p4_protocol,load_e23_settings
from integrations.e23.train import load_stage2_selection,build_forecast_model,_new_memory,_starts,_validation_forecast_metrics
from integrations.e23.validation_cache import ForecastValidationCache
from integrations.e23.forecast_sensitive import parameter_hash
from malr.migration import MigrationCatalog
torch.set_num_threads(4);device=torch.device('cuda:0')
protocol=load_e23_p4_protocol(catalog=MigrationCatalog.load('/home/xyw/paper/E22/migration/paper_catalog.jsonl',project='E23'),protocol_config=Path(p['protocol_config']),mask_artifacts=Path(p['mask_artifacts']),training_seed=2023);settings=load_e23_settings(Path(p['e23_config']),protocol=protocol)
selection=Path(p['paired_original_root'])/'sr010_bl8/stage2/runs/e23-stage2.json';d=json.loads(selection.read_text());payload=load_stage2_selection(selection,protocol,settings,device=device)
m=build_forecast_model(protocol,settings,device=device);m.load_state_dict(payload['model_state']);m.repair.historical.retrieval.load_state_dict(payload['retrieval_state']);bank=_new_memory(protocol);bank.load_state_dict(payload['memory_state']);m.repair.historical.refresh_snapshot(bank.snapshot());m.repair.requires_grad_(False);m.eval()
initial=parameter_hash(m);torch.cuda.synchronize();t=time.perf_counter();full=_validation_forecast_metrics(m,protocol,settings,device=device);torch.cuda.synchronize();full_seconds=time.perf_counter()-t
cache=ForecastValidationCache.build(m,protocol,settings,device=device,starts_fn=_starts);torch.cuda.synchronize();t=time.perf_counter();fast=cache.evaluate(m,protocol,settings,device=device);torch.cuda.synchronize();fast_seconds=time.perf_counter()-t
assert full==fast
assert abs(full['mse']-d['best_val_mse'])<1e-6 and abs(full['mae']-d['best_val_mae'])<1e-6
assert parameter_hash(m)==initial
next(m.repair.parameters()).requires_grad_(True)
try:cache.evaluate(m,protocol,settings,device=device)
except ValueError:guard=True
else:raise AssertionError('unfrozen model accepted cache')
m.repair.requires_grad_(False)
result=dict(status='passed',full=full,cached=fast,exact_metrics_equal=True,full_validation_seconds=full_seconds,cached_validation_seconds=fast_seconds,cache=cache.metadata,frozen_state_unchanged=True,unfrozen_rejected=guard,test_used=False)
(R/'preflight.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
