import sys,json,time,ast,importlib.util,types
from pathlib import Path
W=Path('/home/xyw/paper/E24');NEW=Path('/home/xyw/paper/E23/versions/e23-efficiency-validation-cache-20261008/E23');sys.path[:0]=[str(W/'src'),str(W)]
import torch
from integrations.e24.data import load_dataset_protocol
from integrations.e23.config import load_e23_settings
from integrations.e23.train import build_forecast_model,_new_memory,_starts,_validation_forecast_metrics
from integrations.e23.forecast_sensitive import parameter_hash
from malr.models.e23.model import E23ForecastModel
# Load only the new cache implementation and extracted input preparation helper.
# The canonical E24 source/run is read-only and no optimizer is created.
tree=ast.parse((NEW/'src/malr/models/e23/model.py').read_text());cl=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='E23ForecastModel');method=next(x for x in cl.body if isinstance(x,ast.FunctionDef) and x.name=='explicit_predictor_inputs');ns=dict(E23ForecastModel.forward.__globals__);exec(compile(ast.Module(body=[method],type_ignores=[]),'<isolated cache helper>','exec'),ns)
spec=importlib.util.spec_from_file_location('integrations.e23.validation_cache',NEW/'integrations/e23/validation_cache.py');module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
R=Path(__file__).resolve().parent;device=torch.device('cuda:0');torch.set_num_threads(4)
manifest=W/'artifacts/Weather/seq96_pred96/sr010_bl8/shared_protocol_v1/dataset_manifest.json';identity=json.loads(manifest.read_text());p=load_dataset_protocol(manifest,config_path=Path(identity['config_path']),artifacts=manifest.parent);settings=load_e23_settings(W/'configs/e24_v79.yaml',protocol=p)
selection=W/'artifacts/Weather/seq96_pred96/sr010_bl8/e24_v79_weather_20261007/seed_2023/E24-v79/stage2/runs/e23-stage2.json';d=json.loads(selection.read_text());payload=torch.load(d['checkpoint'],map_location=device,weights_only=False);m=build_forecast_model(p,settings,device=device);m.load_state_dict(payload['model_state']);m.repair.historical.retrieval.load_state_dict(payload['retrieval_state']);bank=_new_memory(p);bank.load_state_dict(payload['memory_state']);m.repair.historical.refresh_snapshot(bank.snapshot());m.repair.requires_grad_(False);m.eval();m.explicit_predictor_inputs=types.MethodType(ns['explicit_predictor_inputs'],m)
hash_before=parameter_hash(m);_validation_forecast_metrics(m,p,settings,device=device)
torch.cuda.synchronize();t=time.perf_counter();full=_validation_forecast_metrics(m,p,settings,device=device);torch.cuda.synchronize();full_seconds=time.perf_counter()-t
cache=module.ForecastValidationCache.build(m,p,settings,device=device,starts_fn=_starts);torch.cuda.synchronize();t=time.perf_counter();fast=cache.evaluate(m,p,settings,device=device);torch.cuda.synchronize();fast_seconds=time.perf_counter()-t
assert fast==full and abs(full['mse']-d['best_val_mse'])<1e-6 and abs(full['mae']-d['best_val_mae'])<1e-6
assert cache.metadata['windows']==5175 and parameter_hash(m)==hash_before
result=dict(status='passed',dataset='Weather',variables=21,validation_windows=5175,validation_batches=41,last_partial_batch_retained=True,metrics=full,exact_metrics_equal=True,uncached_validation_seconds=full_seconds,cached_validation_seconds=fast_seconds,cache=cache.metadata,training=False,test_used=False,canonical_e24_files_modified=False)
(R/'weather_validation_cache.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
