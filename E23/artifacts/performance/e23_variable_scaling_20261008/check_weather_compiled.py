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

from integrations.forecast_compare.protocol import iter_p4_batches
def score():
 m.eval();sq=ab=rs=ra=ts=ta=0.;count=rc=tc=0
 with torch.no_grad():
  for cpu in iter_p4_batches(p,'validation',batch_size=settings.batch_size):
   b=cpu.to(device);starts=_starts(b,stride=m.repair.stride,patches=m.repair.patch_count,device=device);pred,out=m(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts)
   error=pred-b.forecast_target;sq+=error.square().sum().item();ab+=error.abs().sum().item();count+=error.numel()
   hidden=~b.observed_mask;re=(out.explicit.normalized_sequence-b.normalized_target)*b.query_std[:,None];rs+=re.square()[hidden].sum().item();ra+=re.abs()[hidden].sum().item();rc+=hidden.sum().item();ts+=re[:,-32:].square()[hidden[:,-32:]].sum().item();ta+=re[:,-32:].abs()[hidden[:,-32:]].sum().item();tc+=hidden[:,-32:].sum().item()
 return dict(forecast_mse=sq/count,forecast_mae=ab/count,repair96_mse=rs/rc,repair96_mae=ra/rc,repair32_mse=ts/tc,repair32_mae=ta/tc)

import os
os.environ['TORCHINDUCTOR_CACHE_DIR']=str(R/'inductor_cache')
before=parameter_hash(m);baseline=score();m.repair.point_candidate_readout.compile(fullgraph=True,dynamic=False);candidate=score();assert parameter_hash(m)==before
result=dict(dataset='Weather',baseline=baseline,compiled=candidate,metric_deltas={k:candidate[k]-baseline[k] for k in baseline},all_5175_validation_windows=True,last_partial_batch_retained=True,test_used=False,training=False,predictor_not_compiled=True,model_parameters_and_buffers_unchanged=True,scope='FP32 compiled repair readout checkpoint sensitivity; not an end-to-end trained speed experiment')
(R/'weather_compiled_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
