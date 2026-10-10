import sys,json,importlib.util,dataclasses
from pathlib import Path
W=Path('/home/xyw/.cache/codex-worktrees/e23-v102-variable-count-20261008');sys.path[:0]=[str(W/'src'),str(W)]
import torch,yaml
from integrations.e23.config import load_e23_settings
from integrations.e23.train import build_repair_model,build_forecast_model,_new_memory,_starts,_identity
from integrations.e23.forecast_sensitive import parameter_hash
from integrations.forecast_compare.protocol import iter_p4_batches
R=Path(__file__).resolve().parent;torch.set_num_threads(4)
spec=importlib.util.spec_from_file_location('readonly_e24_data','/home/xyw/paper/E24/integrations/e24/data.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
manifest=Path('/home/xyw/paper/E24/artifacts/Weather/seq96_pred96/sr010_bl8/shared_protocol_v1/dataset_manifest.json');identity=json.loads(manifest.read_text());protocol=module.load_dataset_protocol(manifest,config_path=Path(identity['config_path']),artifacts=manifest.parent)
source=Path('/home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_stage1_mean_and_dense_20261008/e23_v97_stage1_mean_and_dense.yaml');document=yaml.safe_load(source.read_text());rows=[];head_hashes=[]
for k in [4,8]:
 document['model']['variable_topk']=k;cfg=R/f'weather_v102_V{k}.yaml';cfg.write_text(yaml.safe_dump(document,sort_keys=False));settings=load_e23_settings(cfg,protocol=protocol,allow_variable_scaling=True)
 torch.manual_seed(2023);model=build_forecast_model(protocol,settings,device=torch.device('cuda:0'));head_hashes.append(parameter_hash(model.predictor));model.eval()
 full=next(iter_p4_batches(protocol,'validation',batch_size=128));batch=dataclasses.replace(full,**{field.name:getattr(full,field.name)[:4] for field in dataclasses.fields(full) if isinstance(getattr(full,field.name),torch.Tensor)},refs=full.refs[:4]).to(torch.device('cuda:0'))
 starts=_starts(batch,stride=model.repair.stride,patches=model.repair.patch_count,device=torch.device('cuda:0'))
 with torch.no_grad():prediction,out=model(batch.normalized_observed,batch.observed_mask,batch.query_mean,batch.query_std,starts)
 assert prediction.shape==batch.forecast_target.shape and torch.isfinite(prediction).all();assert torch.equal(out.explicit.normalized_sequence[batch.observed_mask],batch.normalized_observed[batch.observed_mask])
 meta=_identity(protocol,settings);memory=_new_memory(protocol)
 rows.append(dict(dataset='Weather',variables=21,variable_topk=k,prediction_shape=list(prediction.shape),repair_parameters=sum(p.numel() for p in model.repair.parameters()),dense_features=meta['global_dense_features'],dense_parameters=meta['global_dense_extra_parameters'],predictor_initial_sha256=head_hashes[-1],config=str(cfg),observations_unchanged=True,finite_forecast=True))
 del model;torch.cuda.empty_cache()
assert head_hashes[0]==head_hashes[1]
(R/'weather_protocol_preflight.json').write_text(json.dumps(dict(status='passed',rows=rows,head_unchanged_between_V4_and_V8=True,scope='four real fixed validation inputs; no optimizer, no tests, no trained accuracy claims',dataset_manifest=str(manifest)),indent=2));print(json.dumps(rows,indent=2))
