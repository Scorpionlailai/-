import sys,json,importlib.util,dataclasses
from pathlib import Path
W=Path('/home/xyw/.cache/codex-worktrees/e23-v103-dynamic-candidates-20261008');sys.path[:0]=[str(W/'src'),str(W)]
import torch,yaml
from integrations.e23.config import load_e23_settings
from integrations.e23.train import build_repair_model,build_forecast_model,_new_memory,_starts,_identity
from integrations.e23.forecast_sensitive import parameter_hash
from integrations.forecast_compare.protocol import iter_p4_batches
R=Path(__file__).resolve().parent;torch.set_num_threads(4)
spec=importlib.util.spec_from_file_location('readonly_e24_data','/home/xyw/paper/E24/integrations/e24/data.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
manifest=Path('/home/xyw/paper/E24/artifacts/Weather/seq96_pred96/sr010_bl8/shared_protocol_v1/dataset_manifest.json');identity=json.loads(manifest.read_text());protocol=module.load_dataset_protocol(manifest,config_path=Path(identity['config_path']),artifacts=manifest.parent)

from malr.models.e23.variable import SparseVariableExpert, DYNAMIC_POLICY
from malr.models.e23.contracts import ExpertInputs
from malr.data.patching import patchify_with_state
selection_path=Path('/home/xyw/paper/E24/artifacts/Weather/seq96_pred96/sr010_bl8/e24_v79_weather_20261007/seed_2023/E24-v79/stage1/runs/e23-stage1.json')
selection=json.loads(selection_path.read_text());payload=torch.load(selection['checkpoint'],map_location='cpu',weights_only=False)
expert=SparseVariableExpert(21,16,64,topk=16,selection=DYNAMIC_POLICY,contextual=True).cuda(0).eval()
state={key[len('variable.'):]:value for key,value in payload['model_state'].items() if key.startswith('variable.')}
expert.load_state_dict(state,strict=True)
counts=[];hits=[];mass=[];hist=torch.zeros(17,dtype=torch.long);windows=0
with torch.no_grad():
 for batch in iter_p4_batches(protocol,'validation',batch_size=128):
  batch=batch.to(torch.device('cuda:0'));patches=patchify_with_state(batch.normalized_observed,batch.observed_mask,16,8)
  b,c,n,p=patches.values.shape
  inputs=ExpertInputs(torch.zeros(b,c,n,64,device='cuda:0'),patches.values.masked_fill(~patches.effective_mask,0),patches.effective_mask,patches.validity,patches.rho,
    torch.arange(n,device='cuda:0')[None,:].expand(b,-1)*8,batch.query_mean,batch.query_std,patches.rho.lt(1))
  _,_,keep,stats=expert._dynamic_sources(inputs,epoch=0,explore=False)
  damaged=inputs.damaged
  count=stats['selected_source_count'][damaged].cpu()
  counts.append(count);hits.append(stats['cap_hit'][damaged].cpu());mass.append(stats['retained_probability_mass'][damaged].cpu())
  hist+=torch.bincount(count,minlength=17);windows+=b
counts=torch.cat(counts);hits=torch.cat(hits);mass=torch.cat(mass)
result=dict(validation_windows=windows,damaged_target_patches=counts.numel(),count_histogram={str(i):int(v) for i,v in enumerate(hist)},
 source_count_mean=float(counts.float().mean()),cap_hit_fraction=float(hits.float().mean()),retained_probability_mass_mean=float(mass.mean()),
 source_checkpoint=selection['checkpoint'],source_checkpoint_sha256=selection['checkpoint_sha256'],
 scope='observed-only ranking diagnostic using existing Weather v79 V weights; v103 untrained; no repair/forecast accuracy claim',test_used=False)
(R/'weather_trained_prior_candidate_distribution.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
