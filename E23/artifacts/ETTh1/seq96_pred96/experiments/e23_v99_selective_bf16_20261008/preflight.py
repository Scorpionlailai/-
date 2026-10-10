import sys,json,torch
from pathlib import Path
R=Path(__file__).resolve().parent;p=json.loads((R/'plan.json').read_text());w=Path(p['work']);sys.path[:0]=[str(w/'src'),str(w)]
from evaluate_repair import protocol_for
from integrations.e23.config import load_e23_settings
from integrations.e23.train import build_repair_model,build_forecast_model,_starts
from integrations.e23.forecast_sensitive import parameter_hash
from integrations.forecast_compare.protocol import iter_p4_batches
torch.set_num_threads(4);torch.manual_seed(2023);scene=next(s for s in p['scenarios'] if s['name']==p['first_scene']);protocol=protocol_for(p,scene);device=torch.device('cuda:0');settings=load_e23_settings(Path(p['models'][0]['config']),protocol=protocol)
m=build_repair_model(protocol,settings,device=device);assert sum(x.numel() for x in m.parameters())==504885
b=next(iter_p4_batches(protocol,'validation',batch_size=128)).to(device);starts=_starts(b,stride=m.stride,patches=m.patch_count,device=device);from integrations.e23.selective_precision import SelectiveRepairPrecision
precision=SelectiveRepairPrecision(m,compile_readout=False)
with precision.training():
 out=m(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts);assert torch.isfinite(out.explicit.normalized_sequence).all();assert torch.equal(out.explicit.normalized_sequence[b.observed_mask],b.normalized_observed[b.observed_mask]);error=(out.explicit.normalized_sequence-b.normalized_target)*b.query_std[:,None];error.square()[~b.observed_mask].mean().backward();assert all(x.grad is None or torch.isfinite(x.grad).all() for x in m.parameters());assert out.point_candidate_attention['weights'].shape[-3]==8
torch.manual_seed(2023);f=build_forecast_model(protocol,settings,device=device);assert parameter_hash(f.predictor)==p['predictor_initial_sha256'];coarse=load_e23_settings(Path(p['parent_settings']),protocol=protocol);c=build_repair_model(protocol,coarse,device=device);assert sum(x.numel() for x in c.parameters())==424238
result=dict(status='passed',coarse_repair_parameters=424238,total_repair_parameters=504885,heads=8,head_dimension=8,attention_width=64,expert_CNN_width=64,finite_forward_and_gradients=True,observations_unchanged=True,original_predictor_initial_sha256=parameter_hash(f.predictor),forecast_head_unchanged=True,test_used=False)
(R/'preflight.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
