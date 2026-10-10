from pathlib import Path
import json,torch
from evaluate_repair import ROOT,protocol_for,score,sha
from integrations.e23.config import load_e23_settings
from integrations.e23.train import load_stage1_selection,build_repair_model,_new_memory
p=json.loads((ROOT/'plan.json').read_text());scene=next(s for s in p['scenarios'] if s['name']=='sr010_bl16');protocol=protocol_for(p,scene);settings=load_e23_settings(Path(p['models'][0]['config']),protocol=protocol);device=torch.device('cpu');torch.set_num_threads(4)
selection=ROOT/'E23-v92'/scene['name']/'stage1/runs/e23-stage1.json';r=json.loads(selection.read_text());before=sha(Path(r['checkpoint']));payload=load_stage1_selection(selection,protocol,settings,device=device)
model=build_repair_model(protocol,settings,device=device);model.load_state_dict(payload['model_state'],strict=True);model.historical.retrieval.load_state_dict(payload['retrieval_state']);memory=_new_memory(protocol);memory.load_state_dict(payload['memory_state']);model.historical.refresh_snapshot(memory.snapshot());model.eval().requires_grad_(False)
result={'scene':scene['name'],'split':'fixed_validation','device':'cpu','in_memory_component_intervention_only':True,'checkpoint_unchanged':True,'metrics':{}}
result['metrics']['v92_full']=score(model,protocol,device,'E23','validation')
model.recent_mean_projector.global_dense_refiner=None
result['metrics']['v79_coarse_plus_mean']=score(model,protocol,device,'E23','validation')
with torch.no_grad():
 for network in [model.recent_mean_projector.network,model.recent_mean_projector.global_level_network]:network[-1].weight.zero_();network[-1].bias.zero_()
result['metrics']['v79_coarse_without_mean']=score(model,protocol,device,'E23','validation')
assert sha(Path(r['checkpoint']))==before
(ROOT/'mean-projector-long-validation-diagnostic.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
