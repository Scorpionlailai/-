from pathlib import Path
import json,torch
from evaluate_repair import protocol_for,score,ROOT
from integrations.e23.config import load_e23_settings
from integrations.e23.train import load_stage1_selection,build_repair_model,_new_memory
plan=json.loads((ROOT/'plan.json').read_text());scene=plan['scenarios'][0];entry=plan['models'][0];protocol=protocol_for(plan,scene);device=torch.device('cpu');torch.set_num_threads(4)
settings=load_e23_settings(Path(entry['config']),protocol=protocol);selection=ROOT/entry['label']/scene['name']/'stage1/runs/e23-stage1.json';payload=load_stage1_selection(selection,protocol,settings,device=device)
model=build_repair_model(protocol,settings,device=device);model.load_state_dict(payload['model_state'],strict=True);model.historical.retrieval.load_state_dict(payload['retrieval_state']);memory=_new_memory(protocol);memory.load_state_dict(payload['memory_state']);model.historical.refresh_snapshot(memory.snapshot());model.eval().requires_grad_(False)
model.recent_mean_projector.global_dense_refiner=None
result={'type':'read_only_ablation_no_training_no_checkpoint_selection','model':'E23-v85','scene':scene['name'],'dense_branch':'disabled','validation':score(model,protocol,device,'E23','validation')}
(ROOT/'v85-dense-disabled-diagnostic.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
