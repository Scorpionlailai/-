"""Repair-only evaluation on the sealed Weather development mask bank."""
import json,time,importlib.util
from pathlib import Path
import numpy as np
import torch
from integrations.e23.train import build_repair_model,load_stage1_selection,_new_memory,_starts,_atomic_json
from integrations.e23.forecast_sensitive import parameter_hash
from integrations.forecast_compare.protocol import iter_p4_batches

@torch.no_grad()
def evaluate(protocol,settings,selection,output,device):
 spec=importlib.util.spec_from_file_location('readonly_e24_eval_data','/home/xyw/paper/E24/integrations/e24/data.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 manifest=Path('/home/xyw/paper/E24/artifacts/Weather/seq96_pred96/sr010_bl8/shared_protocol_v1/dataset_manifest.json');record=json.loads(manifest.read_text())
 payload=load_stage1_selection(selection,protocol,settings,device=device)
 model=build_repair_model(protocol,settings,device=device);model.load_state_dict(payload['model_state'],strict=True)
 model.historical.retrieval.load_state_dict(payload['retrieval_state']);memory=_new_memory(protocol);memory.load_state_dict(payload['memory_state']);model.historical.refresh_snapshot(memory.snapshot());model.eval().requires_grad_(False)
 before=parameter_hash(model);del payload
 rows=[];started=time.perf_counter()
 for i,item in enumerate(x for x in record['masks'] if x['split']=='test'):
  current=module.with_test_mask(protocol,record,item)
  squared=absolute=tail_squared=tail_absolute=0.;count=tail_count=windows=0
  for cpu in iter_p4_batches(current,'test',batch_size=settings.batch_size):
   batch=cpu.to(device);starts=_starts(batch,stride=model.stride,patches=model.patch_count,device=device)
   out=model(batch.normalized_observed,batch.observed_mask,batch.query_mean,batch.query_std,starts).explicit
   assert torch.equal(out.normalized_sequence[batch.observed_mask],batch.normalized_observed[batch.observed_mask])
   hidden=~batch.observed_mask & out.sequence_validity
   error=(out.normalized_sequence.double()-batch.normalized_target.double())*batch.query_std[:,None].double()
   squared+=error.square()[hidden].sum().item();absolute+=error.abs()[hidden].sum().item();count+=hidden.sum().item()
   tail_squared+=error[:,-32:].square()[hidden[:,-32:]].sum().item();tail_absolute+=error[:,-32:].abs()[hidden[:,-32:]].sum().item();tail_count+=hidden[:,-32:].sum().item();windows+=len(batch.refs)
  row=dict(draw_index=i,mask_identity=item,windows=windows,repair96_mse=squared/count,repair96_mae=absolute/count,repair32_mse=tail_squared/tail_count,repair32_mae=tail_absolute/tail_count,hidden96_coordinates=count,hidden32_coordinates=tail_count)
  rows.append(row);print(json.dumps({'stage':'stage1_repair_evaluation',**{k:v for k,v in row.items() if k!='mask_identity'}}),flush=True)
 assert parameter_hash(model)==before
 summary={k:dict(mean=float(np.mean([r[k] for r in rows])),mask_std_ddof1=float(np.std([r[k] for r in rows],ddof=1))) for k in ('repair96_mse','repair96_mae','repair32_mse','repair32_mae')}
 _atomic_json(dict(rows=rows,summary=summary,selection=str(selection),evaluation_seconds=time.perf_counter()-started,repair_state_unchanged=True,scope='existing development mask bank; not blind holdout; single training seed2023; no forecast evaluation'),output)
 return output
