"""Fixed-design ETTh1 repair transfer, existing twenty masks and frozen T1 audit."""
from pathlib import Path
from time import perf_counter
import argparse,json,csv,statistics
import torch
from weather_common import ROOT,protocol_for
from stage1_iteration import repair_totals,summarize_draws
from integrations.e23.config import load_e23_settings
from integrations.e23.calibration_ensemble import build_calibration_repair
from integrations.e23.train import load_stage1_selection,_new_memory,_starts,_key_sha256,_file_sha256
from integrations.e23.forecast_sensitive import parameter_hash
from integrations.forecast_compare.protocol import iter_p4_batches
from integrations.forecast_compare.test_mask_bank_v2 import load_test_bank
from integrations.forecast_compare.resources import selected_training_resources
KEYS=['repair96_mse','repair96_mae','repair32_mse','repair32_mae']
@torch.inference_mode()
def score(model,protocol,device):
 sums={k:0 for k in ['repair96_squared','repair96_absolute','repair32_squared','repair32_absolute','hidden96_coordinates','hidden32_coordinates']};windows=0
 for cpu in iter_p4_batches(protocol,'test',batch_size=128):
  b=cpu.to(device);starts=_starts(b,stride=model.stride,patches=model.patch_count,device=device)
  o=model(b.normalized_observed,b.observed_mask,b.query_mean,b.query_std,starts).explicit
  assert torch.equal(o.normalized_sequence[b.observed_mask],b.normalized_observed[b.observed_mask])
  parts=repair_totals(o.normalized_sequence,b.normalized_target,b.observed_mask,o.sequence_validity,b.query_std)
  for k in sums:sums[k]+=parts[k]
  windows+=len(b.refs)
 row=dict(windows=windows,hidden96_coordinates=sums['hidden96_coordinates'],hidden32_coordinates=sums['hidden32_coordinates'])
 for length in [96,32]:
  for name,key in [('mse','squared'),('mae','absolute')]:row[f'repair{length}_{name}']=sums[f'repair{length}_{key}']/sums[f'hidden{length}_coordinates']
 return row
@torch.inference_mode()
def main():
 torch.set_num_threads(4);p=json.loads((ROOT/'plan.json').read_text());scene=p['scenarios'][0];protocol=protocol_for(p,scene)
 settings=load_e23_settings(Path(p['models'][0]['config']),protocol=protocol,allow_variable_scaling=True);device=torch.device(p['device'])
 selection=ROOT/'E23-v125'/scene['name']/'stage1/runs/e23-stage1-mae.json';r=json.loads(selection.read_text())
 payload=load_stage1_selection(selection,protocol,settings,device=device,allow_calibration_ensemble=True)
 model=build_calibration_repair(protocol,settings,r['calibration_ensemble_spec'],device=device);model.load_state_dict(payload['model_state'],strict=True)
 model.historical.retrieval.load_state_dict(payload['retrieval_state']);memory=_new_memory(protocol);memory.load_state_dict(payload['memory_state']);model.historical.refresh_snapshot(memory.snapshot());model.eval().requires_grad_(False)
 before=parameter_hash(model),memory.snapshot_hash(),_key_sha256(model.historical.retrieval.state_dict())
 del payload
 bank=load_test_bank(protocol,Path(scene['test_bank']));audit_path=Path(p['t1_repair_audit']);baseline=json.loads(audit_path.read_text())
 t1=json.loads(Path(scene['t1_repair_selection']).read_text());assert baseline['checkpoint_sha256']==t1['checkpoint_sha256']==_file_sha256(Path(t1['checkpoint']))
 assert baseline['state_unchanged'] and len(baseline['test_groups'])==len(bank)==20
 rows=[];other=[];start=perf_counter()
 for index,(view,b) in enumerate(zip(bank,baseline['test_groups'])):
  assert view.mask_sha256['test']==b['mask_sha256']
  row=dict(group_id=b['group_id'],mask_sha256=view.mask_sha256['test'],**score(model,view,device));assert row['windows']==b['windows']==2688
  ref=dict(group_id=b['group_id'],mask_sha256=b['mask_sha256'],windows=b['windows'])
  for length,region in [(96,'all_hidden'),(32,'last32_hidden')]:
   assert row[f'hidden{length}_coordinates']==b['metrics'][region]['hidden_coordinates']
   ref[f'hidden{length}_coordinates']=b['metrics'][region]['hidden_coordinates']
   for name in ['mse','mae']:ref[f'repair{length}_{name}']=b['metrics'][region][name]
  rows.append(row);other.append(ref);print(json.dumps(dict(index=index,**row)),flush=True)
 assert before==(parameter_hash(model),memory.snapshot_hash(),_key_sha256(model.historical.retrieval.state_dict()))
 resources=selected_training_resources([Path(p['parent_selection']),*[Path(s['selection']) for s in p['calibration_sources']],selection]);assert resources['resource_status']=='measured_all_stages'
 out=ROOT/'E23-v125'/scene['name']/'repair_evaluation_mae';out.mkdir(exist_ok=False)
 result=dict(model='E23-v125',dataset='ETTh1',scene=scene['name'],start_rate=scene['start_rate'],block_len=scene['block_len'],seq_len=96,pred_len=96,n_vars=7,metric_space='train_standardized',draws=rows,summary=summarize_draws(rows),training_resources=resources,selection_manifest=str(selection),checkpoint_sha256=r['checkpoint_sha256'],test_mask_bank=scene['test_bank'],test_mask_bank_sha256=_file_sha256(Path(scene['test_bank'])),parameters_memory_and_keys_unchanged=True,visible_values_exact=True,forecast_evaluated=False,test_role='original20 development masks; no checkpoint selection on test',test_inference_seconds=perf_counter()-start)
 (out/'twenty-test-draws.json').write_text(json.dumps(result,indent=2)+'\n')
 comparison=[]
 for name,draws in [('E23-v125',rows),('T1',other)]:
  row=dict(model=name,scene=scene['name'],test_masks=20)
  for k in KEYS:row[k]=statistics.mean(d[k] for d in draws);row[k+'_mask_std_ddof1']=statistics.stdev(d[k] for d in draws)
  comparison.append(row)
 with (ROOT.parent/'matched_repair_comparison.csv').open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(comparison[0]));w.writeheader();w.writerows(comparison)
 paired={k:dict(mean_difference=comparison[0][k]-comparison[1][k],improved_masks=sum(a[k]<b[k] for a,b in zip(rows,other))) for k in KEYS}
 (ROOT.parent/'paired_repair_differences.json').write_text(json.dumps(dict(scope='original20 mask uncertainty, one training seed',t1_audit_source=str(audit_path),t1_audit_sha256=_file_sha256(audit_path),comparisons=paired),indent=2)+'\n')
 print(json.dumps(dict(status='completed',metrics=comparison[0],t1=comparison[1],paired=paired)),flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--selection',choices=['mae'],default='mae');parser.parse_args();main()
