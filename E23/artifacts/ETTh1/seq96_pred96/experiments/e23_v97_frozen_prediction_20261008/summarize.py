import json,csv,statistics,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent;p=json.loads((R/'plan.json').read_text());rows=[]
for s in p['scenarios']:
 for metric in ['mae','mse']:
  path=R/s['name']/('evaluation_'+metric)/'twenty-test-draws.json'
  if not path.exists():continue
  d=json.loads(path.read_text());t=json.loads(Path(s['t1_pure_evaluation']).read_text());v=json.loads((Path(p['parent_root'])/s['name']/'E23-v79/evaluation_mae/twenty-test-draws.json').read_text())
  for a,b in zip(d['test_draws'],t['test_draws']):assert all(a[k]==b[k] for k in ['group_id','mask_seed','mask_sha256','windows','target_elements'])
  selected=json.loads((R/s['name']/'stage2/runs'/('e23-stage2.json' if metric=='mae' else 'e23-stage2-mse.json')).read_text())
  resources=json.loads(next((R/s['name']/'stage2/resources').glob('*.json')).read_text())
  rows.append(dict(scene=s['name'],selection_metric=metric,training_seed=2023,stage2_epochs=100,repair_frozen=True,loss='pure_mse',test_groups=20,actual_missing_fraction=s['actual_test_missing_fraction'],mse=d['test_mean']['mse'],mae=d['test_mean']['mae'],mse_std=statistics.stdev(x['mse'] for x in d['test_draws']),mae_std=statistics.stdev(x['mae'] for x in d['test_draws']),t1_mse=t['test_mean']['mse'],t1_mae=t['test_mean']['mae'],v79_mae_selected_mse=v['test_mean']['mse'],v79_mae_selected_mae=v['test_mean']['mae'],best_epoch=selected['best_epoch'],stage2_full_workflow_seconds=resources['full_workflow_seconds'],peak_allocated_gib=resources['peak_allocated_bytes']/2**30,peak_reserved_gib=resources['peak_reserved_bytes']/2**30,resource_scope='Stage2 only; reused Stage1 cost excluded',test_role='old20_development_masks'))
D=Path('/home/xyw/paper/E23/docs/experiments');out=D/'E23_v97_frozen_prediction_20261008.csv'
with out.open('w') as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
lines=['# E23-v97 frozen-repair prediction','', 'ETTh1 history96/horizon96; seed2023; Stage2 100 epochs, batch128, Adam lr0.0001, pure MSE; complete repairer frozen; original PatchTST unchanged. Main checkpoint validation MAE; secondary validation MSE. Same 20 development masks as T1 and v79; one training seed. GPU timing and memory below refer only to this new Stage2, not total training cost.','', '| Scene | v97 MSE / MAE | T1 MSE / MAE | v79 MSE / MAE | Stage2 seconds | GPU allocated / reserved GiB |','|---|---|---|---|---|---|']
for x in rows:
 if x['selection_metric']=='mae':lines.append(f"| {x['scene']} | {x['mse']:.6f} / {x['mae']:.6f} | {x['t1_mse']:.6f} / {x['t1_mae']:.6f} | {x['v79_mae_selected_mse']:.6f} / {x['v79_mae_selected_mae']:.6f} | {x['stage2_full_workflow_seconds']:.2f} | {x['peak_allocated_gib']:.3f} / {x['peak_reserved_gib']:.3f} |")
(D/'E23_v97_frozen_prediction_20261008.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(rows,indent=2))
