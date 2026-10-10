"""Read-only verification of formal v30 artifacts; writes its own audit evidence."""
from pathlib import Path
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from scripts.run_e23_v30_trial import verify_freeze
from scripts.report_e23_v30_trial import load_candidate, validate_test_result, choose_v30_candidate

project=Path.cwd()
root=project/'artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/e23_v30_stacked_parallel'
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
verify_freeze(root)
user_csv=project/'docs/experiments/E22_all_datasets_missing_scenarios.csv'
assert sha(user_csv)=='530947f4a58aaba96f23d7a254cc0b279bc94fc78b70c228869fcef484721fa2'
bank_path=root.parent.parent/'test_mask_bank_20261002/manifest.json'
assert sha(bank_path)=='95fe49c78c6b884477e70c05f18a6c6c06f009e75d3afa01946275207b70e3a9'
bank=json.loads(bank_path.read_text())
for group in bank['groups']:
 assert sha(bank_path.parent/group['path'])==group['sha256']
decision=json.loads((root/'validation_selection.json').read_text())
records=[]
for layers,groups in ((1,8),(4,1),(4,8)):
 cell=root/f'l{layers}_g{groups}'
 config=project/f'configs/e23_multi_expert_p4_width64_stack_l{layers}_g{groups}_v30.yaml'
 record=load_candidate(cell,config)
 assert record==next(item for item in decision['candidates'] if item['model']==record['model'])
 validate_test_result(record,json.loads((cell/'three-test-draws.json').read_text()))
 for stage in ('stage1','stage2'):
  lines=(cell/f'{stage}.log').read_text().splitlines()
  completed=[line for line in lines if 'best_epoch=' in line and line.startswith('stage=')]
  assert len(completed)==100,(cell.name,stage,len(completed))
  assert 'epoch=100/100' in completed[-1]
 records.append(record)
winner=choose_v30_candidate(records)
assert decision['winner']==(winner['model'] if winner else None)
output=project/'docs/experiments'
comparison=output/'ETTh1_P4_E23_v30_T1_comparison_2026-10-03.csv'
test_groups=output/'ETTh1_P4_E23_v30_test_groups_2026-10-03.csv'
layer_csv=output/'ETTh1_P4_E23_v30_layer_diagnostics_2026-10-03.csv'
read=lambda path:list(csv.DictReader(path.open(newline='')))
rows=read(comparison);test_rows=read(test_groups);layer_rows=read(layer_csv)
assert len(rows)==7 and len({row['model'] for row in rows})==7
assert len(test_rows)==9 and len(layer_rows)==18
old_rows=read(output/'ETTh1_P4_E23_v28_v29_T1_comparison_2026-10-03.csv')
for old in old_rows:
 row=next(row for row in rows if row['model']==old['model'])
 assert all(row[k]==v for k,v in old.items()),old['model']
for record in records:
 model=record['model'];cell=Path(record['root'])
 row=next(row for row in rows if row['model']==model)
 result=json.loads((cell/'three-test-draws.json').read_text())
 for metric in ('mse','mae'):
  for stat in ('mean','std'):
   assert float(row[f'test_{metric}_{stat}'])==result[f'test_{stat}'][metric]
 for key in ('full_workflow_seconds','training_stages_seconds','peak_allocated_gib','peak_reserved_gib'):
  assert float(row[key])==record['training_resources'][key]
 assert row['validation_gate_passed']==str(record['best_val_mse']<decision['reference_mse'] and record['best_val_mae']<=decision['reference_mae'])
 assert row['selected_candidate']==str(decision['winner']==model)
 group_rows=[row for row in test_rows if row['model']==model]
 assert len(group_rows)==3
 for raw,reported in zip(result['test_groups'],group_rows):
  for metric in ('mse','mae'):assert float(reported[metric])==raw[metric]
  assert reported['sha256']==raw['sha256']
  assert reported['checkpoint_sha256']==record['checkpoint_sha256']
 diag=json.loads((cell/'layer_diagnostics.json').read_text())
 for stage in ('stage1','stage2'):
  reported=[row for row in layer_rows if row['model']==model and row['stage']==stage]
  assert len(reported)==record['recovery_layers']
  for raw,row in zip(diag[stage]['layers'],reported):
   assert int(row['layer_index'])==raw['layer_index']
   assert float(row['global_mse'])==raw['global_mse']
   assert len(json.loads(row['mean_group_weights']))==record['expert_groups']+1
   assert row['checkpoint_sha256']==diag[stage]['checkpoint_sha256']
audit=dict(completed_utc=datetime.now(timezone.utc).isoformat(),all_checks_passed=True,
 formal_trials=3,epochs_per_stage=100,formal_stage_completions=6,test_draws_per_trial=3,new_test_rows=9,
 comparison_rows=7,layer_rows=18,code_frozen_files=len(json.loads((root/'code-freeze.json').read_text())['files']),
 code_freeze_sha256=sha(root/'code-freeze.json'),user_csv_sha256=sha(user_csv),test_mask_bank_sha256=sha(bank_path),
 selected_model=winner['model'] if winner else decision['retained_model_if_no_winner'],
 new_candidate_promoted=winner is not None,
 selection_source=decision['selection_source'],test_results_used_for_training_or_selection=False,
 report_sha256={str(path.relative_to(project)):sha(path) for path in (comparison,test_groups,layer_csv)},
 candidate_checkpoints={record['model']:record['checkpoint_sha256'] for record in records})
(root/'final-audit.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n')
print(json.dumps(audit,indent=2))
