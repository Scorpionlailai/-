"""Read-only final evidence checks; separate from frozen training source."""
from pathlib import Path
import csv,hashlib,json,math
ROOT=Path(__file__).resolve().parent
WORKSPACE=Path('/home/xyw/paper/.e23-v33-refresh/E23')
SHARED=Path('/home/xyw/paper/E23')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 freeze=json.loads((ROOT/'code-freeze.json').read_text())
 assert all(sha(WORKSPACE/n)==h for n,h in freeze['files'].items())
 reuse=json.loads((ROOT/'reused-stage1-reference.json').read_text())
 assert sha(reuse['manifest'])==reuse['manifest_sha256']
 assert sha(reuse['record']['checkpoint'])==reuse['checkpoint_sha256']
 before=json.loads((ROOT/'shared-workspace-before.json').read_text())
 changed=[n for n,h in before.items() if not (SHARED/n).exists() or sha(SHARED/n)!=h]
 assert not changed, changed
 with (WORKSPACE/'docs/experiments/ETTh1_P4_E23_v33_comparison_2026-10-04.csv').open() as f:rows=list(csv.DictReader(f))
 with (WORKSPACE/'docs/experiments/ETTh1_P4_E23_v33_test_groups_2026-10-04.csv').open() as f:groups=list(csv.DictReader(f))
 assert len(rows)==6 and len(groups)==9
 records=[]
 for condition in ('C0','A','B'):
  cell=ROOT/condition;s1=json.loads((cell/'stage1/runs/e23-stage1.json').read_text());s2=json.loads((cell/'stage2/runs/e23-stage2.json').read_text())
  test=json.loads((cell/'three-test-draws.json').read_text());row=next(r for r in rows if r['model']=='E23_v33_'+condition)
  assert s1['epochs_completed']==100 and s2['epochs_completed']==100
  assert sha(s1['checkpoint'])==s1['checkpoint_sha256'] and sha(s2['checkpoint'])==s2['checkpoint_sha256']
  assert s2['loss']=='forecast_mse_only' and s2['selection_metric']=='mae'
  assert s2['predictor_initial_sha256']=='18ad18b5af292a9b526233ea426f81ffb3899d1f4fc850441060843d8adc404a'
  assert sha(cell/'three-test-draws.json')==row['test_result_sha256']
  for metric in ('mse','mae'):
   values=[g[metric] for g in test['test_groups']]
   mean=sum(values)/3;std=math.sqrt(sum((v-mean)**2 for v in values)/3)
   assert abs(float(row['test_'+metric+'_mean'])-mean)<1e-12
   assert abs(float(row['test_'+metric+'_std'])-std)<1e-12
   assert abs(float(row['validation_'+metric])-s2['best_val_'+metric])<1e-12
  for field in ('full_workflow_seconds','peak_allocated_gib','peak_reserved_gib'):
   assert math.isfinite(float(row[field])) and float(row[field])>0
  records.append(dict(condition=condition,stage1_sha256=s1['checkpoint_sha256'],stage2_sha256=s2['checkpoint_sha256'],stage1_best_epoch=s1['best_epoch'],stage2_best_epoch=s2['best_epoch'],test_mean=test['test_mean']))
 result=dict(status='complete',source_freeze_unchanged=True,source_files=len(freeze['files']),prediction_head_unchanged=True,stage2_training_unchanged=True,shared_existing_files_unchanged=True,records=records,tests=9,baseline_stage1_reused_without_retraining=True,decision=json.loads((ROOT/'final-selection.json').read_text()))
 (ROOT/'final-audit.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result))
if __name__=='__main__':main()
