from pathlib import Path
import json,csv,math,statistics
from scipy.stats import t
BASE=Path(__file__).resolve().parent.parent;DOCS=Path('/home/xyw/paper/E23/docs/experiments');rows=[]
for dirname in ['e23_stage1_boundary_coarse_tail_20261007','e23_stage1_boundary_tail_mae_20261007','e23_stage1_repeated_boundary_tail_mae_20261007','e23_stage1_one_sided_boundary_gpu1_20261007','e23_stage1_tail_selection_20261008','e23_stage1_local_one_sided_guide_20261008','e23_stage1_mean_and_dense_20261008']:
 root=BASE/dirname;plan=json.loads((root/'plan.json').read_text());entry=plan['models'][0]
 for scene in plan['scenarios']:
  p=root/entry['label']/scene['name']/'repair-audit.json';bp=root/'T1'/scene['name']/'repair-audit.json'
  if not p.exists():continue
  own=json.loads(p.read_text());baseline=json.loads(bp.read_text());a=own['test_groups'];b=baseline['test_groups'];assert len(a)==len(b)
  for region in ['all_hidden','last32_hidden']:
   for metric in ['mse','mae']:
    assert all(x['mask_sha256']==y['mask_sha256'] for x,y in zip(a,b))
    diff=[x['metrics'][region][metric]-y['metrics'][region][metric] for x,y in zip(a,b)];mean=statistics.mean(diff);sd=statistics.stdev(diff);half=t.ppf(.975,len(diff)-1)*sd/math.sqrt(len(diff))
    rows.append({'model':entry['label'],'scene':scene['name'],'region':region,'metric':metric,'groups':len(diff),'e23_mean':own['test_mean'][region][metric],'t1_mean':own['t1_test_mean'][region][metric],'e23_minus_t1':mean,'paired_mask_std':sd,'paired_mask_ci95_low':mean-half,'paired_mask_ci95_high':mean+half,'masks_e23_lower':sum(d<0 for d in diff),'ci_scope':'mask position randomness only; one training seed; development masks'})
with (DOCS/'E23_stage1_paired_mask_comparison_20261007.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps({'rows':len(rows),'file':str(DOCS/'E23_stage1_paired_mask_comparison_20261007.csv')}))
