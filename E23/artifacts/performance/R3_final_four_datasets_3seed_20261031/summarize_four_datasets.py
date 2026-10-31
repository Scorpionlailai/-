from pathlib import Path
import json,csv,statistics
ROOT=Path(__file__).resolve().parent
summary=[];perseed=[];checked=0
for ds in ['ETTh1','BeijingAirQuality','PhysioNet2012','Exchange-Rate']:
 for scene in ['sr010_bl8','sr015_bl8','sr020_bl8','sr010_bl16']:
  paths=sorted((ROOT/'outputs/evaluation'/ds/scene).glob('seed_*/H_GLOBAL/results.json'))
  results=[json.loads(p.read_text()) for p in paths]
  assert sorted(r['seed'] for r in results)==[2023,2024,2025]
  for r in results:
   assert r['test_masks']==20 and len(r['per_mask'])==20 and r['export_verified'] and not r['smoke_only']
   base=ROOT/'outputs/runs'/ds/scene/f"seed_{r['seed']}"/'H_GLOBAL'
   for stage in ['main','calibration']:
    c=json.loads((base/stage/'complete.json').read_text());assert c['status']=='complete' and c['epochs']==100 and c['updates']==100*c['updates_per_epoch']
    if stage=='calibration':assert c['frozen_before']==c['frozen_after']
   checked+=1
   for stage,field in [('coarse','precal_test'),('final','test')]:
    for scope in ['full96','recent32']:
     metrics=r[field][scope]
     perseed.append(dict(dataset=ds,scene=scene,training_seed=r['seed'],stage=stage,scope=scope,test_masks=20,mse=metrics['mse_mean'],mae=metrics['mae_mean']))
  for stage in ['coarse','final']:
   for scope in ['full96','recent32']:
    rr=[r for r in perseed if (r['dataset'],r['scene'],r['stage'],r['scope'])==(ds,scene,stage,scope)]
    summary.append(dict(dataset=ds,scene=scene,stage=stage,scope=scope,n_training_seeds=3,test_masks_per_seed=20,mse_mean=statistics.mean(r['mse'] for r in rr),mse_std_ddof1=statistics.stdev(r['mse'] for r in rr),mae_mean=statistics.mean(r['mae'] for r in rr),mae_std_ddof1=statistics.stdev(r['mae'] for r in rr)))
assert checked==48
out=ROOT/'summary';out.mkdir(exist_ok=True)
for name,rows in [('R3_four_datasets_all_stages.csv',summary),('R3_four_datasets_final_full96.csv',[r for r in summary if r['stage']=='final' and r['scope']=='full96']),('R3_four_datasets_per_training_seed.csv',perseed)]:
 with (out/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(out/'completion_audit.json').write_text(json.dumps(dict(status='passed',models_checked=checked,epochs_each_stage=100,complete_budget_all=True,frozen_prefix_all=True,test_masks_each_seed=20,export_verified_all=True,statistics='First mean of20 masks within each model, then sampleSD across independent training seeds2023/2024/2025'),indent=2)+'\n')
print('48 models verified; summaries saved in',out)
