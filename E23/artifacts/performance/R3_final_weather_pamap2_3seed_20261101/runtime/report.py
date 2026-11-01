"""Three independent training seeds; mask replicates remain within each model."""
from pathlib import Path
import json,csv
import numpy as np
ROOT=Path(__file__).resolve().parent

def report():
 rows=[]
 for path in sorted((ROOT/'outputs/evaluation').glob('*/*/seed_*/H_GLOBAL/results.json')):
  r=json.loads(path.read_text());assert not r['smoke_only'] and r['test_masks']==20 and r['export_verified']
  g=r['test']['full96'];pre=r['precal_test']['full96'];info=r['info']
  rows.append({'dataset':r['dataset'],'scene':r['scene'],'seed':r['seed'],'validation_mse':r['validation']['full96']['mse'],'validation_mae':r['validation']['full96']['mae'],'test_mse':g['mse_mean'],'test_mae':g['mae_mean'],'precal_test_mse':pre['mse_mean'],'precal_test_mae':pre['mae_mean'],'parameters':r['effective_parameters'],'main_epoch':info['coarse']['epoch'],'calibration_epoch':info['epoch'],'main_seconds':info['coarse']['training_seconds'],'calibration_seconds':info['calibration_seconds'],'export_verified':True})
 means=[]
 for dataset in ['Weather','PAMAP2','Exchange-Rate']:
  for scene in ['sr010_bl8','sr015_bl8','sr020_bl8','sr010_bl16']:
   rr=[r for r in rows if r['dataset']==dataset and r['scene']==scene]
   if len(rr)!=3:continue
   assert sorted(x['seed'] for x in rr)==[2023,2024,2025]
   m={'dataset':dataset,'scene':scene,'training_seeds':3,'test_masks_per_model':20,'effective_parameters':rr[0]['parameters']}
   for k in ['validation_mse','validation_mae','test_mse','test_mae','precal_test_mse','precal_test_mae']:
    m[k+'_mean']=float(np.mean([x[k] for x in rr]));m[k+'_seed_sd']=float(np.std([x[k] for x in rr],ddof=1))
   means.append(m)
 out=ROOT/'reports';out.mkdir(exist_ok=True)
 for name,data in [('per_seed',rows),('three_seed_summary',means)]:
  (out/(name+'.json')).write_text(json.dumps(data,indent=2)+'\n')
  if data:
   with (out/(name+'.csv')).open('w') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 lines=['# R3 三数据集正式训练进度与结果','','Commit `59419e9bbc8f2e8833c8265505d0e84fa2b04de4`；仅修复、无Stage2。','遮挡以各数据集其他基线的原始bank为准；详见baseline_alignment.json。','','|数据集|场景|测试MSE ± seed SD|测试MAE ± seed SD|参数|','|---|---|---:|---:|---:|']
 for m in means:lines.append(f"|{m['dataset']}|{m['scene']}|{m['test_mse_mean']:.6f} ± {m['test_mse_seed_sd']:.6f}|{m['test_mae_mean']:.6f} ± {m['test_mae_seed_sd']:.6f}|{m['effective_parameters']:,}|")
 lines+=['',f'已完成正式模型 {len(rows)}/36；凑齐三个seed的场景 {len(means)}/12。未凑齐的场景只保存逐seed记录，不冒充三seed结论。','20组mask属于单个训练模型内重复评价，不是20个训练seed。其他基线若只有seed2023，不能将其mask SD当训练seed波动。','原始观测硬覆盖、冻结及导出一致性由每模型评价检查。失败运行保留在jobs及logs，不剔除seed。']
 (out/'report.md').write_text('\n'.join(lines)+'\n')
 return len(rows)
if __name__=='__main__':report()
