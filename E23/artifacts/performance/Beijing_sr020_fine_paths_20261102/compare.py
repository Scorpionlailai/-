from pathlib import Path
import json,csv,argparse
R=Path(__file__).resolve().parent;P=json.loads((R/'plan.json').read_text())
p=argparse.ArgumentParser();p.add_argument('--suffix',default='current');a=p.parse_args();models={}
base=json.loads((Path(P['R3_root'])/'outputs/evaluation/BeijingAirQuality/sr020_bl8/seed_2023/H_GLOBAL/results.json').read_text());models['R3_P0']={**base,'coarse':base['precal_test'],'final':base['test'],'backbone':'R3','mode':'P0'}
base=json.loads((Path(P['T1_root'])/'outputs/results.json').read_text());models['T1_P0']={**base,'backbone':'T1','mode':'P0'}
for job in ['R3_P1','R3_P2','R3_P3','T1_P2']:
 path=R/'outputs'/job
 if (path/'complete.json').exists():models[job]=json.loads((path/'results.json').read_text())
rows=[];paired=[]
for job,d in models.items():
 baseline=models[d['backbone']+'_P0'];assert d['test_masks']==20
 assert [x['mask_seed'] for x in d['per_mask']]==P['test_masks']
 for x,y in zip(d['per_mask'],baseline['per_mask']):
  for stage in ['coarse','final']:
   for reg in ['full96','recent32']:assert x[stage][reg]['coordinates']==y[stage][reg]['coordinates']
  for reg in ['full96','recent32']:
   assert abs(x['coarse'][reg]['mse']-y['coarse'][reg]['mse'])<5e-6,(job,reg,'coarse mismatch')
   paired.append(dict(job=job,mask_seed=x['mask_seed'],region=reg,delta_mse_vs_P0=x['final'][reg]['mse']-y['final'][reg]['mse'],delta_mae_vs_P0=x['final'][reg]['mae']-y['final'][reg]['mae']))
 for reg in ['full96','recent32']:
  c=d['coarse'][reg];f=d['final'][reg];b=baseline['final'][reg]
  row=dict(job=job,backbone=d['backbone'],mode=d['mode'],seed=2023,region=reg,coarse_mse=c['mse_mean'],coarse_mae=c['mae_mean'],final_mse=f['mse_mean'],final_mae=f['mae_mean'],delta_mse_vs_P0=f['mse_mean']-b['mse_mean'],delta_mae_vs_P0=f['mae_mean']-b['mae_mean'],mse_increase_vs_P0_pct=100*(f['mse_mean']/b['mse_mean']-1),mae_increase_vs_P0_pct=100*(f['mae_mean']/b['mae_mean']-1))
  if d['mode']!='P0':
   path=R/'outputs'/job;cost=json.loads((path/'resources.json').read_text());done=json.loads((path/'fine/complete.json').read_text());row.update(head_parameters=cost['head_registered_parameters'],P2_constant_input_weights=cost['P2_constant_input_weights'],p50_batch128_ms=cost['p50_batch128_ms'],fine_training_seconds=done['seconds'],peak_train_allocated_GiB=done['resources']['max_allocated_gib'],peak_train_reserved_GiB=done['resources']['max_reserved_gib'])
  rows.append(row)
interp=json.loads((Path(P['S0_S1_root'])/'outputs/S1/results.json').read_text())['coarse']
(R/'summary').mkdir(exist_ok=True)
def csvwrite(path,data):
 keys=list(dict.fromkeys(k for row in data for k in row))
 with path.open('w') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(data)
csvwrite(R/'summary'/f'path_comparison_{a.suffix}.csv',rows);csvwrite(R/'summary'/f'paired_masks_{a.suffix}.csv',paired)
lines=['# Beijing sr020_bl8 精恢复两路径实验','','seed2023，同100轮19000更新，验证全96点MAE选模，相同20组测试遮挡。所有臂固定粗恢复器，不训练粗恢复或预测头。','','| 骨干／精恢复 | 粗MSE | 粗MAE | 最终MSE | 最终MAE | 最终MSE相对本骨干P0变化 |','|---|---:|---:|---:|---:|---:|']
for row in sorted(rows,key=lambda x:x['job']):
 if row['region']=='full96':lines.append(f"| {row['job']} | {row['coarse_mse']:.6f} | {row['coarse_mae']:.6f} | {row['final_mse']:.6f} | {row['final_mae']:.6f} | {row['mse_increase_vs_P0_pct']:+.2f}% |")
lines+=['',f"无训练邻点插值参照：full96 MSE {interp['full96']['mse_mean']:.6f}，MAE {interp['full96']['mae_mean']:.6f}。复用已核验的S1初始化输出，仅取精恢复之前的coarse字段，不是S1训练后成绩。",'','P0：五类输入＋网络有界残差＋带符号邻点门控；P1：同五类输入，仅网络残差；P2：仅粗结果/可见值/mask，末两组输入恒零，仅网络残差；P3：五类输入，仅邻点门控。P1/P2删除780个门控输出参数，P3删除780个网络输出参数，骨干初值相同。P2另有1536个输入权重连接恒零字段，不能虚报为有效能力。','比较P0/P1定位直接门控路径；P1/P2定位邻点输入；P0/P3定位自由网络残差。P0与P1/P2的可达修正范围不同，这是输出路径删除的实际影响，不能将它完全解释为邻点语义。单训练seed不支持跨训练seed显著性结论，20mask不是20个训练seed。']
if all(x in models for x in ['R3_P1','R3_P2','R3_P3','T1_P2']):
 for job in ['R3_P1','R3_P2','R3_P3','T1_P2']:
  done=json.loads((R/'outputs'/job/'fine/complete.json').read_text());assert done['epochs']==100 and done['updates']==19000 and done['frozen_before']==done['frozen_after']
  for epoch in range(1,101):
   x=json.loads((R/'outputs'/job/'fine/epochs'/f'{epoch:03}.json').read_text());b=json.loads((Path(P['R3_root'])/'outputs/runs/BeijingAirQuality/sr020_bl8/seed_2023/H_GLOBAL/calibration/epochs'/f'{epoch:03}.json').read_text());assert x['batch_fingerprint']==b['batch_fingerprint']
 (R/'summary'/f'completion_audit_{a.suffix}.json').write_text(json.dumps(dict(status='passed',new_models=4,reused_P0_models=2,paired_training_epochs_verified=400,epochs_each=100,updates_each=19000,test_masks_each=20,coarse_all_frozen=True,coarse_replay_match_per_mask=True,seed=2023),indent=2)+'\n')
(R/f'report_{a.suffix}.md').write_text('\n'.join(lines)+'\n');print('available',list(models),flush=True)
