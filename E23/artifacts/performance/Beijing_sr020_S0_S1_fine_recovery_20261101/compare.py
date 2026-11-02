from pathlib import Path
import json,csv,argparse
R=Path(__file__).resolve().parent;P=json.loads((R/'plan.json').read_text())
p=argparse.ArgumentParser();p.add_argument('--suffix',default='current');a=p.parse_args();rows=[];models={}
for v in ['S0','S1']:
 path=R/'outputs'/v/'results.json'
 if path.exists():
  d=json.loads(path.read_text());models[v]=d
for v,root in [('E6',Path(P['E6_root'])/'evaluation/BeijingAirQuality/sr020_bl8/seed_2023/E6'),('R3',Path(P['R3_root'])/'outputs/evaluation/BeijingAirQuality/sr020_bl8/seed_2023/H_GLOBAL')]:
 d=json.loads((root/'results.json').read_text());models[v]={**d,'coarse':d['precal_test'],'final':d['test']}
d=json.loads((Path(P['T1_root'])/'results.json').read_text());models['T1']=d
base=models['R3']['final']['full96']
for v in ['S0','S1','E6','R3','T1']:
 if v not in models:continue
 d=models[v];assert d['test_masks']==20
 assert [x['mask_seed'] for x in d['per_mask']]==P['test_masks']
 for x,y in zip(d['per_mask'],models['R3']['per_mask']):
  for stage in ['coarse','final']:
   for reg in ['full96','recent32']:assert x[stage][reg]['coordinates']==y[stage][reg]['coordinates']
 for reg in ['full96','recent32']:
  c=d['coarse'][reg];f=d['final'][reg];b=models['R3']['final'][reg]
  row=dict(model=v,seed=2023,region=reg,initial_mse=c['mse_mean'],initial_mae=c['mae_mean'],final_mse=f['mse_mean'],final_mae=f['mae_mean'],final_mse_increase_vs_R3_pct=100*(f['mse_mean']/b['mse_mean']-1),final_mae_increase_vs_R3_pct=100*(f['mae_mean']/b['mae_mean']-1),test_masks=20)
  if v in ['S0','S1']:
   done=json.loads((R/'outputs'/v/'fine/complete.json').read_text());resource=json.loads((R/'outputs'/v/'resources.json').read_text())
   row.update(nominal_parameters=resource['nominal_trainable_parameters'],fine_training_seconds=done['seconds'],phase_one_training_seconds=0,p50_batch128_ms=resource['latency']['initialization_and_fine']['p50_batch_ms'],train_peak_allocated_gib=done['resources']['max_allocated_gib'],train_peak_reserved_gib=done['resources']['max_reserved_gib'])
  rows.append(row)
(R/'summary').mkdir(exist_ok=True)
keys=list(dict.fromkeys(k for row in rows for k in row))
with (R/'summary'/f'comparison_{a.suffix}.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
lines=['# Beijing sr020_bl8：简单初始化＋精恢复对照','','训练seed2023，各模型20组相同测试mask。E6、R3、T1复用既有结果。新增S0/S1只有精恢复网络，初始填补不含可训练网络；每臂独立从相同seed初始化并训练100轮19000更新。','','| 输入 | 初始化MSE | 初始化MAE | 精恢复后MSE | 精恢复后MAE | 最终MSE相对R3变化 |','|---|---:|---:|---:|---:|---:|']
for x in rows:
 if x['region']=='full96':lines.append(f"| {x['model']} | {x['initial_mse']:.6f} | {x['initial_mae']:.6f} | {x['final_mse']:.6f} | {x['final_mae']:.6f} | {x['final_mse_increase_vs_R3_pct']:+.2f}% |")
lines+=['','本轮为单训练seed开发比较；不把20mask当作20个独立训练seed。不依据测试更改结构或重选checkpoint。S1的guide-coarse理论为零，保留同款邻点门控并说明其条件不活跃；名义参数不能解释为每臂实际功能维度完全相同。','初始填补和邻点引导只使用当前窗口内合法可见值。S0零定义在训练集标准化尺度，S1全变量无观测时也回退零；不访问测试隐藏真值。','S0/S1无阶段一训练，成本为0；E6、R3的阶段一成本另在原资源记录中计入，不能只比较阶段二耗时来声称总成本等同。']
if all(v in models for v in ['S0','S1']):
 for v in ['S0','S1']:
  for epoch in range(1,101):
   x=json.loads((R/'outputs'/v/'fine/epochs'/f'{epoch:03}.json').read_text());y=json.loads((Path(P['R3_root'])/'outputs/runs/BeijingAirQuality/sr020_bl8/seed_2023/H_GLOBAL/calibration/epochs'/f'{epoch:03}.json').read_text());assert x['batch_fingerprint']==y['batch_fingerprint']
 (R/'summary'/f'completion_audit_{a.suffix}.json').write_text(json.dumps(dict(status='passed',new_models=2,phase_one_network_absent=True,epochs_each=100,updates_each=19000,paired_training_epochs_verified=200,test_masks_each=20,training_seeds=[2023],scored_coordinates_match_all_arms=True),indent=2)+'\n')
lines.append('')
(R/f'report_{a.suffix}.md').write_text('\n'.join(lines))
print('available_models',list(models),flush=True)
