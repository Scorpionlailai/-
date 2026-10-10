"""Summarize the completed single-scene v99 trial; never changes training inputs."""
import csv
import json
import statistics
from pathlib import Path

E = Path('/home/xyw/paper/E23')
X = E/'artifacts/ETTh1/seq96_pred96/experiments'
R = X/'e23_v99_selective_bf16_20261008'
D = E/'docs/experiments'
S = 'sr015_bl8'
C = R/'E23-v99'/S


def read(p):
    return json.loads(Path(p).read_text())


def csv_write(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


assert read(R/'status.json')['status'] == 'pilot_completed'
plan = read(R/'plan.json')
for path, sha in read(R/'frozen-definition.json')['sources'].items():
    import hashlib
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha

v98_cost = read(X/'e23_v98_compact_readout_20261008/complete_training_cost.json')
resource_rows = []
costs = {m: v98_cost[m] for m in ['v97', 'v98']}
stages = []
for phase, path in [
    ('stage1_coarse', R/'coarse'/S/'E23-v99-coarse/stage1/resources'),
    ('stage1_refinement', C/'stage1/resources'),
    ('stage2_forecast', C/'stage2/resources'),
]:
    paths = list(path.glob('*.json'))
    assert len(paths) == 1
    p = paths[0]
    r = read(p)
    assert r['status'] == 'completed' and r['epochs_completed'] == 100
    stages.append(dict(resource_path=str(p), full_workflow_seconds=r['full_workflow_seconds'],
                       peak_allocated_gib=r['peak_allocated_bytes']/2**30,
                       peak_reserved_gib=r['peak_reserved_bytes']/2**30))
    resource_rows.append(dict(model='v99', scene=S, phase=phase, epochs=100,
                              full_workflow_seconds=r['full_workflow_seconds'],
                              peak_allocated_gib=r['peak_allocated_bytes']/2**30,
                              peak_reserved_gib=r['peak_reserved_bytes']/2**30,
                              precision=r['precision'], amp=r['amp'], source=str(p)))
costs['v99'] = dict(stages=stages,
                    full_workflow_seconds=sum(s['full_workflow_seconds'] for s in stages),
                    stage1_full_workflow_seconds=sum(s['full_workflow_seconds'] for s in stages[:2]),
                    peak_allocated_gib=max(s['peak_allocated_gib'] for s in stages),
                    peak_reserved_gib=max(s['peak_reserved_gib'] for s in stages))
resource_rows.append(dict(model='v99', scene=S, phase='all_training_phases', epochs=300,
                          full_workflow_seconds=costs['v99']['full_workflow_seconds'],
                          peak_allocated_gib=costs['v99']['peak_allocated_gib'],
                          peak_reserved_gib=costs['v99']['peak_reserved_gib'],
                          precision='Stage1 selective BF16 training/FP32 validation; Stage2 FP32',
                          amp='Stage1 only', source=';'.join(s['resource_path'] for s in stages)))
csv_write(D/'E23_v99_selective_precision_resources_20261008.csv', resource_rows)
costs['scope'] = 'Sum of cold coarse Stage1, refined Stage1 and Stage2 full workflows, each 100 epochs; excludes tests and separate probes.'
(R/'complete_training_cost.json').write_text(json.dumps(costs, indent=2)+'\n')

a = read(C/'repair-audit.json')
b = read(X/'e23_stage1_mean_and_dense_20261008/E23-v97'/S/'repair-audit.json')
for own, base in zip(a['test_groups'], b['test_groups']):
    assert own['mask_sha256'] == base['mask_sha256']
with (D/'E23_v97_frozen_prediction_20261008.csv').open() as f:
    base_forecast = next(r for r in csv.DictReader(f) if r['scene'] == S and r['selection_metric'] == 'mae')
rows = []
for split, metrics in [('validation', a['split_validation']['metrics']), ('old20_development_test', a['test_mean'])]:
    for region in ['all_hidden', 'last32_hidden']:
        std = {k: statistics.stdev(g['metrics'][region][k] for g in a['test_groups'])
               for k in ['mse', 'mae']} if split != 'validation' else dict(mse='', mae='')
        rows.append(dict(model='v99', scene=S, task='repair', scope=region, split=split,
                         selection_metric='fixed_validation_full_hidden_global_mse', training_seed=2023,
                         mask_groups=20 if split != 'validation' else 1,
                         mse=metrics[region]['mse'], mae=metrics[region]['mae'],
                         mse_mask_std_ddof1=std['mse'], mae_mask_std_ddof1=std['mae'],
                         repair_frozen=True, forecast_loss='pure_mse', source=str(C/'repair-audit.json')))
forecast = {}
for metric in ['mae', 'mse']:
    j = C/f'evaluation_{metric}/twenty-test-draws.json'
    f = read(j)
    selection = read(C/'stage2/runs'/('e23-stage2.json' if metric == 'mae' else 'e23-stage2-mse.json'))
    assert selection['repair_trainable_parameters'] == 0
    assert selection['loss'] == 'forecast_mse_only'
    assert f['predictor_initial_sha256'] == plan['predictor_initial_sha256']
    forecast[metric] = f
    rows.append(dict(model='v99', scene=S, task='forecast', scope='all_future96',
                     split='old20_development_test', selection_metric=metric, training_seed=2023,
                     mask_groups=20, mse=f['test_mean']['mse'], mae=f['test_mean']['mae'],
                     mse_mask_std_ddof1=f['test_mask_std_ddof1']['mse'],
                     mae_mask_std_ddof1=f['test_mask_std_ddof1']['mae'],
                     repair_frozen=True, forecast_loss='pure_mse', source=str(j)))
csv_write(D/'E23_v99_selective_precision_results_20261008.csv', rows)

coarse = read(R/'coarse'/S/'E23-v99-coarse/stage1/runs/e23-stage1.json')
refined = read(C/'stage1/runs/e23-stage1.json')
primary = forecast['mae']['test_mean']
speed = 100*(1-costs['v99']['full_workflow_seconds']/costs['v97']['full_workflow_seconds'])
memory = 100*(1-costs['v99']['peak_allocated_gib']/costs['v97']['peak_allocated_gib'])
summary = dict(scene=S, forecast_mae_selected=primary,
                repair_test=a['test_mean'], repair_validation=a['split_validation']['metrics'],
                full_workflow_seconds=costs['v99']['full_workflow_seconds'],
                peak_allocated_gib=costs['v99']['peak_allocated_gib'],
                time_reduction_vs_v97_percent=speed, peak_allocated_reduction_vs_v97_percent=memory,
                forecast_both_better_than_v97=all(primary[k]<float(base_forecast[k]) for k in ['mse','mae']),
                forecast_both_better_than_t1=all(primary[k]<float(base_forecast['t1_'+k]) for k in ['mse','mae']))
(R/'pilot_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
table = '\n'.join(f"| {region} {metric.upper()} | {b['test_mean'][region][metric]:.6f} | {a['test_mean'][region][metric]:.6f} | {a['t1_test_mean'][region][metric]:.6f} |"
                   for region in ['all_hidden','last32_hidden'] for metric in ['mse','mae'])
time_table = '\n'.join(f"| {model} | {costs[model]['full_workflow_seconds']/60:.3f} | {costs[model]['peak_allocated_gib']:.3f} | {costs[model]['peak_reserved_gib']:.3f} |" for model in ['v97','v98','v99'])
md = f'''# E23-v99 局部混合精度试验结果（2026-10-08）

## 配置与比较范围

采用 [预先固定的设计](/home/xyw/paper/E23/docs/experiments/2026-10-08_E23_v99_selective_precision_plan.md)：v97 原64维八头结构无参数压缩；仅 Stage1 训练的窗口编码/候选读出启用 BF16，并编译候选读出。所有验证/测试、历史查询、损失、权重、Adam 以及 Stage2 均保持原精度。Stage2 整个修复器冻结、原PatchTST、纯MSE、启用固定验证缓存。

单场景 ETTh1，seq96/pred96，起点率0.15/块长8，seed2023/batch128；粗修复、细化和预测各100轮。粗修复选第{coarse['best_epoch']}轮、细化选第{refined['best_refinement_epoch']}轮，但三阶段均完整跑满100轮。20组测试缺失位置沿用开发实验，非盲测、非训练多种子。

## 修复测试20组均值

| 指标 | v97 | v99 | T1 已固定参考 |
|---|---:|---:|---:|
{table}

FP32验证全窗口 hidden MSE/MAE = {a['split_validation']['metrics']['all_hidden']['mse']:.6f}/{a['split_validation']['metrics']['all_hidden']['mae']:.6f}；近32点 = {a['split_validation']['metrics']['last32_hidden']['mse']:.6f}/{a['split_validation']['metrics']['last32_hidden']['mae']:.6f}。正式checkpoint按验证全hidden MSE锁定，测试未用于选epoch。

## 冻结修复器后的预测测试

| 模型/选模 | MSE | MAE |
|---|---:|---:|
| v97 验证MAE | {float(base_forecast['mse']):.6f} | {float(base_forecast['mae']):.6f} |
| v99 验证MAE（主结果） | {primary['mse']:.6f} | {primary['mae']:.6f} |
| v99 验证MSE（副结果） | {forecast['mse']['test_mean']['mse']:.6f} | {forecast['mse']['test_mean']['mae']:.6f} |
| T1 冻结纯MSE已固定参考 | {float(base_forecast['t1_mse']):.6f} | {float(base_forecast['t1_mae']):.6f} |

主结果比v97两项更好：{summary['forecast_both_better_than_v97']}；比T1两项更好：{summary['forecast_both_better_than_t1']}。缺失位置标准差在结果CSV。预测头初始SHA与原v97一致，Stage2修复可训练参数为0。

## 包含父粗修复的完整训练资源

| 模型 | 完整训练分钟 | 峰值allocated GiB | 峰值reserved GiB |
|---|---:|---:|---:|
{time_table}

v99 相比v97完整时间减少 {speed:.2f}%、峰值allocated减少 {memory:.2f}%。这是单次、GPU0实测；包含所有三阶段准备、训练、验证、首次编译和保存，不包含最终测试及独立微测。v99时间收益同时包含局部混合精度、编译和Stage2验证缓存，不能归因于一个模块。自动 evaluation JSON 聚合的训练资源只包含细化与预测，不能用作总成本；正式引用 [complete_training_cost.json]({R}/complete_training_cost.json)。T1本轮未用同样执行优化重跑，不据此比较架构效率。

## 查验与后续

- [精度CSV](/home/xyw/paper/E23/docs/experiments/E23_v99_selective_precision_results_20261008.csv)
- [资源CSV](/home/xyw/paper/E23/docs/experiments/E23_v99_selective_precision_resources_20261008.csv)
- 所有运行命令、模型权重、散列冻结源码及原始指标：`{R}`。

只完成一个场景，尚无其余三个场景或完整多变量训练结果。保留v97的四场景证据；BF16改变训练数值，需要后续数据集验证。E24运行源码和封存v79均未修改。
'''
(D/'2026-10-08_E23_v99_selective_precision_results.md').write_text(md)
print(json.dumps(summary, indent=2))
