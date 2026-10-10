"""Generate the efficiency pilot report from measured artifacts, including parents."""
import csv
import json
import statistics
from pathlib import Path

E23 = Path('/home/xyw/paper/E23')
EXP = E23 / 'artifacts/ETTh1/seq96_pred96/experiments'
DOC = E23 / 'docs/experiments'
PILOT = EXP / 'e23_v98_compact_readout_20261008'
SCENE = 'sr015_bl8'
ROOT98 = PILOT / 'E23-v98' / SCENE
ROOT97 = EXP / 'e23_stage1_mean_and_dense_20261008/E23-v97' / SCENE


def read(path):
    return json.loads(Path(path).read_text())


def resource(directory):
    matches = list(Path(directory).glob('*.json'))
    assert len(matches) == 1, matches
    return matches[0], read(matches[0])


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


phases = {
    'v97': [
        EXP / 'e23_v79_v69_global_level_four_pureMSE_20261007' / SCENE / 'E23-v79/stage1/resources',
        ROOT97 / 'stage1/resources',
        EXP / 'e23_v97_frozen_prediction_20261008' / SCENE / 'stage2/resources',
    ],
    'v98': [
        PILOT / 'coarse' / SCENE / 'E23-v98-coarse/stage1/resources',
        ROOT98 / 'stage1/resources',
        ROOT98 / 'stage2/resources',
    ],
}
cost = {}
resource_rows = []
for model, paths in phases.items():
    records = []
    for phase, directory in zip(['stage1_coarse', 'stage1_refinement', 'stage2_forecast'], paths):
        p, d = resource(directory)
        assert d['status'] == 'completed' and d['epochs_completed'] == 100
        records.append((p, d))
        resource_rows.append(dict(model=model, scene=SCENE, phase=phase, epochs=100,
                                  full_workflow_seconds=d['full_workflow_seconds'],
                                  peak_allocated_gib=d['peak_allocated_bytes']/2**30,
                                  peak_reserved_gib=d['peak_reserved_bytes']/2**30,
                                  source=str(p), scope=d['timing_scope']))
    cost[model] = dict(
        full_workflow_seconds=sum(d['full_workflow_seconds'] for _, d in records),
        stage1_full_workflow_seconds=sum(d['full_workflow_seconds'] for _, d in records[:2]),
        peak_allocated_gib=max(d['peak_allocated_bytes']/2**30 for _, d in records),
        peak_reserved_gib=max(d['peak_reserved_bytes']/2**30 for _, d in records),
        stages=[dict(resource_path=str(p), full_workflow_seconds=d['full_workflow_seconds'],
                     peak_allocated_gib=d['peak_allocated_bytes']/2**30,
                     peak_reserved_gib=d['peak_reserved_bytes']/2**30) for p, d in records],
    )
    resource_rows.append(dict(model=model, scene=SCENE, phase='all_training_phases', epochs=300,
                              full_workflow_seconds=cost[model]['full_workflow_seconds'],
                              peak_allocated_gib=cost[model]['peak_allocated_gib'],
                              peak_reserved_gib=cost[model]['peak_reserved_gib'],
                              source=';'.join(str(p) for p, _ in records),
                              scope='Sum of all three measured training workflows including parent; excludes final tests and separate audit jobs'))

write_csv(DOC / 'E23_v98_compact32_resources_20261008.csv', resource_rows)
cost['scope'] = 'Includes cold coarse Stage1 + refined Stage1 + Stage2; each 100 epochs. Excludes final tests and audit/benchmark jobs.'
cost['display_correction'] = 'evaluation_mae/training_resources reports refinement+Stage2 only; the cold parent must additionally be counted. Use this complete cost record for full-pipeline reporting.'
(PILOT / 'complete_training_cost.json').write_text(json.dumps(cost, indent=2)+'\n')

audits = {'v97': read(ROOT97 / 'repair-audit.json'), 'v98': read(ROOT98 / 'repair-audit.json')}
forecast98 = read(ROOT98 / 'evaluation_mae/twenty-test-draws.json')
with (DOC / 'E23_v97_frozen_prediction_20261008.csv').open() as f:
    forecast97 = next(r for r in csv.DictReader(f) if r['scene'] == SCENE and r['selection_metric'] == 'mae')
rows = []
for model, audit in audits.items():
    for split, metric_key in [('validation', 'split_validation'), ('old20_development_test', 'test_mean')]:
        metrics = audit[metric_key]['metrics'] if split == 'validation' else audit[metric_key]
        for scope in ['all_hidden', 'last32_hidden']:
            std = {m: statistics.stdev(g['metrics'][scope][m] for g in audit['test_groups'])
                   for m in ['mse', 'mae']} if split != 'validation' else {'mse': '', 'mae': ''}
            rows.append(dict(model=model, scene=SCENE, task='repair', scope=scope, split=split,
                             selection='fixed_validation_full_hidden_global_mse', training_seed=2023,
                             mask_groups=20 if split != 'validation' else 1,
                             mse=metrics[scope]['mse'], mae=metrics[scope]['mae'],
                             mse_mask_std_ddof1=std['mse'], mae_mask_std_ddof1=std['mae'],
                             repair_frozen_in_stage2=True, stage2_loss='pure_mse',
                             source=str((ROOT97 if model == 'v97' else ROOT98) / 'repair-audit.json')))
for scope, values in audits['v98']['t1_test_mean'].items():
    rows.append(dict(model='T1_reference', scene=SCENE, task='repair', scope=scope,
                     split='old20_development_test', selection='existing_frozen_reference',
                     training_seed='see original reference record', mask_groups=20,
                     mse=values['mse'], mae=values['mae'], mse_mask_std_ddof1='', mae_mask_std_ddof1='',
                     repair_frozen_in_stage2=True, stage2_loss='pure_mse',
                     source=str(ROOT98 / 'repair-audit.json')))
for model, mse, mae, ms, mas, source in [
    ('v97', forecast97['mse'], forecast97['mae'], forecast97['mse_std'], forecast97['mae_std'],
     DOC / 'E23_v97_frozen_prediction_20261008.csv'),
    ('v98', forecast98['test_mean']['mse'], forecast98['test_mean']['mae'],
     forecast98['test_mask_std_ddof1']['mse'], forecast98['test_mask_std_ddof1']['mae'],
     ROOT98 / 'evaluation_mae/twenty-test-draws.json'),
    ('T1_reference', forecast97['t1_mse'], forecast97['t1_mae'], '', '',
     DOC / 'E23_v97_frozen_prediction_20261008.csv'),
    ('v79_reference', forecast97['v79_mae_selected_mse'], forecast97['v79_mae_selected_mae'], '', '',
     DOC / 'E23_v97_frozen_prediction_20261008.csv'),
]:
    rows.append(dict(model=model, scene=SCENE, task='forecast', scope='all_future96',
                     split='old20_development_test', selection='validation_mae', training_seed=2023 if model in ['v97', 'v98'] else 'see original reference record',
                     mask_groups=20, mse=mse, mae=mae, mse_mask_std_ddof1=ms, mae_mask_std_ddof1=mas,
                     repair_frozen_in_stage2=True, stage2_loss='pure_mse', source=str(source)))
write_csv(DOC / 'E23_v98_compact32_results_20261008.csv', rows)

speed = 100*(1-cost['v98']['full_workflow_seconds']/cost['v97']['full_workflow_seconds'])
stage1speed = 100*(1-cost['v98']['stage1_full_workflow_seconds']/cost['v97']['stage1_full_workflow_seconds'])
memory = 100*(1-cost['v98']['peak_allocated_gib']/cost['v97']['peak_allocated_gib'])
md = f'''# 2026-10-08 E23 计算开销优化与 v98 单场景试验

## 结论

本轮完成验证缓存、V 来源筛选原型、读出层编译原型，以及 v98 读出层压缩的完整单场景训练。保留 v97 作为四场景对照版本；v98 尚无另外三个场景或完整多变量数据集结果。

## 1. 可直接复用的固定验证修复缓存

修复器完全冻结后，把固定验证窗口的修复结果与预测器输入缓存到 CPU；每轮仍运行当前预测器验证，训练仍使用原来的动态缺失窗口。缓存构建计入耗时，检测修复参数、缓冲区和随机状态。默认关闭，命令增加 `--cache-fixed-validation` 开启。

ETTh1 v97 sr010_bl8 完整 Stage2 100 轮：287.8438 → 239.7063 秒，减少 **16.72%**。按验证 MAE 和 MSE 选择的两个完整模型状态分别与旧流程逐张量完全相同；使用此前已验证的测试结果，不额外重测。Weather 只读检查覆盖全部 5175 验证窗口和最后不足一个 batch 的窗口，指标完全一致；单次热验证 2.5195 → 0.0619 秒，首次构建另需 2.5119 秒、CPU 缓存约 160 MiB。这不是 Weather 全训练时间下降 97.5% 的证据。

代码：[缓存实现](/home/xyw/paper/E23/versions/e23-efficiency-validation-cache-20261008/E23/integrations/e23/validation_cache.py)。证据：[完整 100 轮验收]({EXP}/e23_v97_cached_validation_20261008/acceptance.json)。若用于正式资源比较，应给 T1 同样的验证缓存机会。

## 2. v98：八头读出总宽度 64 → 32

只压缩逐点候选读出的 query/key/value/output：八头保留，每头 8 → 4 维。专家与上下文宽度仍为 64，其余修复结构、候选数、预测头不变。修复总参数 504885 → 486677。未使用本轮 V 筛选或编译原型，Stage2 启用了已验证的缓存。

### 实验口径

- ETTh1，回溯/预测均 96；起点率 0.15、缺失块长 8，20 组实际缺失比例均值约 72.93%。
- 单个训练种子 2023，batch 128，GPU0 RTX 5090，FP32。
- 从头粗修复 100 轮，Adam lr 0.001；修复细化 100 轮，Adam lr 0.0001，仅均值投影与 dense refiner 的 97161 参数可训练，专家与历史库冻结。
- 细化损失：全窗口 hidden MSE + 0.25 × 最近 32 点 hidden MSE + 0.05 × 最近 32 点 hidden MAE；均在训练集全局标准化尺度。Stage1 按固定验证全 hidden MSE 选模。
- Stage2 整个修复器冻结，原 PatchTST、纯 MSE、Adam lr 0.0001，100 轮；验证 MAE 主选、MSE 副选。本次二者均选择第 100 轮。
- 使用现有同一 20 组缺失位置；它们已经用于开发。标准差描述缺失位置变化，不是训练多种子标准差，也不构成盲测。
- 粗修复选第 96 轮，细化选第 18 轮，两阶段实际均完整跑满 100 轮。

### 20 组测试均值

| 指标 | v97 | v98 | T1 固定参考 |
|---|---:|---:|---:|
| 全窗口 hidden 修复 MSE | 0.176235 | 0.172707 | 0.193593 |
| 全窗口 hidden 修复 MAE | 0.264315 | 0.262832 | 0.280481 |
| 最近 32 点 hidden 修复 MSE | 0.180031 | 0.177621 | 0.196546 |
| 最近 32 点 hidden 修复 MAE | 0.269464 | 0.269068 | 0.282877 |
| 未来 96 点预测 MSE | 0.398613 | 0.398360 | 0.408589 |
| 未来 96 点预测 MAE | 0.407094 | 0.407680 | 0.414075 |

v98 预测 MSE 降低约 0.064%，MAE 增加约 0.144%。验证集修复全窗口 MSE 0.236775 → 0.238095、最近 32 点 MSE 0.259775 → 0.263599，均略退步；预测验证 MAE 也略退步。不能依据开发测试的小幅改善声称它全面优于 v97。

### 完整训练资源（包含粗修复父模型）

| 阶段 | v97 秒 | v98 秒 |
|---|---:|---:|
| 粗修复 100 轮 | {resource_rows[0]['full_workflow_seconds']:.3f} | {resource_rows[4]['full_workflow_seconds']:.3f} |
| 细化修复 100 轮 | {resource_rows[1]['full_workflow_seconds']:.3f} | {resource_rows[5]['full_workflow_seconds']:.3f} |
| Stage2 100 轮 | {resource_rows[2]['full_workflow_seconds']:.3f} | {resource_rows[6]['full_workflow_seconds']:.3f} |
| 全训练流程 | {cost['v97']['full_workflow_seconds']:.3f} | {cost['v98']['full_workflow_seconds']:.3f} |
| 全流程峰值 allocated GiB | {cost['v97']['peak_allocated_gib']:.3f} | {cost['v98']['peak_allocated_gib']:.3f} |
| 全流程峰值 reserved GiB | {cost['v97']['peak_reserved_gib']:.3f} | {cost['v98']['peak_reserved_gib']:.3f} |

Stage1 两阶段合计减少 **{stage1speed:.2f}%**；全流程减少 **{speed:.2f}%**、峰值 allocated 减少 **{memory:.2f}%**。全流程收益同时包含压缩和缓存，不能全部归因于压缩。单次资源实测，非反复测量置信区间；各进程完整耗时包含数据准备、模型建立、全部训练/验证及保存，不包含最终测试与单独诊断任务。

原 `evaluation_mae/twenty-test-draws.json` 中自动聚合的 `training_resources` 只包含细化和 Stage2，遗漏父粗修复成本。正式全流程报告使用补充的 [complete_training_cost.json]({PILOT}/complete_training_cost.json) 及本表，不能把 521.55 秒作为总训练耗时。

## 3. 多变量扩展的两项实验实现

**V 候选筛选：**先用全局先验选 16 来源，再计算窗口可见性和一致性、取最终四个。局部张量从 B×patch×C×C×patch_length 减为 B×patch×C×16×patch_length；全局先验仍有 C² 成本。自动策略只在 C≥64 启用，C=7/21 完全走旧算法。合成未训练 V 模块的 C=128/321 来源选择耗时约下降 80.9%/93.6%；不代表整个模型或训练准确率。Weather 现有权重只读全验证的筛选输出与指标一致，未重训。

**读出层编译：**保持参数和结构，Weather FP32 forward/backward 微测 174.14 → 156.19 ms，约下降 10.3%；首次编译另需 12.34 秒。全验证六项指标最大绝对差约 2.13e-8。微测关闭 dropout、无优化器，尚未验证完整训练的速度和质量。

相关隔离版本：[V 自动筛选实现](/home/xyw/paper/E23/versions/e23-efficiency-shortlist-20261008/E23/README.md)。当前副本的数据与 dense 模块仍有 ETTh1 七变量约束；迁移时需要在新的隔离副本适配变量数，不能直接覆盖 E24 的在跑代码。E24 后台、v79 封存版本均未改动。

## 文件与后续选择

- [精度 CSV](/home/xyw/paper/E23/docs/experiments/E23_v98_compact32_results_20261008.csv)：修复全窗口/最近 32 点、预测、测试缺失位置标准差及原始来源。
- [资源 CSV](/home/xyw/paper/E23/docs/experiments/E23_v98_compact32_resources_20261008.csv)：每阶段与包含父模型的总耗时、allocated/reserved。
- [优化证据总表](/home/xyw/paper/E23/docs/experiments/E23_efficiency_iteration_20261008.csv)。
- v98 隔离源码：`/home/xyw/paper/E23/versions/e23-efficiency-compact32-20261008/E23`；计划、原始配置、命令日志、权重、源码归档及冻结散列位于 `{PILOT}`。

当前保留 v97 四场景结果，v98 为单场景效率候选。对多变量数据集，首先复用冻结验证缓存；随后在独立副本评估读出层编译和更大压缩幅度。V 自动筛选针对 C≥64 的数据，不应宣称它解决了 Weather 21 变量的主耗时。
'''
(DOC / '2026-10-08_E23_efficiency_and_v98_pilot.md').write_text(md)

registry = DOC / 'E23_efficiency_iteration_20261008.csv'
with registry.open() as f:
    reader = csv.DictReader(f)
    fields = reader.fieldnames
    previous = [r for r in reader if not r['method'].startswith('compact32_')]
for phase_index, scope in [(0, 'ETTh1 cold Stage1 coarse 100epochs'), (1, 'ETTh1 Stage1 refinement 100epochs'), (2, 'ETTh1 Stage2 100epochs; compression plus validation cache'), (3, 'ETTh1 complete coarse+refinement+Stage2; 300epochs')]:
    a, b = resource_rows[phase_index], resource_rows[phase_index+4]
    previous.append(dict(method='compact32_'+a['phase'], scope=scope, variables=7, batch_size=128,
                         original_seconds=a['full_workflow_seconds'], candidate_seconds=b['full_workflow_seconds'],
                         time_reduction_percent=100*(1-b['full_workflow_seconds']/a['full_workflow_seconds']),
                         entire_model_state_exact_equal=False,
                         quality_scope='sr015_bl8 pilot; old20 development masks; forecast MSE slightly better, MAE slightly worse; validation repair slightly worse',
                         test_rerun=True, original_peak_allocated_gib=a['peak_allocated_gib'],
                         candidate_peak_allocated_gib=b['peak_allocated_gib']))
with registry.open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(previous)
print(json.dumps(cost, indent=2))
print('Reports written; no training source or sealed model modified.')
