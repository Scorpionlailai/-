"""Require exact checkpoint equivalence before reusing v97 test metrics."""
import csv
import hashlib
import json
from pathlib import Path

import torch

R = Path(__file__).resolve().parent
E = Path('/home/xyw/paper/E23')
D = E/'docs/experiments'
X = E/'artifacts/ETTh1/seq96_pred96/experiments'


def read(path):
    return json.loads(Path(path).read_text())


def measured(directory):
    paths = list(Path(directory).glob('*.json'))
    assert len(paths) == 1
    record = read(paths[0])
    assert record['status'] == 'completed' and record['epochs_completed'] == 100
    return paths[0], record


assert read(R/'status.json')['status'] == 'training_completed'
plan = read(R/'plan.json')
assert all(hashlib.sha256(Path(f).read_bytes()).hexdigest() == sha
           for f, sha in read(R/'frozen-definition.json')['sources'].items())
original = Path(plan['original_stage2'])
new = R/'sr015_bl8/stage2'
acceptance = dict(scene='sr015_bl8', choices={})
metrics = []
for metric, name in [('mae','e23-stage2.json'),('mse','e23-stage2-mse.json')]:
    before = read(original/'runs'/name)
    after = read(new/'runs'/name)
    a = torch.load(before['checkpoint'], map_location='cpu', weights_only=False)
    b = torch.load(after['checkpoint'], map_location='cpu', weights_only=False)
    same = a['model_state'].keys() == b['model_state'].keys() and all(
        torch.equal(a['model_state'][key], b['model_state'][key]) for key in a['model_state'])
    checks = dict(entire_repair_and_predictor_state_exact_equal=same,
                  best_epoch_equal=before['best_epoch'] == after['best_epoch'],
                  memory_sha_equal=before['memory_sha256'] == after['memory_sha256'],
                  key_sha_equal=before['key_sha256'] == after['key_sha256'],
                  predictor_initial_sha_equal=before['predictor_initial_sha256'] == after['predictor_initial_sha256'],
                  repair_fully_frozen=after['repair_trainable_parameters'] == 0,
                  pure_forecast_mse=after['loss'] == 'forecast_mse_only',
                  epochs_completed=after['epochs_completed'] == 100,
                  cache_enabled=after['fixed_validation_cache'] is not None)
    assert all(checks.values()), checks
    source = original.parent/f'evaluation_{metric}/twenty-test-draws.json'
    scores = read(source)
    acceptance['choices'][metric] = dict(checks=checks, best_epoch=after['best_epoch'],
                                         selected_manifest=str(new/'runs'/name),
                                         reused_test_result=str(source),
                                         test_result_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                                         test_mean=scores['test_mean'],
                                         mask_std_ddof1=scores['test_mask_std_ddof1'])
    metrics.append(dict(scene='sr015_bl8', model='v101_execution_same_as_v97',
                        selection_metric=metric, training_seed=2023, stage2_epochs=100,
                        repair_frozen=True, forecast_loss='pure_mse',
                        mse=scores['test_mean']['mse'], mae=scores['test_mean']['mae'],
                        mse_mask_std_ddof1=scores['test_mask_std_ddof1']['mse'],
                        mae_mask_std_ddof1=scores['test_mask_std_ddof1']['mae'],
                        test_groups=20, test_role='old20_development', test_reused=True,
                        entire_selected_state_exact_equal=True, source=str(source)))

stage1 = read(Path(plan['stage1_training_resource_parent'])/'extended_acceptance.json')
assert stage1['acceptance_passed'] and stage1['every_epoch_validation_history_exact_equal']
acceptance.update(acceptance_passed=True, stage1_equivalence=str(Path(plan['stage1_training_resource_parent'])/'extended_acceptance.json'),
                  tests_rerun=False, metric_space='train_standardized')
(R/'acceptance.json').write_text(json.dumps(acceptance,indent=2)+'\n')

coarse_path, coarse = measured(X/'e23_v79_v69_global_level_four_pureMSE_20261007/sr015_bl8/E23-v79/stage1/resources')
refine_path, refine = measured(Path(plan['stage1_training_resource_parent'])/'E23-v100/sr015_bl8/stage1/resources')
forecast_path, forecast = measured(new/'resources')
_, old_forecast = measured(original/'resources')
old_cost = read(X/'e23_v98_compact_readout_20261008/complete_training_cost.json')['v97']
rows = []
for phase, path, rec, reused in [('coarse',coarse_path,coarse,True),
                                 ('refinement',refine_path,refine,True),
                                 ('forecast',forecast_path,forecast,False)]:
    rows.append(dict(model='v101_execution_same_as_v97',scene='sr015_bl8',phase=phase,epochs=100,
                     full_workflow_seconds=rec['full_workflow_seconds'],
                     peak_allocated_gib=rec['peak_allocated_bytes']/2**30,
                     peak_reserved_gib=rec['peak_reserved_bytes']/2**30,
                     measured_training_record_reused=reused, source=str(path)))
total = sum(row['full_workflow_seconds'] for row in rows)
peak = max(row['peak_allocated_gib'] for row in rows)
reserved = max(row['peak_reserved_gib'] for row in rows)
reduction = 100*(1-total/old_cost['full_workflow_seconds'])
stage2_reduction = 100*(1-forecast['full_workflow_seconds']/old_forecast['full_workflow_seconds'])
rows.append(dict(model='v101_execution_same_as_v97', scene='sr015_bl8',phase='all_training_phases_accounted',epochs=300,
                 full_workflow_seconds=total,peak_allocated_gib=peak,peak_reserved_gib=reserved,
                 measured_training_record_reused='coarse and refinement',source=';'.join(row['source'] for row in rows)))
cost = dict(stages=rows[:-1],full_workflow_seconds=total,peak_allocated_gib=peak,
            peak_reserved_gib=reserved,reduction_vs_v97_percent=reduction,
            stage2_reduction_percent=stage2_reduction,
            scope='sum of separately measured coarse+v100 refinement+this Stage2; not a new uninterrupted cold run; excludes tests')
(R/'complete_training_cost.json').write_text(json.dumps(cost,indent=2)+'\n')
for name, values in [('E23_v101_combined_cache_results_20261008.csv',metrics),
                      ('E23_v101_combined_cache_resources_20261008.csv',rows)]:
    with (D/name).open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)

md = f'''# v101：Stage1 与 Stage2 固定验证缓存组合验收

## 完成范围

ETTh1 sr015_bl8、seed2023、FP32、batch128，结构和预测头均与v97一致。v100 已完成100轮修复细化，逐轮验证记录和全部选中修复权重与v97相同；本轮以该修复器运行100轮Stage2，只启用固定验证缓存。Stage2修复器完全冻结、纯MSE、原PatchTST配置与初始权重不变。

按验证MAE和MSE选出的两个完整模型状态（全部修复器与预测器张量）均与原v97逐张量完全相同，选中轮次、历史库/key与预测头初始散列相同。因此沿用原同20组缺失位置的测试结果，不重复测试。

## 精度与资源

- 主结果预测MSE **{metrics[0]['mse']:.6f}**、MAE **{metrics[0]['mae']:.6f}**，与v97相同。
- Stage1全窗口hidden修复MSE/MAE仍为0.176235/0.264315，最近32点为0.180031/0.269464。
- 本次Stage2完整实测 **{forecast['full_workflow_seconds']:.3f}秒**，原 **{old_forecast['full_workflow_seconds']:.3f}秒**，减少 **{stage2_reduction:.2f}%**。
- 计入父粗修复{coarse['full_workflow_seconds']:.3f}秒、v100细化{refine['full_workflow_seconds']:.3f}秒后，三阶段成本合计 **{total/60:.3f}分钟**；原v97为{old_cost['full_workflow_seconds']/60:.3f}分钟，减少 **{reduction:.2f}%**。
- 三阶段峰值allocated **{peak:.3f}GiB**，reserved **{reserved:.3f}GiB**。全流程GPU峰值由粗修复决定，验证缓存未降低它。Stage1 CPU缓存约579MiB，另有Stage2CPU缓存。

该成本合计来自三个分别实测的训练记录，粗修复与细化沿用已有记录，**本轮没有重新跑一个连续的300轮冷启动流程**。各完整计时均包括准备、验证、保存及缓存构建，不含最终测试或独立诊断；单次速度实测，不是多次计时均值。

## 查验

- [精度CSV](/home/xyw/paper/E23/docs/experiments/E23_v101_combined_cache_results_20261008.csv)
- [资源CSV](/home/xyw/paper/E23/docs/experiments/E23_v101_combined_cache_resources_20261008.csv)
- [严格等价验收]({R}/acceptance.json)
- [完整成本及原始来源]({R}/complete_training_cost.json)

v101为执行配置标签，学习结果仍是v97，不能算作一个新的精度改良模型。多变量数据集尚未完成这套组合的完整训练验收。E24后台和封存版本未修改。后续正式效率比较应给各冻结修复基线相同验证缓存机会。
'''
(D/'2026-10-08_E23_v101_combined_validation_cache.md').write_text(md)
(R/'status.json').write_text(json.dumps(dict(status='completed',exact_selected_model_equivalence=True)))
print(json.dumps(dict(acceptance_passed=True,complete_accounted_seconds=total,
                      stage2_seconds=forecast['full_workflow_seconds'],
                      accounted_time_reduction_percent=reduction,
                      stage2_time_reduction_percent=stage2_reduction,
                      unchanged_main_forecast_metrics=metrics[0]),indent=2))
