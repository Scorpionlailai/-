"""Compare only the matched repair metrics and complete Stage1 resource chain."""
from pathlib import Path
import csv, hashlib, json, statistics, sys
import numpy as np

ROOT = Path(__file__).resolve().parent
workflow = json.loads((ROOT/'workflow.json').read_text())
sys.path[:0] = [workflow['source'], workflow['source']+'/src',
               '/home/xyw/paper/E23/artifacts/performance/weather_iteration_20261008']
from repair_resource_scope import stage1_manifests
from integrations.forecast_compare.resources import selected_training_resources

KEYS = ['repair96_mse', 'repair96_mae', 'repair32_mse', 'repair32_mae']
BASE = Path('/home/xyw/paper/E24/artifacts/Weather/seq96_pred96/sr015_bl8')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


ours_path = Path(workflow['ensemble_root'])/'E23-v125/sr015_bl8/repair_evaluation_mae/twenty-test-draws.json'
ours = read(ours_path)
assert ours['start_rate'] == .15 and ours['block_len'] == 8 and not ours['forecast_evaluated']
assert ours['visible_values_exact'] and ours['parameters_memory_and_keys_unchanged']
sources = [dict(model='E23-v125', source=ours_path),
           *[dict(model=name, source=BASE/'weather_author_comparison_sr015_20261008'/name/'evaluation/mae/twenty-test-draws.json') for name in ['T1', 'SAITS']],
           dict(model='Weather-v79', source=BASE/'e24_v79_weather_sr015_20261008/seed_2023/E24-v79/evaluation_mae/twenty-test-draws.json')]
fixture_keys = ['seed', 'mask_sha256', 'windows', 'hidden96_coordinates', 'hidden32_coordinates']
fixtures = [[d[k] for k in fixture_keys] for d in ours['draws']]
assert len(fixtures) == 20 and all(x[2] == 10444 for x in fixtures)
rows = []
paired = {}
rng = np.random.default_rng(20261009)
bootstrap_indices = rng.integers(0, 20, size=(20000, 20))
for item in sources:
    evaluation = read(item['source'])
    assert evaluation['start_rate'] == .15 and evaluation['block_len'] == 8
    assert [[d[k] for k in fixture_keys] for d in evaluation['draws']] == fixtures
    assert evaluation['dataset_manifest_sha256'] == ours['dataset_manifest_sha256']
    selected, paths = stage1_manifests(evaluation)
    resources = selected_training_resources(paths)
    assert resources['resource_status'] == 'measured_all_stages'
    for resource_path in resources['resource_record_paths']:
        resource = read(resource_path)
        assert resource['status'] == 'completed'
        assert all('forecast' not in stage for stage in resource['stages']), resource_path
    row = dict(model=item['model'], stage1_selection_metric=read(selected).get('selection_metric', read(selected)['selection']), test_masks=20)
    for key in KEYS:
        values = [d[key] for d in evaluation['draws']]
        row[key] = statistics.mean(values)
        row[key+'_mask_std_ddof1'] = statistics.stdev(values)
        assert abs(row[key]-evaluation['summary'][key]['mean']) < 1e-12
    row.update(stage1_full_seconds=resources['full_workflow_seconds'],
               stage1_full_minutes=resources['full_workflow_seconds']/60,
               stage1_peak_allocated_gib=resources['peak_allocated_gib'],
               stage1_peak_reserved_gib=resources['peak_reserved_gib'],
               source=str(item['source']), source_sha256=sha(item['source']),
               stage1_selection_paths=json.dumps([str(p) for p in paths]),
               stage1_resource_paths=json.dumps(resources['resource_record_paths']))
    rows.append(row)
    if item['model'] != 'E23-v125':
        paired[item['model']] = {}
        for key in KEYS:
            diff = np.array([a[key]-b[key] for a, b in zip(ours['draws'], evaluation['draws'])])
            paired[item['model']][key] = dict(mean_difference=float(diff.mean()),
                relative_reduction_percent=100*(row[key]-rows[0][key])/row[key],
                improved_masks=int((diff < 0).sum()),
                paired_mask_bootstrap95=np.quantile(diff[bootstrap_indices].mean(1), [.025, .975]).tolist())
with (ROOT/'repair_comparison.csv').open('x', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
audit = dict(scope='same Weather sr015/bl8 development masks; one training seed per model; mask uncertainty only',
             comparisons=paired,
             all_four_means_lower_than_external=all(v['mean_difference'] < 0 for name in ['T1', 'SAITS'] for v in paired[name].values()),
             all_four_external_mask95_upper_negative=all(v['paired_mask_bootstrap95'][1] < 0 for name in ['T1', 'SAITS'] for v in paired[name].values()),
             average_window_missing_fraction=statistics.mean(d['window_missing_fraction'] for d in ours['draws']))
(ROOT/'paired_repair_mask_differences.json').write_text(json.dumps(audit, indent=2)+'\n')
resources = ours['training_resources']
cost = dict(full_stage1_seconds=resources['full_workflow_seconds'], full_stage1_minutes=resources['full_workflow_seconds']/60,
            peak_allocated_gib=resources['peak_allocated_gib'], peak_reserved_gib=resources['peak_reserved_gib'],
            resource_record_paths=resources['resource_record_paths'],
            scope='same-scene reused coarse100 once + independent100 C64 + independent100 C256 + fixed combination validation; no Stage2; actual source workflow sum')
(ROOT/'complete_stage1_training_cost.json').write_text(json.dumps(cost, indent=2)+'\n')
selected = read(ours['selection_manifest'])
lines = ['# Weather E23-v125固定设计迁移：0.15/8', '',
         '结构、目标和1:1比例保持0.10/8版；本场景父粗修复从头训练100轮，两条校准各重新训练100轮。原预测头不变，本轮预测未运行。', '',
         '| 模型 | 全96 MSE | 全96 MAE | 近32 MSE | 近32 MAE | 完整Stage1分钟 | allocated GiB |',
         '|---|---:|---:|---:|---:|---:|---:|']
for row in rows:
    lines.append('| '+row['model']+' | '+' | '.join(f'{row[key]:.8f}' for key in KEYS)+f" | {row['stage1_full_minutes']:.2f} | {row['stage1_peak_allocated_gib']:.3f} |")
lines.extend(['', f"实际窗口平均缺失率：{audit['average_window_missing_fraction']*100:.4f}%。全部模型共用20份位置，每组10444窗口；只统计隐藏有效坐标，全局训练标准化尺度。",
              f"四项均值均低于T1/SAITS：{audit['all_four_means_lower_than_external']}；四项配对95%区间上界均负：{audit['all_four_external_mask95_upper_negative']}。",
              f"64/256校准验证MAE主选择精炼轮数：{selected['source_refinement_epochs']}；固定系数{selected['calibration_ensemble_spec']['weights']}，没有重新搜比例。",
              f"完整训练链reserved峰值{cost['peak_reserved_gib']:.3f}GiB；包含全部复用父Stage1成本，共同粗训练去重，不含父Stage2或最终测试。不是连续冷启动墙钟。",
              '', '## 配对位置差值', '', '| 对照 | 指标 | v125−对照均值 | 配对95%区间 | 改善位置组数 |', '|---|---|---:|---|---:|'])
for name, metrics in paired.items():
    for key, v in metrics.items():
        lo, hi = v['paired_mask_bootstrap95']
        lines.append(f"| {name} | {key} | {v['mean_difference']:.8f} | [{lo:.8f}, {hi:.8f}] | {v['improved_masks']}/20 |")
lines.extend(['', '训练seed：E23 2023、T1作者502、SAITS2023；验证掩码seed2021，测试seed2026100801至2026100820。20组表示缺失位置，不是20次独立训练。当前为开发比较，结果不外推到所有缺失场景或最终盲测。', '',
              '[正式CSV](repair_comparison.csv) · [配对核验](paired_repair_mask_differences.json) · [完整成本](complete_stage1_training_cost.json)', '',
              '[组合配对记录](ensemble/E23-v125/sr015_bl8/stage1/runs/e23-stage1-mae.json) · [正式20组结果](ensemble/E23-v125/sr015_bl8/repair_evaluation_mae/twenty-test-draws.json)', '',
              '后续Stage2必须按组合配方显式恢复模型，冻结完整修复器且eval，原预测头使用纯MSE。'])
(ROOT/'result.md').write_text('\n'.join(lines)+'\n')
with (ROOT/'progress.md').open('a') as stream:
    stream.write('\nTask3/4/5: complete — fixed0.5 combination strict preflight,20mask repair evaluation,matched fixture/resource/paired audit and reports finished.\n')
print(json.dumps(dict(status='completed', metrics={k: rows[0][k] for k in KEYS}, **{k:audit[k] for k in ['all_four_means_lower_than_external','all_four_external_mask95_upper_negative']})), flush=True)
