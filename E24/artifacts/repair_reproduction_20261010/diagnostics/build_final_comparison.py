"""Join completed baseline results with traceable v125/T1 reference records."""
import csv
import json
from pathlib import Path
from integrations.repair_baselines.data import file_sha256

ROOT = Path('/home/xyw/paper/E24/artifacts/repair_reproduction_20261010')
PROJECT = Path('/home/xyw/paper')
old_forecasts = {}
for dataset in ('ETTh1', 'Weather'):
    source = PROJECT / f'E23/docs/experiments/{dataset}_E23_v125_completed_baselines_forecast_20261010.csv'
    for row in csv.DictReader(source.open()):
        if row['model'] not in ('E23-v125', 'T1', 'T1-frozen'):
            continue
        assert row['selection_metric'] == 'mae' and int(row['test_masks']) == 20
        assert file_sha256(Path(row['source'])) == row['source_sha256']
        key = (dataset, row['scene'], 'T1' if row['model'].startswith('T1') else row['model'])
        old_forecasts[key] = row
new_forecasts = {(row['dataset'], row['scene'], row['model']): row
                 for row in csv.DictReader((ROOT / 'results.csv').open())}
rows = []
for record in csv.DictReader((ROOT / 'comparisons/completed_repair_vs_v125_t1_20261012.csv').open()):
    assert int(record['test_masks']) == 20
    assert file_sha256(Path(record['source'])) == record['source_sha256']
    key = (record['dataset'], record['scene'], record['model'])
    record['repair_source'] = record.pop('source')
    record['repair_source_sha256'] = record.pop('source_sha256')
    forecast = old_forecasts.get(key)
    if forecast:
        for field in ('forecast_mse', 'forecast_mae', 'forecast_mse_mask_std_ddof1', 'forecast_mae_mask_std_ddof1'):
            record[field] = forecast[field]
        record['forecast_source'] = forecast['source']
        record['forecast_source_sha256'] = forecast['source_sha256']
        record['forecast_training_protocol'] = 'legacy_reference_protocol'
        record['forecast_status'] = 'completed_reference'
    elif key in new_forecasts:
        forecast = new_forecasts[key]
        for metric in ('mse', 'mae'):
            record['forecast_' + metric] = forecast['main_forecast_' + metric + '_mean']
            record['forecast_' + metric + '_mask_std_ddof1'] = forecast['main_forecast_' + metric + '_mask_std_ddof1']
        source = ROOT / key[0] / key[1] / key[2] / 'seed_2023/evaluation/forecast_mae_summary.json'
        record['forecast_source'] = str(source)
        record['forecast_source_sha256'] = file_sha256(source)
        record['forecast_training_protocol'] = forecast['forecast_training_protocol']
        record['forecast_status'] = 'completed_new_baseline'
    else:
        record['forecast_status'] = 'not_available_in_local_reference_assets'
    rows.append(record)
assert len(rows) == 48
assert sum(row['forecast_status'] == 'completed_new_baseline' for row in rows) == 32
assert sum(row['forecast_status'] == 'completed_reference' for row in rows) == 14
target = ROOT / 'comparisons/final_repair_forecast_vs_v125_t1_20261013.csv'
with target.open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
    writer.writeheader()
    writer.writerows(rows)
lines = ['# 两个数据集修复与预测结果：新增基线、v125与T1', '',
         '新增四个模型的32项修复和32项预测全部完成并通过缓存与权重核验。所有已列出的结果均为20组缺失测试均值；每格按MSE / MAE排列，预测统一报告验证MAE主选分支。', '',
         'ETTh1新增基线使用用户确认的独立重建掩码，v125/T1保留旧协议，数值只能作为参考，不能声明逐坐标配对。Weather测试掩码资产已核验，但新增基线的预测训练统一采用固定训练掩码缓存，与旧参考训练协议不同。', '',
         '本地参考资产尚无Weather v125的20%×8、10%×16完整预测评估，明确留空；这两项并不属于此次新增32项任务。', '',
         '| 数据集 | 场景 | 模型 | 全96修复 MSE / MAE | 近32修复 MSE / MAE | 预测 MSE / MAE |',
         '|---|---|---|---|---|---|']
def pair(row, prefix):
    return f"{float(row[prefix + '_mse']):.6f} / {float(row[prefix + '_mae']):.6f}" if row.get(prefix + '_mse') else '本地无完整结果'
for row in rows:
    lines.append('| ' + ' | '.join([row['dataset'], row['scene'], row['model'],
                 pair(row, 'repair96'), pair(row, 'repair32'), pair(row, 'forecast')]) + ' |')
lines += ['', '完整精度、测试掩码标准差、参考配置差异和原始记录SHA见同名CSV。此次新增基线的完整审核证据见上级目录 final_saved_output_audit.json、review_status.json。']
target.with_suffix('.md').write_text('\n'.join(lines) + '\n')
print(json.dumps(dict(comparison_rows=len(rows), new_completed_forecasts=32, completed_reference_forecasts=14,
                      unavailable_reference_forecasts=2, csv=str(target))))
