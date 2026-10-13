"""Regenerate local review artifacts from completed, traceable checkpoints."""
import csv
import json
from pathlib import Path

from .data import SCENES, write_json

MODELS = ('TimesNet', 'ImputeFormer', 'BRITS', 'CSDI')


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def render(root):
    root = Path(root)
    rows, audit = [], []
    for dataset in ('ETTh1', 'Weather'):
        for scene in SCENES:
            for model in MODELS:
                folder = root / dataset / scene / model / 'seed_2023'
                row = dict(dataset=dataset, scene=scene, model=model, training_seed=2023,
                           status=(read(folder / 'status.json') or {}).get('phase', 'pending'))
                plan = read(folder / 'plan.json')
                if plan:
                    row['parameter_status'] = plan['profile']['parameter_status']
                    row['variant'] = plan['profile'].get('variant', 'author_config_adapted')
                    row['parameter_source'] = plan['profile']['source']
                    row['data_manifest_sha256'] = plan['data_manifest_sha256']
                stage1 = read(folder / 'stage1/selection.json')
                stage2 = read(folder / 'stage2/selection.json')
                for name, record in (('stage1', stage1), ('stage2', stage2)):
                    if record:
                        row[name+'_epochs'] = record.get('actual_epochs', record.get('epochs'))
                        row[name+'_optimizer_steps'] = record['optimizer_steps']
                        row[name+'_training_seconds'] = record['training_seconds']
                        row[name+'_peak_allocated_gib'] = record['resources']['max_allocated_gib']
                        row[name+'_peak_reserved_gib'] = record['resources']['max_reserved_gib']
                if stage1:
                    row['repair_checkpoint_sha256'] = stage1['checkpoint_sha256']
                if stage2:
                    row['head_initial_sha256'] = stage2['head_initial_sha256']
                    row['forecast_main_checkpoint_sha256'] = stage2['checkpoints']['mae']
                repair_result = read(folder / 'evaluation/repair_summary.json')
                main = read(folder / 'evaluation/forecast_mae_summary.json')
                secondary = read(folder / 'evaluation/forecast_mse_summary.json')
                for result, prefix in ((repair_result, 'repair'), (main, 'main'), (secondary, 'secondary')):
                    if result:
                        row[prefix+'_test_seconds'] = result['test_seconds']
                        for section, summary in result['summary'].items():
                            if prefix != 'repair' and section != 'forecast':
                                continue
                            for name in ('mse_mean', 'mae_mean', 'mse_mask_std_ddof1', 'mae_mask_std_ddof1'):
                                row[prefix+'_'+section+'_'+name] = summary[name]
                if repair_result and main:
                    consistent = all(a['full96'] == b['full96'] and a['recent32'] == b['recent32']
                                     for a, b in zip(repair_result['groups'], main['groups']))
                    row['stage2_repair_metrics_identical'] = consistent
                    audit.append(dict(dataset=dataset, scene=scene, model=model,
                                      repair_unchanged_after_stage2=consistent))
                rows.append(row)
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with (root / 'results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    ready = all(r.get('main_forecast_mse_mean') is not None and
                r.get('secondary_forecast_mse_mean') is not None and
                r.get('stage2_repair_metrics_identical') for r in rows)
    head_shas = {r['head_initial_sha256'] for r in rows if 'head_initial_sha256' in r}
    if len(head_shas) > 1:
        ready = False
    write_json(root / 'review_status.json', dict(ready_for_user_review=ready,
        expected_runs=32, forecast_test_mask_groups=20,
        completed_repairs=sum('repair_full96_mse_mean' in r for r in rows),
        completed_forecasts=sum('main_forecast_mse_mean' in r for r in rows),
        common_head_initial_sha256=list(head_shas), freeze_metric_audit=audit,
        github_uploaded=False, github_policy='await_user_review'))
    lines = ['# ETTh1 / Weather 新增基线复现报告', '',
             '**状态：' + ('全部完成，待用户审核。' if ready else '实验进行中；空白项尚未完成。') + '**', '',
             '范围：TimesNet、ImputeFormer、BRITS、CSDI；两个数据集、四种块缺失场景，共32次独立修复训练，随后32次冻结修复器的公共 PatchTST 训练。', '',
             '主表均在训练集全局标准化尺度计算。全96点及近32点只计隐藏有效坐标；每组先累积误差/坐标数，再对20组求均值与样本标准差。单个训练 seed=2023；位置标准差不代表训练种子标准差。', '',
             'Weather 掩码内容校验和与历史资产一致。ETTh1 使用独立重建协议；20组缺失比例及窗口统计与历史记录一致，但原二进制资产缺失，文件SHA不同，未验证与旧实验逐坐标一致。此次四个模型使用同一套封存资产；不能直接把本表 ETTh1 与旧表声称为严格配对结果。', '',
             'TimesNet 使用作者 ETTh1/Weather 专用插补脚本及入口默认参数。ImputeFormer 使用作者 LA 基础配置；BRITS 使用 README 示例；CSDI 使用 base.yaml。后三者明确标为非目标数据集最佳配置，已获用户授权。', '',
             '用户所指预测阶段标准为测试采用20组缺失：每组独立计算预测MSE/MAE，再汇总均值和样本标准差。当前实现满足这一要求。Stage2 当前使用公共 PatchTST，完整冻结修复器并 eval，100轮、batch128、FP32、Adam lr=1e-4、纯预测MSE。主权重按验证MAE选择；验证MSE分支另报，不按测试择优。', '',
             '用户授权加速版：BRITS 保留隐藏维108及原损失，采用TorchScript循环，训练上限由1000轮改为200轮，验证MAE选模、patience20。CSDI 保留网络、200训练轮及50扩散步，推理由100样本改为10样本、每批并行5样本并取中位数。两者标为加速变体，不声称完全复现作者配置；精度影响以最终评估为准。', '',
             '| 数据集 | 场景 | 模型 | 全96修复 MSE / MAE | 近32修复 MSE / MAE | 预测 MSE / MAE（验证MAE主选） | 状态 |',
             '|---|---|---|---|---|---|---|']
    def pair(row, prefix):
        first, second = row.get(prefix+'_mse_mean'), row.get(prefix+'_mae_mean')
        return f'{first:.6f} / {second:.6f}' if first is not None else '—'
    for row in rows:
        lines.append('| ' + ' | '.join([row['dataset'], row['scene'], row['model'],
                     pair(row, 'repair_full96'), pair(row, 'repair_recent32'),
                     pair(row, 'main_forecast'), row['status']]) + ' |')
    lines += ['', '完整均值、位置标准差、次选预测分支、实际轮数/步数、训练和推理耗时、显存、参数来源及权重SHA见 [results.csv](results.csv)。各运行目录保存plan、逐轮曲线、选模记录、20组逐组指标和资源记录。', '',
              '适配说明：公共切分/标准化/块缺失替换作者原数据设置；回溯96、变量数按数据集调整；只向修复器输入回溯及可见性。TimesNet 全缺失变量分母 clamp_min(1)，不取隐藏真值。BRITS 无分类标签，label_weight=0，并采用上述加速设置。ImputeFormer 保留值通道+作者日周期sin/cos两个外生通道和原Freg；FFT用FP32处理非二次幂窗口，训练其余部分仍使用作者混合FP16。CSDI 固定50扩散步、10样本、torch.median 聚合，记录完整采样成本。', '',
              '所有文件仅保存在本地，用户审核后再上传 GitHub。']
    (root / 'report.md').write_text('\n'.join(lines) + '\n')
    return rows


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    render(parser.parse_args().root)
