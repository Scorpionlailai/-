"""Copy the completed five-seed report to experiment docs; never publish partial means."""
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parent
DOCS = Path('/home/xyw/paper/E23/docs/experiments')
STEM = 'ETTh1_P4_E23_v31B_T1_five_seed_twenty_masks_20261004'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def status(value):
    temp = ROOT/'publish-status.tmp'
    temp.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')
    temp.replace(ROOT/'publish-status.json')


def publish(statistics):
    if statistics['expected_training_seeds'] != [102, 202, 302, 402, 502] or statistics['completed_runs'] != 10:
        raise ValueError('completed five-seed protocol required')
    definition = ROOT/'trial-definition.json'
    if definition.with_suffix('.sha256').read_text().strip() != digest(definition):
        raise ValueError('trial definition seal mismatch')
    with (ROOT/'comparison_summary.csv').open() as stream:
        summary = list(csv.DictReader(stream))
    with (ROOT/'per_training_seed.csv').open() as stream:
        seeds = list(csv.DictReader(stream))
    with (ROOT/'per_training_seed_and_mask.csv').open() as stream:
        draws = list(csv.DictReader(stream))
    if len(summary) != 2 or len(seeds) != 10 or len(draws) != 200:
        raise ValueError('complete report row counts required')
    if any(int(row['independent_training_runs']) != 5 or int(row['test_masks']) != 20 for row in summary):
        raise ValueError('report replicate counts changed')
    outputs = {}
    for source, name in [('comparison_summary.csv', STEM+'.csv'),
                         ('per_training_seed.csv', STEM+'_per_seed.csv'),
                         ('per_training_seed_and_mask.csv', STEM+'_all_draws.csv'),
                         ('comparison_statistics.json', STEM+'_statistics.json')]:
        target = DOCS/name
        if target.exists() and digest(target) != digest(ROOT/source):
            raise FileExistsError(f'refusing to overwrite a different completed report: {target}')
        if not target.exists():
            shutil.copyfile(ROOT/source, target)
        outputs[str(target)] = digest(target)
    lines = ['# ETTh1 P4：v31-B 与 T1 的五次独立训练对照', '',
             '正式实验已完成：每个模型从头完整训练 5 次；每个 checkpoint 测试同一批 20 组新缺失位置。', '',
             '训练种子 102、202、302、402、502；96→96；原 PatchTST D16 预测头。选模只使用原固定验证集 MAE。原 seed=2021 权重和旧三组记录保持原样。', '',
             '| 模型 | 平均 MSE | 训练标准差 MSE | 缺失位置标准差 MSE | 平均 MAE | 训练标准差 MAE | 缺失位置标准差 MAE |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for row in summary:
        cols = [row['model']]+[f"{float(row[key]):.6f}" for key in ['mse_mean','mse_training_seed_std_ddof1','mse_test_mask_std_ddof1','mae_mean','mae_training_seed_std_ddof1','mae_test_mask_std_ddof1']]
        lines.append('| '+' | '.join(cols)+' |')
    lines.extend(['', '标准差为样本标准差（ddof=1）。训练标准差由各 checkpoint 的 20 掩码均值计算；缺失位置标准差由各掩码的 5 训练种子均值计算。', '',
                  '| 指标 | E23−T1 平均差异 | 配对交叉 bootstrap 95% 区间 | 训练种子胜出数 | 掩码胜出数 |',
                  '|---|---:|---|---:|---:|'])
    for metric in ['mse', 'mae']:
        d = statistics['paired_difference'][metric]
        lo, hi = d['crossed_bootstrap_ci95']
        lines.append(f"| {metric.upper()} | {d['mean_difference_ours_minus_T1']:.6f} | [{lo:.6f}, {hi:.6f}] | {d['seed_pair_win_count']}/5 | {d['mask_pair_win_count']}/20 |")
    lines.extend(['', '负差异代表 E23 更好。区间用同一模型配对、训练种子与掩码两个轴分别重采样 10,000 次计算。100 个单元没有被当成独立样本；区间仅针对此固定 ETTh1 切分下的训练随机性和缺失位置变化。', '',
                  '| 模型 | 完整训练时间均值±标准差（秒） | Stage 1 时间均值（秒） | Stage 2 时间均值（秒） | 峰值 allocated 均值/最大值（GiB） | 峰值 reserved 均值/最大值（GiB） |',
                  '|---|---:|---:|---:|---:|---:|'])
    for row in summary:
        lines.append(f"| {row['model']} | {float(row['full_training_seconds_mean']):.2f} ± {float(row['full_training_seconds_std_ddof1']):.2f} | {float(row['repair_stage_seconds_mean']):.2f} | {float(row['forecast_stage_seconds_mean']):.2f} | {float(row['peak_allocated_gib_mean']):.3f}/{float(row['peak_allocated_gib_max']):.3f} | {float(row['peak_reserved_gib_mean']):.3f}/{float(row['peak_reserved_gib_max']):.3f} |")
    lines.extend(['', '完整训练时间包括入口导入、准备、全部训练阶段、验证与 checkpoint 保存，排除最终测试。所有训练顺序使用同一张 RTX 5090 GPU 1。T1 Stage 1 保留官方 batch=16/BF16/早停；E23 Stage 1 为 batch=128/FP32/100 轮；Stage 2 均为 batch=128/FP32/100 轮。具体实际轮数见逐种子 CSV。', '',
                  'E23 使用原 v31-B 的混合预测损失与均匀修复保持；T1 使用原 MSE 预测训练。因此预测头结构相同，不宣称训练损失完全相同。', '',
                  '该表不证明跨数据集、所有缺失场景最优。T1 原论文的 40% 点缺失训练和跨点缺失率/长块缺失测试尚未包含于本轮 P4 对照。', '',
                  f'[方案与来源](2026-10-04_T1_five_seed_P4_comparison.md) · [汇总 CSV]({STEM}.csv) · [逐种子 CSV]({STEM}_per_seed.csv) · [全部掩码结果]({STEM}_all_draws.csv)', '',
                  f'完整产物：`{ROOT}`', ''])
    target = DOCS/(STEM+'.md')
    with target.open('x') as stream:
        stream.write('\n'.join(lines))
    outputs[str(target)] = digest(target)
    status(dict(status='published_local_files', finished_utc=datetime.now(timezone.utc).isoformat(),
                outputs=outputs, source_sha256=digest(__file__), trial_definition_sha256=digest(definition)))


def main():
    status(dict(status='waiting_for_complete_five_seed_results', source_sha256=digest(__file__)))
    for _ in range(288):  # At most 24 hours; observe at the requested five-minute cadence.
        path = ROOT/'comparison_statistics.json'
        if path.exists():
            try:
                result = json.loads(path.read_text())
            except json.JSONDecodeError:
                result = {}
            if result.get('status') == 'complete':
                publish(result)
                return
        if 'Traceback (most recent call last)' in (ROOT/'coordinator.log').read_text():
            status(dict(status='coordinator_failed', log=str(ROOT/'coordinator.log')))
            return
        time.sleep(300)
    status(dict(status='timed_out_after_24_hours'))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status(dict(status='publication_failed', error=repr(error)))
        raise
