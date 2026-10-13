"""Regenerate local review artifacts from completed, traceable checkpoints."""
import csv
import json
import math
from pathlib import Path

from .data import SCENES, file_sha256, write_json

MODELS = ('TimesNet', 'ImputeFormer', 'BRITS', 'CSDI')
REPAIR_METRIC_ABSOLUTE_TOLERANCE = 2 ** -23  # One FP32 epsilon; no relative tolerance.


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def audit_repair(folder, plan, stage1, stage2, original, main, secondary):
    banks = [record['groups'] for record in (original, main, secondary)]
    reference = {group['mask_seed']: group for group in banks[0]}
    support_matches = all(len(bank) == 20 and len({group['mask_seed'] for group in bank}) == 20
                          and {group['mask_seed'] for group in bank} == set(reference) for bank in banks)
    metrics_identical = support_matches
    metrics_consistent = support_matches
    max_delta = 0.
    for bank in banks[1:]:
        for group in bank:
            expected = reference.get(group['mask_seed'])
            if expected is None:
                continue
            support_matches &= all(group[key] == expected[key] for key in ('mask_sha256', 'windows'))
            for section in ('full96', 'recent32'):
                left, right = expected[section], group[section]
                support_matches &= left['coordinates'] == right['coordinates']
                metrics_identical &= left == right
                for key in ('mse', 'mae'):
                    delta = abs(left[key] - right[key])
                    max_delta = max(max_delta, delta)
                    metrics_consistent &= math.isclose(left[key], right[key], rel_tol=0.,
                                                      abs_tol=REPAIR_METRIC_ABSOLUTE_TOLERANCE)
    main_by_seed = {group['mask_seed']: group for group in banks[1]}
    branches_identical = support_matches and all(
        all(group[section] == main_by_seed[group['mask_seed']][section] for section in ('full96', 'recent32'))
        for group in banks[2])
    checkpoint_matches = bool(stage1 and stage2) and (
        stage1['checkpoint_sha256'] == stage2['parent_stage1_sha256'] == file_sha256(folder / 'stage1/selected.pt'))
    cache_records = (read(folder / 'stage2/cache_audit.json') or {}).get('records', {})
    expected_cache_keys = {'test_{}'.format(seed) for seed in reference}
    test_cache_keys = {key for key in cache_records if key.startswith('test_')}
    cache_matches = bool(stage2 and plan) and expected_cache_keys == test_cache_keys and all(
        record['binding']['checkpoint_sha256'] == stage2['parent_stage1_sha256'] and
        record['binding']['state_sha256'] == stage2['repair_state_sha256'] and
        record['binding']['data_manifest_sha256'] == plan['data_manifest_sha256']
        for record in cache_records.values())
    return dict(repair_unchanged_after_stage2=bool(support_matches and metrics_consistent and
                                                 branches_identical and checkpoint_matches and cache_matches),
                repair_metrics_bitwise_identical=bool(metrics_identical),
                repair_metrics_numerically_consistent=bool(metrics_consistent and support_matches),
                forecast_branches_repair_metrics_identical=bool(branches_identical),
                mask_and_coordinate_support_matches=bool(support_matches),
                repair_checkpoint_matches=bool(checkpoint_matches), cache_bindings_match=bool(cache_matches),
                max_repair_metric_absolute_difference=max_delta,
                metric_absolute_tolerance=REPAIR_METRIC_ABSOLUTE_TOLERANCE)


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
                    row['variant'] = 'CSDI' if model == 'CSDI' else plan['profile'].get('variant', 'author_config_adapted')
                    row['stage1_configured_epochs'] = plan['profile']['epochs']
                    if model == 'CSDI':
                        row['evaluation_samples'] = plan['model_config']['sampling']['samples']
                        row['diffusion_steps'] = plan['model_config']['sampling']['diffusion_steps']
                        row['learning_rate_milestones'] = str(plan['profile'].get('milestones', [150, 180]))
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
                    row['forecast_training_protocol'] = stage2.get('training_protocol')
                    row['cache_generation_seconds'] = sum(r['generation_seconds'] for r in stage2.get('cache_records', {}).values())
                    row['csdi_training_samples'] = stage2.get('csdi_training_samples')
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
                if repair_result and main and secondary:
                    check = audit_repair(folder, plan, stage1, stage2, repair_result, main, secondary)
                    row['stage2_repair_metrics_identical'] = check['repair_metrics_bitwise_identical']
                    row['stage2_repair_audit_passed'] = check['repair_unchanged_after_stage2']
                    row['stage2_repair_max_metric_absolute_difference'] = check['max_repair_metric_absolute_difference']
                    audit.append(dict(dataset=dataset, scene=scene, model=model, **check))
                rows.append(row)
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with (root / 'results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    ready = all(r.get('main_forecast_mse_mean') is not None and
                r.get('secondary_forecast_mse_mean') is not None and
                r.get('stage2_repair_audit_passed') and r['status'] == 'stage2_complete' for r in rows)
    head_shas = {r['head_initial_sha256'] for r in rows if 'head_initial_sha256' in r}
    if len(head_shas) != 1:
        ready = False
    saved_output_audit = read(root / 'final_saved_output_audit.json') or {}
    saved_outputs_verified = (saved_output_audit.get('passed') is True and
                              len(saved_output_audit.get('cases', [])) == 32 and
                              saved_output_audit.get('total_test_groups') == 640)
    ready = ready and saved_outputs_verified
    upload_authorization = read(root / 'github_publication_authorization_20261013.json') or {}
    upload_authorized = upload_authorization.get('approved') is True
    publication = read(root / 'github_publication_status.json') or {}
    github_uploaded = publication.get('remote_verified') is True
    write_json(root / 'review_status.json', dict(ready_for_user_review=ready,
        expected_runs=32, forecast_test_mask_groups=20,
        completed_repairs=sum('repair_full96_mse_mean' in r for r in rows),
        completed_forecasts=sum('main_forecast_mse_mean' in r for r in rows),
        common_head_initial_sha256=list(head_shas), freeze_metric_audit=audit,
        saved_repair_outputs_verified=saved_outputs_verified,
        github_uploaded=github_uploaded, github_authorized=upload_authorized,
        github_policy='user_authorized_publication' if upload_authorized else 'await_user_review'))
    lines = ['# ETTh1 / Weather 新增基线复现报告', '',
             '**状态：' + (('全部完成，已核验，用户已授权同步GitHub。' if upload_authorized else '全部完成，已核验，待用户审核。') if ready else '实验或完整性核验尚未完成；详见 review_status.json。') + '**', '',
             '范围：TimesNet、ImputeFormer、BRITS、CSDI；两个数据集、四种块缺失场景，共32次独立修复训练，随后32次冻结修复器的公共 PatchTST 训练。', '',
             '主表均在训练集全局标准化尺度计算。全96点及近32点只计隐藏有效坐标；每组先累积误差/坐标数，再对20组求均值与样本标准差。单个训练 seed=2023；位置标准差不代表训练种子标准差。', '',
             '冻结核验分别检查修复权重SHA、缓存绑定、20组掩码与坐标数量，并要求两个预测选模分支的修复指标完全一致。原修复评估与缓存评估的指标允许一个FP32 epsilon（2^-23）绝对误差、无相对容差；逐位一致性另行保留，原始结果不改写。实际最大指标差异见 results.csv 与 review_status.json。', '',
             'Weather 掩码内容校验和与历史资产一致。ETTh1 使用独立重建协议；20组缺失比例及窗口统计与历史记录一致，但原二进制资产缺失，文件SHA不同，未验证与旧实验逐坐标一致。此次四个模型使用同一套封存资产；不能直接把本表 ETTh1 与旧表声称为严格配对结果。', '',
             'TimesNet 使用作者 ETTh1/Weather 专用插补脚本及入口默认参数。ImputeFormer 使用作者 LA 基础配置；BRITS 使用 README 示例；CSDI 使用 base.yaml。后三者明确标为非目标数据集最佳配置，已获用户授权。', '',
             '用户所指预测阶段标准为测试采用20组缺失：每组独立计算预测MSE/MAE，再汇总均值和样本标准差。当前实现满足这一要求。Stage2 四个新增基线统一使用固定训练掩码及FP32修复缓存；这是对原动态训练掩码协议的调整。CSDI训练缓存用1次采样，验证和测试沿用各次修复实验封存的评价采样数。缓存只含历史修复值与可见性，未来真值不输入修复器。Stage2 使用公共 PatchTST，完整冻结修复器并 eval，100轮、batch128、FP32、Adam lr=1e-4、纯预测MSE。主权重按验证MAE选择；验证MSE分支另报，不按测试择优。', '',
             '用户授权加速版：BRITS 保留隐藏维108及原损失，采用TorchScript循环，训练上限由1000轮改为200轮，验证MAE选模、patience20。CSDI 按各次实验封存配置报告：后续任务100训练轮、75/90轮学习率衰减、50扩散步，评价5样本取中位数，预测训练缓存1样本；已运行任务保留其原封存配置。模型名称写为CSDI，实际参数差异如实披露。', '',
             '| 数据集 | 场景 | 模型 | 全96修复 MSE / MAE | 近32修复 MSE / MAE | 预测 MSE / MAE（验证MAE主选） | 状态 |',
             '|---|---|---|---|---|---|---|']
    def pair(row, prefix):
        first, second = row.get(prefix+'_mse_mean'), row.get(prefix+'_mae_mean')
        return f'{first:.6f} / {second:.6f}' if first is not None else '—'
    comparison_csv = root / 'comparisons/final_repair_forecast_vs_v125_t1_20261013.csv'
    references = []
    if comparison_csv.exists():
        with comparison_csv.open() as stream:
            for reference in csv.DictReader(stream):
                if reference['model'] not in ('E23-v125', 'T1'):
                    continue
                item = dict(dataset=reference['dataset'], scene=reference['scene'],
                            model='v125' if reference['model'] == 'E23-v125' else 'T1',
                            status='历史参考；预测记录缺失' if not reference.get('forecast_mse') else '历史参考')
                for source, target in (('repair96', 'repair_full96'), ('repair32', 'repair_recent32'), ('forecast', 'main_forecast')):
                    for metric in ('mse', 'mae'):
                        if reference.get(source+'_'+metric):
                            item[target+'_'+metric+'_mean'] = float(reference[source+'_'+metric])
                references.append(item)
        table_start = lines.index('| 数据集 | 场景 | 模型 | 全96修复 MSE / MAE | 近32修复 MSE / MAE | 预测 MSE / MAE（验证MAE主选） | 状态 |')
        lines[table_start:table_start] = ['v125与T1作为历史参考列入同表，不计入此次新增32项任务。ETTh1与新基线不构成逐坐标配对；Weather新增基线使用固定训练掩码缓存，旧参考训练协议不同。旧Weather v125的sr020_bl8及sr010_bl16在本地资产中缺少完整预测评估，明确留空。', '']
    displayed = []
    for row in rows:
        if row['model'] == MODELS[0]:
            displayed.extend(reference for reference in references
                             if reference['dataset'] == row['dataset'] and reference['scene'] == row['scene'])
        displayed.append(row)
    for row in displayed:
        lines.append('| ' + ' | '.join([row['dataset'], row['scene'], row['model'],
                     pair(row, 'repair_full96'), pair(row, 'repair_recent32'),
                     pair(row, 'main_forecast'), row['status']]) + ' |')
    lines += ['', '完整均值、位置标准差、次选预测分支、实际轮数/步数、训练和推理耗时、显存、参数来源及权重SHA见 [results.csv](results.csv)。各运行目录保存plan、逐轮曲线、选模记录、20组逐组指标和资源记录。', '',
              '适配说明：公共切分/标准化/块缺失替换作者原数据设置；回溯96、变量数按数据集调整；只向修复器输入回溯及可见性。TimesNet 全缺失变量分母 clamp_min(1)，不取隐藏真值。BRITS 无分类标签，label_weight=0，并采用上述加速设置。ImputeFormer 保留值通道+作者日周期sin/cos两个外生通道和原Freg；FFT用FP32处理非二次幂窗口，训练其余部分仍使用作者混合FP16。CSDI 保留50扩散步，采样数以各运行plan为准，torch.median 聚合。', '',
              '用户已授权将实验结果、配置、评估记录与复现代码同步GitHub；实际发布状态见 github_publication_status.json。' if upload_authorized else '所有文件仅保存在本地，用户审核后再上传 GitHub。']
    if saved_outputs_verified:
        lines += ['', '完整输出核验通过：32项各有训练、验证及20组测试缓存，共640组测试修复输出；1408个缓存文件SHA校验通过，保存权重的实际状态SHA与预测阶段冻结记录一致。均值与ddof=1标准差已重新核算，核验未调用修复模型。证据见 [final_saved_output_audit.json](final_saved_output_audit.json)。']
    comparison = root / 'comparisons/final_repair_forecast_vs_v125_t1_20261013.md'
    if comparison.exists():
        lines += ['', '[新增基线与v125/T1的两个数据集完整对照](comparisons/final_repair_forecast_vs_v125_t1_20261013.md)。旧参考的缺失项及协议差异单独标注。']
    (root / 'report.md').write_text('\n'.join(lines) + '\n')
    return rows


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    render(parser.parse_args().root)
