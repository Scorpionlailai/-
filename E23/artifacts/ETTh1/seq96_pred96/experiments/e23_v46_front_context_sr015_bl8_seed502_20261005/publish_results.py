"""Publish the completed v46 trial with existing matched v41/T1 references."""
import csv
import hashlib
import json
from pathlib import Path
import tarfile
import numpy as np

ROOT = Path(__file__).resolve().parent
DOCS = Path('/home/xyw/paper/E23/docs/experiments')
PREFIX = DOCS/'ETTh1_E23_v46_front_context_sr015_bl8_seed502_20261005'


def read(path):return json.loads(Path(path).read_text())
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path, value):
    with Path(path).open('x') as f:json.dump(value, f, indent=2);f.write('\n')
def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys);writer.writeheader();writer.writerows(rows)


def main():
    completed = read(ROOT/'completed-audit.json');assert completed['status'] == 'complete'
    plan = read(ROOT/'plan.json');definition = read(ROOT/'frozen-definition.json');work = Path(plan['work'])
    for relative, sha in definition['sources'].items():assert digest(work/relative) == sha
    for path, sha in definition['protected'].items():assert digest(path) == sha
    assert digest(ROOT/'plan.json') == definition['plan_sha256']
    assert digest(plan['scenario_manifest']) == definition['scenario_sha256']
    assert digest(plan['test_bank']) == definition['test_bank_sha256']
    assert digest(plan['config']) == definition['config_sha256']
    with tarfile.open(ROOT/'frozen-source.tar.gz') as archive:
        for relative, sha in definition['sources'].items():
            assert hashlib.sha256(archive.extractfile(relative).read()).hexdigest() == sha, relative
    current = read(ROOT/'evaluation/twenty-test-draws.json')
    selection = ROOT/'stage2/runs/e23-stage2.json';s2 = read(selection);s1 = read(ROOT/'stage1/runs/e23-stage1.json')
    resources = current['training_resources'];preflight = read(ROOT/'preflight.json')
    assert resources['resource_status'] == 'measured_all_stages'
    assert s1['epochs_completed'] == s2['epochs_completed'] == 100
    assert len(current['test_draws']) == 20
    assert s2['predictor_initial_sha256'] == plan['predictor_initial_sha256']
    assert s2['predictor_learning_rate'] == 1e-4 and s2['repair_learning_rate'] == 1e-5
    assert s2['loss'] == 'forecast_mse_plus_0p2_mae' and s2['repair_retention'] == 'uniform_v1'
    assert s2.get('forecast_mixer', 'none') == 'none' and s2['forecast_input'] == 'explicit_v1'
    old_csv = DOCS/'ETTh1_T1_v41_slow_higher_missing_seed502_20261004.csv'
    with old_csv.open() as f:rows = [r for r in csv.DictReader(f) if r['scenario'] == 'sr015_bl8']
    assert len(rows) == 2
    refs = {name:read(path) for name, path in plan['cached_references'].items()}
    for row in rows:
        ref = refs[row['model']]
        assert float(row['mse']) == ref['test_mean']['mse'] and float(row['mae']) == ref['test_mean']['mae']
        assert ref['test_mask_bank_sha256'] == current['test_mask_bank_sha256']
        assert ref['validation_mask_sha256'] == current['validation_mask_sha256']
        assert ref['predictor_initial_sha256'] == current['predictor_initial_sha256']
        row['cached_reference'] = True
        row['repair_parameters'] = 174362 if row['model'] == 'E23-v41-slow' else ''
        row['front_context_parameters'] = 0
    stage_resources = [read(p) for p in resources['resource_record_paths']]
    assert len(stage_resources) == 2 and all(r['status'] == 'completed' for r in stage_resources)
    uuid_set = {r['device']['uuid'] for r in stage_resources}
    for ref in refs.values():
        uuid_set.update(read(p)['device']['uuid'] for p in ref['training_resources']['resource_record_paths'])
    assert len(uuid_set) == 1
    bystage = {name:v for r in stage_resources for name, v in r['stages'].items()}
    new = dict(scenario='sr015_bl8', start_rate=.15, block_len=8,
        actual_test_missing_fraction=plan['actual_test_missing_fraction'], model=plan['label'], training_seed=502,
        seq_len=96, pred_len=96, test_masks=20, mse=current['test_mean']['mse'], mae=current['test_mean']['mae'],
        mask_mse_std=current['test_mask_std_ddof1']['mse'], mask_mae_std=current['test_mask_std_ddof1']['mae'],
        validation_mse=s2['best_val_mse'], validation_mae=s2['best_val_mae'],
        full_training_seconds=resources['full_workflow_seconds'], peak_allocated_gib=resources['peak_allocated_gib'],
        peak_reserved_gib=resources['peak_reserved_gib'], gpu=resources['gpu'], training_batch_sizes='128',
        predictor_parameters=35427, predictor_initial_sha256=plan['predictor_initial_sha256'],
        validation_mask_sha256=current['validation_mask_sha256'], test_bank_sha256=current['test_mask_bank_sha256'],
        checkpoint_sha256=s2['checkpoint_sha256'], selection=str(selection),
        evaluation=str(ROOT/'evaluation/twenty-test-draws.json'), cached_reference=False,
        stage1_epochs=100, stage2_epochs=100, selected_stage2_epoch=s2['best_epoch'],
        stage1_repair_mse=s1['best_val_repair_mse'],
        repair_stage_seconds=bystage['repair']['wall_seconds'], forecast_stage_seconds=bystage['forecast']['wall_seconds'],
        stage1_batch_size=128, stage2_batch_size=128, stage1_precision='float32', stage2_precision='float32',
        stage2_loss=s2['loss'], repair_update_policy=s2['repair_update_policy'],
        repair_parameters=preflight['repair_parameters'], front_context_parameters=preflight['front_context_parameters'])
    rows.append(new);write_csv(PREFIX.with_suffix('.csv'), rows)
    all_draws = []
    for label, result in {**refs, plan['label']:current}.items():
        for draw in result['test_draws']:
            all_draws.append(dict(model=label, training_seed=502, scenario='sr015_bl8', cached_reference=label in refs, **draw))
    write_csv(Path(str(PREFIX)+'_all_draws.csv'), all_draws)
    comparisons = []
    for label, baseline in refs.items():
        deltas = {metric:[] for metric in ('mse', 'mae')}
        for a, b in zip(current['test_draws'], baseline['test_draws']):
            assert (a['group_id'], a['mask_seed'], a['mask_sha256'], a['target_elements']) == (b['group_id'], b['mask_seed'], b['mask_sha256'], b['target_elements'])
            for metric in deltas:deltas[metric].append(a[metric]-b[metric])
        comparisons.append(dict(candidate=plan['label'], baseline=label, same_masks=True,
            mse_delta=float(np.mean(deltas['mse'])), mae_delta=float(np.mean(deltas['mae'])),
            relative_mse_improvement_pct=(baseline['test_mean']['mse']-current['test_mean']['mse'])/baseline['test_mean']['mse']*100,
            mse_wins=sum(d<0 for d in deltas['mse']), mae_wins=sum(d<0 for d in deltas['mae']), total_masks=20))
    write_csv(Path(str(PREFIX)+'_paired.csv'), comparisons)
    diagnostics = read(ROOT/'recovery-diagnostics.json')
    lines = ['# E23-v46：专家前端可见上下文编码', '',
        'ETTh1，回溯96/预测96，起点率0.15、块长8，20组实际平均缺失72.9252%；训练seed502。v46从头训练100轮Stage1＋100轮Stage2，原v41-slow与T1复用同场景已有结果。按固定验证MAE选权重后锁定测试。', '',
        '## 预测与训练资源', '', '| 模型 | 测试MSE均值±位置标准差 | 测试MAE均值±位置标准差 | 验证MSE / MAE | 完整训练分钟 | allocated / reserved GiB |',
        '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['model']} | {float(row['mse']):.6f} ± {float(row['mask_mse_std']):.6f} | {float(row['mae']):.6f} ± {float(row['mask_mae_std']):.6f} | {float(row['validation_mse']):.6f} / {float(row['validation_mae']):.6f} | {float(row['full_training_seconds'])/60:.2f} | {float(row['peak_allocated_gib']):.3f} / {float(row['peak_reserved_gib']):.3f} |")
    gate = completed['validation']['matched_v41_validation_gate']
    lines += ['', '## 结果判断', '', f"预先规定的验证门槛（MSE更低、MAE不差）{'通过' if gate else '未通过'}；v46选中第{s2['best_epoch']}轮，Stage1选中第{s1['best_epoch']}轮。测试不参与checkpoint选择或版本筛选。"]
    for comparison in comparisons:
        lines += ['', f"相对{comparison['baseline']}：测试MSE差值{comparison['mse_delta']:+.6f}，MAE差值{comparison['mae_delta']:+.6f}；MSE相对改善{comparison['relative_mse_improvement_pct']:+.2f}%。20组位置分别赢下MSE {comparison['mse_wins']}/20、MAE {comparison['mae_wins']}/20。"]
    lines += ['', '## 修复诊断', '', '| 保存的修复器 | 所有隐藏点MSE | 最近32点隐藏MSE | 最近32点水平误差MSE | 最近32点斜率误差MSE |', '|---|---:|---:|---:|---:|']
    for d in diagnostics:lines.append(f"| {d['stage']} | {d['metrics']['all96']['mse']:.6f} | {d['metrics']['last32']['mse']:.6f} | {d['last32_level_mse']:.6f} | {d['last32_slope_mse']:.8f} |")
    lines += ['', '修复误差在训练全局标准化尺度统计，每个原始隐藏坐标只计一次。水平/斜率指标先保留可见值、再对最近32点误差拟合，包含可见坐标的零误差。诊断只读取固定验证集和锁定权重。', '',
        '## 实现和实验口径', '',
        '仅新增H/V/T前端时间CNN：每变量共享，5种可见输入，64维、2块、kernel7/31、dilation1/2，patch有效均值池化，零初始化潜表示增量。增加43200参数，修复器总217562。原H检索、V硬候选选择、T候选条件、P16/S8、单轮T→V→H、最后32点修复CNN均保留。没有新增跨变量注意力、逐点路由或多轮候选刷新。', '',
        '预测头源码/配置/初始化不变，原PatchTST参数35427。Stage1全局隐藏MSE＋0.1检索；Stage2原MSE＋0.2MAE和第11轮起0.05均匀修复保留，预测器lr1e-4、修复器lr1e-5。T1沿用作者修复配置及既有预测扩展，Stage2纯MSE、联合lr1e-4；因此比较的是既定方案，两者训练配方尚有差别。', '',
        '同一RTX5090，训练时间包含两个完整阶段的准备、所有训练与验证及权重保存；最终20组测试与诊断另计。allocated/reserved为PyTorch训练峰值，不是整张卡占用。T1 Stage1为batch16/BF16并按作者规则早停71轮，Stage2为batch128/FP32；E23两个阶段均batch128/FP32。', '',
        '20组标准差只反映缺失位置变化；一个训练seed的结构开发结果不能替代多训练seed结论。', '',
        f"[汇总CSV]({PREFIX.name}.csv) · [60个测试记录]({PREFIX.name}_all_draws.csv) · [配对差值]({PREFIX.name}_paired.csv)", '',
        f"隔离代码：`{work}`。实验权重、完整日志、资源sidecar、归档源码与实现补丁：`{ROOT}`。"]
    with PREFIX.with_suffix('.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    design = DOCS/'2026-10-05_E23_v46_front_context_design.md'
    doc = design.read_text().replace('状态：实现、45项针对性测试、GPU检查通过；从头训练中。', '状态：实现、45项针对性测试、GPU检查通过；从头训练100＋100轮、20组测试及验证修复诊断全部完成。')
    design.write_text(doc+f"\n[完成报告]({PREFIX.name}.md)\n")
    audit = dict(status='complete', source_files_verified=len(definition['sources']),
        archived_source_files_verified=len(definition['sources']), protected_files_verified=len(definition['protected']),
        new_stage1_runs=1, new_stage2_runs=1, new_test_draws=20, cached_reference_draws=40,
        tests_passed=45, gpu_preflight_passed=True, predictor_source_and_initialization_unchanged=True,
        physical_gpu_uuid=next(iter(uuid_set)), test_used_for_selection=False,
        validation_gate_passed=gate, source_archive_sha256=digest(ROOT/'frozen-source.tar.gz'),
        new_full_training_seconds=resources['full_workflow_seconds'], reports=[str(PREFIX.with_suffix('.csv')), str(PREFIX.with_suffix('.md'))])
    write_json(ROOT/'delivery-audit.json', audit)
    print(json.dumps(dict(audit=audit, comparisons=comparisons, new=new)), flush=True)


if __name__ == '__main__':main()
