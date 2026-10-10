from pathlib import Path
import csv
import hashlib
import json

ROOT = Path('/home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v72_long_gap_moments_seed2023_20261005')
PREFIX = Path('/home/xyw/paper/E23/docs/experiments/ETTh1_E23_v72_long_gap_moments_seed2023_20261005')
read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    audit = read(ROOT / 'report-audit.json')
    assert audit['all_scenes_complete']
    plan = read(ROOT / 'plan.json')
    counts = {r['scene']: r for r in read(ROOT / 'validation-coverage-counts.json')}
    rows = []
    for scene in plan['scenarios']:
        count = counts[scene['name']]
        assert count['validation_mask_sha256'] == scene['validation_mask_sha256']
        assert count['windows_without_observation'] == 0
        reference_cell = Path(scene['v69_evaluation']).parent.parent
        own_cell = ROOT / scene['name'] / 'E23-v72-mae'
        for model, cell in [('E23-v69', reference_cell), ('E23-v72', own_cell)]:
            path = cell / 'recovery-diagnostics.json'
            diagnostics = read(path)
            diag = next(r for r in diagnostics if r['stage'] == 'stage1')
            stage2 = next(r for r in diagnostics if r['stage'] == 'stage2_mae')
            assert diag['metrics'] == stage2['metrics']
            assert diag['bank_state_unchanged'] and stage2['bank_state_unchanged']
            assert sha(diag['selection_manifest']) == diag['selection_sha256']
            manifest = read(diag['selection_manifest'])
            assert sha(manifest['checkpoint']) == diag['checkpoint_sha256']
            assert diag['metric_space'] == 'train_standardized'
            assert diag['damaged_patches'] == count['damaged_patches']
            full = count['fully_hidden_patches']
            row = dict(
                scenario=scene['name'], model=model, training_seed=2023,
                split='fixed_validation', metric_space=diag['metric_space'],
                actual_test_missing_fraction=scene['actual_test_missing_fraction'],
                validation_mask_sha256=scene['validation_mask_sha256'],
                selected_stage1_epoch=diag['selected_epoch'],
            )
            for region, metrics in diag['metrics'].items():
                for metric in ('mse', 'mae', 'hidden_coordinates'):
                    row[region + '_' + metric] = metrics[metric]
            row.update(
                last32_level_mse=diag['last32_level_mse'],
                last32_slope_mse=diag['last32_slope_mse'],
                damaged_patches=count['damaged_patches'],
                fully_hidden_patches=full,
                h_available_damaged_patches=diag['h_available_damaged_patches'],
                h_available_damaged_fraction=diag['h_available_damaged_patches'] / count['damaged_patches'],
                h_available_fully_hidden_patches=diag['h_available_fully_hidden_patches'],
                h_available_fully_hidden_fraction=diag['h_available_fully_hidden_patches'] / full,
                point_candidate_null_mass=diag['point_candidate_null_mass'],
                stage2_repair_same_as_stage1=True,
                diagnostic_path=str(path), diagnostic_sha256=sha(path),
            )
            rows.append(row)
        old, new = rows[-2:]
        for region in ('all96', 'before64', 'last32', 'block_boundaries'):
            assert old[region + '_hidden_coordinates'] == new[region + '_hidden_coordinates']
        assert new['h_available_fully_hidden_patches'] == 0
        assert 0 <= new['h_available_damaged_patches'] <= count['damaged_patches']
    path = Path(str(PREFIX) + '_repair_diagnostics.csv')
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    marker = '\n## 固定验证修复诊断\n'
    report = PREFIX.with_suffix('.md')
    lines = [
        '', '## 固定验证修复诊断', '',
        '下表是固定验证集缺失坐标的修复误差，采用全局训练标准化尺度；与上面的20组测试预测误差分开。两版本Stage2均冻结修复器，所以其修复结果与各自Stage1相同。', '',
        '| 场景 | 模型 | 全96点修复MSE | 前64点 | 最近32点 | 全缺失patch的H覆盖 |',
        '|---|---|---:|---:|---:|---:|',
    ]
    for r in rows:
        lines.append(f"| {r['scenario']} | {r['model']} | {r['all96_mse']:.6f} | {r['before64_mse']:.6f} | {r['last32_mse']:.6f} | {r['h_available_fully_hidden_patches']}/{r['fully_hidden_patches']} |")
    lines += ['',
        'H覆盖只表示候选可用，不表示准确。本轮只修改Stage1监督，完全缺失patch仍不读取H；分区域诊断检查全窗口及近期信息是否同时改善。',
        '本轮同时监督长段水平与趋势，尚未分离两项贡献；没有用干预实验确认回溯位置对预测的因果影响。', '',
        f'[分区域修复与H覆盖CSV]({path.name})', '',
    ]
    report.write_text(report.read_text().split(marker)[0] + '\n'.join(lines))
    summary = dict(rows=len(rows), csv=str(path), csv_sha256=sha(path), report_sha256=sha(report),
                   fixed_validation_only=True, stage2_repair_frozen=True, coverage_verified_against_actual_masks=True)
    (ROOT / 'diagnostic-report-audit.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
