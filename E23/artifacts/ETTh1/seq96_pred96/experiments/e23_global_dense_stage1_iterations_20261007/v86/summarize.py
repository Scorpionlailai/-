"""Verify paired forecasts and separately report matched Stage1 repair criteria."""
import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DOCS = Path('/home/xyw/paper/E23/docs/experiments')
STEM = 'ETTh1_E23_v86_v79_T1_pureMSE_frozen_three_scenarios_20261007'


def read(path):
    return json.loads(Path(path).read_text())


def main(partial=False):
    plan = read(ROOT / 'plan.json')
    spec = importlib.util.spec_from_file_location('baseline_summary', Path(plan['baseline_root']) / 'summarize.py')
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    frozen = read(ROOT / 'frozen-definition.json')
    assert helpers.sha(ROOT / 'plan.json') == frozen['plan_sha256']
    assert all(helpers.sha(p) == h for p, h in frozen['protected'].items())
    assert all(helpers.sha(Path(plan['work']) / p) == h for p, h in frozen['sources'].items())
    assert read(ROOT / 'preflight.json')['status'] == 'passed'
    if not partial:
        assert read(ROOT / 'complete.json')['status'] == 'completed'
    repair_ref = {d['scene']: d for d in read(plan['repair_reference'])}
    clean = read(plan['clean_reference'])['metrics']
    rows, repairs, draws, paired, summaries = [], [], [], [], []
    for scene in plan['scenarios']:
        cell = ROOT / scene['name'] / 'E23-v86'
        if partial and not (cell / 'evaluation_mse/twenty-test-draws.json').exists():
            continue
        stage1 = read(cell / 'stage1/runs/e23-stage1.json')
        assert stage1['epochs_completed'] == 100 and stage1['optimizer_updates'] == 6600
        assert stage1['stage1_objective'] == 'global_mse_v1'
        assert stage1.get('stage1_structure','none') == plan['stage1_structure']
        assert stage1.get('stage1_gap_structure', 'none') == 'none'
        assert stage1.get('stage1_aux_forecast', 'none') == 'none'
        assert stage1['recent_mean_projection'] == plan['recent_mean_projection']
        assert helpers.sha(stage1['checkpoint']) == stage1['checkpoint_sha256']
        rm = stage1['validation_repair_metrics']
        assert abs(rm['global_mse'] - stage1['best_val_repair_mse']) < 1e-12
        ref = repair_ref[scene['name']]
        assert rm['hidden_coordinates'] == ref['t1']['all_hidden']['hidden_coordinates']
        rrow = dict(scene=scene['name'], split='fixed_validation', metric_space='train_standardized',
                    hidden_coordinates=rm['hidden_coordinates'], selected_epoch=stage1['best_epoch'],
                    v86_mse=rm['global_mse'], v86_mae=rm['global_mae'],
                    v79_mse=ref['v79']['all_hidden']['mse'], v79_mae=ref['v79']['all_hidden']['mae'],
                    t1_mse=ref['t1']['all_hidden']['mse'], t1_mae=ref['t1']['all_hidden']['mae'])
        rrow.update(repair_beats_t1_mse=rrow['v86_mse'] < rrow['t1_mse'],
                    repair_beats_t1_mae=rrow['v86_mae'] < rrow['t1_mae'])
        repairs.append(rrow)
        selections = {}
        for metric in ['mae', 'mse']:
            paths = {'E23-v86': cell / f'evaluation_{metric}/twenty-test-draws.json',
                     'E23-v79': Path(scene['v79_pure_evaluation']).parent.parent / f'evaluation_{metric}/twenty-test-draws.json',
                     'T1': Path(scene['t1_pure_evaluation']).parent.parent / f'evaluation_{metric}/twenty-test-draws.json'}
            values = {}
            for label, path in paths.items():
                value, record = helpers.verify_record(path, scene, metric)
                assert record['repair_trainable_parameters'] == 0
                if label == 'T1':
                    assert record['optimizer_steps'] == 6600
                else:
                    assert record['batch_size'] == 128
                if label == 'E23-v86':
                    assert record.get('recent_band_extra_parameters',0) == plan['recent_band_extra_parameters']
                    assert record['global_dense_extra_parameters'] == 79751
                    lock = read(cell / f'validation-{metric}-lock-before-tests.json')
                    assert lock['selection_sha256'] == value['selection_manifest_sha256']
                    assert lock['test_metrics_seen'] is False
                values[label] = value
                row = dict(scene=scene['name'], start_rate=scene['start_rate'], block_len=scene['block_len'],
                    actual_missing_fraction=scene['actual_test_missing_fraction'], model=label,
                    selection_metric=metric, selected_epoch=record['best_epoch'],
                    mse_mean=value['test_mean']['mse'], mae_mean=value['test_mean']['mae'],
                    mse_std_ddof1=value['test_mask_std_ddof1']['mse'],
                    mae_std_ddof1=value['test_mask_std_ddof1']['mae'], test_groups=20,
                    clean_mse=clean['mse'], clean_mae=clean['mae'],
                    mse_excess_above_clean=value['test_mean']['mse'] - clean['mse'],
                    mae_excess_above_clean=value['test_mean']['mae'] - clean['mae'],
                    forecast_loss='pure_mse', repair_update_policy='frozen_v1',
                    stage1_training_seed=502 if label == 'T1' else 2023,
                    stage2_training_seed=2023, stage2_epochs=100, stage2_optimizer_steps=6600,
                    stage2_batch_size=128, stage2_learning_rate=1e-4,
                    predictor_initial_sha256=record['predictor_initial_sha256'],
                    predictor_parameters=35427, predictor_trainable_parameters=35424,
                    forecast_padding='replicate_last_value_v1' if label == 'T1' else 'zero_invalid_v1',
                    evaluation_manifest=str(path), selection_manifest=value['selection_manifest'],
                    **helpers.resource_details(value))
                if label != 'T1':
                    row['resource_scope'] = 'measured_fresh_stage1_and_stage2_commands_tests_excluded'
                rows.append(row)
                for draw in value['test_draws']:
                    draws.append(dict(scene=scene['name'], model=label, selection_metric=metric,
                        group_id=draw['group_id'], mask_seed=draw['mask_seed'], mse=draw['mse'],
                        mae=draw['mae'], mask_sha256=draw['mask_sha256']))
            candidate = values['E23-v86']
            for label in ['E23-v79', 'T1']:
                for a, b in zip(candidate['test_draws'], values[label]['test_draws']):
                    assert all(a[k] == b[k] for k in ['group_id', 'mask_seed', 'mask_sha256', 'windows', 'target_elements'])
                    paired.append(dict(scene=scene['name'], selection_metric=metric, reference=label,
                        group_id=a['group_id'], mask_seed=a['mask_seed'],
                        candidate_minus_reference_mse=a['mse'] - b['mse'],
                        candidate_minus_reference_mae=a['mae'] - b['mae']))
            selections[metric] = {label: v['test_mean'] for label, v in values.items()}
        summaries.append(dict(scene=scene['name'], repair=rrow, selections=selections))
    if partial:
        path = ROOT / 'partial-results.json'
        path.write_text(json.dumps(dict(status='partial', completed_scenes=len(summaries),
                                       clean=clean, scenarios=summaries), indent=2, allow_nan=False) + '\n')
        print(json.dumps(dict(completed_scenes=len(summaries), scenarios=summaries)), flush=True)
        return
    helpers.csv_write(DOCS / f'{STEM}.csv', rows)
    helpers.csv_write(DOCS / f'{STEM}_repair.csv', repairs)
    helpers.csv_write(DOCS / f'{STEM}_all_draws.csv', draws)
    helpers.csv_write(DOCS / f'{STEM}_paired.csv', paired)
    report = ['# E23-v86：最近区段修复与 clean 差距', '',
        plan['label_description'], '',
        'Stage2双方纯MSE、完整冻结修复器、原预测头及同初始SHA、seed2023、batch128、lr1e-4、100轮/6600更新。主表验证MAE选模，验证MSE次选同轨迹另列CSV。', '',
        f'同口径 clean：MSE {clean["mse"]:.6f}，MAE {clean["mae"]:.6f}。这是全可见历史下实测参考，不是理论下界。', '',
        '| 场景 | v79 MSE | v86 MSE | T1 MSE | v79 MAE | v86 MAE | T1 MAE |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        a, b, c = [s['selections']['mae'][k] for k in ['E23-v79', 'E23-v86', 'T1']]
        report.append(f'| {s["scene"]} | {a["mse"]:.6f} | {b["mse"]:.6f} | {c["mse"]:.6f} | {a["mae"]:.6f} | {b["mae"]:.6f} | {c["mae"]:.6f} |')
    report += ['', '## Stage1 修复：固定验证集缺失坐标', '',
        '| 场景 | v79 MSE | v86 MSE | T1 MSE | v79 MAE | v86 MAE | T1 MAE | 超过T1 MSE/MAE |',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for r in repairs:
        report.append(f'| {r["scene"]} | {r["v79_mse"]:.6f} | {r["v86_mse"]:.6f} | {r["t1_mse"]:.6f} | {r["v79_mae"]:.6f} | {r["v86_mae"]:.6f} | {r["t1_mae"]:.6f} | {r["repair_beats_t1_mse"]}/{r["repair_beats_t1_mae"]} |')
    report += ['', '## 时间与显存', '',
        '| 场景 | 模型 | 两阶段记录成本秒 | Stage1秒 | Stage2秒 | allocated / reserved GiB |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        if r['selection_metric'] == 'mae':
            report.append(f'| {r["scene"]} | {r["model"]} | {r["two_stage_recorded_cost_seconds"]:.2f} | {r["stage1_recorded_seconds"]:.2f} | {r["stage2_training_seconds"]:.2f} | {r["peak_allocated_gib"]:.3f} / {r["peak_reserved_gib"]:.3f} |')
    report += ['', '## 统计范围', '',
        '- 修复允许相对上一版退步，但必须超过T1；同时分别核对修复MSE和MAE，不用预测优势替代修复优势。',
        '- 旧20组是开发测试。20组标准差是缺失位置方差，不是训练种子方差。已曝光的v79确认组也不能作为新盲测。',
        '- v86、v79两阶段分别实测；T1成本为历史Stage1与匹配新Stage2组合。三者同物理RTX5090；测试时间不计入训练。',
        '- T1保留自身末尾复制补齐，E23保留自身零补齐。没有更改预测头。',
        '- 缺失真值干预仅用于前期固定验证诊断；模型推理只读取可见历史和自身修复。',
        f'- 封存配置、代码、原始记录、逐窗口误差与五分钟检查记录：{ROOT}', '',
        f'[预测CSV](./{STEM}.csv) · [修复CSV](./{STEM}_repair.csv) · [逐组误差](./{STEM}_all_draws.csv)']
    with (DOCS / f'{STEM}.md').open('x') as handle:
        handle.write('\n'.join(report) + '\n')
    verification = dict(status='verified', scenarios=summaries, clean=clean,
        repair_beats_t1_all_metrics_all_scenes=all(r['repair_beats_t1_mse'] and r['repair_beats_t1_mae'] for r in repairs),
        forecast_better_than_v79_all_metrics_all_scenes=all(
            s['selections']['mae']['E23-v86'][k] < s['selections']['mae']['E23-v79'][k]
            for s in summaries for k in ['mse', 'mae']),
        report=str(DOCS / f'{STEM}.md'), csv=str(DOCS / f'{STEM}.csv'))
    with (ROOT / 'verification.json').open('x') as handle:
        json.dump(verification, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps(verification), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--partial', action='store_true')
    main(parser.parse_args().partial)
