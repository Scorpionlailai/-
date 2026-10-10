"""Verify and report one shared repair change under the matched downstream protocol."""
import csv
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DOCS = Path('/home/xyw/paper/E23/docs/experiments')
STEM = 'ETTh1_E23_v79_v69_T1_pureMSE_frozen_four_scenarios_20261007'


def read(path):
    return json.loads(Path(path).read_text())


def main():
    plan = read(ROOT / 'plan.json')
    baseline = Path(plan['baseline_root'])
    module_spec = importlib.util.spec_from_file_location('baseline_summary', baseline / 'summarize.py')
    helpers = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(helpers)
    sha = helpers.sha
    assert read(ROOT / 'complete.json')['status'] == 'completed'
    frozen = read(ROOT / 'frozen-definition.json')
    assert sha(ROOT / 'plan.json') == frozen['plan_sha256']
    assert all(sha(p) == h for p, h in frozen['protected'].items())
    assert all(sha(Path(plan['work']) / p) == h for p, h in frozen['sources'].items())
    assert read(ROOT / 'preflight.json')['status'] == 'passed'
    baseline_rows = list(csv.DictReader((DOCS / 'ETTh1_E23_v69_T1_pureMSE_frozen_four_scenarios_20261007.csv').open()))
    rows, draws, paired, results = [], [], [], []
    for scene in plan['scenarios']:
        candidate_cell = ROOT / scene['name'] / 'E23-v79'
        stage1 = read(candidate_cell / 'stage1/runs/e23-stage1.json')
        assert stage1['epochs_completed'] == 100 and stage1['optimizer_updates'] == 6600
        assert stage1['stage1_objective'] == 'global_mse_v1'
        assert stage1['stage1_structure'] == 'recent_mean_v2'
        assert stage1.get('stage1_gap_structure', 'none') == 'none'
        assert stage1.get('stage1_aux_forecast', 'none') == 'none'
        assert stage1['recent_mean_projection'] == 'global_level_mean_v2'
        selections = {}
        for metric in ['mae', 'mse']:
            path = candidate_cell / f'evaluation_{metric}/twenty-test-draws.json'
            candidate, record = helpers.verify_record(path, scene, metric)
            assert record['repair_trainable_parameters'] == 0
            assert read(ROOT / 'preflight.json')['scenarios'][0]['head_parameters'] == 35427
            assert record['recent_mean_global_extra_parameters'] == 3265
            row = dict(scene=scene['name'], start_rate=scene['start_rate'], block_len=scene['block_len'],
                actual_missing_fraction=scene['actual_test_missing_fraction'], model='E23-v79-pureMSE',
                selection_metric=metric, selected_epoch=record['best_epoch'],
                mse_mean=candidate['test_mean']['mse'], mae_mean=candidate['test_mean']['mae'],
                mse_std_ddof1=candidate['test_mask_std_ddof1']['mse'],
                mae_std_ddof1=candidate['test_mask_std_ddof1']['mae'], test_groups=20,
                windows_per_group=2688, forecast_loss='pure_mse', repair_update_policy='frozen_v1',
                stage1_training_seed=2023, stage1_epochs_completed=100, stage1_reused=False,
                stage2_training_seed=2023, stage2_epochs=100, stage2_optimizer_steps=6600,
                stage2_batch_size=128, stage2_learning_rate=1e-4, stage2_weight_decay=0.,
                predictor_initial_sha256=record['predictor_initial_sha256'], predictor_parameters=35427,
                predictor_trainable_parameters=35424, forecast_padding='zero_invalid_v1',
                evaluation_manifest=str(path), selection_manifest=candidate['selection_manifest'],
                **helpers.resource_details(candidate))
            row['resource_scope'] = 'measured_fresh_stage1_and_stage2_commands_tests_excluded'
            rows.append(row)
            references = {}
            for label, directory in [('E23-v69-pureMSE', 'E23-v69'), ('T1-pureMSE-frozen', 'T1')]:
                ref_path = baseline / directory / scene['name'] / f'evaluation_{metric}/twenty-test-draws.json'
                value, _ = helpers.verify_record(ref_path, scene, metric)
                references[label] = value
                matching = [r for r in baseline_rows if r['scene'] == scene['name'] and r['model'] == label and r['selection_metric'] == metric]
                assert len(matching) == 1
                assert set(matching[0]) == set(row)
                rows.append(matching[0])
                for a, b in zip(candidate['test_draws'], value['test_draws']):
                    assert all(a[k] == b[k] for k in ['group_id', 'mask_seed', 'mask_sha256', 'windows', 'target_elements'])
                    paired.append(dict(scene=scene['name'], selection_metric=metric, reference=label,
                        group_id=a['group_id'], mask_seed=a['mask_seed'],
                        candidate_mse=a['mse'], reference_mse=b['mse'], candidate_minus_reference_mse=a['mse']-b['mse'],
                        candidate_mae=a['mae'], reference_mae=b['mae'], candidate_minus_reference_mae=a['mae']-b['mae']))
            for label, value in [('E23-v79-pureMSE', candidate), *references.items()]:
                for d in value['test_draws']:
                    draws.append(dict(scene=scene['name'], model=label, selection_metric=metric,
                        group_id=d['group_id'], mask_seed=d['mask_seed'], mse=d['mse'], mae=d['mae'], mask_sha256=d['mask_sha256']))
            t1 = references['T1-pureMSE-frozen']['test_mean']
            selections[metric] = dict(candidate=candidate['test_mean'],
                v69=references['E23-v69-pureMSE']['test_mean'], t1=t1,
                both_better=all(candidate['test_mean'][k] < t1[k] for k in ['mse', 'mae']),
                paired_group_mse_wins=sum(a['mse'] < b['mse'] for a, b in zip(candidate['test_draws'], references['T1-pureMSE-frozen']['test_draws'])),
                paired_group_mae_wins=sum(a['mae'] < b['mae'] for a, b in zip(candidate['test_draws'], references['T1-pureMSE-frozen']['test_draws'])))
        results.append(dict(scene=scene['name'], stage1_best_epoch=stage1['best_epoch'],
            stage1_validation_repair_mse=stage1['best_val_repair_mse'], selections=selections))
    helpers.csv_write(DOCS / f'{STEM}.csv', rows)
    helpers.csv_write(DOCS / f'{STEM}_all_draws.csv', draws)
    helpers.csv_write(DOCS / f'{STEM}_paired.csv', paired)
    report = ['# v69 增加绝对水平校正：四场景纯 MSE 冻结对照', '',
        'v79 只将 v69 的 recent_mean_projection 从 mean_v1 改为 global_level_mean_v2，新增3265个修复器参数。没有加入长缺失矩损失或Stage1辅助预测。四场景使用同一结构和配置；分别训练。', '',
        '双方Stage2纯MSE、完整冻结修复器、原预测头、相同初始头SHA、seed2023、Adam lr1e-4、batch128、FP32、100轮/6600更新。主表验证MAE选模；验证MSE次选同轨迹另列CSV。', '',
        '| 起点率/块长 | 实际缺失 | v69 MSE | v79 MSE | T1 MSE | v69 MAE | v79 MAE | T1 MAE |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for scene, result in zip(plan['scenarios'], results):
        v = result['selections']['mae']; a, b, c = v['v69'], v['candidate'], v['t1']
        report.append(f'| {scene["start_rate"]:.2f}/{scene["block_len"]} | {100*scene["actual_test_missing_fraction"]:.2f}% | {a["mse"]:.6f} | {b["mse"]:.6f} | {c["mse"]:.6f} | {a["mae"]:.6f} | {b["mae"]:.6f} | {c["mae"]:.6f} |')
    report += ['', '## 时间与显存', '', '| 场景 | 模型 | 两阶段记录成本秒 | Stage2训练秒 | allocated / reserved GiB |', '|---|---|---:|---:|---:|']
    for r in rows:
        if r['selection_metric'] == 'mae':
            report.append(f'| {r["scene"]} | {r["model"]} | {float(r["two_stage_recorded_cost_seconds"]):.2f} | {float(r["stage2_training_seconds"]):.2f} | {float(r["peak_allocated_gib"]):.3f} / {float(r["peak_reserved_gib"]):.3f} |')
    report += ['', '## 统计范围', '',
        '- 每场景20组共享缺失位置，每组2688窗口。标准差反映缺失位置变化；只使用一个训练种子，不能作为多训练种子方差。',
        '- 固定验证集选checkpoint并在测试前锁定；这些旧20组已多次用于开发。结果用于迭代，正式结论需另用未参与开发的共享缺失位置确认。',
        '- v79两个训练阶段本次实测；v69/T1 Stage1为对应原始记录，Stage2为本次新测，两阶段成本为组合记录。GPU为同一物理RTX5090。测试不计入训练时间。',
        '- T1原生末尾复制补齐和E23原生零补齐遵循用户要求；预测头架构没有修改。',
        f'- 完整记录和源代码封存：{ROOT}', '',
        f'[CSV](./{STEM}.csv) · [逐组误差](./{STEM}_all_draws.csv) · [配对差值](./{STEM}_paired.csv)']
    with (DOCS / f'{STEM}.md').open('x') as f:
        f.write('\n'.join(report) + '\n')
    verification = dict(status='verified', scenarios=results,
        all_four_better=all(r['selections']['mae']['both_better'] for r in results),
        report=str(DOCS / f'{STEM}.md'), csv=str(DOCS / f'{STEM}.csv'))
    with (ROOT / 'verification.json').open('x') as f:
        json.dump(verification, f, indent=2); f.write('\n')
    print(json.dumps(verification), flush=True)


if __name__ == '__main__':
    main()
