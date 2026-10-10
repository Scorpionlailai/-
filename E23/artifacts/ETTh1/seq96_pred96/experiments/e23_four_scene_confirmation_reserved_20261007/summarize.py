"""Verify the single reserved confirmation and publish paired results."""
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
BASELINE = Path('/home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_t1_pure_mse_frozen_four_scenarios_20261007')
DOCS = Path('/home/xyw/paper/E23/docs/experiments')


def read(path):
    return json.loads(Path(path).read_text())


def main():
    lock = read(ROOT / 'candidate-lock-before-confirmation.json')
    spec = importlib.util.spec_from_file_location('matched_checks', BASELINE / 'summarize.py')
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    assert read(ROOT / 'complete.json')['status'] == 'completed'
    assert all(helper.sha(p) == h for p, h in lock['protected'].items())
    stem = f'ETTh1_{lock["candidate_label"].replace("-", "_")}_T1_pureMSE_frozen_reserved20_20261007'
    rows, draws, pairs, scenes = [], [], [], []
    baseline_plan = read(BASELINE / 'plan.json')
    for cell in lock['scenarios']:
        bank = read(cell['test_bank'])
        values = {}
        original = next(s for s in baseline_plan['scenarios'] if s['name'] == cell['scene'])
        for label, directory in [(lock['candidate_label'], 'E23'), ('T1-pureMSE-frozen', 'T1')]:
            path = ROOT / directory / cell['scene'] / 'evaluation_mae/twenty-test-draws.json'
            value, record = helper.verify_record(path, dict(test_bank_sha256=cell['bank_sha256'],
                validation_mask_sha256=cell['validation_mask_sha256']), 'mae')
            assert tuple(value['mask_seeds']) == tuple(cell['seeds'])
            assert value['selection_manifest'] == cell['e23_selection' if directory == 'E23' else 't1_selection']
            values[directory] = value
            resource = helper.resource_details(value)
            resource['resource_scope'] = 'original_training_measurements_no_retraining_in_confirmation'
            rows.append(dict(scene=cell['scene'], start_rate=original['start_rate'], block_len=original['block_len'],
                actual_missing_fraction=np.mean([g['missing_evaluation_fraction'] for g in bank['groups']]),
                model=label, mse_mean=value['test_mean']['mse'], mae_mean=value['test_mean']['mae'],
                mse_mask_std_ddof1=value['test_mask_std_ddof1']['mse'], mae_mask_std_ddof1=value['test_mask_std_ddof1']['mae'],
                test_groups=20, windows_per_group=2688, training_seeds=1,
                stage2_seed=2023, stage2_epochs=100, stage2_optimizer_updates=6600,
                forecast_loss='pure_mse', repair_update_policy='frozen_v1', selection_metric='mae', selected_epoch=record['best_epoch'],
                predictor_initial_sha256=record['predictor_initial_sha256'],
                forecast_padding='zero_invalid_v1' if directory == 'E23' else 'replicate_last_value_v1',
                evaluation_manifest=str(path), selection_manifest=value['selection_manifest'], **resource))
            for d in value['test_draws']:
                draws.append(dict(scene=cell['scene'], model=label, group_id=d['group_id'], mask_seed=d['mask_seed'],
                    mse=d['mse'], mae=d['mae'], mask_sha256=d['mask_sha256']))
        a, b = values['E23'], values['T1']
        for x, y in zip(a['test_draws'], b['test_draws']):
            assert all(x[k] == y[k] for k in ['group_id', 'mask_seed', 'mask_sha256', 'windows', 'target_elements'])
            pairs.append(dict(scene=cell['scene'], group_id=x['group_id'], mask_seed=x['mask_seed'],
                e23_mse=x['mse'], t1_mse=y['mse'], e23_minus_t1_mse=x['mse']-y['mse'],
                e23_mae=x['mae'], t1_mae=y['mae'], e23_minus_t1_mae=x['mae']-y['mae']))
        scenes.append(dict(scene=cell['scene'], e23=a['test_mean'], t1=b['test_mean'],
            both_better=all(a['test_mean'][k] < b['test_mean'][k] for k in ['mse', 'mae']),
            mse_group_wins=sum(x['mse'] < y['mse'] for x, y in zip(a['test_draws'], b['test_draws'])),
            mae_group_wins=sum(x['mae'] < y['mae'] for x, y in zip(a['test_draws'], b['test_draws']))))
    helper.csv_write(DOCS / f'{stem}.csv', rows)
    helper.csv_write(DOCS / f'{stem}_all_draws.csv', draws)
    helper.csv_write(DOCS / f'{stem}_paired.csv', pairs)
    publish_report(lock, stem, rows, scenes)


def publish_report(lock, stem, rows, scenes):
    report = [f'# {lock["candidate_label"]} / T1：封存新20组确认', '',
        '新位置在候选训练前封存，未用于开发或checkpoint选择。先锁定四个场景的MAE-selected权重，再逐组测试；不重训、不根据新位置改配置。', '',
        '双方Stage2纯MSE、完整冻结修复器、原预测头和同初始SHA、seed2023、Adam lr1e-4、batch128、100轮/6600更新。T1保留原复制补齐，E23保留零补齐。', '',
        '| 场景 | E23 MSE | T1 MSE | E23 MAE | T1 MAE | MSE / MAE胜出组数 |',
        '|---|---:|---:|---:|---:|---:|']
    for s in scenes:
        a, b = s['e23'], s['t1']
        report.append(f'| {s["scene"]} | {a["mse"]:.6f} | {b["mse"]:.6f} | {a["mae"]:.6f} | {b["mae"]:.6f} | {s["mse_group_wins"]}/20 · {s["mae_group_wins"]}/20 |')
    report += ['', '## 原训练资源', '',
        '| 场景 | 模型 | 两阶段记录成本/分钟 | allocated / reserved GiB |', '|---|---|---:|---:|']
    for r in rows:
        report.append(f'| {r["scene"]} | {r["model"]} | {r["two_stage_recorded_cost_seconds"]/60:.2f} | {r["peak_allocated_gib"]:.3f} / {r["peak_reserved_gib"]:.3f} |')
    all_better = all(s['both_better'] for s in scenes)
    report += ['', f'四场景新位置均值双指标低于T1：**{all_better}**。', '',
        '- 单训练种子；20组标准差反映缺失位置变化，测试时间区间仍为同一ETTh1留出区间。',
        '- 本次确认只做推理。CSV中的时间/显存来自对应原训练记录：E23候选两阶段实测，T1为历史Stage1加本次纯MSE冻结Stage2；同物理RTX5090。',
        '- 相同缺失参数会产生略有不同的实际缺失比例，新组比例单列CSV。',
        f'- 权重、配置、mask、逐窗口误差及SHA锁定文件：{ROOT}', '',
        f'[CSV](./{stem}.csv) · [逐组误差](./{stem}_all_draws.csv) · [配对差值](./{stem}_paired.csv)']
    with (DOCS / f'{stem}.md').open('x') as f:
        f.write('\n'.join(report) + '\n')
    result = dict(status='verified', candidate=lock['candidate_label'], scenarios=scenes,
        all_four_better=all_better, training_seed_count=1, groups_per_scene=20,
        no_checkpoint_selection=True, report=str(DOCS / f'{stem}.md'), csv=str(DOCS / f'{stem}.csv'))
    with (ROOT / 'verification.json').open('x') as f:
        json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
