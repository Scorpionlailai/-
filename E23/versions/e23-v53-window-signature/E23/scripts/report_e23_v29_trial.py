"""Lock the validation-only v29 decision, then report measured trial artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

REFERENCE_MSE = 0.7402293330687785
REFERENCE_MAE = 0.5855096656989619


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _metric(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError('validation metric must be finite and nonnegative')
    return float(value)


def choose_v29_candidate(
    records: Sequence[Mapping[str, object]], *,
    reference_mse: float = REFERENCE_MSE, reference_mae: float = REFERENCE_MAE,
) -> Mapping[str, object] | None:
    reference_mse, reference_mae = _metric(reference_mse), _metric(reference_mae)
    passed = []
    for record in records:
        rounds = record.get('recovery_rounds')
        if (record.get('stage') != 'forecast'
                or record.get('recovery_schedule') != 'iterative_tv_h_v1'
                or type(rounds) is not int or rounds not in (1, 3)
                or record.get('epochs_completed') != 100
                or record.get('stage1_epochs_completed') != 100
                or record.get('selection_metric') != 'mae'
                or record.get('loss') != 'forecast_mse_plus_0p2_mae'
                or record.get('forecast_input') != 'explicit_v1'
                or record.get('stage1_objective') != 'global_mse_v1'):
            raise ValueError('incomplete or incompatible v29 training identity')
        if not all(_sha(record.get(k)) for k in
                   ('checkpoint_sha256', 'e23_config_sha256', 'stage1_checkpoint_sha256')):
            raise ValueError('missing checkpoint/config identity')
        mse, mae = _metric(record.get('best_val_mse')), _metric(record.get('best_val_mae'))
        replay = record.get('validation_replay')
        if (not isinstance(replay, Mapping) or replay.get('split') != 'validation'
                or replay.get('windows') != 2688 or replay.get('coordinates') != 1806336
                or replay.get('stage2_checkpoint_sha256') != record['checkpoint_sha256']
                or replay.get('recovery_schedule') != record['recovery_schedule']
                or replay.get('recovery_rounds') != rounds):
            raise ValueError('validation replay identity mismatch')
        for kind in ('key', 'bank'):
            before, after = replay.get(f'h_{kind}_sha256_before'), replay.get(f'h_{kind}_sha256_after')
            if not _sha(before) or before != after:
                raise ValueError('historical state changed or missing during validation')
        if abs(_metric(replay.get('mse')) - mse) > 1e-6 or abs(_metric(replay.get('mae')) - mae) > 1e-6:
            raise ValueError('validation replay metric mismatch')
        if mse <= reference_mse and mae < reference_mae:
            passed.append(record)
    return min(passed, key=lambda r: (r['best_val_mae'], r['best_val_mse'], r['recovery_rounds'])) if passed else None


def load_candidate(root: Path, config: Path) -> dict:
    from integrations.forecast_compare.resources import selected_training_resources
    root, config = root.resolve(), config.resolve()
    s2_path = root / 'stage2/runs/e23-stage2.json'
    record = json.loads(s2_path.read_text())
    s1_path = Path(record['stage1_manifest'])
    s1 = json.loads(s1_path.read_text())
    if s1_path.resolve() != root / 'stage1/runs/e23-stage1.json':
        raise ValueError('trial must use its own fresh Stage1 selection')
    for item in (record, s1):
        if _digest(Path(item['checkpoint'])) != item['checkpoint_sha256'] or _digest(config) != item['e23_config_sha256']:
            raise ValueError('checkpoint or config hash changed')
    if (_digest(s1_path) != record['stage1_manifest_sha256']
            or s1['checkpoint_sha256'] != record['stage1_checkpoint_sha256']
            or s1.get('recovery_schedule') != record.get('recovery_schedule')
            or s1.get('recovery_rounds') != record.get('recovery_rounds')):
        raise ValueError('paired Stage1 identity mismatch')
    record.update(root=str(root), config=str(config), model=f"E23_v29_r{record['recovery_rounds']}",
                  stage1_epochs_completed=s1['epochs_completed'],
                  validation_replay=json.loads((root / 'validation_replay.json').read_text()),
                  training_resources=selected_training_resources([s1_path, s2_path]))
    if record['training_resources']['resource_status'] != 'measured_all_stages':
        raise ValueError('both stages require measured resource sidecars')
    choose_v29_candidate([record])  # Validate even a non-passing candidate.
    return record


def write_selection(records: Sequence[dict], output: Path) -> dict:
    winner = choose_v29_candidate(records)
    decision = {
        'selection_source': 'original fixed validation only; no test metrics consumed',
        'reference_mse': REFERENCE_MSE, 'reference_mae': REFERENCE_MAE,
        'winner': None if winner is None else winner['model'],
        'candidates': list(records),
        'retained_model_if_no_winner': 'E23_v28',
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(decision, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return decision


def validate_round_diagnostics(record: Mapping, diagnostic: Mapping) -> None:
    rounds = record['recovery_rounds']
    if (diagnostic.get('split') != 'original_fixed_validation'
            or diagnostic.get('metric_space') != 'train_standardized'
            or diagnostic.get('diagnostic_only') is not True
            or type(diagnostic.get('recovery_rounds')) is not int
            or diagnostic.get('recovery_rounds') != rounds):
        raise ValueError('round diagnostic validation identity mismatch')
    for stage, sha in (('stage1', record['stage1_checkpoint_sha256']),
                       ('stage2', record['checkpoint_sha256'])):
        result = diagnostic.get(stage)
        if (not isinstance(result, Mapping) or result.get('checkpoint_sha256') != sha
                or result.get('windows') != 2688
                or not isinstance(result.get('rounds'), list)
                or len(result['rounds']) != rounds):
            raise ValueError('round diagnostic checkpoint or round count mismatch')
        for kind in ('key', 'bank'):
            before, after = result.get(f'h_{kind}_sha256_before'), result.get(f'h_{kind}_sha256_after')
            if not _sha(before) or before != after:
                raise ValueError('round diagnostic historical state missing or changed')
        for i, item in enumerate(result['rounds']):
            if item.get('round_index') != i:
                raise ValueError('round diagnostic index mismatch')
            _metric(item.get('global_mse')); _metric(item.get('global_mae'))


def report(decision_path: Path, output_dir: Path, old_comparison: Path) -> None:
    decision = json.loads(decision_path.read_text())
    records = decision['candidates']
    winner = choose_v29_candidate(records)
    if decision['winner'] != (None if winner is None else winner['model']):
        raise ValueError('saved decision does not match locked validation rule')
    # Recheck paired files; changing a candidate after the decision is forbidden.
    for old in records:
        fresh = load_candidate(Path(old['root']), Path(old['config']))
        if fresh != old:
            raise ValueError('candidate changed after validation decision')
    rows = [row.copy() for row in json.loads(old_comparison.read_text())['models']
            if row['model'] in ('E23_v28', 'T1')]
    groups = []
    for row in rows:
        row['status'] = 'existing_saved_result'
        result = json.loads(Path(row['result_path']).read_text())
        for item in result['test_groups']:
            groups.append(dict(model=row['model'], checkpoint_sha256=result['checkpoint_sha256'], **item))
    for record in records:
        resources = record['training_resources']
        stage_resources = [json.loads(Path(p).read_text()) for p in resources['resource_record_paths']]
        diag = json.loads((Path(record['root']) / 'round_diagnostics.json').read_text())
        validate_round_diagnostics(record, diag)
        row = {
            'dataset': 'ETTh1', 'scenario': 'P4', 'input_len': 96, 'pred_len': 96,
            'model': record['model'], 'recovery_rounds': record['recovery_rounds'],
            'recovery_schedule': record['recovery_schedule'], 'training_runs': 1,
            'stage1_epochs_completed': record['stage1_epochs_completed'], 'stage2_epochs_completed': record['epochs_completed'],
            'validation_mse': record['best_val_mse'], 'validation_mae': record['best_val_mae'],
            'checkpoint_sha256': record['checkpoint_sha256'], 'e23_config_sha256': record['e23_config_sha256'],
            'selection_metric': 'fixed_validation_mae', 'stage1_objective': 'global_mse_v1',
            'stage1_validation_repair_mse_global': diag['stage1']['rounds'][-1]['global_mse'],
            'stage1_validation_repair_mae_global': diag['stage1']['rounds'][-1]['global_mae'],
            'stage2_validation_repair_mse_global': diag['stage2']['rounds'][-1]['global_mse'],
            'stage2_validation_repair_mae_global': diag['stage2']['rounds'][-1]['global_mae'],
            'repair_parameters': diag['repair_parameters'],
            'status': 'not_tested_validation_gate_or_selection', 'test_draws': 0,
            'stage1_batch_size': 128, 'stage2_batch_size': 128, 'training_seed': 2021,
            'gpu_uuid': ';'.join(sorted({s['device']['uuid'] for s in stage_resources})),
            'stage1_full_workflow_seconds': stage_resources[0]['full_workflow_seconds'],
            'stage2_full_workflow_seconds': stage_resources[1]['full_workflow_seconds'],
            'stage1_training_seconds': stage_resources[0]['training_stages_seconds'],
            'stage2_training_seconds': stage_resources[1]['training_stages_seconds'],
            'resource_record_paths': ';'.join(resources['resource_record_paths']),
            'result_path': record['root'],
            'notes': 'FP32; GPU1; concurrent conditions in formal_training_environment.json; no extra repair loss in Stage2',
        }
        for key in ('full_workflow_seconds', 'training_stages_seconds', 'peak_allocated_gib', 'peak_reserved_gib', 'gpu'):
            row[key] = resources[key]
        if winner is not None and record['model'] == winner['model']:
            result_path = Path(record['root']) / 'three-test-draws.json'
            result = json.loads(result_path.read_text())
            if (result['checkpoint_sha256'] != record['checkpoint_sha256']
                    or not result['h_state_unchanged'] or result['test_draws'] != 3
                    or [g['seed'] for g in result['test_groups']] != [2021, 2026100201, 2026100202]
                    or any(g['windows'] != 2688 or g['target_elements'] != 1806336 for g in result['test_groups'])):
                raise ValueError('test result identity or locked draw protocol mismatch')
            row.update(status='selected_validation_then_tested', test_draws=3, result_path=str(result_path))
            for metric in ('mse', 'mae'):
                row[f'test_{metric}_mean'] = result['test_mean'][metric]
                row[f'test_{metric}_std'] = result['test_std'][metric]
            for item in result['test_groups']:
                groups.append(dict(model=row['model'], checkpoint_sha256=result['checkpoint_sha256'], **item))
        elif (Path(record['root']) / 'three-test-draws.json').exists():
            raise ValueError('unselected trial unexpectedly tested')
        rows.append(row)
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    table = output_dir / 'ETTh1_P4_E23_v28_v29_T1_comparison_2026-10-03.csv'
    with table.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    with (output_dir / 'ETTh1_P4_E23_v28_v29_T1_test_groups_2026-10-03.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in groups for k in row)))
        writer.writeheader(); writer.writerows(groups)
    lines = ['# E23 v29：T→V→H 共享迭代 P4 实验', '',
             'ETTh1 96→96；修复宽度64、原预测头宽度16；每个新模型独立完成100轮 Stage1 与100轮 Stage2。', '',
             'Stage1：全局 hidden MSE + 0.1 retrieval，按全局验证修复 MSE 选模；Stage2：MSE + 0.2 MAE，按原固定验证 MAE 选模。', '',
             f"验证门槛：MAE < {REFERENCE_MAE} 且同一权重 MSE ≤ {REFERENCE_MSE}。唯一晋级者：{decision['winner'] or '无；保留 v28，未运行新测试'}。", '',
             '| 模型 | 验证 MSE | 验证 MAE | 三组测试 MSE ± std | 三组测试 MAE ± std | 全训练秒 | 显存 allocated/reserved GiB |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        mse = f"{row['test_mse_mean']:.6f} ± {row['test_mse_std']:.6f}" if 'test_mse_mean' in row else '未测试'
        mae = f"{row['test_mae_mean']:.6f} ± {row['test_mae_std']:.6f}" if 'test_mae_mean' in row else '未测试'
        lines.append(f"| {row['model']} | {row['validation_mse']:.6f} | {row['validation_mae']:.6f} | {mse} | {mae} | {row['full_workflow_seconds']:.2f} | {row['peak_allocated_gib']:.3f}/{row['peak_reserved_gib']:.3f} |")
    lines += ['', '## 固定权重逐轮诊断', '',
              '每轮都用同一 decoder 诊断；正式预测仍仅在最终轮解码一次。以下诊断只用原固定验证集，不参与额外选模。', '',
              '| 模型 | 阶段 | 轮次 | 全局修复 MSE | 全局修复 MAE | 平均状态更新范数 |', '|---|---|---:|---:|---:|---:|']
    for record in records:
        diag = json.loads((Path(record['root']) / 'round_diagnostics.json').read_text())
        for stage in ('stage1', 'stage2'):
            for r in diag[stage]['rounds']:
                lines.append(f"| {record['model']} | {stage} | {r['round_index'] + 1} | {r['global_mse']:.6f} | {r['global_mae']:.6f} | {r['mean_update_norm']:.6f} |")
    lines += ['', '## 统计口径与证据', '',
              '- 全训练时间为两个训练入口的实测全流程之和，包含准备、全部训练、验证与保存，排除预检、诊断和最终测试。',
              '- 显存为训练进程 PyTorch allocated/reserved 峰值；stage 时间、GPU UUID、参数数、资源与权重 SHA 在 CSV 和原始 JSON 中。',
              '- std 是同一次训练权重下三组测试缺失位置的总体标准差；没有统计训练随机种子的方差。',
              '- T1/v28 沿用已有记录；T1 Stage1 实际74轮、batch16/BF16，新版 Stage1/2 各100轮、batch128/FP32。硬件并发条件见实验目录记录。',
              '- 三组 DDR 未计算：现有 Missing 参考只对应一组位置。',
              f'- 验证决定：`{decision_path.resolve()}`。', f'- 汇总 CSV：`{table.resolve()}`。', '']
    with (output_dir / 'E23_v29_iterative_TV_H_P4_trial.md').open('x') as stream:
        stream.write('\n'.join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('select', 'report'))
    parser.add_argument('--root-one', type=Path)
    parser.add_argument('--root-three', type=Path)
    parser.add_argument('--config-one', type=Path)
    parser.add_argument('--config-three', type=Path)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('docs/experiments'))
    parser.add_argument('--old-comparison', type=Path, default=Path('artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/e23_v28_global_stage1/comparison.json'))
    args = parser.parse_args()
    if args.mode == 'select':
        if any(getattr(args, k) is None for k in ('root_one', 'root_three', 'config_one', 'config_three')):
            parser.error('select requires both trial roots and configs')
        records = [load_candidate(args.root_one, args.config_one), load_candidate(args.root_three, args.config_three)]
        decision = write_selection(records, args.selection)
        print(json.dumps({'winner': decision['winner'], 'selection': str(args.selection)}))
    else:
        report(args.selection, args.output_dir, args.old_comparison)


if __name__ == '__main__':
    main()
