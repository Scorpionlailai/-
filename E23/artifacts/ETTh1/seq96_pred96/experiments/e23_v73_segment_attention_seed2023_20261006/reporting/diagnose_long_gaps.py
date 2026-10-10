"""Read-only selected Stage1 diagnostics on fixed validation long hidden runs."""
from pathlib import Path
import csv
import hashlib
import json
import os

import numpy as np
import torch
from integrations.e23.config import load_e23_p4_protocol, load_e23_settings
from integrations.e23.train import build_repair_model, load_stage1_selection, _starts, _new_memory, _key_sha256
from integrations.e23.long_gap import long_gap_moments
from integrations.forecast_compare.protocol import iter_p4_batches
from malr.migration import MigrationCatalog

ROOT = Path(__file__).resolve().parents[1]
PREFIX = Path('/home/xyw/paper/E23/docs/experiments/ETTh1_E23_v73_segment_attention_seed2023_20261006')
read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def geometry(observed):
    lengths = np.zeros(observed.shape, dtype=np.int16)
    distances = np.zeros(observed.shape, dtype=np.int16)
    B, T, C = observed.shape
    for b in range(B):
        for c in range(C):
            m = observed[b, :, c]
            visible = np.flatnonzero(m)
            edges = np.flatnonzero(np.diff(np.r_[False, ~m, False].astype(np.int8)))
            for start, stop in zip(edges[::2], edges[1::2]):
                lengths[b, start:stop, c] = stop-start
            distances[b, :, c] = (np.min(np.abs(np.arange(T)[:, None]-visible[None]), axis=1)
                                  if len(visible) else T)
    return lengths, distances


def main():
    assert read(ROOT/'completed-audit.json')['status'] == 'complete'
    plan = read(ROOT/'plan.json')
    scene = plan['scenarios'][0]
    work = Path(plan['work'])
    os.environ['MALR_MISSING_SCENARIO'] = scene['scenario_manifest']
    torch.set_num_threads(4)
    device = torch.device(plan['device'])
    protocol = load_e23_p4_protocol(
        catalog=MigrationCatalog.load(work.parent/'E22/migration/paper_catalog.jsonl', project='E23'),
        protocol_config=Path(plan['protocol_config']),
        mask_artifacts=work.parent/'E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context',
        training_seed=2023,
    )
    assert protocol.mask_sha256['validation'] == scene['validation_mask_sha256']
    own = ROOT/scene['name']/'E23-v73-mae'
    reference = Path(scene['v72_evaluation']).parent.parent
    outputs = []
    for name, cell, config in [('E23-v72', reference, read(reference/'plan.json')['config']),
                                ('E23-v73', own, plan['config'])]:
        selection = cell/'stage1/runs/e23-stage1.json'
        record = read(selection)
        assert sha(record['checkpoint']) == record['checkpoint_sha256']
        selection_sha = sha(selection)
        settings = load_e23_settings(Path(config), protocol=protocol)
        payload = load_stage1_selection(selection, protocol, settings, device=device)
        model = build_repair_model(protocol, settings, device=device)
        model.load_state_dict(payload['model_state'], strict=True)
        model.historical.retrieval.load_state_dict(payload['retrieval_state'])
        memory = _new_memory(protocol)
        memory.load_state_dict(payload['memory_state'])
        model.historical.refresh_snapshot(memory.snapshot())
        before = memory.snapshot_hash(), _key_sha256(model.historical.retrieval.state_dict())
        model.eval()
        totals = {r: [0., 0., 0] for r in ['gap_ge16', 'gap_ge32', 'gap_ge16_last32', 'interior_distance_ge9']}
        level_sum = trend_sum = moment_count = segment_count = 0.
        with torch.inference_mode():
            for cpu in iter_p4_batches(protocol, 'validation', batch_size=128):
                length, distance = geometry(cpu.observed_mask.numpy())
                b = cpu.to(device)
                starts = _starts(b, stride=model.stride, patches=model.patch_count, device=device)
                out = model(b.normalized_observed, b.observed_mask, b.query_mean, b.query_std, starts)
                repaired = out.explicit.normalized_sequence
                assert torch.equal(repaired[b.observed_mask], b.normalized_observed[b.observed_mask])
                error = (repaired-b.normalized_target)*b.query_std[:, None]
                hidden = ~b.observed_mask
                L = torch.from_numpy(length).to(device)
                D = torch.from_numpy(distance).to(device)
                recent = torch.arange(96, device=device)[None, :, None] >= 64
                masks = dict(gap_ge16=hidden & (L >= 16), gap_ge32=hidden & (L >= 32),
                             gap_ge16_last32=hidden & (L >= 16) & recent,
                             interior_distance_ge9=hidden & (D >= 9))
                for region, mask in masks.items():
                    e = error[mask].double()
                    totals[region][0] += e.square().sum().item()
                    totals[region][1] += e.abs().sum().item()
                    totals[region][2] += e.numel()
                level, trend, count, segments = long_gap_moments(repaired, b.normalized_target, hidden, b.query_std)
                n = count.item()
                level_sum += level.item()*n
                trend_sum += trend.item()*n
                moment_count += n
                segment_count += segments.item()
        assert before == (memory.snapshot_hash(), _key_sha256(model.historical.retrieval.state_dict()))
        assert selection_sha == sha(selection) and sha(record['checkpoint']) == record['checkpoint_sha256']
        assert int(moment_count) == totals['gap_ge16'][2]
        row = dict(model=name, scene=scene['name'], split='fixed_validation',
                   metric_space='train_standardized', training_seed=2023,
                   selected_stage1_epoch=record['best_epoch'],
                   gap_min_length=16, gap_level_mse=level_sum/moment_count,
                   gap_trend_amplitude_mse=trend_sum/moment_count,
                   long_gap_segments=int(segment_count), long_gap_coordinates=int(moment_count),
                   validation_mask_sha256=scene['validation_mask_sha256'],
                   source_csv_sha256=protocol.source_sha256,
                   stage1_selection=str(selection), checkpoint_sha256=record['checkpoint_sha256'],
                   bank_state_unchanged=True, no_new_training=True,
                   count_unit='occurrences across overlapping validation lookbacks')
        for region, (squared, absolute, count) in totals.items():
            row.update({region+'_mse': squared/count, region+'_mae': absolute/count,
                        region+'_hidden_coordinates': count})
        outputs.append(row)
        del model, memory, payload, out, b, repaired
        torch.cuda.empty_cache()
    for region in totals:
        assert outputs[0][region+'_hidden_coordinates'] == outputs[1][region+'_hidden_coordinates']
    assert outputs[0]['long_gap_segments'] == outputs[1]['long_gap_segments']
    assert outputs[0]['long_gap_coordinates'] == outputs[1]['long_gap_coordinates']
    raw = ROOT/scene['name']/'long-gap-recovery-diagnostics.json'
    with raw.open('x') as f:
        json.dump(outputs, f, indent=2); f.write('\n')
    csv_path = Path(str(PREFIX)+'_long_gap_diagnostics.csv')
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(outputs[0]))
        writer.writeheader(); writer.writerows(outputs)
    report = PREFIX.with_suffix('.md')
    report_before = sha(report)
    assert report_before == read(ROOT/'report-audit.json')['reports'][str(report)]
    lines = ['', '## 固定验证长缺失段诊断', '',
        '缺失段按原始mask划分，长度在96点窗口处截断；以下计数是重叠回溯中的坐标出现次数。只读已选Stage1权重；Stage2全冻结，修复器相同。T1未追加修复推理，此表仅直接比较v72与v73。', '',
        '| 模型 | ≥16段内修复MSE | ≥32段内 | ≥16段且最近32点 | 距可见点≥9 | 段均值误差 | 段趋势幅度误差 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in outputs:
        lines.append(f"| {r['model']} | {r['gap_ge16_mse']:.6f} | {r['gap_ge32_mse']:.6f} | {r['gap_ge16_last32_mse']:.6f} | {r['interior_distance_ge9_mse']:.6f} | {r['gap_level_mse']:.6f} | {r['gap_trend_amplitude_mse']:.6f} |")
    lines += ['', '段均值与趋势幅度均为重建残差的平方误差，按段长汇总；不把段内线性拟合当作信号真值。修复改善不等于预测改善，本轮仍按固定验证预测指标裁定。', '',
              f'[长段诊断CSV]({csv_path.name})', '']
    report.write_text(report.read_text().split('\n## 固定验证长缺失段诊断\n')[0]+'\n'.join(lines))
    audit = dict(rows=2, raw=str(raw), raw_sha256=sha(raw), csv_sha256=sha(csv_path),
                 report_before_long_gap_sha256=report_before,
                 report_sha256=sha(report), fixed_validation_only=True, same_gap_coordinates=True,
                 no_new_training=True, forecast_targets_used=False)
    (ROOT/'long-gap-diagnostic-audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
