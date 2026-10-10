"""Add recorded training resources to the existing P4 CSV without retraining."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from integrations.forecast_compare.resources import selected_training_resources

ROOT = Path(__file__).resolve().parent.parent
TABLE = ROOT / 'docs/experiments/ETTh1_P4_extended_baselines_2026-10-02.csv'
GEOMETRY = 'artifacts/ETTh1/seq96_pred96/sr010_bl8'
V12 = ROOT / GEOMETRY / 'experiments/e23_v3_window_v12'


def saved_manifest_map() -> dict[str, list[Path]]:
    mapping = {
        'E23_v12': [V12 / 'stage1/runs/e23-stage1.json', V12 / 'stage2/runs/e23-stage2.json'],
        'E22': [ROOT.parent / 'E22' / GEOMETRY / 'experiments/two_stage_v5_dynamic_temporal_context/runs' / name
                for name in ('ETTh1-ours-8dda1f69aa67.json', 'ETTh1-ours-dea387986927.json')],
        'SAITS': [ROOT.parent / 'SAITS' / GEOMETRY / 'saits' / stage / 'runs' / name
                  for stage, name in [('stage1_p4_v2', 'saits-stage1.json'),
                                      ('stage2_p4_v2', 'saits-stage2.json')]],
        'S4M': [ROOT.parent / 'S4M' / GEOMETRY / 'official_p4_v2/runs/S4M-ETTh1-missing-el2-lr0p005-do0p1-adam.json'],
    }
    for key, directory, stem in [('CRIB', 'crib', 'crib'), ('CoIFNet', 'coifnet', 'coifnet'),
                                 ('BiTGraph', 'bitgraph', 'bitgraph'),
                                 ('ChannelTokenFormer', 'ctf', 'ctf')]:
        mapping[key] = [ROOT / f'artifacts/{directory}_etth1_p4/{stem}-selection.json']
    for key, stem in [('T1+shared_PatchTST', 't1'), ('BRITS+shared_PatchTST', 'brits')]:
        mapping[key] = [ROOT / f'artifacts/{stem}_etth1_p4/{stem}-stage{stage}-selection.json'
                        for stage in (1, 2)]
    return mapping


def update_table(table: Path, output: Path, manifest_map: dict[str, list[Path]]) -> list[dict]:
    with table.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames)
        rows = list(reader)
    extra_fields = ['metric_space', 'missing_start_rate', 'missing_block_length',
                    'actual_missing_fraction', 'legacy_training_loop_seconds',
                    'stage1_legacy_loop_seconds', 'stage2_legacy_loop_seconds',
                    'full_workflow_seconds', 'full_workflow_minutes', 'training_stages_seconds',
                    'peak_allocated_gib', 'peak_reserved_gib', 'gpu', 'batch_size',
                    'hardware_consistent', 'resource_status', 'selection_manifest_paths',
                    'resource_record_paths', 'resource_notes']
    for row in rows:
        resources = selected_training_resources(manifest_map[row['model']])
        row.update(resources)
        row.update(metric_space='train_standardized', missing_start_rate=0.10,
                   missing_block_length=8, actual_missing_fraction=0.5667107337726758)
        seconds = resources['full_workflow_seconds']
        row['full_workflow_minutes'] = None if seconds is None else seconds / 60
        if resources['batch_size'] is None:
            row['batch_size'] = 16 if row['model'] == 'S4M' else 128
        row['selection_manifest_paths'] = json.dumps(resources['selection_manifest_paths'])
        row['resource_record_paths'] = json.dumps(resources['resource_record_paths'])
        row['resource_notes'] = (
            'Legacy timer covers training loops and validation; full command time and training '
            'memory were not recorded. GPU model unknown; do not rank legacy speed. '
            'S4M batch=16; other rows batch=128.' if seconds is None else
            'All stages measured; time summed and memory maximum taken across stages. '
            'Final test and hyperparameter search excluded.'
        )
        if row['model'] == 'ChannelTokenFormer':
            row['resource_notes'] += (' Adapter splits the effective batch into visibility groups; '
                                      'forward microbatches can be smaller than 128. '
                                      'Time includes this mask-correctness adaptation overhead.')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields + [f for f in extra_fields if f not in fields])
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--table', type=Path, default=TABLE)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--manifest-map', type=Path,
                        help='JSON mapping CSV model names to all training-stage manifests')
    args = parser.parse_args()
    mapping = saved_manifest_map() if args.manifest_map is None else {
        name: [Path(path) for path in paths]
        for name, paths in json.loads(args.manifest_map.read_text()).items()
    }
    rows = update_table(args.table, args.output or args.table, mapping)
    print(json.dumps({'table': str(args.output or args.table), 'rows': len(rows),
                      'resources_measured': sum(r['resource_status'] == 'measured_all_stages' for r in rows)}))


if __name__ == '__main__':
    main()
