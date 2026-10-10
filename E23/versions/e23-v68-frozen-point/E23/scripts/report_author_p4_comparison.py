"""Write the new author-configured three-test-draw CSV, including unfinished runs."""

import argparse
import csv
import json
from pathlib import Path

from integrations.forecast_compare.author_profiles import ROOT, load_author_profiles


def write_comparison(root: Path, output: Path) -> list[dict]:
    profiles = load_author_profiles()
    names = ['E23_v12', *profiles['models']]
    rows = []
    for name in names:
        cfg = profiles['models'].get(name)
        row = {'model': name, 'dataset': 'ETTh1', 'scenario': 'P4', 'status': 'not_started',
               'profile_version': cfg['version'] if cfg else 'E23 v12 width64; frozen configuration',
               'test_draws': 3, 'training_runs': 1, 'missing_start_rate': 0.10, 'block_length': 8,
               'validation': 'original fixed mask', 'metric_space': 'train_standardized'}
        path = root / name / 'three-test-draws.json'
        progress = root / name / 'progress.json'
        if progress.exists():
            row['status'] = json.loads(progress.read_text())['state']
        if path.exists():
            result = json.loads(path.read_text())
            if result['test_draws'] != 3 or len(result['test_groups']) != 3:
                raise ValueError(f'incomplete test groups: {path}')
            row.update(status='completed', mse_mean=result['test_mean']['mse'],
                       mae_mean=result['test_mean']['mae'], mse_std=result['test_std']['mse'],
                       mae_std=result['test_std']['mae'], std_ddof=0, result_path=str(path.resolve()))
            for index, group in enumerate(result['test_groups']):
                for key in ('mse', 'mae', 'seed', 'missing_evaluation_fraction'):
                    row[f'group_{index}_{key}'] = group[key]
            resources = result['training_resources']
            for key in ('full_workflow_seconds', 'training_stages_seconds', 'peak_allocated_gib',
                        'peak_reserved_gib', 'gpu', 'resource_status'):
                row[key] = resources[key]
            stage_details = []
            for resource_path in resources['resource_record_paths']:
                resource = json.loads(Path(resource_path).read_text())
                row['gpu_uuid'] = resource['device'].get('uuid')
                stage_details.extend({'stage': stage, 'seconds': value['wall_seconds'],
                                      'batch_size': value.get('batch_size', resource['batch_size']),
                                      'precision': value.get('precision', resource['precision'])}
                                     for stage, value in resource['stages'].items())
            row['training_stages'] = json.dumps(stage_details, ensure_ascii=False)
            manifests = resources['selection_manifest_paths']
            selections = [json.loads(Path(p).read_text()) for p in manifests]
            final = selections[-1]
            row['trainable_parameters'] = final.get('trainable_parameters')
            row['epochs_completed'] = ';'.join(str(d['epochs_completed']) for d in selections)
            if final.get('repair_selection_manifest'):
                repair = json.loads(Path(final['repair_selection_manifest']).read_text())
                row['epochs_completed'] = f"{repair['epochs_completed']};{final['epochs_completed']}"
            row['s4_kernel_backend'] = final.get('s4_kernel_backend')
            row['checkpoint_sha256'] = result['checkpoint_sha256']
        rows.append(row)
    fixed = ['model', 'dataset', 'scenario', 'status', 'mse_mean', 'mae_mean', 'mse_std', 'mae_std',
             'std_ddof', 'full_workflow_seconds', 'training_stages_seconds', 'peak_allocated_gib',
             'peak_reserved_gib', 'gpu', 'gpu_uuid', 'trainable_parameters', 'epochs_completed',
             'training_stages', 'profile_version', 'resource_status']
    fields = fixed + sorted({key for row in rows for key in row} - set(fixed))
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + '.tmp')
    with temp.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    temp.replace(output)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/experiments/ETTh1_P4_author_three_test_draws_2026-10-02.csv')
    args = parser.parse_args()
    write_comparison(args.root.resolve(), args.output.resolve())


if __name__ == '__main__':
    main()
