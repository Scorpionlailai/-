"""Write seed-level and crossed-design reports, including measured resources."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

from integrations.forecast_compare.replicate_statistics import TRAINING_SEEDS, summarize_matrix, crossed_paired_ci
from integrations.forecast_compare.test_mask_bank_v2 import TEST_SEEDS
from malr.evaluation.report import file_sha256


def write_csv(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def summarize(root):
    root = Path(root)
    models = ['E23-v31-B', 'T1']
    records = {m: {} for m in models}
    rows, draw_rows = [], []
    physical_devices = set()
    bank_hash = file_sha256(root/'test_masks/manifest.json')
    for seed in TRAINING_SEEDS:
        for model in models:
            path = root/f'seed_{seed}'/model/'evaluation/twenty-test-draws.json'
            if not path.exists():
                continue
            result = json.loads(path.read_text())
            if (result['model'], result['training_seed'], tuple(result['mask_seeds'])) != (model, seed, TEST_SEEDS):
                raise ValueError('replicate report identity mismatch')
            if result['test_mask_bank_sha256'] != bank_hash or tuple(r['mask_seed'] for r in result['test_draws']) != TEST_SEEDS:
                raise ValueError('replicate test mask bank/order changed')
            if file_sha256(Path(result['selection_manifest'])) != result['selection_manifest_sha256']:
                raise ValueError('selected manifest changed after test')
            resources = result['training_resources']
            if resources['resource_status'] != 'measured_all_stages':
                raise ValueError('complete measured training resources required')
            records[model][seed] = result
            stage_seconds = {'repair': 0., 'forecast': 0.}
            repair_epochs = forecast_epochs = 0
            gpu_identity = set()
            for sidecar in resources['resource_record_paths']:
                r = json.loads(Path(sidecar).read_text())
                gpu_identity.add(json.dumps(r['device'], sort_keys=True))
                physical_devices.add(r['device'].get('uuid', 'unavailable'))
                for name, stage in r['stages'].items():
                    stage_seconds[name] = stage_seconds.get(name, 0.) + stage['wall_seconds']
            selection = json.loads(Path(result['selection_manifest']).read_text())
            if model == 'T1':
                repair = json.loads(Path(selection['repair_selection_manifest']).read_text())
            else:
                repair = json.loads(Path(selection['stage1_manifest']).read_text())
            rows.append(dict(model=model, training_seed=seed, test_masks=len(TEST_SEEDS),
                             mse=result['test_mean']['mse'], mae=result['test_mean']['mae'],
                             mask_mse_std=result['test_mask_std_ddof1']['mse'],
                             mask_mae_std=result['test_mask_std_ddof1']['mae'],
                             full_training_seconds=resources['full_workflow_seconds'],
                             repair_stage_seconds=stage_seconds['repair'], forecast_stage_seconds=stage_seconds['forecast'],
                             peak_allocated_gib=resources['peak_allocated_gib'],
                             peak_reserved_gib=resources['peak_reserved_gib'], gpu=resources['gpu'],
                             repair_epochs=repair['epochs_completed'], forecast_epochs=selection['epochs_completed'],
                             repair_batch_size=128 if model == 'E23-v31-B' else 16,
                             repair_precision='float32' if model == 'E23-v31-B' else 'bfloat16',
                             forecast_batch_size=128, forecast_precision='float32',
                             validation_mse=result['validation_replay']['mse'],
                             validation_mae=result['validation_replay']['mae'],
                             checkpoint_sha256=result['checkpoint_sha256'], result=str(path.resolve())))
            for draw in result['test_draws']:
                draw_rows.append(dict(model=model, training_seed=seed, **{k: draw[k] for k in ['mask_seed', 'mask_sha256', 'mse', 'mae', 'windows', 'target_elements']}))
        if all(seed in records[m] for m in models) and records[models[0]][seed]['predictor_initial_sha256'] != records[models[1]][seed]['predictor_initial_sha256']:
            raise ValueError('paired methods did not start from identical predictor initialization')
    if len(physical_devices) > 1 or (physical_devices and 'unavailable' in physical_devices):
        raise ValueError('formal training used different/unidentified physical GPUs')
    if rows:
        write_csv(root/'per_training_seed.csv', rows)
        write_csv(root/'per_training_seed_and_mask.csv', draw_rows)
    complete_seeds = [s for s in TRAINING_SEEDS if all(s in records[m] for m in models)]
    status = dict(status='complete' if len(complete_seeds) == 5 else 'partial',
                  paired_completed_seeds=complete_seeds, expected_training_seeds=list(TRAINING_SEEDS),
                  test_mask_seeds=list(TEST_SEEDS), completed_runs=len(rows))
    if len(complete_seeds) >= 2:
        matrices = {m: {k: np.array([[r[k] for r in records[m][s]['test_draws']] for s in complete_seeds])
                       for k in ['mse', 'mae']} for m in models}
        status['models'] = {m: {k: summarize_matrix(matrices[m][k]) for k in ['mse', 'mae']} for m in models}
        status['paired_difference'] = {k: crossed_paired_ci(matrices[models[0]][k], matrices[models[1]][k]) for k in ['mse', 'mae']}
        summary = []
        for model in models:
            resource_rows = [r for r in rows if r['model'] == model and r['training_seed'] in complete_seeds]
            entry = dict(model=model, independent_training_runs=len(complete_seeds), test_masks=20)
            for metric in ['mse', 'mae']:
                for key in ['mean', 'training_seed_std_ddof1', 'test_mask_std_ddof1']:
                    entry[f'{metric}_{key}'] = status['models'][model][metric][key]
            for key in ['full_training_seconds', 'repair_stage_seconds', 'forecast_stage_seconds', 'peak_allocated_gib', 'peak_reserved_gib']:
                vals = np.array([r[key] for r in resource_rows])
                entry[key+'_mean'] = float(vals.mean())
                entry[key+'_std_ddof1'] = float(vals.std(ddof=1))
                entry[key+'_max'] = float(vals.max())
            entry['total_training_seconds'] = sum(r['full_training_seconds'] for r in resource_rows)
            summary.append(entry)
        write_csv(root/'comparison_summary.csv', summary)
    (root/'comparison_statistics.json').write_text(json.dumps(status, indent=2, sort_keys=True)+'\n')
    return status


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(summarize(args.root), indent=2))
