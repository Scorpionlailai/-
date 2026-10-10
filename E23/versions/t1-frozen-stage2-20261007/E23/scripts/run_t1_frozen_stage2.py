"""One frozen T1 Stage2 run with the recorded E23-v69 optimization recipe."""
from time import perf_counter
ENTRY_STARTED = perf_counter()

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import random
import hashlib
import subprocess

import numpy as np
import torch

from integrations.e23.config import load_e23_p4_protocol, load_e23_settings
from integrations.e23.train import build_forecast_model, stage2_objective
from integrations.e23.forecast_sensitive import parameter_hash
from integrations.forecast_compare.author_profiles import build_author_model
from integrations.forecast_compare.protocol import iter_p4_batches
from integrations.forecast_compare.resources import TrainingResources, selected_training_resources
from integrations.forecast_compare.t1_frozen import FrozenT1ForecastModel
from integrations.forecast_compare.test_mask_bank_v2 import load_test_bank, TEST_SEEDS
from malr.evaluation.report import file_sha256
from malr.migration import MigrationCatalog


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def setup(plan, scene):
    os.environ.pop('MALR_MISSING_SCENARIO', None)
    if scene['scenario_manifest']:
        os.environ['MALR_MISSING_SCENARIO'] = scene['scenario_manifest']
    protocol = load_e23_p4_protocol(
        catalog=MigrationCatalog.load('/home/xyw/paper/E22/migration/paper_catalog.jsonl', project='E23'),
        protocol_config=Path(plan['protocol_config']), mask_artifacts=Path(plan['mask_artifacts']),
        training_seed=plan['stage2_training_seed'])
    assert protocol.config.missing.start_rate == scene['start_rate']
    assert protocol.config.missing.block_len == scene['block_len']
    assert protocol.mask_sha256['validation'] == scene['validation_mask_sha256']
    assert file_sha256(Path(scene['test_bank'])) == scene['test_bank_sha256']
    views = load_test_bank(protocol, Path(scene['test_bank']))
    assert len(views) == 20
    parent_path = Path(scene['t1_repair_selection'])
    assert file_sha256(parent_path) == scene['t1_repair_selection_sha256']
    parent = read(parent_path)
    assert parent['stage'] == 'repair'
    assert parent.get('training_seed', parent['seed']) == plan['stage1_training_seed']
    assert parent['source_csv_sha256'] == protocol.source_sha256
    assert parent['validation_mask_sha256'] == protocol.mask_sha256['validation']
    checkpoint = Path(parent['checkpoint'])
    assert file_sha256(checkpoint) == parent['checkpoint_sha256']
    official_root = Path('/home/xyw/paper/E23') / parent['profile']['root']
    git = '/home/xyw/miniforge3/envs/malr/bin/git'
    revision = subprocess.check_output([git, '-C', str(official_root), 'rev-parse', 'HEAD'], text=True).strip()
    assert revision == parent['upstream_revision']
    tracked = subprocess.check_output([git, '-C', str(official_root), 'ls-files', '-z']).decode().split('\0')
    digest = hashlib.sha256()
    for name in sorted(tracked):
        path = official_root/name
        if path.suffix in ('.py', '.yaml', '.yml', '.sh', '.ini') and path.is_file():
            digest.update(name.encode())
            digest.update(path.read_bytes())
    assert digest.hexdigest() == parent['upstream_source_sha256']
    device = torch.device(plan['device'])
    imputer = build_author_model('T1', parent['profile'], official_root, device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    imputer.load_state_dict(payload['model_state'], strict=True)
    settings = load_e23_settings(Path(plan['e23_config']), protocol=protocol)
    assert (settings.seed, settings.batch_size, settings.stage2_lr, settings.weight_decay) == (
        plan['stage2_training_seed'], plan['batch_size'], plan['learning_rate'], plan['weight_decay'])
    reference = read(scene['e23_selection'])
    assert reference['loss'] == 'forecast_mse_plus_0p2_mae'
    assert reference['repair_update_policy'] == 'frozen_v1'
    assert reference['epochs_completed'] == plan['epochs']
    assert reference['validation_mask_sha256'] == protocol.mask_sha256['validation']
    # Reproduce the exact head and RNG constructor sequence of recorded E23-v69.
    # The temporary E23 repairer is never used in T1 repair or forecasting.
    random.seed(settings.seed)
    np.random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    reference_model = build_forecast_model(protocol, settings, device=device)
    predictor = reference_model.predictor
    assert parameter_hash(predictor) == plan['predictor_initial_sha256'] == reference['predictor_initial_sha256']
    assert sum(p.numel() for p in predictor.parameters()) == plan['predictor_parameters']
    del reference_model
    model = FrozenT1ForecastModel(imputer, predictor).to(device)
    return protocol, views, parent, model


def score(model, protocol, split, device, *, per_window=False):
    model.eval()
    squared = absolute = count = windows = 0
    window_mse, window_mae, starts = [], [], []
    refs = dict(protocol.refs)
    refs[split] = refs[split][:len(refs[split])//128*128]
    with torch.no_grad():
        for cpu_batch in iter_p4_batches(replace(protocol, refs=refs), split, batch_size=128):
            batch = cpu_batch.to(device)
            error = model(batch)-batch.forecast_target
            if not torch.isfinite(error).all():
                raise FloatingPointError('nonfinite evaluation')
            squared += error.square().sum().item()
            absolute += error.abs().sum().item()
            count += error.numel()
            windows += len(batch.refs)
            if per_window:
                window_mse.extend(error.double().square().mean((1, 2)).cpu().tolist())
                window_mae.extend(error.double().abs().mean((1, 2)).cpu().tolist())
                starts.extend(ref.target_start for ref in batch.refs)
    assert (windows, count) == (2688, 2688*7*96)
    result = dict(mse=squared/count, mae=absolute/count, windows=windows, target_elements=count)
    return (result, (window_mse, window_mae, starts)) if per_window else result


def assert_frozen(model, initial):
    assert all(not module.training for module in model.imputer.modules())
    assert all(not p.requires_grad and p.grad is None for p in model.imputer.parameters())
    assert parameter_hash(model.imputer) == initial


def preflight(plan, scene):
    protocol, _, parent, model = setup(plan, scene)
    initial = parameter_hash(model.imputer)
    optimizer = torch.optim.Adam(model.predictor.parameters(), lr=plan['learning_rate'], weight_decay=0)
    batch = next(iter_p4_batches(protocol, 'train', batch_size=128, epoch=0)).to(plan['device'])
    predictor_before = parameter_hash(model.predictor)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = stage2_objective(model(batch), batch.forecast_target, loss_mode=plan['forecast_loss'])
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.predictor.parameters() if p.grad is not None)
    optimizer.step()
    assert predictor_before != parameter_hash(model.predictor)
    assert_frozen(model, initial)
    # A target-only edit must have no effect on deterministic model input/output.
    model.eval()
    before = model(batch)
    tampered = replace(batch, normalized_target=torch.full_like(batch.normalized_target, 999),
                       forecast_target=torch.full_like(batch.forecast_target, -999))
    assert torch.equal(before, model(tampered))
    result = dict(status='passed', scene=scene['name'], frozen_parameters_and_buffers=True,
                  predictor_parameters=35427, predictor_initial_sha256=plan['predictor_initial_sha256'],
                  stage1_checkpoint_sha256=parent['checkpoint_sha256'],
                  frozen_imputer_sha256=initial, loss=float(loss.detach()),
                  forecast_loss=plan['forecast_loss'], test_scoring=False)
    write(Path(plan['root'])/scene['name']/'preflight.json', result)
    print(json.dumps(result), flush=True)


def cached_stage1(parent_path, output):
    parent_path = Path(parent_path)
    parent = read(parent_path)
    matches = []
    for path in (parent_path.parent/'resources').glob('*.json'):
        row = read(path)
        if row.get('status') == 'completed' and 'repair' in row.get('stages', {}):
            matches.append((path, row))
    assert len(matches) == 1
    path, original = matches[0]
    selected = read(original['selection_manifest'])
    assert selected.get('repair_selection_sha256') == file_sha256(parent_path) or (
        Path(original['selection_manifest']).resolve() == parent_path.resolve())
    stage = dict(original['stages']['repair'])
    stage.update(batch_size=parent['training']['batch_size'], precision=parent['training']['precision'])
    overhead = original['preparation_and_reporting_seconds']
    result = {**original, 'stages': {'repair': stage}, 'cached_stage1': True,
              'source_resource_record': str(path.resolve()), 'source_resource_record_sha256': file_sha256(path),
              'selection_manifest': str(parent_path.resolve()), 'selection_manifest_sha256': file_sha256(parent_path),
              'checkpoint_sha256': parent['checkpoint_sha256'],
              'epochs_completed': parent['epochs_completed'],
              'full_workflow_seconds': stage['wall_seconds']+overhead,
              'training_stages_seconds': stage['wall_seconds'],
              'legacy_training_seconds': parent['training_seconds'],
              'peak_allocated_bytes': stage['peak_allocated_bytes'],
              'peak_reserved_bytes': stage['peak_reserved_bytes'],
              'batch_size': parent['training']['batch_size'], 'precision': parent['training']['precision'],
              'timing_scope': 'cached original repair stage + all original entrypoint non-stage overhead; excludes old Stage2; recomposed cost, not a fresh end-to-end run'}
    write(output, result)


def train(plan, scene):
    output = Path(plan['root'])/scene['name']/'training'
    output.mkdir(parents=True, exist_ok=False)
    with TrainingResources(output, device=plan['device'], batch_size=128, model='T1-frozen-E23-aligned',
                           started_at=ENTRY_STARTED, precision='float32', amp=False) as resources:
        protocol, views, parent, model = setup(plan, scene)
        frozen_sha = parameter_hash(model.imputer)
        torch.cuda.synchronize(plan['device'])
        torch.cuda.empty_cache()
        cached_path = output/'cached-stage1-resources.json'
        cached_stage1(scene['t1_repair_selection'], cached_path)
        optimizer = torch.optim.Adam(model.predictor.parameters(), lr=plan['learning_rate'],
                                     weight_decay=plan['weight_decay'])
        assert sum(p.numel() for group in optimizer.param_groups for p in group['params']) == 35427
        best = {'mae': float('inf'), 'mse': float('inf')}
        selections = {}
        with resources.stage('forecast'):
            started = perf_counter()
            history_path = output/'epochs.jsonl'
            for epoch in range(1, plan['epochs']+1):
                model.train()
                batches = windows = 0
                summed_loss = 0.
                for cpu_batch in iter_p4_batches(protocol, 'train', batch_size=128, epoch=epoch-1):
                    batch = cpu_batch.to(plan['device'])
                    optimizer.zero_grad(set_to_none=True)
                    prediction = model(batch)
                    loss = stage2_objective(prediction, batch.forecast_target, loss_mode=plan['forecast_loss'])
                    if not torch.isfinite(loss):
                        raise FloatingPointError('nonfinite training loss')
                    loss.backward()
                    optimizer.step()
                    summed_loss += float(loss.detach())
                    batches += 1
                    windows += len(batch.refs)
                assert (batches, windows) == (66, 8448)
                validation = score(model, protocol, 'validation', plan['device'])
                assert_frozen(model, frozen_sha)
                for metric in ['mae', 'mse']:
                    if validation[metric] < best[metric]:
                        best[metric] = validation[metric]
                        path = output/f'forecast-{metric}-selected.pt'
                        torch.save(dict(model_state=model.state_dict(), epoch=epoch, validation=validation,
                                        frozen_imputer_sha256=frozen_sha, selection_metric=metric), path)
                        selections[metric] = dict(best_epoch=epoch, best_validation=validation,
                                                   checkpoint=str(path.resolve()))
                row = dict(scene=scene['name'], epoch=epoch, train_forecast_loss=summed_loss/batches,
                           validation=validation, best_epochs={k: v['best_epoch'] for k, v in selections.items()},
                           frozen_imputer_sha256=frozen_sha, elapsed_seconds=perf_counter()-started)
                with history_path.open('a') as stream:
                    stream.write(json.dumps(row)+'\n')
                print(json.dumps(row), flush=True)
            training_seconds = perf_counter()-started
            assert file_sha256(Path(scene['t1_repair_selection'])) == scene['t1_repair_selection_sha256']
            for metric, chosen in selections.items():
                record = dict(model='T1-frozen-E23-aligned', stage='forecast', stage1_reused=True,
                    stage1_training_seed=502, seed=2023, training_seed=2023,
                    profile=parent['profile'], profile_version=parent['profile_version'],
                    repair_selection_manifest=scene['t1_repair_selection'],
                    repair_selection_sha256=scene['t1_repair_selection_sha256'],
                    stage1_checkpoint_sha256=parent['checkpoint_sha256'],
                    cached_stage1_resource_record=str(cached_path.resolve()),
                    cached_stage1_resource_sha256=file_sha256(cached_path),
                    repair_update_policy='frozen_v1', repair_trainable_parameters=0,
                    repair_training_mode='eval', frozen_imputer_sha256=frozen_sha,
                    frozen_state_verified_epochs=100, trainable_parameters=35427,
                    predictor_parameters=35427, predictor_initial_sha256=plan['predictor_initial_sha256'],
                    predictor_final_sha256=parameter_hash(model.predictor),
                    epochs_completed=100, training_windows=844800, optimizer_steps=6600,
                    training=dict(batch_size=128, epochs=100, learning_rate=1e-4, weight_decay=0.,
                                  loss='forecast_mse_plus_0p2_mae', lradj='constant', precision='float32',
                                  selection_metric=metric, patience=None),
                    loss='forecast_mse_plus_0p2_mae', selection_metric=metric,
                    forecast_padding=plan['forecast_padding'],
                    repair_retention='omitted_frozen_constant_no_predictor_gradient',
                    source_csv_sha256=protocol.source_sha256,
                    validation_mask_sha256=protocol.mask_sha256['validation'],
                    original_test_mask_sha256=protocol.mask_sha256['test'],
                    test_mask_bank_sha256=scene['test_bank_sha256'],
                    checkpoint_sha256=file_sha256(Path(chosen['checkpoint'])),
                    training_seconds=training_seconds, status='validation_finalized',
                    test_metrics_seen=False, runner_sha256=file_sha256(Path(__file__)),
                    e23_reference_selection=scene['e23_selection'],
                    e23_reference_selection_sha256=file_sha256(Path(scene['e23_selection'])), **chosen)
                if metric == 'mse':
                    record.update(training_resource_parent_manifest=str((output/'forecast-mae-selection.json').resolve()),
                                  training_resource_parent_sha256=file_sha256(output/'forecast-mae-selection.json'))
                write(output/f'forecast-{metric}-selection.json', record)
        resources.selection_manifest = output/'forecast-mae-selection.json'
    for metric in ['mae', 'mse']:
        evaluate(plan, scene, protocol, views, model, output/f'forecast-{metric}-selection.json', metric)


def evaluate(plan, scene, protocol, views, model, selection_path, metric):
    output = Path(plan['root'])/scene['name']/f'evaluation_{metric}'
    record = read(selection_path)
    checkpoint = Path(record['checkpoint'])
    selection_sha = file_sha256(selection_path)
    assert file_sha256(checkpoint) == record['checkpoint_sha256']
    payload = torch.load(checkpoint, map_location=plan['device'], weights_only=False)
    model.load_state_dict(payload['model_state'], strict=True)
    assert_frozen(model, record['frozen_imputer_sha256'])
    validation = score(model, protocol, 'validation', plan['device'])
    assert all(abs(validation[k]-record['best_validation'][k]) <= 1e-6 for k in ['mse', 'mae'])
    lock = dict(model='T1-frozen-E23-aligned', training_seed=2023, stage1_training_seed=502,
                checkpoint_selection_metric=metric, selection_manifest=str(selection_path.resolve()),
                selection_manifest_sha256=selection_sha, checkpoint_sha256=record['checkpoint_sha256'],
                test_mask_bank_sha256=scene['test_bank_sha256'],
                validation_mask_sha256=protocol.mask_sha256['validation'], validation_replay=validation,
                predictor_parameters=35427, predictor_initial_sha256=plan['predictor_initial_sha256'],
                repair_update_policy='frozen_v1', forecast_loss=plan['forecast_loss'],
                frozen_imputer_sha256=record['frozen_imputer_sha256'], test_metrics_seen=False)
    write(output/'validation-lock.json', lock)
    entries = read(scene['test_bank'])['groups']
    results = []
    for entry, view in zip(entries, views):
        metrics, arrays = score(model, view, 'test', plan['device'], per_window=True)
        path = output/f"{entry['group_id']}-windows.npz"
        np.savez_compressed(path, mse=arrays[0], mae=arrays[1], target_start=arrays[2])
        row = dict(group_id=entry['group_id'], mask_seed=entry['seed'], mask_sha256=entry['sha256'],
                   window_metrics=str(path.resolve()), window_metrics_sha256=file_sha256(path), **metrics)
        results.append(row)
        print(json.dumps(dict(event='test_draw', scene=scene['name'], selection=metric, **row)), flush=True)
    assert file_sha256(selection_path) == selection_sha
    assert file_sha256(Path(scene['test_bank'])) == scene['test_bank_sha256']
    load_test_bank(protocol, Path(scene['test_bank']))
    assert_frozen(model, record['frozen_imputer_sha256'])
    result = {**lock, 'test_metrics_seen': True, 'mask_seeds': list(TEST_SEEDS),
              'metric_space': 'train_standardized', 'split': 'test', 'test_draws': results,
              'test_mean': {k: float(np.mean([r[k] for r in results])) for k in ['mse', 'mae']},
              'test_mask_std_ddof1': {k: float(np.std([r[k] for r in results], ddof=1)) for k in ['mse', 'mae']},
              'training_resources': selected_training_resources([selection_path])}
    write(output/'twenty-test-draws.json', result)
    print(json.dumps(dict(event='complete', scene=scene['name'], selection=metric,
                          test_mean=result['test_mean'])), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--scene', required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    plan = read(args.plan)
    scene = next(s for s in plan['scenarios'] if s['name'] == args.scene)
    torch.set_num_threads(plan['cpu_threads'])
    if args.preflight:
        preflight(plan, scene)
    else:
        assert read(Path(plan['root'])/scene['name']/'preflight.json')['status'] == 'passed'
        train(plan, scene)


if __name__ == '__main__':
    main()
