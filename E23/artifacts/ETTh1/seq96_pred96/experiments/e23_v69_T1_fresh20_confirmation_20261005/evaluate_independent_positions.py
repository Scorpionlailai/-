"""Replay fixed validation, then test one locked replicate on 20 frozen masks."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from integrations.e23.config import load_e23_p4_protocol, load_e23_settings
from integrations.e23.train import (
    _new_memory, _key_sha256, _starts, _validation_forecast_metrics,
    build_forecast_model, load_stage2_selection,
)
from integrations.forecast_compare.author_profiles import build_author_model, load_author_profiles
from integrations.forecast_compare.author_provenance import author_source_identity, write_new_record
from integrations.forecast_compare.protocol import iter_p4_batches
from integrations.forecast_compare.resources import selected_training_resources
from integrations.forecast_compare.t1_p4 import T1ForecastModel
from integrations.forecast_compare.test_mask_bank_v2 import load_test_bank
from malr.evaluation.report import file_sha256
from malr.migration import MigrationCatalog
from scripts.run_author_aligned_p4 import _forecast_head, score as author_score

import os
ROOT = Path(os.environ["MALR_E23_CONFIRM_WORK"])
TEST_SEEDS = (2026100501, 2026100502, 2026100503, 2026100504, 2026100505, 2026100506, 2026100507, 2026100508, 2026100509, 2026100510, 2026100511, 2026100512, 2026100513, 2026100514, 2026100515, 2026100516, 2026100517, 2026100518, 2026100519, 2026100520)
MASKS = ROOT.parent/'E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=['E23-v31-B', 'T1'], required=True)
    p.add_argument('--training-seed', type=int, required=True)
    p.add_argument('--model-label', help='Exploratory E23 variant label; does not change model loading')
    p.add_argument('--e23-config', type=Path)
    p.add_argument('--profiles', type=Path)
    p.add_argument('--selection', type=Path, required=True)
    p.add_argument('--test-mask-bank', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:1')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(f'refusing duplicate test evaluation: {args.output}')
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    protocol = load_e23_p4_protocol(
        catalog=MigrationCatalog.load(ROOT.parent/'E22/migration/paper_catalog.jsonl', project='E23'),
        protocol_config=ROOT/'configs/stage1_seq96_dynamic_temporal_repair.yaml',
        mask_artifacts=MASKS, training_seed=args.training_seed)
    views = load_test_bank(protocol, args.test_mask_bank, expected_seeds=TEST_SEEDS)
    entries = json.loads(args.test_mask_bank.read_text())['groups']
    record = json.loads(args.selection.read_text())
    selection_hash, bank_hash = file_sha256(args.selection), file_sha256(args.test_mask_bank)
    if record['seed'] != args.training_seed or record.get('selection_metric', record.get('training', {}).get('selection_metric')) not in ('mae','mse'):
        raise ValueError('replicate seed/selection metric mismatch')
    if args.model == 'E23-v31-B':
        settings = load_e23_settings(args.e23_config, protocol=protocol)
        common = record.get('downstream_protocol') == 't1_mse_v1'
        if common:
            if (record['loss'], record.get('repair_retention','none'), record.get('forecast_padding'), settings.recovery_rounds) != (
                    'forecast_mse_only','none','replicate_last_value_v1',1):
                raise ValueError('expected common pure-MSE downstream training')
        elif (record.get('repair_retention'), record['loss'], settings.recovery_rounds,
            record.get('forecast_mixer'), record.get('forecast_input')) != (
                'uniform_v1', 'forecast_mse_plus_0p2_mae', 1, 'none', 'explicit_v1') and not (
                record.get('repair_update_policy') in ('freeze_front_v1','frozen_v1') and
                (record.get('repair_retention','none'),record['loss'],settings.recovery_rounds,
                 record.get('forecast_mixer'),record.get('forecast_input')) ==
                ('none','forecast_mse_only',1,'none','explicit_v1')):
            raise ValueError('expected original or pure-MSE/frozen-front training/model identity')
        payload = load_stage2_selection(args.selection, protocol, settings, device=device)
        model = build_forecast_model(protocol, settings, device=device)
        if common or record.get('forecast_padding_alignment_only', False):
            model.forecast_padding = 'replicate_last_value_v1'
        model.load_state_dict(payload['model_state'], strict=True)
        model.repair.historical.retrieval.load_state_dict(payload['retrieval_state'])
        memory = _new_memory(protocol)
        memory.load_state_dict(payload['memory_state'])
        model.repair.historical.refresh_snapshot(memory.snapshot())
        historical_identity = memory.snapshot_hash(), _key_sha256(model.repair.historical.retrieval.state_dict())
        validation = _validation_forecast_metrics(model, protocol, settings, device=device)
        expected_validation = {'mse': record['best_val_mse'], 'mae': record['best_val_mae']}
        manifests = [Path(record['stage1_manifest']), args.selection]
    else:
        profiles = load_author_profiles(args.profiles)
        cfg = profiles['models']['T1']
        identity = author_source_identity(args.profiles, cfg, protocol)
        if any(record.get(k) != v for k, v in identity.items()):
            raise ValueError('T1 saved source/config/seed identity changed')
        if record['status'] != 'validation_finalized' or record.get('stage1_aux_forecast', 'none') != 'none':
            raise ValueError('expected completed unmodified T1 run')
        repair_selection = Path(record['repair_selection_manifest'])
        repair = json.loads(repair_selection.read_text())
        if (file_sha256(repair_selection) != record['repair_selection_sha256'] or
            file_sha256(Path(repair['checkpoint'])) != repair['checkpoint_sha256'] or
            file_sha256(Path(record['checkpoint'])) != record['checkpoint_sha256']):
            raise ValueError('T1 selected checkpoint/parent hash mismatch')
        payload = torch.load(record['checkpoint'], map_location=device, weights_only=False)
        if any(payload.get(k) != v for k, v in identity.items()):
            raise ValueError('T1 payload identity mismatch')
        model = T1ForecastModel(build_author_model('T1', cfg, ROOT/cfg['root'], device),
                                _forecast_head(protocol)).to(device)
        model.load_state_dict(payload['model_state'], strict=True)
        validation = author_score('T1', model, protocol, 'validation', device, None, record['training'])
        expected_validation = record['best_validation']
        manifests = [args.selection]  # combined repair+forecast resource sidecar
    if any(abs(validation[k]-expected_validation[k]) > 1e-6 for k in ['mse', 'mae']):
        raise ValueError('fixed validation replay mismatch')
    predictor_parameters = sum(p.numel() for p in model.predictor.parameters())
    if predictor_parameters != 35427:
        raise ValueError('original prediction head parameter count changed')
    lock = dict(model=args.model_label or args.model, training_seed=args.training_seed,
                checkpoint_selection_metric=record.get('selection_metric',record.get('training',{}).get('selection_metric')),
                selection_manifest=str(args.selection.resolve()), selection_manifest_sha256=selection_hash,
                checkpoint_sha256=record['checkpoint_sha256'], test_mask_bank_sha256=bank_hash,
                validation_mask_sha256=protocol.mask_sha256['validation'],
                validation_replay=validation, predictor_parameters=predictor_parameters,
                predictor_initial_sha256=record['predictor_initial_sha256'], test_metrics_seen=False)
    write_new_record(args.output/'validation-lock.json', lock)
    model.eval()
    results = []
    for entry, view in zip(entries, views):
        squared = absolute = count = windows = 0
        window_mse, window_mae, target_starts = [], [], []
        with torch.inference_mode():
            for cpu_batch in iter_p4_batches(view, 'test', batch_size=128):
                batch = cpu_batch.to(device)
                if args.model == 'T1':
                    prediction = model(batch)
                else:
                    starts = _starts(batch, stride=model.repair.stride, patches=model.repair.patch_count, device=device)
                    prediction, _ = model(batch.normalized_observed, batch.observed_mask,
                                          batch.query_mean, batch.query_std, starts)
                error = prediction-batch.forecast_target
                if not torch.isfinite(error).all():
                    raise FloatingPointError('nonfinite test errors')
                squared += error.square().sum().item()
                absolute += error.abs().sum().item()
                count += error.numel()
                windows += len(prediction)
                window_mse.extend(error.double().square().mean((1, 2)).cpu().tolist())
                window_mae.extend(error.double().abs().mean((1, 2)).cpu().tolist())
                target_starts.extend(ref.target_start for ref in batch.refs)
        if (windows, count) != (2688, 2688*96*7):
            raise ValueError('test evaluation support changed')
        if args.model == 'E23-v31-B' and historical_identity != (
                memory.snapshot_hash(), _key_sha256(model.repair.historical.retrieval.state_dict())):
            raise ValueError('historical state changed during test')
        arrays = args.output/f"{entry['group_id']}-windows.npz"
        np.savez_compressed(arrays, mse=window_mse, mae=window_mae, target_start=target_starts)
        result = dict(group_id=entry['group_id'], mask_seed=entry['seed'], mask_sha256=entry['sha256'],
                      mse=squared/count, mae=absolute/count, windows=windows, target_elements=count,
                      window_metrics=str(arrays.resolve()), window_metrics_sha256=file_sha256(arrays))
        results.append(result)
        print(json.dumps(dict(model=args.model, training_seed=args.training_seed, **result)), flush=True)
    if file_sha256(args.selection) != selection_hash or file_sha256(args.test_mask_bank) != bank_hash:
        raise ValueError('locked selection/bank changed during test')
    load_test_bank(protocol, args.test_mask_bank, expected_seeds=TEST_SEEDS)
    result = {**lock, 'test_metrics_seen': True, 'mask_seeds': list(TEST_SEEDS),
              'metric_space': 'train_standardized', 'split': 'test', 'test_draws': results,
              'test_mean': {k: float(np.mean([r[k] for r in results])) for k in ['mse', 'mae']},
              'test_mask_std_ddof1': {k: float(np.std([r[k] for r in results], ddof=1)) for k in ['mse', 'mae']},
              'training_resources': selected_training_resources(manifests)}
    write_new_record(args.output/'twenty-test-draws.json', result)
    print(json.dumps(dict(model=args.model, training_seed=args.training_seed, test_mean=result['test_mean'])), flush=True)


if __name__ == '__main__':
    main()
