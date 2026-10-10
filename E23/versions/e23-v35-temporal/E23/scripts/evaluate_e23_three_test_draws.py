"""Evaluate a locked E23 checkpoint on the frozen test bank; no model selection."""

import argparse
import json
from pathlib import Path

import torch

from integrations.e23.config import load_e23_p4_protocol, load_e23_settings
from integrations.e23.train import (
    _new_memory, _key_sha256, _starts, _trial_identity, build_forecast_model, load_stage2_selection,
)
from integrations.forecast_compare.author_provenance import write_new_record
from integrations.forecast_compare.protocol import iter_p4_batches
from integrations.forecast_compare.resources import selected_training_resources
from integrations.forecast_compare.three_draw_evaluation import evaluate_three_draws
from malr.evaluation.report import file_sha256
from malr.migration import MigrationCatalog

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--e23-config', type=Path, default=ROOT / 'configs/e23_multi_expert_p4_width64_window_v12.yaml')
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--original-test-result', type=Path)
    parser.add_argument('--test-mask-bank', type=Path, default=ROOT / 'artifacts/ETTh1/seq96_pred96/sr010_bl8/test_mask_bank_20261002/manifest.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-label', default='E23_v12')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'three-draw evaluation already finalized: {args.output}')
    torch.set_num_threads(4)
    protocol = load_e23_p4_protocol(
        catalog=MigrationCatalog.load(ROOT.parent / 'E22/migration/paper_catalog.jsonl', project='E23'),
        protocol_config=ROOT / 'configs/stage1_seq96_dynamic_temporal_repair.yaml',
        mask_artifacts=ROOT.parent / 'E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context',
    )
    settings = load_e23_settings(args.e23_config, protocol=protocol)
    if settings.stage1_objective != 'window_smooth_l1' and args.model_label == 'E23_v12':
        raise ValueError('a new Stage 1 objective requires an explicit --model-label')
    device = torch.device(args.device)
    payload = load_stage2_selection(args.selection, protocol, settings, device=device)
    record = json.loads(args.selection.read_text())
    original = None
    if args.original_test_result is not None:
        original = json.loads(args.original_test_result.read_text())
        expected = {'selection_manifest_sha256': file_sha256(args.selection),
                    'checkpoint_sha256': record['checkpoint_sha256'],
                    'source_csv_sha256': protocol.source_sha256,
                    'validation_mask_sha256': protocol.mask_sha256['validation'],
                    'test_mask_sha256': protocol.mask_sha256['test'],
                    'e23_config_sha256': settings.config_sha256,
                    'metric_space': 'train_standardized', 'split': 'test',
                    'windows': 2688, 'target_elements': 2688 * 96 * 7}
        if any(original.get(k) != v for k, v in expected.items()):
            raise ValueError('cached original test does not match the locked E23 checkpoint/protocol')
    model = build_forecast_model(protocol, settings, device=device,
                                forecast_mixer=record.get('forecast_mixer', 'none'),
                                history_adapter=record.get('history_adapter', 'none'),
                                forecast_input=record.get('forecast_input', 'explicit_v1'))
    model.load_state_dict(payload['model_state'], strict=True)
    model.repair.historical.retrieval.load_state_dict(payload['retrieval_state'])
    memory = _new_memory(protocol); memory.load_state_dict(payload['memory_state'])
    model.repair.historical.refresh_snapshot(memory.snapshot()); model.eval()
    bank_before = memory.snapshot_hash()
    keys_before = _key_sha256(model.repair.historical.retrieval.state_dict())

    def score(group):
        if original is not None and group.mask_sha256['test'] == protocol.mask_sha256['test']:
            return {k: original[k] for k in ('mse', 'mae', 'windows', 'target_elements')}
        squared = absolute = count = windows = 0
        with torch.inference_mode():
            for cpu_batch in iter_p4_batches(group, 'test', batch_size=settings.batch_size):
                batch = cpu_batch.to(device)
                starts = _starts(batch, stride=model.repair.stride, patches=model.repair.patch_count, device=device)
                forecast, _ = model(batch.normalized_observed, batch.observed_mask,
                                    batch.query_mean, batch.query_std, starts)
                error = forecast - batch.forecast_target
                squared += error.square().sum().item(); absolute += error.abs().sum().item()
                count += error.numel(); windows += forecast.shape[0]
        if memory.snapshot_hash() != bank_before or _key_sha256(model.repair.historical.retrieval.state_dict()) != keys_before:
            raise ValueError('E23 historical bank/key state changed during test evaluation')
        return {'mse': squared / count, 'mae': absolute / count,
                'windows': windows, 'target_elements': count}

    result = evaluate_three_draws(protocol, args.test_mask_bank, args.selection, score)
    profile = f'locked width{settings.d_model}/{settings.variable_selection}'
    if settings.stage1_objective != 'window_smooth_l1':
        profile += f'/{settings.stage1_objective}'
    if settings.recovery_schedule != 'parallel_v1':
        profile += f'/{settings.recovery_schedule}/r{settings.recovery_rounds}'
    if settings.recovery_schedule == "stacked_parallel_tv_h_v1":
        profile += f"/l{settings.recovery_layers}/g{settings.expert_groups}"
    from integrations.e23.forecast_sensitive import retention_identity
    result.update(**retention_identity(record.get("repair_retention", "none")))
    from integrations.e23.prediction_preservation import checked_preservation
    result.update(**checked_preservation(record))
    result.update(**_trial_identity(settings))
    result.update(model=args.model_label, profile_version=profile,
                  stage1_objective=settings.stage1_objective,
                  recovery_schedule=settings.recovery_schedule,
                  recovery_rounds=settings.recovery_rounds,
                  e23_config_sha256=settings.config_sha256,
                  h_state_unchanged=True,
                  training_resources=selected_training_resources([Path(record['stage1_manifest']), args.selection]))
    if args.original_test_result is not None:
        result.update(original_test_reused=str(args.original_test_result.resolve()),
                      original_test_result_sha256=file_sha256(args.original_test_result))
    write_new_record(args.output, result)
    print(json.dumps({'test_mean': result['test_mean'], 'test_std': result['test_std']}))


if __name__ == '__main__':
    main()
