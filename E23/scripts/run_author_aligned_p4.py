"""Author-configured P4 training and test-only three-draw evaluation (20261002)."""

from __future__ import annotations

from time import perf_counter
_PROCESS_STARTED = perf_counter()

import argparse
from contextlib import nullcontext
from dataclasses import replace
import json
from pathlib import Path
import random

import numpy as np
import torch

from integrations.forecast_compare.author_profiles import (
    ROOT, DEFAULT_PROFILES, load_author_profiles, build_author_model,
    epoch_end_learning_rate, author_coifnet_loss,
)
from integrations.forecast_compare.author_provenance import (
    author_source_identity as _code_identity, write_new_record as _write,
)
from integrations.forecast_compare.bitgraph_p4 import bitgraph_inputs
from integrations.forecast_compare.coifnet_p4 import coifnet_inputs, load_timestamp_features
from integrations.forecast_compare.crib_p4 import crib_input, crib_training_loss
from integrations.forecast_compare.ctf_p4 import ctf_inputs, group_by_full_patch_visibility
from integrations.forecast_compare.protocol import iter_p4_batches, load_p4_protocol
from integrations.forecast_compare.resources import TrainingResources, selected_training_resources
from integrations.forecast_compare.t1_p4 import T1ForecastModel, t1_inputs, t1_clean_history
from integrations.forecast_compare.three_draw_evaluation import evaluate_three_draws
from malr.evaluation.report import file_sha256
from malr.models.patchtst import PatchTSTCore
from malr.models.repaired_forecast import MaskAwareForecastModel

DEFAULT_MASKS = ROOT.parent / 'E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context'
DEFAULT_BANK = ROOT / 'artifacts/ETTh1/seq96_pred96/sr010_bl8/test_mask_bank_20261002/manifest.json'


def _seed():
    random.seed(2021)
    np.random.seed(2021)
    torch.manual_seed(2021)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2021)


def _autocast(training, device):
    dtype = {'float16': torch.float16, 'bfloat16': torch.bfloat16}.get(training['precision'])
    return torch.autocast('cuda', dtype=dtype) if dtype is not None and device.type == 'cuda' else nullcontext()


def _forecast_head(protocol):
    cfg = protocol.config.model.for_dataset('ETTh1')
    core = PatchTSTCore(c_in=7, seq_len=96, pred_len=96, patch_len=16, stride=8,
                        d_model=cfg.d_model, n_heads=cfg.n_heads, d_ff=cfg.d_ff,
                        n_layers=cfg.encoder_layers, dropout=cfg.dropout,
                        fc_dropout=cfg.fc_dropout, head_dropout=cfg.head_dropout,
                        attn_dropout=cfg.attn_dropout)
    return MaskAwareForecastModel(core)


def _predict(name, model, batch, stamps):
    if isinstance(model, T1ForecastModel):
        return model(batch)
    if name == 'CRIB':
        return model(crib_input(batch), test_flag=True)[4].permute(0, 2, 1)
    if name == 'CoIFNet':
        values, mask, time = coifnet_inputs(batch, stamps)
        return model.impute(values, mask, feat=time)[:, 96:].permute(0, 2, 1)
    if name == 'BiTGraph':
        values, mask = bitgraph_inputs(batch)
        return model(values, mask, 0).squeeze(-1).permute(0, 2, 1)
    values, mask = ctf_inputs(batch)
    prediction = torch.empty_like(batch.forecast_target)
    for indices in group_by_full_patch_visibility(mask):
        part = model(values[indices], None, None, None, missing_flag=mask[indices])
        prediction[indices] = part.permute(0, 2, 1)
    return prediction


def _repair_loss(name, model, batch):
    values, mask = t1_inputs(batch)
    if name == 'BRITS':
        return model.original_loss(values, mask)
    if name == 'SAITS':
        return model.original_loss(values, mask, t1_clean_history(batch))
    candidate = model(values, mask=mask)
    return (candidate - t1_clean_history(batch)).square()[~batch.observed_mask].mean()


def _loss(name, model, batch, stamps, cfg, stage):
    if stage == 'repair':
        return _repair_loss(name, model, batch)
    if isinstance(model, T1ForecastModel):
        return (model(batch) - batch.forecast_target).square().mean()
    if name == 'CRIB':
        return crib_training_loss(model(crib_input(batch)), batch.forecast_target.permute(0, 2, 1))
    if name == 'CoIFNet':
        values, mask, time = coifnet_inputs(batch, stamps)
        return author_coifnet_loss(model.impute(values, mask, feat=time), values, mask,
                                   batch.forecast_target.permute(0, 2, 1),
                                   weight=cfg['model']['loss_lambda'])
    error = _predict(name, model, batch, stamps) - batch.forecast_target
    return error.abs().mean() if name == 'BiTGraph' else error.square().mean()


def score(name, model, protocol, split, device, stamps, training, *, stage='forecast', limit=None, auxiliary_predictor=None):
    model.eval()
    squared = absolute = count = windows = 0
    auxiliary_squared = auxiliary_absolute = auxiliary_count = 0
    if auxiliary_predictor is not None:
        auxiliary_predictor.eval()
    # Same original first 2688 windows, independently of author training batch size.
    refs = dict(protocol.refs)
    refs[split] = refs[split][:len(refs[split]) // 128 * 128]
    view = replace(protocol, refs=refs)
    batch_size = 32 if name == 'ChannelTokenFormer' else 128
    with torch.no_grad():
        for index, cpu_batch in enumerate(iter_p4_batches(view, split, batch_size=batch_size)):
            batch = cpu_batch.to(device)
            with _autocast(training, device):
                if stage == 'repair':
                    values, mask = t1_inputs(batch)
                    candidate = model(values, mask=mask)
                    error = (candidate.float() - t1_clean_history(batch))[~batch.observed_mask]
                    if auxiliary_predictor is not None:
                        from integrations.forecast_compare.auxiliary_forecast import predict_repaired
                        repaired = (candidate - batch.query_mean[:, None, :]) / batch.query_std[:, None, :]
                        future_error = predict_repaired(auxiliary_predictor, repaired, batch).float() - batch.forecast_target.float()
                        auxiliary_squared += future_error.square().sum().item()
                        auxiliary_absolute += future_error.abs().sum().item()
                        auxiliary_count += future_error.numel()
                else:
                    prediction = _predict(name, model, batch, stamps).float()
                    if prediction.shape != batch.forecast_target.shape:
                        raise ValueError(f'{name} forecast output shape mismatch')
                    error = prediction - batch.forecast_target
            squared += error.square().sum().item(); absolute += error.abs().sum().item()
            count += error.numel(); windows += len(batch.refs)
            if limit is not None and index + 1 >= limit:
                break
    if count == 0:
        raise ValueError('evaluation produced no target elements')
    return {'mse': squared / count, 'mae': absolute / count,
            'windows': windows, 'target_elements': count,
            **({'auxiliary_forecast_mse': auxiliary_squared / auxiliary_count,
                'auxiliary_forecast_mae': auxiliary_absolute / auxiliary_count} if auxiliary_predictor is not None else {})}


def _train_stage(name, model, protocol, stamps, cfg, training, stage, output, device, identity,
                 *, smoke=False, auxiliary_forecast='none'):
    from integrations.forecast_compare.auxiliary_forecast import (
        auxiliary_identity, build_auxiliary_predictor, auxiliary_forecast_loss, t1_repaired_sequence,
    )
    from integrations.e23.forecast_sensitive import parameter_hash
    auxiliary_metadata = auxiliary_identity(auxiliary_forecast)
    if auxiliary_metadata and (name != 'T1' or stage != 'repair'):
        raise ValueError('author auxiliary supervision is restricted to T1 repair stage')
    auxiliary_predictor = build_auxiliary_predictor(protocol, device=device) if auxiliary_metadata else None
    if auxiliary_predictor is not None:
        auxiliary_metadata.update(stage1_aux_predictor_initial_sha256=parameter_hash(auxiliary_predictor),
                                  stage1_aux_predictor_parameters=sum(p.numel() for p in auxiliary_predictor.parameters()))
    _seed()
    optimizer = torch.optim.Adam(list(model.parameters()) + (list(auxiliary_predictor.parameters()) if auxiliary_predictor is not None else []), lr=training['learning_rate'],
                                 weight_decay=training['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', enabled=training['precision'] == 'float16' and device.type == 'cuda')
    checkpoint, selection = output / f'{stage}-selected.pt', output / f'{stage}-selection.json'
    if selection.exists() or checkpoint.exists():
        raise FileExistsError(f'refusing to replace a finalized or partial stage: {output}/{stage}')
    output.mkdir(parents=True, exist_ok=True)
    best = float('inf'); stale = 0; best_epoch = 0
    started = perf_counter()
    auxiliary_gradient_batches = auxiliary_windows = 0
    for epoch in range(1 if smoke else training['epochs']):
        model.train(); losses = []
        if auxiliary_predictor is not None:
            auxiliary_predictor.train()
        for batch_index, cpu_batch in enumerate(iter_p4_batches(protocol, 'train', batch_size=training['batch_size'], epoch=epoch), 1):
            batch = cpu_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(training, device):
                if auxiliary_predictor is None:
                    loss = _loss(name, model, batch, stamps, cfg, stage)
                else:
                    loss, repaired = t1_repaired_sequence(model, batch)
                    loss = loss + .05 * auxiliary_forecast_loss(auxiliary_predictor, repaired, batch,
                                                                epoch=epoch+1, batch_index=batch_index)
                    auxiliary_windows += len(batch.normalized_observed)
                    auxiliary_gradient_batches += int(epoch+1 > 10)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'{name} {stage} nonfinite loss in epoch {epoch + 1}')
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            losses.append(loss.item())
            if smoke:
                break
        if not losses:
            raise ValueError('training produced no batches at the configured author batch size')
        validation = score(name, model, protocol, 'validation', device, stamps, training,
                           stage=stage, limit=1 if smoke else None,
                           **({'auxiliary_predictor': auxiliary_predictor} if auxiliary_predictor is not None else {}))
        if validation[training['selection_metric']] < best:
            best, stale, best_epoch = validation[training['selection_metric']], 0, epoch + 1
            best_validation = validation
            payload = {'model_state': model.state_dict(), 'best_epoch': best_epoch, **identity, **auxiliary_metadata}
            if auxiliary_predictor is not None:
                payload['auxiliary_predictor_state'] = auxiliary_predictor.state_dict()
            torch.save(payload, checkpoint)
        else:
            stale += 1
        print(json.dumps({'model': name, 'stage': stage, 'epoch': epoch + 1,
                          'train_loss': float(np.mean(losses)), 'validation': validation,
                          'best_epoch': best_epoch}), flush=True)
        if training['patience'] is not None and stale >= training['patience']:
            break
        for group in optimizer.param_groups:
            group['lr'] = epoch_end_learning_rate(training, epoch + 1)
    record = {**identity, 'model': name, 'profile_version': cfg['version'], 'profile': cfg,
              'stage': stage, 'training': training, 'seed': 2021,
              'best_epoch': best_epoch, 'epochs_completed': epoch + 1,
              'best_validation': best_validation, 'training_seconds': perf_counter() - started,
              'checkpoint': str(checkpoint.resolve()), 'checkpoint_sha256': file_sha256(checkpoint),
              'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
              'status': 'smoke' if smoke else 'validation_finalized', 'test_metrics_seen': False,
              **auxiliary_metadata,
              **({'stage1_aux_repair_gradient_batches': auxiliary_gradient_batches,
                  'stage1_aux_training_windows': auxiliary_windows} if auxiliary_metadata else {})}
    _write(selection, record)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)['model_state'])
    return selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=list(load_author_profiles()['models']), required=True)
    parser.add_argument('--profiles', type=Path, default=DEFAULT_PROFILES)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/stage1_seq96_dynamic_temporal_repair.yaml')
    parser.add_argument('--masks', type=Path, default=DEFAULT_MASKS)
    parser.add_argument('--test-mask-bank', type=Path, default=DEFAULT_BANK)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--stage1-aux-forecast', choices=('none','forecast_mse_v1'), default='none')
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--smoke', action='store_true')
    action.add_argument('--train', action='store_true')
    action.add_argument('--evaluate-selected', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    profiles = load_author_profiles(args.profiles); cfg = profiles['models'][args.model]
    if args.stage1_aux_forecast != 'none' and args.model != 'T1':
        raise ValueError('Stage1 auxiliary baseline is only defined for T1')
    if cfg['kind'] == 's4m':
        raise ValueError('use scripts/run_author_s4m_p4.py in the dedicated S4M environment')
    device = torch.device(args.device)
    with TrainingResources(args.output, device=device, batch_size=cfg['training']['batch_size'],
                           model=args.model, mode='evaluation' if args.evaluate_selected else 'smoke' if args.smoke else 'training',
                           started_at=_PROCESS_STARTED, precision=cfg['training']['precision'],
                           amp=cfg['training']['precision'] != 'float32') as resources:
        protocol = load_p4_protocol(args.config, args.masks)
        identity = _code_identity(args.profiles, cfg, protocol)
        if args.stage1_aux_forecast != 'none':
            from integrations.forecast_compare.auxiliary_forecast import auxiliary_identity
            identity.update(auxiliary_identity(args.stage1_aux_forecast))
        stamps = load_timestamp_features(Path(protocol.config.data.datasets['ETTh1'].path)) if args.model == 'CoIFNet' else None
        _seed()
        model = build_author_model(args.model, cfg, ROOT / cfg['root'], device)
        if args.evaluate_selected:
            if cfg['kind'] == 'imputer':
                model = T1ForecastModel(model, _forecast_head(protocol)).to(device)
            selection = args.output / 'forecast-selection.json'
            record = json.loads(selection.read_text())
            if record['status'] != 'validation_finalized':
                raise ValueError('smoke/partial runs cannot enter formal test evaluation')
            if cfg['kind'] == 'imputer':
                repair_selection = Path(record['repair_selection_manifest'])
                if file_sha256(repair_selection) != record['repair_selection_sha256']:
                    raise ValueError('selected repair-stage manifest changed')
                repair_record = json.loads(repair_selection.read_text())
                if file_sha256(Path(repair_record['checkpoint'])) != repair_record['checkpoint_sha256']:
                    raise ValueError('selected repair-stage checkpoint changed')
                if args.stage1_aux_forecast != 'none':
                    from integrations.forecast_compare.auxiliary_forecast import verify_auxiliary_identity
                    verify_auxiliary_identity(repair_record, torch.load(repair_record['checkpoint'], map_location=device, weights_only=False))
            for key, value in identity.items():
                if record.get(key) != value:
                    raise ValueError(f'selected run {key} changed')
            checkpoint = Path(record['checkpoint'])
            if file_sha256(checkpoint) != record['checkpoint_sha256']:
                raise ValueError('selected checkpoint SHA-256 mismatch')
            payload = torch.load(checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(payload['model_state'], strict=True)
            training = record['training']
            validation = score(args.model, model, protocol, 'validation', device, stamps, training)
            if any(abs(validation[k] - record['best_validation'][k]) > 1e-6 for k in ('mse', 'mae')):
                raise ValueError('selected validation failed replay')
            result = evaluate_three_draws(protocol, args.test_mask_bank, selection,
                lambda group: score(args.model, model, group, 'test', device, stamps, training))
            result.update(model=args.model, profile_version=cfg['version'], validation=validation,
                          training_resources=selected_training_resources([selection]))
            _write(args.output / 'three-test-draws.json', result)
            print(json.dumps(result['test_mean']), flush=True)
        else:
            if cfg['kind'] == 'imputer':
                with resources.stage('repair'):
                    repair_selection = _train_stage(args.model, model, protocol, stamps, cfg, cfg['training'],
                                                    'repair', args.output, device, identity, smoke=args.smoke,
                                                    auxiliary_forecast=args.stage1_aux_forecast)
                resources.stages['repair'].update(batch_size=cfg['training']['batch_size'],
                                                 precision=cfg['training']['precision'])
                identity.update(repair_selection_manifest=str(repair_selection.resolve()),
                                repair_selection_sha256=file_sha256(repair_selection))
                _seed()
                model = T1ForecastModel(model, _forecast_head(protocol)).to(device)
                from integrations.e23.forecast_sensitive import parameter_hash
                identity['predictor_initial_sha256'] = parameter_hash(model.predictor)
            training = profiles['shared_forecast_stage'] if cfg['kind'] == 'imputer' else cfg['training']
            # The shared forecast stage has no learning-rate decay.
            training = {**training, 'lradj': training.get('lradj', 'constant')}
            with resources.stage('forecast'):
                selection = _train_stage(args.model, model, protocol, stamps, cfg, training,
                                         'forecast', args.output, device, identity, smoke=args.smoke)
            resources.stages['forecast'].update(batch_size=training['batch_size'], precision=training['precision'])
            resources.selection_manifest = selection


if __name__ == '__main__':
    main()
