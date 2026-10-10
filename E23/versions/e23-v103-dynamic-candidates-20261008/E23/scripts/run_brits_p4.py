"""Two-stage BRITS imputer and shared PatchTST head on locked ETTh1 P4."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
from time import monotonic, perf_counter

_PROCESS_STARTED = perf_counter()

import torch

from integrations.forecast_compare.brits_p4 import (
    BRITS_SETTINGS, BritsForecastModel, brits_inputs, brits_repair_loss, load_brits,
)
from integrations.forecast_compare.protocol import iter_p4_batches, load_p4_protocol
from integrations.forecast_compare.resources import TrainingResources, selected_training_resources
from malr.evaluation.report import file_sha256
from malr.models.patchtst import PatchTSTCore
from malr.models.repaired_forecast import MaskAwareForecastModel


DEFAULT_CONFIG = Path('configs/stage1_seq96_dynamic_temporal_repair.yaml')
DEFAULT_MASKS = Path('../E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context')
DEFAULT_UPSTREAM = Path('../SAITS')
DEFAULT_OUTPUT = Path('artifacts/brits_etth1_p4')
BATCH_SIZE = 128
EPOCHS = 100


def upstream_revision(root: Path) -> str:
    return subprocess.check_output(
        ['/home/xyw/miniforge3/envs/malr/bin/git', '-C', str(root), 'rev-parse', 'HEAD'],
        text=True,
    ).strip()


def model_and_head(protocol, root: Path, device: torch.device) -> BritsForecastModel:
    profile = protocol.config.model.for_dataset('ETTh1')
    core = PatchTSTCore(
        c_in=7, seq_len=96, pred_len=96, patch_len=16, stride=8,
        d_model=profile.d_model, n_heads=profile.n_heads, d_ff=profile.d_ff,
        n_layers=profile.encoder_layers, dropout=profile.dropout,
        fc_dropout=profile.fc_dropout, head_dropout=profile.head_dropout,
        attn_dropout=profile.attn_dropout,
    )
    return BritsForecastModel(load_brits(root, device=device), MaskAwareForecastModel(core)).to(device)


def score_repair(model, protocol, device: torch.device, *, limit: int | None = None) -> dict:
    model.eval()
    absolute = squared = count = 0
    with torch.no_grad():
        for index, cpu_batch in enumerate(iter_p4_batches(protocol, 'validation', batch_size=BATCH_SIZE)):
            batch = cpu_batch.to(device)
            candidate, _ = model.impute(brits_inputs(batch, supervised=False))
            clean = (batch.normalized_target * batch.query_std[:, None, :]
                     + batch.query_mean[:, None, :])
            error = (candidate - clean)[~batch.observed_mask]
            absolute += error.abs().sum().item()
            squared += error.square().sum().item()
            count += error.numel()
            if limit is not None and index + 1 >= limit:
                break
    return {'mae': absolute / count, 'mse': squared / count, 'hidden_elements': count}


def score_forecast(model, protocol, split: str, device: torch.device,
                   *, limit: int | None = None) -> dict:
    model.eval()
    absolute = squared = count = windows = 0
    with torch.no_grad():
        for index, cpu_batch in enumerate(iter_p4_batches(protocol, split, batch_size=BATCH_SIZE)):
            batch = cpu_batch.to(device)
            prediction = model(batch)
            if prediction.shape != batch.forecast_target.shape:
                raise ValueError('BRITS forecast output shape mismatch')
            error = prediction - batch.forecast_target
            absolute += error.abs().sum().item()
            squared += error.square().sum().item()
            count += error.numel()
            windows += error.shape[0]
            if limit is not None and index + 1 >= limit:
                break
    return {'mae': absolute / count, 'mse': squared / count,
            'windows': windows, 'target_elements': count}


def identity(protocol, root: Path) -> dict:
    adapter = Path(__file__).resolve().parent.parent / 'integrations' / 'forecast_compare' / 'brits_p4.py'
    return {
        'source_csv_sha256': protocol.source_sha256,
        'validation_mask_sha256': protocol.mask_sha256['validation'],
        'test_mask_sha256': protocol.mask_sha256['test'],
        'upstream_revision': upstream_revision(root),
        'official_source_sha256': file_sha256(root / 'modeling' / 'brits.py'),
        'adapter_sha256': file_sha256(adapter),
        'settings': BRITS_SETTINGS,
    }


def checked_record(path: Path, protocol, root: Path) -> dict:
    record = json.loads(path.read_text(encoding='utf-8'))
    for key, expected in identity(protocol, root).items():
        if record.get(key) != expected:
            raise ValueError(f'BRITS selected {key} mismatch')
    if record.get('epochs_completed') != EPOCHS:
        raise ValueError('BRITS stage did not finish all 100 epochs')
    if file_sha256(Path(record['checkpoint'])) != record['checkpoint_sha256']:
        raise ValueError('BRITS selected checkpoint SHA-256 mismatch')
    return record


def train_stage1(protocol, root: Path, output: Path, device: torch.device, *, smoke: bool) -> None:
    torch.manual_seed(2021)
    torch.cuda.manual_seed_all(2021)
    model = load_brits(root, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    if smoke:
        model.train()
        batch = next(iter_p4_batches(protocol, 'train', batch_size=BATCH_SIZE, epoch=0)).to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = brits_repair_loss(model, batch)
        if not torch.isfinite(loss):
            raise FloatingPointError('BRITS stage1 smoke loss not finite')
        loss.backward()
        optimizer.step()
        print(json.dumps({'stage1_smoke': 'ok', 'loss': loss.item(),
                          'validation': score_repair(model, protocol, device, limit=1)}))
        return
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / 'brits-stage1-selected.pt'
    selection = output / 'brits-stage1-selection.json'
    if selection.exists():
        raise FileExistsError(f'BRITS stage1 already selected: {selection}')
    started = monotonic()
    best = {'mae': float('inf')}
    best_epoch = 0
    for epoch in range(EPOCHS):
        model.train()
        total = batches = 0
        for cpu_batch in iter_p4_batches(protocol, 'train', batch_size=BATCH_SIZE, epoch=epoch):
            batch = cpu_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = brits_repair_loss(model, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'BRITS stage1 non-finite loss at epoch {epoch + 1}')
            loss.backward()
            optimizer.step()
            total += loss.item()
            batches += 1
        validation = score_repair(model, protocol, device)
        if validation['mae'] < best['mae']:
            best, best_epoch = validation, epoch + 1
            torch.save({'model_state': model.state_dict(), 'best_epoch': best_epoch}, checkpoint)
        print(json.dumps({'stage': 1, 'epoch': epoch + 1, 'train_loss': total / batches,
                          'validation': validation, 'best_epoch': best_epoch}), flush=True)
    record = {
        **identity(protocol, root), 'model': 'BRITS', 'dataset': 'ETTh1', 'scenario': 'P4',
        'stage': 'imputation', 'checkpoint': str(checkpoint.resolve()),
        'checkpoint_sha256': file_sha256(checkpoint), 'best_epoch': best_epoch,
        'epochs_completed': EPOCHS, 'best_validation': best,
        'selection': 'earliest minimum fixed-validation hidden-coordinate MAE',
        'batch_size': BATCH_SIZE, 'learning_rate': 1e-3, 'seed': 2021,
        'loss': 'official_BRITS_consistency+reconstruction+MIT',
        'training_seconds': monotonic() - started,
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    selection.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(f'BRITS_STAGE1_SELECTION {selection}', flush=True)


def train_stage2(protocol, root: Path, output: Path, device: torch.device,
                 *, smoke: bool) -> None:
    stage1_path = output / 'brits-stage1-selection.json'
    stage1 = checked_record(stage1_path, protocol, root)
    torch.manual_seed(2021)
    torch.cuda.manual_seed_all(2021)
    model = model_and_head(protocol, root, device)
    payload = torch.load(stage1['checkpoint'], map_location=device, weights_only=False)
    model.imputer.load_state_dict(payload['model_state'], strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    if smoke:
        model.train()
        batch = next(iter_p4_batches(protocol, 'train', batch_size=BATCH_SIZE, epoch=0)).to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(batch), batch.forecast_target)
        if not torch.isfinite(loss):
            raise FloatingPointError('BRITS stage2 smoke loss not finite')
        loss.backward()
        optimizer.step()
        print(json.dumps({'stage2_smoke': 'ok', 'loss': loss.item(),
                          'validation': score_forecast(model, protocol, 'validation', device, limit=1)}))
        return
    checkpoint = output / 'brits-stage2-selected.pt'
    selection = output / 'brits-stage2-selection.json'
    if selection.exists():
        raise FileExistsError(f'BRITS stage2 already selected: {selection}')
    started = monotonic()
    best = {'mae': float('inf')}
    best_epoch = 0
    for epoch in range(EPOCHS):
        model.train()
        total = batches = 0
        for cpu_batch in iter_p4_batches(protocol, 'train', batch_size=BATCH_SIZE, epoch=epoch):
            batch = cpu_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(batch), batch.forecast_target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'BRITS stage2 non-finite loss at epoch {epoch + 1}')
            loss.backward()
            optimizer.step()
            total += loss.item()
            batches += 1
        validation = score_forecast(model, protocol, 'validation', device)
        if validation['mae'] < best['mae']:
            best, best_epoch = validation, epoch + 1
            torch.save({'model_state': model.state_dict(), 'best_epoch': best_epoch}, checkpoint)
        print(json.dumps({'stage': 2, 'epoch': epoch + 1, 'train_loss': total / batches,
                          'validation': validation, 'best_epoch': best_epoch}), flush=True)
    record = {
        **identity(protocol, root), 'model': 'BRITS+shared PatchTST',
        'dataset': 'ETTh1', 'scenario': 'P4', 'stage': 'forecast',
        'checkpoint': str(checkpoint.resolve()),
        'checkpoint_sha256': file_sha256(checkpoint), 'best_epoch': best_epoch,
        'epochs_completed': EPOCHS, 'best_validation': best,
        'stage1_selection_sha256': file_sha256(stage1_path),
        'stage1_checkpoint_sha256': stage1['checkpoint_sha256'],
        'selection': 'earliest minimum fixed-validation forecast MAE',
        'batch_size': BATCH_SIZE, 'learning_rate': 1e-4, 'seed': 2021,
        'loss': 'forecast_MSE', 'metric_space': 'train_standardized',
        'predictor_profile': asdict(protocol.config.model.for_dataset('ETTh1')),
        'training_seconds': monotonic() - started,
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    selection.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(f'BRITS_STAGE2_SELECTION {selection}', flush=True)


def evaluate_selected(protocol, root: Path, output: Path, device: torch.device) -> None:
    selection = output / 'brits-stage2-selection.json'
    evaluation = output / 'brits-test.json'
    if evaluation.exists():
        raise FileExistsError(f'BRITS test already finalized: {evaluation}')
    record = checked_record(selection, protocol, root)
    stage1_path = output / 'brits-stage1-selection.json'
    stage1 = checked_record(stage1_path, protocol, root)
    if file_sha256(stage1_path) != record['stage1_selection_sha256']:
        raise ValueError('BRITS Stage1 selection changed')
    if stage1['checkpoint_sha256'] != record['stage1_checkpoint_sha256']:
        raise ValueError('BRITS Stage1 checkpoint changed')
    model = model_and_head(protocol, root, device)
    payload = torch.load(record['checkpoint'], map_location=device, weights_only=False)
    model.load_state_dict(payload['model_state'], strict=True)
    if payload['best_epoch'] != record['best_epoch']:
        raise ValueError('BRITS selected checkpoint epoch mismatch')
    validation = score_forecast(model, protocol, 'validation', device)
    for key in ('mse', 'mae'):
        if abs(validation[key] - record['best_validation'][key]) > 1e-7:
            raise ValueError(f'BRITS validation {key} failed replay')
    test = score_forecast(model, protocol, 'test', device)
    result = {
        'model': 'BRITS+shared PatchTST', 'dataset': 'ETTh1', 'scenario': 'P4',
        'selected_epoch': record['best_epoch'], 'validation': validation,
        'test': test, 'selection_sha256': file_sha256(selection),
        'checkpoint_sha256': record['checkpoint_sha256'],
        'source_csv_sha256': protocol.source_sha256,
        'test_mask_sha256': protocol.mask_sha256['test'],
        'metric_space': 'train_standardized',
    }
    result['training_resources'] = selected_training_resources([stage1_path, selection])
    evaluation.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(f'TRAINING_RESOURCE_SUMMARY {json.dumps(result["training_resources"])}', flush=True)
    print(f'BRITS_TEST {json.dumps(test)}', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--masks', type=Path, default=DEFAULT_MASKS)
    parser.add_argument('--upstream', type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--device', default='cuda:0')
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--smoke-stage1', action='store_true')
    action.add_argument('--smoke-stage2', action='store_true')
    action.add_argument('--train-stage1', action='store_true')
    action.add_argument('--train-stage2', action='store_true')
    action.add_argument('--evaluate-selected', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if args.evaluate_selected:
        protocol = load_p4_protocol(args.config, args.masks)
        evaluate_selected(protocol, args.upstream, args.output, device)
        return
    stage = 'repair' if args.smoke_stage1 or args.train_stage1 else 'forecast'
    smoke = args.smoke_stage1 or args.smoke_stage2
    with TrainingResources(args.output, device=device, batch_size=BATCH_SIZE,
                           model='BRITS+shared_PatchTST', mode='smoke' if smoke else 'training',
                           started_at=_PROCESS_STARTED) as resources:
        protocol = load_p4_protocol(args.config, args.masks)
        with resources.stage(stage):
            if stage == 'repair':
                train_stage1(protocol, args.upstream, args.output, device, smoke=smoke)
            else:
                train_stage2(protocol, args.upstream, args.output, device, smoke=smoke)
        if not smoke:
            resources.selection_manifest = args.output / 'brits-stage1-selection.json' if stage == 'repair' else args.output / 'brits-stage2-selection.json'


if __name__ == '__main__':
    main()
