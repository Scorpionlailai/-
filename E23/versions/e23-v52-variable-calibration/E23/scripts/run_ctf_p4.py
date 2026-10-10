"""Train official CTF missing model with correct per-sample P4 patch deletion."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from time import monotonic

import torch

from integrations.forecast_compare.ctf_p4 import (
    ctf_inputs, group_by_full_patch_visibility, load_official_ctf,
)
from integrations.forecast_compare.protocol import iter_p4_batches, load_p4_protocol
from malr.evaluation.report import file_sha256


DEFAULT_CONFIG = Path('configs/stage1_seq96_dynamic_temporal_repair.yaml')
DEFAULT_MASKS = Path('../E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context')
DEFAULT_UPSTREAM = Path('artifacts/upstream/ChannelTokenFormer')
DEFAULT_OUTPUT = Path('artifacts/ctf_etth1_p4')


def upstream_revision(root: Path) -> str:
    return subprocess.check_output(
        ['/home/xyw/miniforge3/envs/malr/bin/git', '-C', str(root), 'rev-parse', 'HEAD'],
        text=True,
    ).strip()


def official_source_hash(root: Path) -> str:
    digest = sha256()
    for relative in ('models/ChannelTokenFormer_missing.py', 'layers/CTF_Embed.py',
                     'layers/SelfAttention_Family.py', 'layers/Embed.py'):
        digest.update(relative.encode())
        digest.update((root / relative).read_bytes())
    return digest.hexdigest()


def grouped_predictions(model, values: torch.Tensor, mask: torch.Tensor):
    """Yield subgroup predictions; each subgroup shares every fully empty patch."""
    for indices in group_by_full_patch_visibility(mask, patch_len=24):
        prediction = model(values.index_select(0, indices), None, None, None,
                           missing_flag=mask.index_select(0, indices))
        yield indices, prediction


def score(model, protocol, split: str, device: torch.device, *, limit: int | None = None) -> dict:
    model.eval()
    squared = absolute = count = windows = 0
    with torch.no_grad():
        for index, cpu_batch in enumerate(iter_p4_batches(protocol, split, batch_size=128)):
            batch = cpu_batch.to(device)
            values, mask = ctf_inputs(batch)
            targets = batch.forecast_target.permute(0, 2, 1)
            for indices, prediction in grouped_predictions(model, values, mask):
                selected_target = targets.index_select(0, indices)
                if prediction.shape != selected_target.shape:
                    raise ValueError('CTF forecast shape does not match P4 targets')
                error = prediction - selected_target
                squared += error.square().sum().item()
                absolute += error.abs().sum().item()
                count += error.numel()
                windows += error.shape[0]
            if limit is not None and index + 1 >= limit:
                break
    if count == 0:
        raise ValueError(f'{split} has no forecast targets')
    return {'mse': squared / count, 'mae': absolute / count,
            'windows': windows, 'target_elements': count}


def train(protocol, root: Path, output: Path, device: torch.device, *, smoke: bool) -> None:
    torch.manual_seed(2021)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(2021)
    model = load_official_ctf(root, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    if smoke:
        model.train()
        batch = next(iter_p4_batches(protocol, 'train', batch_size=128, epoch=0)).to(device)
        values, mask = ctf_inputs(batch)
        target = batch.forecast_target.permute(0, 2, 1)
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        groups = 0
        for indices, prediction in grouped_predictions(model, values, mask):
            loss = torch.nn.functional.mse_loss(prediction, target.index_select(0, indices))
            if not torch.isfinite(loss):
                raise FloatingPointError('CTF smoke loss is not finite')
            (loss * (indices.numel() / target.shape[0])).backward()
            total_loss += loss.item() * indices.numel() / target.shape[0]
            groups += 1
        optimizer.step()
        validation = score(model, protocol, 'validation', device, limit=1)
        print(json.dumps({'smoke': 'ok', 'loss': total_loss,
                          'groups': groups, 'validation': validation}))
        return

    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / 'ctf-selected.pt'
    selection = output / 'ctf-selection.json'
    if selection.exists():
        raise FileExistsError(f'CTF selection already finalized: {selection}')
    started = monotonic()
    best_epoch = 0
    best_validation = {'mae': float('inf'), 'mse': float('inf')}
    for epoch in range(100):
        model.train()
        running_loss = 0.0
        batches = 0
        for cpu_batch in iter_p4_batches(protocol, 'train', batch_size=128, epoch=epoch):
            batch = cpu_batch.to(device)
            values, mask = ctf_inputs(batch)
            target = batch.forecast_target.permute(0, 2, 1)
            optimizer.zero_grad(set_to_none=True)
            for indices, prediction in grouped_predictions(model, values, mask):
                loss = torch.nn.functional.mse_loss(prediction, target.index_select(0, indices))
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'CTF non-finite loss at epoch {epoch + 1}')
                weight = indices.numel() / target.shape[0]
                (loss * weight).backward()
                running_loss += loss.item() * weight
            optimizer.step()
            batches += 1
        validation = score(model, protocol, 'validation', device)
        if validation['mae'] < best_validation['mae']:
            best_validation = validation
            best_epoch = epoch + 1
            torch.save({'model_state': model.state_dict(), 'best_epoch': best_epoch}, checkpoint)
        print(json.dumps({'epoch': epoch + 1, 'train_loss': running_loss / batches,
                          'validation': validation, 'best_epoch': best_epoch}), flush=True)
    record = {
        'model': 'ChannelTokenFormer_missing', 'dataset': 'ETTh1', 'scenario': 'P4',
        'upstream_revision': upstream_revision(root),
        'official_source_sha256': official_source_hash(root),
        'adapter_sha256': file_sha256(Path(__file__)),
        'source_csv_sha256': protocol.source_sha256,
        'validation_mask_sha256': protocol.mask_sha256['validation'],
        'test_mask_sha256': protocol.mask_sha256['test'],
        'config_path': str(protocol.config_path),
        'checkpoint': str(checkpoint.resolve()),
        'checkpoint_sha256': file_sha256(checkpoint),
        'selection': 'earliest minimum fixed-validation forecast MAE',
        'best_epoch': best_epoch, 'epochs_completed': 100,
        'best_validation': best_validation,
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'batch_size': 128, 'learning_rate': 1e-4, 'seed': 2021,
        'patch_len': 24, 'keep_prob': 0.6,
        'loss': 'forecast_MSE',
        'mask_adapter': 'group equal full-patch visibility before upstream first-sample deletion',
        'metric_space': 'train_standardized',
        'training_seconds': monotonic() - started,
    }
    selection.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(f'CTF_SELECTION {selection}', flush=True)


def evaluate_selected(protocol, root: Path, output: Path, device: torch.device) -> None:
    selection = output / 'ctf-selection.json'
    evaluation = output / 'ctf-test.json'
    if evaluation.exists():
        raise FileExistsError(f'CTF test already finalized: {evaluation}')
    record = json.loads(selection.read_text(encoding='utf-8'))
    for key, expected in (
        ('source_csv_sha256', protocol.source_sha256),
        ('validation_mask_sha256', protocol.mask_sha256['validation']),
        ('test_mask_sha256', protocol.mask_sha256['test']),
    ):
        if record.get(key) != expected:
            raise ValueError(f'selected CTF {key} mismatch')
    if record.get('epochs_completed') != 100 or not 1 <= int(record.get('best_epoch', 0)) <= 100:
        raise ValueError('selected CTF run did not complete 100 epochs')
    if record['upstream_revision'] != upstream_revision(root) or record['official_source_sha256'] != official_source_hash(root):
        raise ValueError('official CTF source changed since training')
    if record['adapter_sha256'] != file_sha256(Path(__file__)):
        raise ValueError('CTF adapter changed since training')
    checkpoint = Path(record['checkpoint'])
    if file_sha256(checkpoint) != record['checkpoint_sha256']:
        raise ValueError('selected CTF checkpoint SHA-256 mismatch')
    model = load_official_ctf(root, device=device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload['model_state'], strict=True)
    if payload['best_epoch'] != record['best_epoch']:
        raise ValueError('selected CTF checkpoint epoch mismatch')
    replay = score(model, protocol, 'validation', device)
    for key in ('mse', 'mae'):
        if abs(replay[key] - record['best_validation'][key]) > 1e-7:
            raise ValueError(f'CTF selected validation {key} failed replay')
    test = score(model, protocol, 'test', device)
    result = {'model': 'ChannelTokenFormer_missing', 'dataset': 'ETTh1', 'scenario': 'P4',
              'selected_epoch': record['best_epoch'], 'validation': replay,
              'test': test, 'selection_sha256': file_sha256(selection),
              'checkpoint_sha256': record['checkpoint_sha256'],
              'source_csv_sha256': protocol.source_sha256,
              'test_mask_sha256': protocol.mask_sha256['test'],
              'metric_space': 'train_standardized'}
    evaluation.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(f'CTF_TEST {json.dumps(test)}', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--masks', type=Path, default=DEFAULT_MASKS)
    parser.add_argument('--upstream', type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--smoke', action='store_true')
    action.add_argument('--train', action='store_true')
    action.add_argument('--evaluate-selected', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    protocol = load_p4_protocol(args.config, args.masks)
    device = torch.device(args.device)
    if args.evaluate_selected:
        evaluate_selected(protocol, args.upstream, args.output, device)
    else:
        train(protocol, args.upstream, args.output, device, smoke=args.smoke)


if __name__ == '__main__':
    main()
