"""Stage 1 repair and frozen common PatchTST Stage 2, with auditable results."""
_PROCESS_STARTED = __import__('time').perf_counter()
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch

from .adapters import build_adapter, training_profile
from .data import ROOT, SCENES, SharedData, write_json, file_sha256


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def state_sha(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def resources(device):
    return dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), cpu_threads=torch.get_num_threads(),
                max_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30 if device.type == 'cuda' else 0.,
                max_reserved_gib=torch.cuda.max_memory_reserved(device) / 2**30 if device.type == 'cuda' else 0.)


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def metric(squared, absolute, count):
    return dict(mse=squared / count if count else None, mae=absolute / count if count else None,
                coordinates=count, squared_error_sum=squared, absolute_error_sum=absolute)


def aggregate(groups):
    result = {}
    for section in ('full96', 'recent32', 'forecast'):
        if section not in groups[0]:
            continue
        result[section] = {}
        for name in ('mse', 'mae'):
            values = np.array([g[section][name] for g in groups], dtype=float)
            result[section][name + '_mean'] = float(values.mean())
            result[section][name + '_mask_std_ddof1'] = float(values.std(ddof=1))
        count = sum(g[section]['coordinates'] for g in groups)
        result[section]['coordinate_weighted'] = metric(
            sum(g[section]['squared_error_sum'] for g in groups),
            sum(g[section]['absolute_error_sum'] for g in groups), count)
    return result


def repair(model, values, observed, starts, sampling_seed):
    # Fixed inference RNG is separate from optimization RNG. No hidden truth
    # is accepted here; CSDI's configured sample median is reproducible for each batch.
    if model.__class__.__name__ == 'CsdiAdapter':
        devices = [values.device.index or 0] if values.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(sampling_seed)
            return model(values, observed, starts).float()
    # Retain the author's mixed-FP16 inference for ImputeFormer while the
    # downstream shared PatchTST always receives FP32 values.
    with torch.cuda.amp.autocast(enabled=values.is_cuda and model.__class__.__name__ == 'ImputeFormerAdapter'):
        return model(values, observed, starts).float()


def forecast(head, repaired, observed, data, starts):
    from malr.data.normalization import TrainVariableStats, masked_window_normalize
    from malr.data.patching import patchify_preserving_values
    # Shared normalization uses only actual visible history, with the same
    # train-standardized fallback as the existing T1/SAITS forecasting bridge.
    train = torch.from_numpy(data.values[:data.bounds.train.end])
    if not hasattr(data, '_fallback'):
        from malr.data.normalization import fit_train_variable_stats
        data._fallback = fit_train_variable_stats(train, eps=0.)
    visible = torch.where(observed, repaired, 0.)
    normalized = masked_window_normalize(visible, observed, data._fallback, eps=1e-5)
    completed = (repaired - normalized.mean[:, None]) / normalized.std[:, None]
    patches = patchify_preserving_values(completed, observed, 16, 8)
    return head(patches.values, patches.effective_mask, normalized.mean, normalized.std,
                patch_validity=patches.validity)


@torch.no_grad()
def evaluate(model, data, split, device, *, batch_size, mask_seed=None, head=None, max_batches=None, cache=None):
    model.eval()
    if head is not None:
        head.eval()
    sums = {k: [0., 0., 0] for k in ('full96', 'recent32', 'forecast')}
    windows = 0
    batches = cache.batches(split, batch_size, mask_seed) if cache is not None else data.batches(split, batch_size, mask_seed=mask_seed, max_batches=max_batches)
    for inputs, observed, starts in batches:
        observed = observed.to(device)
        truth = torch.from_numpy(np.stack([data.values[s:s+96] for s in starts])).to(device)
        values = torch.where(observed, truth, 0.)
        completed = inputs.to(device) if cache is not None else repair(model, values, observed, starts, 910000 + (mask_seed or 2021) + int(starts[0]))
        if not torch.isfinite(completed).all():
            raise FloatingPointError('non-finite repair output')
        if not torch.equal(completed[observed], values[observed]):
            raise AssertionError('observed coordinates changed')
        for key, offset in (('full96', 0), ('recent32', 64)):
            valid = torch.from_numpy(data.valid_for(starts)).to(device)
            score = (~observed & valid)[:, offset:]
            error = (completed[:, offset:] - truth[:, offset:])[score].double()
            sums[key][0] += error.square().sum().item()
            sums[key][1] += error.abs().sum().item()
            sums[key][2] += error.numel()
        if head is not None:
            prediction = forecast(head, completed, observed, data, starts)
            future = torch.from_numpy(np.stack([data.values[s+96:s+192] for s in starts])).to(device).permute(0, 2, 1)
            future_valid = torch.from_numpy(data.valid_for(starts, offset=96)).to(device).permute(0, 2, 1)
            error = (prediction - future)[future_valid].double()
            sums['forecast'][0] += error.square().sum().item()
            sums['forecast'][1] += error.abs().sum().item()
            sums['forecast'][2] += error.numel()
        windows += len(starts)
    result = dict(split=split, mask_seed=mask_seed or 2021, windows=windows,
                  full96=metric(*sums['full96']), recent32=metric(*sums['recent32']))
    if head is not None:
        result['forecast'] = metric(*sums['forecast'])
    if max_batches is None and windows != len(data.starts[split]):
        raise AssertionError('evaluation window count differs from shared support')
    return result


def source_seal(name):
    directory = (ROOT / 'E23/versions/e23-v125-shared-calibration-20261009/E23') if name=='v125' else ROOT / name
    files = {str(p.relative_to(ROOT)): file_sha256(p) for p in directory.rglob('*.py')
             if '__pycache__' not in p.parts}
    for p in Path(__file__).parent.glob('*.py'):
        files[str(p.relative_to(ROOT))] = file_sha256(p)
    registry = json.loads((ROOT / 'E23/docs/experiments/repair_baseline_registry_20261010.json').read_text())
    return dict(files=files, registry=registry)


def train_stage1(args, model, data, output, device, profile):
    stage = output / 'stage1'
    stage.mkdir()
    optimizer = torch.optim.Adam(model.parameters(), lr=profile['learning_rate'], weight_decay=profile['weight_decay'])
    scheduler = None
    if profile['schedule'] == 'magic':
        from imputeformer.scheduler import CosineSchedulerWithRestarts
        scheduler = CosineSchedulerWithRestarts(optimizer, num_warmup_steps=12, num_training_steps=300,
                                                min_factor=.1, linear_decay=.67, num_cycles=3)
    elif profile['schedule'] == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=profile.get('milestones', [150, 180]), gamma=.1)
    amp = profile['precision'] == 'float16_mixed' and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    best, best_epoch, stale, steps = float('inf'), 0, 0, 0
    started = time.perf_counter()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(profile['epochs']):
        model.train()
        losses = []
        for truth, observed, starts in data.batches('train', profile['batch_size'], epoch=epoch,
                seed=profile['seed'], max_batches=profile['max_batches']):
            truth, observed = truth.to(device), observed.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                loss = model.loss(truth, observed, starts)
            if not torch.isfinite(loss):
                raise FloatingPointError('non-finite training loss')
            scaler.scale(loss).backward()
            if profile['grad_clip'] is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), profile['grad_clip'])
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
            steps += 1
        record = dict(epoch=epoch+1, steps=steps, train_loss=float(np.mean(losses)), learning_rate=optimizer.param_groups[0]['lr'])
        if profile['selection'] in ('mse', 'mae'):
            validation = evaluate(model, data, 'validation', device, batch_size=profile['batch_size'])
            value = validation['full96'][profile['selection']]
            record['validation'] = validation
            if value < best:
                best, best_epoch, stale = value, epoch+1, 0
                torch.save(model.state_dict(), stage / 'selected.pt')
            else:
                stale += 1
        else:
            # CSDI retains the author's final-epoch checkpoint policy.
            best_epoch = epoch+1
            torch.save(model.state_dict(), stage / 'selected.pt')
            if args.model == 'CSDI' and (epoch+1) % 20 == 0:
                model.eval()
                noise_sum, coordinates = 0., 0
                with torch.no_grad(), torch.random.fork_rng(devices=[device.index or 0] if device.type=='cuda' else []):
                    torch.manual_seed(720000)
                    for truth, observed, starts in data.batches('validation', profile['batch_size']):
                        truth, observed = truth.to(device), observed.to(device)
                        mask, side = model.side(truth, observed)
                        valid = torch.from_numpy(data.valid_for(starts)).to(device).permute(0,2,1).to(mask.dtype)
                        clean = torch.where(valid.bool(), truth.permute(0,2,1), 0.)
                        noise_loss = model.core.calc_loss_valid(clean, mask, valid, side, 0)
                        count = int(((~observed) & valid.permute(0,2,1).bool()).sum())
                        noise_sum += float(noise_loss) * count
                        coordinates += count
                record['validation_diffusion_noise_mse'] = noise_sum / coordinates
                record['validation_diffusion_loss_steps'] = 50
        synchronize(device)
        record['elapsed_seconds'] = time.perf_counter() - started
        with (stage / 'curve.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        write_json(output / 'status.json', dict(phase='stage1_training', **record))
        print(json.dumps(dict(dataset=data.dataset, scene=data.scene, model=args.model, **record)), flush=True)
        if profile['patience'] is not None and stale >= profile['patience']:
            break
        if scheduler is not None:
            scheduler.step()
        elif profile['schedule'] == 'type1':
            for group in optimizer.param_groups:
                group['lr'] = profile['learning_rate'] * .5 ** epoch
    synchronize(device)
    training_seconds = time.perf_counter() - started
    summary = dict(profile=profile, actual_epochs=epoch+1, optimizer_steps=steps,
                   selected_epoch=best_epoch, checkpoint_sha256=file_sha256(stage / 'selected.pt'),
                   training_seconds=training_seconds, resources=resources(device))
    write_json(stage / 'selection.json', summary)
    model.load_state_dict(torch.load(stage / 'selected.pt', map_location=device, weights_only=True))
    return summary


def build_head(data, device):
    from malr.models.patchtst import PatchTSTCore
    from malr.models.repaired_forecast import MaskAwareForecastModel
    seed_all(2023)
    return MaskAwareForecastModel(PatchTSTCore(c_in=data.n_vars, seq_len=96, pred_len=96,
        patch_len=16, stride=8, d_model=16, n_heads=4, d_ff=128, n_layers=3,
        dropout=.3, fc_dropout=.3, head_dropout=0., attn_dropout=0.)).to(device)


def train_stage2(args, model, data, output, device, cache):
    stage = output / 'stage2'
    stage.mkdir()
    model.requires_grad_(False).eval()
    head = build_head(data, device)
    initial_sha, repair_sha = state_sha(head), state_sha(model)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-4, weight_decay=0.)
    assert not {id(p) for p in model.parameters()} & {id(p) for g in optimizer.param_groups for p in g['params']}
    best = dict(mae=float('inf'), mse=float('inf'))
    selected = {}
    started, steps = time.perf_counter(), 0
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(100):
        head.train()
        model.eval()
        for completed, observed, starts in cache.batches('train', 128, epoch=epoch):
            completed, observed = completed.to(device), observed.to(device)
            prediction = forecast(head, completed, observed, data, starts)
            future = torch.from_numpy(np.stack([data.values[s+96:s+192] for s in starts])).to(device).permute(0, 2, 1)
            optimizer.zero_grad(set_to_none=True)
            future_valid = torch.from_numpy(data.valid_for(starts, offset=96)).to(device).permute(0,2,1)
            loss = (prediction - future)[future_valid].square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('non-finite forecast loss')
            loss.backward()
            if any(p.grad is not None for p in model.parameters()):
                raise AssertionError('repair gradient present during frozen stage2')
            optimizer.step()
            steps += 1
        validation = evaluate(model, data, 'validation', device, batch_size=128, head=head, cache=cache)
        if state_sha(model) != repair_sha:
            raise AssertionError('repair state changed during frozen stage2')
        for key in best:
            value = validation['forecast'][key]
            if value < best[key]:
                best[key], selected[key] = value, epoch+1
                torch.save(head.state_dict(), stage / f'selected_{key}.pt')
        record = dict(epoch=epoch+1, optimizer_steps=steps, validation=validation,
                      repair_state_sha256=repair_sha, elapsed_seconds=time.perf_counter()-started)
        with (stage / 'curve.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        write_json(output / 'status.json', dict(phase='stage2_training', **record))
        print(json.dumps(record), flush=True)
    expected_steps = len(data.starts['train']) // 128 * 100
    assert steps == expected_steps
    summary = dict(epochs=100, optimizer_steps=steps, loss='pure_forecast_mse', batch_size=128,
        learning_rate=1e-4, weight_decay=0., precision='float32', seed=2023,
        training_protocol='fixed_epoch0_masks_cached_repair', cache_records=cache.records,
        csdi_training_samples=1 if args.model == 'CSDI' else None,
        main_selection='validation_mae', secondary_selection='validation_mse', selected_epochs=selected,
        head_initial_sha256=initial_sha, repair_state_sha256=repair_sha,
        parent_stage1_sha256=file_sha256(output / 'stage1/selected.pt'),
        freeze_checks='optimizer_exclusion,requires_grad_false,eval,no_grad,state_sha_each_epoch',
        checkpoints={k: file_sha256(stage / f'selected_{k}.pt') for k in best},
        training_seconds=time.perf_counter()-started, resources=resources(device))
    write_json(stage / 'selection.json', summary)
    return head


def evaluate_bank(model, data, output, device, batch_size, *, head=None, label='repair', cache=None):
    if len(data.test_seeds) != 20 or len(set(data.test_seeds)) != 20:
        raise AssertionError('repair and forecast tests require 20 distinct missing-mask groups')
    if head is not None:
        if cache is None:
            raise ValueError('forecast testing requires stored repair outputs')
        # Check the entire bank before evaluation: forecasting must never generate
        # missing repair outputs implicitly through RepairCache.load.
        missing = [seed for seed in data.test_seeds
                   if not (cache.directory / f'test_{seed}' / 'manifest.json').is_file()]
        if missing:
            raise FileNotFoundError(f'forecast testing requires completed repair caches for masks: {missing}')
    folder = output / 'evaluation'
    folder.mkdir(exist_ok=True)
    groups = []
    synchronize(device)
    started = time.perf_counter()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    for seed in data.test_seeds:
        item = evaluate(model, data, 'test', device, batch_size=batch_size, mask_seed=seed, head=head, cache=cache)
        item['mask_sha256'] = next(x['sha256'] for x in data.manifest['masks'] if x['split']=='test' and x['seed']==seed)
        groups.append(item)
        write_json(folder / f'{label}_group_{seed}.json', item)
        write_json(output / 'status.json', dict(phase=f'{label}_test', completed_masks=len(groups), total_masks=20))
        print(json.dumps(dict(evaluation=label, **item)), flush=True)
    synchronize(device)
    record = dict(groups=groups, summary=aggregate(groups), averaging='unweighted_mean_of_20_coordinate_weighted_mask_metrics',
                  training_runs=1, training_seed=2023, test_seconds=time.perf_counter()-started, resources=resources(device))
    write_json(folder / f'{label}_summary.json', record)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['TimesNet', 'BRITS', 'ImputeFormer', 'CSDI', 'v125', 'T1'], required=True)
    parser.add_argument('--dataset', choices=['PAMAP2'], required=True)
    parser.add_argument('--scene', choices=SCENES, required=True)
    parser.add_argument('--phase', choices=['stage1', 'stage2'], required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.set_num_threads(4)
    seed_all(2023)
    device = torch.device(args.device)
    data = SharedData(args.dataset, args.scene, args.root / 'shared_assets')
    output = args.root / args.dataset / args.scene / args.model / 'seed_2023'
    output.mkdir(parents=True, exist_ok=True)
    if (output / args.phase).exists():
        raise FileExistsError(f'refusing to overwrite existing {args.phase}: {output}')
    total_started = _PROCESS_STARTED
    profile = training_profile(args.model)
    model = build_adapter(args.model, data, device)
    if args.phase == 'stage1':
        write_json(output / 'plan.json', dict(model=args.model, dataset=args.dataset, scene=args.scene,
            profile=profile, model_config=model.config, data_manifest_sha256=data.manifest_sha,
            source_seal=source_seal(args.model), model_parameters=sum(p.numel() for p in model.parameters()),
            adaptations=['shared_split_scaler_window_indices', 'shared_dynamic_block_mask',
                'visible_values_preserved', 'no_future_model_inputs',
                'TimesNet empty-variable count clamped to one' if args.model=='TimesNet' else 'task_dimensions'],
            github_policy='local_only_until_user_review'))
        if args.model == 'v125':
            from .references import train_v125
            training = train_v125(args, model, data, output, device, profile)
        else:
            training = train_stage1(args, model, data, output, device, profile)
        write_json(output / 'stage1/resources.json', dict(training=training,
            entry_through_training_seconds=time.perf_counter()-total_started))
        from .cache import RepairCache
        test_cache = RepairCache(model, data, output, device, profile['batch_size'])
        # All repair outputs, including training/validation, are sealed in
        # Stage1. Stage2 has no permission to generate additional repairs.
        test_cache.load('train')
        test_cache.load('validation')
        evaluate_bank(model, data, output, device, profile['batch_size'], cache=test_cache)
    else:
        authorization = args.root / 'protocol_accepted.json'
        preflight = args.root / f'preflight_{args.model}.json'
        if not authorization.exists() or not preflight.exists() or not json.loads(preflight.read_text()).get('passed'):
            raise RuntimeError('cached forecasting protocol requires recorded approval and passed preflight')
        model.load_state_dict(torch.load(output / 'stage1/selected.pt', map_location=device, weights_only=True))
        # Restore each completed run's sealed inference settings, including older CSDI jobs.
        plan = json.loads((output / 'plan.json').read_text())
        if args.model == 'CSDI':
            model.n_samples = plan['model_config']['sampling']['samples']
            model.sample_chunk_size = plan['model_config']['sampling']['sample_chunk_size']
        from .cache import RepairCache
        model.requires_grad_(False).eval()
        cache = RepairCache(model, data, output, device, profile['batch_size'], allow_generation=False)
        head = train_stage2(args, model, data, output, device, cache)
        for key in ('mae', 'mse'):
            head.load_state_dict(torch.load(output / f'stage2/selected_{key}.pt', map_location=device, weights_only=True))
            evaluate_bank(model, data, output, device, profile['batch_size'], head=head, label='forecast_'+key, cache=cache)
        write_json(output / 'stage2/cache_audit.json', dict(records=cache.records, total_generation_seconds=sum(r['generation_seconds'] for r in cache.records.values()), test_mask_groups=20, reuse='same cached repairs for both forecast checkpoint selections'))
        if state_sha(model) != json.loads((output / 'stage2/selection.json').read_text())['repair_state_sha256']:
            raise AssertionError('repair state changed during evaluation')
    write_json(output / 'status.json', dict(phase=args.phase+'_complete', total_seconds=time.perf_counter()-total_started))


if __name__ == '__main__':
    main()
