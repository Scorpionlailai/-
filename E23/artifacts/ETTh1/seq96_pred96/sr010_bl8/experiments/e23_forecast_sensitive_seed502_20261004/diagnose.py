"""Read-only fixed-validation interventions; original checkpoints and heads."""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from integrations.e23.config import load_e23_settings
from integrations.e23.forecast_sensitive import parameter_hash
from integrations.e23.train import (
    _key_sha256, _new_memory, _starts, build_forecast_model, load_stage2_selection,
)
from integrations.forecast_compare import author_provenance
from integrations.forecast_compare.author_profiles import build_author_model, load_author_profiles
from integrations.forecast_compare.protocol import load_p4_protocol, with_training_seed, iter_p4_batches
from integrations.forecast_compare.t1_p4 import T1ForecastModel, t1_inputs
from malr.data.patching import patchify_preserving_values
from scripts.run_author_aligned_p4 import _forecast_head

OUT = Path(__file__).resolve().parent
TRIALS = OUT.parent
FORMAL = TRIALS / 'e23_t1_five_seed_standard_20261004'
LEARNING = TRIALS / 'e23_v37_v39_t1_learning_seed502_20261004'
WORK = Path('/home/xyw/paper/.e23-v39-depth/E23')
STANDARD = Path('/home/xyw/paper/.e23-t1-standard/E23')
CANONICAL = Path('/home/xyw/paper/E23')
MASKS = CANONICAL.parent / 'E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context'
VARIABLES = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
LABELS = ['v31B', 'v38', 'T1']
GROUPS = [(c, t, t + 16) for t in range(0, 96, 16) for c in range(7)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def write_csv(path, rows):
    with Path(path).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def protect_sources():
    files = [OUT / 'diagnose.py', OUT / 'plan.json']
    for root in [WORK, STANDARD]:
        for directory in ['src', 'integrations', 'scripts', 'configs']:
            files.extend(p for p in (root / directory).rglob('*')
                         if p.is_file() and p.suffix in {'.py', '.json', '.yaml', '.yml'})
    files.extend(p for p in (CANONICAL / 'docs/experiments').glob('*')
                 if p.is_file() and p.suffix in {'.csv', '.md', '.json'})
    files.extend((CANONICAL / 'src').rglob('*.py'))
    files.extend((CANONICAL / 'integrations').rglob('*.py'))
    files.append(CANONICAL / 'configs/stage1_seq96_dynamic_temporal_repair.yaml')
    return {str(p): sha(p) for p in sorted(set(files))}


def load_models(protocol, device):
    models, memories, records, protected = {}, {}, {}, {}
    definitions = [
        ('v31B', FORMAL / 'configs/e23_seed_502.yaml',
         FORMAL / 'seed_502/E23-v31-B/stage2/runs/e23-stage2.json'),
        ('v38', LEARNING / 'v38/e23_seed502.yaml',
         LEARNING / 'v38/stage2/runs/e23-stage2.json'),
    ]
    for label, config, selection in definitions:
        settings = load_e23_settings(config, protocol=protocol)
        record = json.loads(selection.read_text())
        assert sha(Path(record['checkpoint'])) == record['checkpoint_sha256']
        payload = load_stage2_selection(selection, protocol, settings, device=device)
        model = build_forecast_model(protocol, settings, device=device)
        model.load_state_dict(payload['model_state'], strict=True)
        model.repair.historical.retrieval.load_state_dict(payload['retrieval_state'])
        memory = _new_memory(protocol)
        memory.load_state_dict(payload['memory_state'])
        model.repair.historical.refresh_snapshot(memory.snapshot())
        models[label] = model.eval().requires_grad_(False)
        memories[label] = memory
        records[label] = record
        for p in [config, selection, Path(record['checkpoint'])]:
            protected[str(p)] = sha(p)
        del payload
    profile_path = FORMAL / 'configs/author_seed_502.json'
    selection = FORMAL / 'seed_502/T1/training/forecast-selection.json'
    record = json.loads(selection.read_text())
    cfg = load_author_profiles(profile_path)['models']['T1']
    previous_root = author_provenance.ROOT
    author_provenance.ROOT = STANDARD
    try:
        identity = author_provenance.author_source_identity(profile_path, cfg, protocol)
    finally:
        author_provenance.ROOT = previous_root
    assert all(record.get(k) == v for k, v in identity.items()), (identity, record)
    assert record['status'] == 'validation_finalized' and record['seed'] == 502
    checkpoint = Path(record['checkpoint'])
    assert sha(checkpoint) == record['checkpoint_sha256']
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    assert all(payload.get(k) == v for k, v in identity.items())
    model = T1ForecastModel(build_author_model('T1', cfg, STANDARD / cfg['root'], device),
                            _forecast_head(protocol)).to(device)
    model.load_state_dict(payload['model_state'], strict=True)
    models['T1'] = model.eval().requires_grad_(False)
    records['T1'] = record
    for p in [profile_path, selection, checkpoint]:
        protected[str(p)] = sha(p)
    head_initial = {r['predictor_initial_sha256'] for r in records.values()}
    assert len(head_initial) == 1
    assert all(sum(p.numel() for p in m.predictor.parameters()) == 35427 for m in models.values())
    return models, memories, records, protected, identity


def state_identity(models, memories):
    result = {}
    for label, model in models.items():
        result[label] = {'model': parameter_hash(model)}
        if label in memories:
            result[label].update(key=_key_sha256(model.repair.historical.retrieval.state_dict()),
                                 memory=memories[label].snapshot_hash())
    return result


def add_metric(acc, prediction, target):
    error = prediction - target
    assert torch.isfinite(error).all()
    acc[0] += error.square().double().sum().item()
    acc[1] += error.abs().double().sum().item()
    acc[2] += error.numel()


@torch.inference_mode()
def run():
    if (OUT / 'diagnosis.json').exists():
        raise FileExistsError('completed diagnosis already exists')
    start = perf_counter()
    torch.set_num_threads(4)
    torch.manual_seed(502)
    device = torch.device('cuda:1')
    protocol = with_training_seed(load_p4_protocol(
        WORK / 'configs/stage1_seq96_dynamic_temporal_repair.yaml', MASKS), 502)
    protected = protect_sources()
    models, memories, records, extra, t1_identity = load_models(protocol, device)
    protected.update(extra)
    write_json(OUT / 'protected-files.json', protected)
    before = state_identity(models, memories)
    torch.cuda.reset_peak_memory_stats(device)

    times = torch.arange(12, device=device)[:, None] * 8 + torch.arange(16, device=device)[None]
    selectors = torch.zeros(42, 7, 12, 16, device=device, dtype=torch.bool)
    for g, (c, left, right) in enumerate(GROUPS):
        selectors[g, c] = (times >= left) & (times < right)
    assert torch.equal(selectors.sum(0), (times < 96)[None].expand(7, -1, -1).long())
    repair_sums = {s: np.zeros((42, 3), dtype=np.float64) for s in LABELS}
    baseline = {(s, h, mode): np.zeros(3) for s in LABELS for h in LABELS
                for mode in ['native', 'zero', 'replicate', 'oracle_native', 'oracle_zero']}
    interventions = [(s, h, 'oracle', 1.) for s in LABELS for h in LABELS]
    interventions += [(s, s, 'oracle_half', .5) for s in LABELS]
    interventions += [(s, s, 'T1_donor', 1.) for s in ['v31B', 'v38']]
    accumulated = {key: {'gain': [], 'mae': [], 'horizon': []} for key in interventions}
    counts_all, starts_all, repair_windows = [], [], {s: [] for s in LABELS}
    windows = coordinates = 0

    for batch_index, cpu in enumerate(iter_p4_batches(protocol, 'validation', batch_size=128)):
        b = cpu.to(device)
        B = len(b.normalized_observed)
        clean = patchify_preserving_values(b.normalized_target, b.observed_mask, 16, 8)
        clean_values = clean.values.masked_fill(~clean.validity, 0)
        observed, valid = clean.effective_mask, clean.validity
        hidden = ~observed & valid
        timeline_hidden = ~b.observed_mask
        counts = torch.stack([timeline_hidden[:, left:right, c].sum(1)
                              for c, left, right in GROUPS], 1)
        assert counts.sum().item() == timeline_hidden.sum().item()
        counts_all.append(counts.cpu().numpy())
        starts_all.extend(ref.target_start for ref in b.refs)
        native, zero, replicate, sequences = {}, {}, {}, {}
        for s, model in models.items():
            if s == 'T1':
                values, mask = t1_inputs(b)
                candidate = model.imputer(values, mask=mask)
                seq = (torch.where(b.observed_mask, values, candidate) - b.query_mean[:, None]) / b.query_std[:, None]
                seq = torch.where(b.observed_mask, b.normalized_observed, seq)
                native[s] = patchify_preserving_values(seq, b.observed_mask, 16, 8).values
            else:
                starts = _starts(b, stride=8, patches=12, device=device)
                x = model.repair(b.normalized_observed, b.observed_mask,
                                 b.query_mean, b.query_std, starts).explicit
                seq, native[s] = x.normalized_sequence, x.repaired_patches
                assert torch.equal(x.observation_patches, observed) and torch.equal(x.validity_patches, valid)
            assert torch.equal(native[s][observed], clean_values[observed])
            assert torch.equal(seq[b.observed_mask], b.normalized_observed[b.observed_mask])
            sequences[s] = seq
            zero[s] = native[s].masked_fill(~valid, 0)
            replicate[s] = torch.where(valid, native[s], seq[:, -1, :, None, None].expand_as(native[s]))
            error = (seq - b.normalized_target) * b.query_std[:, None]
            for g, (c, left, right) in enumerate(GROUPS):
                e = error[:, left:right, c][timeline_hidden[:, left:right, c]]
                repair_sums[s][g] += [e.square().double().sum().item(), e.abs().double().sum().item(), e.numel()]
            repair_windows[s].append((error.square() * timeline_hidden).sum((1, 2)).cpu().numpy()
                                     / timeline_hidden.sum((1, 2)).clamp_min(1).cpu().numpy())
        predictions = {}
        for s in LABELS:
            for h in LABELS:
                head = models[h].predictor
                def predict(x):
                    return head(x, observed, b.query_mean, b.query_std, patch_validity=valid)
                predictions[s, h] = predict(zero[s])
                for mode, x in [('native', native[s]), ('zero', zero[s]), ('replicate', replicate[s]),
                                ('oracle_native', torch.where(hidden, clean_values, native[s])),
                                ('oracle_zero', torch.where(hidden, clean_values, zero[s]))]:
                    prediction = predictions[s, h] if mode == 'zero' else predict(x)
                    add_metric(baseline[s, h, mode], prediction, b.forecast_target)
                    if mode == 'native' and s == h and batch_index == 0:
                        direct = models[s](b) if s == 'T1' else models[s](
                            b.normalized_observed, b.observed_mask, b.query_mean, b.query_std,
                            _starts(b, stride=8, patches=12, device=device))[0]
                        torch.testing.assert_close(prediction, direct, rtol=0, atol=0)
        for key in interventions:
            s, h, kind, strength = key
            head = models[h].predictor
            values = zero[s]
            donor = zero['T1'] if kind == 'T1_donor' else clean_values
            replacement = (values + (donor - values) * strength) if strength != 1 else donor
            base_error = predictions[s, h] - b.forecast_target
            base_mse = base_error.square().mean((1, 2))
            base_mae = base_error.abs().mean((1, 2))
            base_horizon = base_error.square().reshape(B, 7, 4, 24).mean((1, 3))
            gains, maes, horizons = [], [], []
            for left in range(0, 42, 7):
                selected = selectors[left:left+7, None] & hidden[None]
                variants = torch.where(selected, replacement[None], values[None])
                # Check observed coordinates, padding and all overlapping copies.
                assert torch.equal(variants.masked_select(~selected),
                                   values[None].expand_as(variants).masked_select(~selected))
                prediction = head(variants.reshape(7*B, 7, 12, 16), observed.repeat(7, 1, 1, 1),
                                  b.query_mean.repeat(7, 1), b.query_std.repeat(7, 1),
                                  patch_validity=valid.repeat(7, 1, 1, 1)).reshape(7, B, 7, 96)
                error = prediction - b.forecast_target[None]
                gain = (base_mse[None] - error.square().mean((2, 3))).T
                mae = (base_mae[None] - error.abs().mean((2, 3))).T
                horizon = (base_horizon[None] - error.square().reshape(7, B, 7, 4, 24).mean((2, 4))).permute(1, 0, 2)
                empty = counts[:, left:left+7] == 0
                if empty.any():
                    assert gain[empty].abs().max().item() < 2e-5
                gains.append(gain.masked_fill(empty, 0).cpu().numpy())
                maes.append(mae.masked_fill(empty, 0).cpu().numpy())
                horizons.append(horizon.masked_fill(empty[..., None], 0).cpu().numpy())
            for name, parts in [('gain', gains), ('mae', maes), ('horizon', horizons)]:
                array = np.concatenate(parts, 1)
                assert np.isfinite(array).all()
                accumulated[key][name].append(array)
        windows += B
        coordinates += b.forecast_target.numel()
        if batch_index % 5 == 0 or batch_index == 20:
            print(json.dumps({'validation_batches': batch_index+1, 'windows': windows,
                              'seconds': perf_counter()-start}), flush=True)

    assert windows == 2688 and coordinates == 1806336
    assert starts_all == list(range(8640, 11328))
    counts = np.concatenate(counts_all)
    assert counts.sum() == 1023670
    baseline_rows = []
    for (s, h, mode), acc in baseline.items():
        metric = dict(source=s, predictor=h, padding=mode, mse=float(acc[0]/acc[2]),
                      mae=float(acc[1]/acc[2]), windows=windows, forecast_coordinates=coordinates)
        baseline_rows.append(metric)
        if s == h and mode == 'native':
            expected = records[s]['best_validation'] if s == 'T1' else {
                'mse': records[s]['best_val_mse'], 'mae': records[s]['best_val_mae']}
            assert all(abs(metric[k]-expected[k]) < 1e-6 for k in ['mse', 'mae']), metric
    arrays = {'target_starts': np.asarray(starts_all), 'region_hidden_count': counts}
    region_rows = []
    for key, acc in accumulated.items():
        s, h, kind, strength = key
        gains, maes, horizons = [np.concatenate(acc[name]) for name in ['gain', 'mae', 'horizon']]
        prefix = f'{s}__{h}__{kind}'
        arrays[prefix+'__mse_gain'] = gains
        arrays[prefix+'__mae_gain'] = maes
        arrays[prefix+'__horizon_gain'] = horizons
        for g, (c, left, right) in enumerate(GROUPS):
            active = counts[:, g] > 0
            stats = repair_sums[s][g]
            region_rows.append(dict(source=s, predictor=h, intervention=kind, strength=strength,
                padding='zero', variable=VARIABLES[c], history_start=left, history_end_exclusive=right,
                hidden_coordinates=int(stats[2]), repair_mse=float(stats[0]/stats[2]),
                repair_mae=float(stats[1]/stats[2]), mean_mse_gain=float(gains[:, g].mean(dtype=np.float64)),
                mean_mae_gain=float(maes[:, g].mean(dtype=np.float64)),
                positive_fraction_active=float((gains[active, g] > 0).mean()), active_windows=int(active.sum()),
                **{f'horizon_{24*i+1}_{24*(i+1)}_mse_gain': float(horizons[:, g, i].mean(dtype=np.float64))
                   for i in range(4)}))
    repair_rows = []
    for s, stats in repair_sums.items():
        repair_rows.append(dict(model=s, mse=float(stats[:, 0].sum()/stats[:, 2].sum()),
                                mae=float(stats[:, 1].sum()/stats[:, 2].sum()),
                                hidden_coordinates=int(stats[:, 2].sum())))
        arrays[s+'__repair_mse'] = np.concatenate(repair_windows[s])
    assert before == state_identity(models, memories)
    changes = [p for p, digest in protected.items() if sha(p) != digest]
    assert not changes, changes
    author_provenance.ROOT = STANDARD
    try:
        assert author_provenance.author_source_identity(FORMAL/'configs/author_seed_502.json',
            records['T1']['profile'], protocol) == t1_identity
    finally:
        author_provenance.ROOT = WORK
    write_csv(OUT / 'baseline_head_padding.csv', baseline_rows)
    write_csv(OUT / 'position_interventions.csv', region_rows)
    write_csv(OUT / 'repair_metrics.csv', repair_rows)
    np.savez_compressed(OUT / 'per_window_gains.npz', **arrays)
    outputs = {name: sha(OUT/name) for name in ['baseline_head_padding.csv',
               'position_interventions.csv', 'repair_metrics.csv', 'per_window_gains.npz']}
    write_json(OUT / 'diagnosis.json', dict(status='complete', completed_utc=datetime.now(timezone.utc).isoformat(),
        training_seed=502, seed_scope='exploratory; seed selected from existing formal test results',
        split='original_fixed_validation', windows=windows, forecast_coordinates=coordinates,
        new_training=False, test_predictions_computed=False, checkpoint_selection_changed=False,
        metric_space='train_standardized', original_heads_frozen=True, predictor_parameters=35427,
        regions=42, region_combinations=len(interventions), position_rows=len(region_rows),
        signed_gains_retained=True, half_oracle_strength=.5, original_masks_and_query_stats_fixed=True,
        overlapping_windows_not_independent=True, oracle_not_a_guaranteed_bound=True,
        cross_swaps_diagnostic_only=True, real_world_causality_identified=False,
        regional_interventions_padding='all invalid padding fixed to zero to compare valid repaired content',
        native_padding_replay_and_padding_controls=True, source_csv_sha256=protocol.source_sha256,
        validation_mask_sha256=protocol.mask_sha256['validation'],
        checkpoint_sha256={s: r['checkpoint_sha256'] for s,r in records.items()},
        state_hashes_unchanged=before, protected_files_unchanged=len(protected),
        outputs=outputs, repair_metrics=repair_rows,
        diagnostic_seconds=perf_counter()-start, device=str(device), gpu=torch.cuda.get_device_name(device),
        peak_allocated_gib=torch.cuda.max_memory_allocated(device)/1024**3,
        peak_reserved_gib=torch.cuda.max_memory_reserved(device)/1024**3))
    print(json.dumps({'status': 'complete', 'repair': repair_rows, 'output': str(OUT)}, indent=2), flush=True)


if __name__ == '__main__':
    run()
