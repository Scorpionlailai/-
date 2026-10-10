"""Throwaway, train-lookback-only candidate diagnosis; no training or model edits."""
from pathlib import Path
from itertools import islice
import hashlib
import json
import os
import numpy as np
import torch
from integrations.e23.config import load_e23_p4_protocol, load_e23_settings
from integrations.e23.train import _starts, _new_memory, _key_sha256, build_repair_model, load_stage1_selection
from integrations.forecast_compare.protocol import iter_p4_batches
from malr.data.patching import patchify_preserving_values
from malr.migration import MigrationCatalog

WORK = Path('/home/xyw/paper/.e23-v69-other-missing/E23')
BASE = Path('/home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments')
OUT = BASE / 'e23_h_candidate_quality_probe_seed2023_20261005'
SCENE = BASE / 't1_v41_higher_missing_seed502_20261004/sr015_bl8/scenario.json'
SELECTION = BASE / 'e23_v69_signature_position_sr015_bl8_20261005/seed_2023_shared/stage1/runs/e23-stage1.json'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    os.environ['MALR_MISSING_SCENARIO'] = str(SCENE)
    torch.set_num_threads(4)
    torch.manual_seed(2023)
    device = torch.device('cuda:1')
    protocol = load_e23_p4_protocol(
        catalog=MigrationCatalog.load(WORK.parent / 'E22/migration/paper_catalog.jsonl', project='E23'),
        protocol_config=WORK / 'configs/stage1_seq96_dynamic_temporal_repair.yaml',
        mask_artifacts=WORK.parent / 'E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context',
        training_seed=2023,
    )
    settings = load_e23_settings(WORK / 'configs/e23_v69_signature_position_seed2023.yaml', protocol=protocol)
    payload = load_stage1_selection(SELECTION, protocol, settings, device=device)
    repair = build_repair_model(protocol, settings, device=device)
    repair.load_state_dict(payload['model_state'], strict=True)
    repair.historical.retrieval.load_state_dict(payload['retrieval_state'])
    memory = _new_memory(protocol)
    memory.load_state_dict(payload['memory_state'])
    repair.historical.refresh_snapshot(memory.snapshot())
    repair.eval()
    before = memory.snapshot_hash(), _key_sha256(repair.historical.retrieval.state_dict()), sha(SELECTION)
    capture = {}
    original = repair.historical.prepare_candidates

    def observe(inputs):
        cache = original(inputs)
        capture.update(inputs=inputs, cache=cache)
        return cache

    repair.historical.prepare_candidates = observe
    methods = ['legacy_weighted', 'visible_best', 'key_best', 'oracle_best',
               'observed_mean_fallback', 'ridge_offset_same_weights', 'ridge_offset_reranked']
    sums = {region: {name: [0., 0] for name in methods} for region in ('all96', 'last32')}
    records = []
    available = damaged = fully_hidden = candidates_n = 0
    exact_visible_oracle = exact_key_oracle = competing = 0
    h_router_sum = h_point_sum = router_count = point_count = 0
    windows = 0
    with torch.inference_mode():
        for cpu in islice(iter_p4_batches(protocol, 'train', batch_size=128, epoch=89), 8):
            batch = cpu.to(device)
            starts = _starts(batch, stride=repair.stride, patches=repair.patch_count, device=device)
            out = repair(batch.normalized_observed, batch.observed_mask, batch.query_mean, batch.query_std, starts)
            inp, cache = capture['inputs'], capture['cache']
            truth = batch.normalized_target * batch.query_std[:, None] + batch.query_mean[:, None]
            patches = patchify_preserving_values(truth, batch.observed_mask, 16, 8)
            hidden = patches.validity & ~patches.effective_mask
            observed = patches.effective_mask
            usable = cache.available & hidden.any(-1)
            available += usable.sum().item()
            damaged += inp.damaged.sum().item()
            fully_hidden += (inp.damaged & ~observed.any(-1)).sum().item()
            windows += len(cpu.refs)
            if cache.empty or not usable.any():
                continue
            raw = cache.provenance['candidate_raw'].to(truth.dtype)
            valid = cache.valid
            weights = out.evidence['historical'].candidates.weights
            cand_error = ((raw - patches.values[..., None, :]).square() * hidden[..., None, :]).sum(-1) / hidden.sum(-1)[..., None].clamp_min(1)
            cand_error = cand_error.masked_fill(~valid, torch.inf)
            oracle = cand_error.argmin(-1)
            visible_best = cache.provenance['visible_error'].masked_fill(~valid, torch.inf).argmin(-1)
            key_best = cache.provenance['key_similarity'].masked_fill(~valid, -torch.inf).argmax(-1)
            many = usable & (valid.sum(-1) >= 2)
            competing += many.sum().item()
            exact_visible_oracle += ((visible_best == oracle) & many).sum().item()
            exact_key_oracle += ((key_best == oracle) & many).sum().item()

            def pick(index):
                return raw.gather(-2, index[..., None, None].expand(*index.shape, 1, 16)).squeeze(-2)

            n = observed.sum(-1)[..., None]
            difference = (patches.values[..., None, :] - raw).masked_fill(~observed[..., None, :], 0)
            delta = difference.sum(-1) / (n + 4)
            limit = 2 * torch.sqrt(inp.query_std.square() + .1 ** 2)[:, :, None, None]
            delta = torch.maximum(torch.minimum(delta, limit), -limit)
            adjusted = raw + delta[..., None]
            global_visible_error = (adjusted - patches.values[..., None, :]).abs().masked_fill(~observed[..., None, :], 0).sum(-1) / n.clamp_min(1)
            rerank_score = (cache.provenance['key_similarity'] - global_visible_error / inp.query_std[:, :, None, None]).masked_fill(~valid, -torch.inf)
            adjusted_weights = torch.softmax(rerank_score, -1).masked_fill(~valid, 0)
            adjusted_weights = torch.where(usable[..., None], adjusted_weights, 0)
            estimates = dict(
                legacy_weighted=(raw * weights[..., None]).sum(-2),
                visible_best=pick(visible_best), key_best=pick(key_best), oracle_best=pick(oracle),
                observed_mean_fallback=inp.query_mean[:, :, None, None].expand_as(patches.values),
                ridge_offset_same_weights=(adjusted * weights[..., None]).sum(-2),
                ridge_offset_reranked=(adjusted * adjusted_weights[..., None]).sum(-2),
            )
            positions = torch.arange(12, device=device)[:, None] * 8 + torch.arange(16, device=device)[None, :]
            for region in sums:
                chosen = hidden & usable[..., None]
                if region == 'last32':
                    chosen = chosen & (positions[None, None] >= 64)
                for name, estimate in estimates.items():
                    e = (estimate - patches.values)[chosen].double()
                    sums[region][name][0] += e.square().sum().item()
                    sums[region][name][1] += e.numel()
            h_router_sum += out.router.weights[..., 1][usable].double().sum().item()
            router_count += usable.sum().item()
            attention = out.point_candidate_attention['weights']
            h_mass = attention[..., 1:5].sum(-1)
            h_chosen = (hidden & usable[..., None])[..., None, :].expand_as(h_mass)
            h_point_sum += h_mass[h_chosen].double().sum().item()
            point_count += h_chosen.sum().item()
            candidates_n += valid[usable].sum().item()
            records.append(torch.stack((
                cache.provenance['visible_error'][usable].min(-1).values,
                cache.provenance['key_similarity'][usable].max(-1).values,
                cand_error[usable].min(-1).values,
                inp.patch_mask.sum(-1)[usable].to(truth.dtype),
            ), -1).cpu().numpy())
    assert before == (memory.snapshot_hash(), _key_sha256(repair.historical.retrieval.state_dict()), sha(SELECTION))
    data = np.concatenate(records)
    correlation = np.corrcoef(data[:, :3].T).tolist()
    result = dict(
        purpose='train-only candidate diagnosis; numerical estimates are not end-to-end model results',
        source_work=str(WORK), scenario=str(SCENE), training_seed=2023, split='train', mask_epoch=89,
        windows=windows, sampled_shuffled_batches=8, stage1_selection=str(SELECTION), selection_sha256=sha(SELECTION),
        metric_space='train_standardized', counting='patch copies, including overlaps; same coordinates per method',
        h_available_damaged_patches=available, damaged_patches=damaged, fully_hidden_patches=fully_hidden,
        average_candidates=candidates_n / available,
        candidate_raw_mse={region: {method: v[0] / v[1] for method, v in methods_.items()} for region, methods_ in sums.items()},
        counted_coordinates={region: next(iter(methods_.values()))[1] for region, methods_ in sums.items()},
        multi_candidate_patches=competing, visible_best_matches_oracle=exact_visible_oracle / competing,
        key_best_matches_oracle=exact_key_oracle / competing,
        correlations_visible_error_key_similarity_best_hidden_mse=correlation,
        mean_h_router_weight=h_router_sum / router_count, mean_h_point_attention=h_point_sum / point_count,
        state_unchanged=True, forecast_targets_read=False, validation_or_test_labels_read=False,
        offset_probe='mean residual with four virtual prior observations; clamp two robust local scales',
        candidate_per_patch_array_columns=['minimum_visible_error_local', 'max_key_similarity', 'minimum_hidden_mse_global', 'real_visible_count'],
    )
    OUT.mkdir()
    with (OUT / 'diagnosis.json').open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    np.savez_compressed(OUT / 'train-candidate-summary.npz', data=data)
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
