"""Read-only round diagnostics on the original fixed validation split."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from integrations.e23.config import load_e23_p4_protocol, load_e23_settings
from integrations.e23.train import (
    _key_sha256, _new_memory, _starts, build_repair_model, build_forecast_model,
    load_stage1_selection, load_stage2_selection,
)
from integrations.forecast_compare.protocol import iter_p4_batches
from malr.data.patching import patchify_with_state
from malr.models.explicit_repair import build_explicit_repair
from malr.migration import MigrationCatalog


def decode_rounds(model, output, values, mask, mean, std, *, valid_mask=None):
    if output.iterations is None:
        raise ValueError('round diagnostics require iterative recovery')
    patches = patchify_with_state(values, mask, model.patch_length, model.stride, validity=valid_mask)
    original = patches.values.masked_fill(~patches.effective_mask, 0)
    return tuple(build_explicit_repair(
        model.decoder(item.latents), original, patches.effective_mask, patches.validity,
        mean, std, window_len=model.window_length, patch_len=model.patch_length, stride=model.stride,
    ).normalized_sequence for item in output.iterations)


@torch.inference_mode()
def diagnose(model, protocol, settings, device):
    model.eval()
    sums = [dict(squared=0., absolute=0., update=0.,
                 route=torch.zeros(4, dtype=torch.float64),
                 gates=torch.zeros(2, dtype=torch.float64)) for _ in range(settings.recovery_rounds)]
    count = targets = windows = 0
    for cpu in iter_p4_batches(protocol, 'validation', batch_size=settings.batch_size):
        batch = cpu.to(device)
        starts = _starts(batch, stride=model.stride, patches=model.patch_count, device=device)
        output = model(batch.normalized_observed, batch.observed_mask, batch.query_mean, batch.query_std, starts)
        sequences = decode_rounds(model, output, batch.normalized_observed, batch.observed_mask,
                                  batch.query_mean, batch.query_std)
        torch.testing.assert_close(sequences[-1], output.explicit.normalized_sequence, rtol=0, atol=0)
        hidden = ~batch.observed_mask
        damaged = output.explicit.observation_patches.logical_not() & output.explicit.validity_patches
        damaged = damaged.any(-1)
        for r, sequence, item in zip(sums, sequences, output.iterations, strict=True):
            error = (sequence - batch.normalized_target) * batch.query_std[:, None, :]
            r['squared'] += error.square()[hidden].sum().item()
            r['absolute'] += error.abs()[hidden].sum().item()
            r['update'] += item.update_norm[damaged].sum().item()
            r['route'] += item.weights[damaged].double().sum(0).cpu()
            r['gates'] += item.preview_gates[damaged].double().sum(0).cpu()
        count += hidden.sum().item(); targets += damaged.sum().item(); windows += len(batch.normalized_observed)
    return dict(windows=windows, hidden_coordinates=count, damaged_patch_targets=targets,
                rounds=[dict(round_index=i, global_mse=r['squared']/count, global_mae=r['absolute']/count,
                             mean_update_norm=r['update']/targets,
                             mean_route_weights=(r['route']/targets).tolist(),
                             mean_preview_gates=(r['gates']/targets).tolist()) for i, r in enumerate(sums)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--e23-config', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    repo = Path(__file__).resolve().parents[1]
    protocol = load_e23_p4_protocol(
        catalog=MigrationCatalog.load(repo.parent/'E22/migration/paper_catalog.jsonl', project='E23'),
        protocol_config=repo/'configs/stage1_seq96_dynamic_temporal_repair.yaml',
        mask_artifacts=repo.parent/'E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context')
    settings = load_e23_settings(args.e23_config, protocol=protocol)
    device = torch.device(args.device)
    result = dict(split='original_fixed_validation', metric_space='train_standardized',
                  diagnostic_only=True, recovery_rounds=settings.recovery_rounds,
                  route_order=['none', 'historical', 'variable', 'temporal'], preview_order=['temporal', 'variable'])
    for stage in ('stage1', 'stage2'):
        manifest = args.root/stage/'runs'/f'e23-{stage}.json'
        record = json.loads(manifest.read_text())
        if stage == 'stage1':
            payload = load_stage1_selection(manifest, protocol, settings, device=device)
            model = build_repair_model(protocol, settings, device=device)
        else:
            payload = load_stage2_selection(manifest, protocol, settings, device=device)
            model = build_forecast_model(protocol, settings, device=device)
        model.load_state_dict(payload['model_state'], strict=True)
        repair = model if stage == 'stage1' else model.repair
        repair.historical.retrieval.load_state_dict(payload['retrieval_state'])
        memory = _new_memory(protocol); memory.load_state_dict(payload['memory_state'])
        repair.historical.refresh_snapshot(memory.snapshot())
        before, bank = _key_sha256(repair.historical.retrieval.state_dict()), memory.snapshot_hash()
        if before != record['key_sha256']:
            raise ValueError('selected H key identity mismatch')
        metrics = diagnose(repair, protocol, settings, device)
        after, bank_after = _key_sha256(repair.historical.retrieval.state_dict()), memory.snapshot_hash()
        if before != after or bank != bank_after:
            raise ValueError('diagnosis changed historical state')
        if stage == 'stage1' and abs(metrics['rounds'][-1]['global_mse']-record['best_val_repair_mse']) > 1e-6:
            raise ValueError('diagnosed final Stage1 repair differs from selected manifest')
        metrics.update(checkpoint_sha256=record['checkpoint_sha256'], selected_epoch=record['best_epoch'],
                       h_key_sha256_before=before, h_key_sha256_after=after,
                       h_bank_sha256_before=bank, h_bank_sha256_after=bank_after)
        result[stage] = metrics
        result['repair_parameters'] = sum(p.numel() for p in repair.parameters())
        del model, repair, memory, payload
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
