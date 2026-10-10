"""Original fixed-validation repair, forecast and evidence-refresh diagnostics."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import torch
from scripts.diagnose_e23_v30_stack import load_protocol
from scripts.diagnose_e23_v29_rounds import decode_rounds
from integrations.e23.config import load_e23_settings
from integrations.e23.train import (load_stage1_selection, load_stage2_selection,
    build_repair_model, build_forecast_model, _new_memory, _key_sha256, _starts)
from integrations.forecast_compare.protocol import iter_p4_batches
from integrations.e23.round_supervision import decode_repair_rounds
from malr.data.patching import patchify_preserving_values


@torch.inference_mode()
def diagnose(model, protocol, settings, device, stage):
    model.eval(); repair = model if stage == 'stage1' else model.repair
    totals = [dict(sq=0., ab=0., fsq=0., fab=0., gate=0., h_changed=0, v_changed=0)
              for _ in range(3)]
    hidden_count = windows = forecast_count = damaged_count = 0
    for cpu in iter_p4_batches(protocol, 'validation', batch_size=settings.batch_size):
        b = cpu.to(device)
        starts = _starts(b, stride=8, patches=12, device=device)
        out = repair(b.normalized_observed, b.observed_mask, b.query_mean, b.query_std, starts)
        sequences = decode_repair_rounds(repair, out, b)
        hidden = ~b.observed_mask
        damaged = (~out.explicit.observation_patches & out.explicit.validity_patches).any(-1)
        previous_h = previous_v = None
        for acc, item, sequence in zip(totals, out.iterations, sequences):
            error = ((sequence - b.normalized_target) * b.query_std[:, None, :])[hidden].double()
            acc['sq'] += error.square().sum().item(); acc['ab'] += error.abs().sum().item()
            gate = torch.ones_like(damaged, dtype=sequence.dtype) if item.update_gate is None else item.update_gate
            acc['gate'] += gate[damaged].double().sum().item()
            h = item.historical_starts.masked_fill(~item.historical_valid, -1).sort(-1).values
            v = item.variable_sources.sort(-1).values
            if previous_h is not None:
                acc['h_changed'] += ((h != previous_h).any(-1) & damaged).sum().item()
                acc['v_changed'] += ((v != previous_v).any(-1) & damaged).sum().item()
            previous_h, previous_v = h, v
            if stage == 'stage2':
                patches = patchify_preserving_values(sequence, b.observed_mask, 16, 8)
                prediction = model.predictor(patches.values.masked_fill(~patches.validity, 0), patches.effective_mask,
                    b.query_mean, b.query_std, patch_validity=patches.validity)
                fe = (prediction - b.forecast_target).double()
                acc['fsq'] += fe.square().sum().item(); acc['fab'] += fe.abs().sum().item()
        hidden_count += hidden.sum().item(); damaged_count += damaged.sum().item()
        windows += len(b.normalized_observed); forecast_count += b.forecast_target.numel()
    rows = []
    for i, acc in enumerate(totals, 1):
        row = dict(round_index=i, global_mse=acc['sq']/hidden_count, global_mae=acc['ab']/hidden_count,
                   mean_update_gate=acc['gate']/damaged_count,
                   historical_candidate_set_change_fraction=acc['h_changed']/damaged_count,
                   variable_source_set_change_fraction=acc['v_changed']/damaged_count)
        if stage == 'stage2':
            row.update(forecast_mse=acc['fsq']/forecast_count, forecast_mae=acc['fab']/forecast_count)
        rows.append(row)
    return dict(rounds=rows, windows=windows, hidden_coordinates=hidden_count,
                forecast_coordinates=forecast_count, damaged_patch_targets=damaged_count)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--cell', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True); p.add_argument('--device', default='cuda:1')
    a = p.parse_args(); protocol = load_protocol(); settings = load_e23_settings(a.config, protocol=protocol)
    device = torch.device(a.device); results = dict(split='original_fixed_validation', metric_space='train_standardized')
    for stage in ('stage1', 'stage2'):
        started = perf_counter(); path = a.cell/stage/'runs'/f'e23-{stage}.json'
        record = json.loads(path.read_text())
        if stage == 'stage1':
            payload = load_stage1_selection(path, protocol, settings, device=device)
            model = build_repair_model(protocol, settings, device=device); repair = model
        else:
            payload = load_stage2_selection(path, protocol, settings, device=device)
            model = build_forecast_model(protocol, settings, device=device); repair = model.repair
        model.load_state_dict(payload['model_state'], strict=True)
        repair.historical.retrieval.load_state_dict(payload['retrieval_state'])
        memory = _new_memory(protocol); memory.load_state_dict(payload['memory_state'])
        repair.historical.refresh_snapshot(memory.snapshot())
        before = (_key_sha256(repair.historical.retrieval.state_dict()), memory.snapshot_hash())
        result = diagnose(model, protocol, settings, device, stage)
        if before != (_key_sha256(repair.historical.retrieval.state_dict()), memory.snapshot_hash()):
            raise ValueError('validation diagnosis changed historical state')
        if stage == 'stage1' and abs(result['rounds'][-1]['global_mse'] - record['best_val_repair_mse']) > 1e-6:
            raise ValueError('final Stage1 repair replay mismatch')
        if stage == 'stage2' and any(abs(result['rounds'][-1]['forecast_'+k] - record['best_val_'+k]) > 1e-6 for k in ('mse','mae')):
            raise ValueError('final forecast replay mismatch')
        result.update(checkpoint_sha256=record['checkpoint_sha256'], e23_config_sha256=record['e23_config_sha256'],
            repair_parameters=sum(p.numel() for p in repair.parameters()),
            predictor_parameters=0 if stage == 'stage1' else sum(p.numel() for p in model.predictor.parameters()),
            h_key_sha256_before=before[0], h_key_sha256_after=before[0],
            h_bank_sha256_before=before[1], h_bank_sha256_after=before[1], diagnostic_seconds=perf_counter()-started)
        results[stage] = result
        del model, repair, memory, payload
    with (a.cell/'round_diagnostics.json').open('x') as stream:
        json.dump(results, stream, indent=2); stream.write('\n')
    print(json.dumps(results))


if __name__ == '__main__': main()
