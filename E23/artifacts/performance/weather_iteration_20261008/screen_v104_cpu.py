"""Small, validation-only Stage1 objective screen on the v97 architecture.

The two repeated training mask banks are for screening only. They do not replace
the epoch-varying masks of a later full training run. No future target is cached.
"""
from pathlib import Path
from types import SimpleNamespace
from dataclasses import fields
from time import perf_counter
import copy
import hashlib
import itertools
import json
import os
import sys

R = Path(__file__).resolve().parent
W = Path('/home/xyw/paper/E23/artifacts/Weather/seq96_pred96/sr010_bl8/e23_v97_weather_cached_20261008')
sys.path.insert(0, str(W))
from weather_common import protocol_for
from train_refinement import coarse_hash, bank_content
import torch
from integrations.e23.config import load_e23_settings
from integrations.e23.train import build_repair_model, load_stage1_selection, _new_memory, _starts
from integrations.e23.repair_validation_cache import explicit_to
from integrations.forecast_compare.protocol import iter_p4_batches

EPOCHS = 20
BATCHES_PER_BANK = 8
CANDIDATES = [('v97_loss_control', 0.0), ('v104_A_full_mae_0p2', 0.2),
              ('v104_B_full_mae_0p5', 0.5)]


def suffix(refiner, row):
    explicit, latents, points, mean, std, target, mask = row
    output, _ = refiner(explicit, SimpleNamespace(latents=latents), points, mean, std)
    assert torch.equal(output.normalized_sequence[mask], explicit.observed_sequence[mask])
    return output, (output.normalized_sequence-target)*std[:, None]


@torch.no_grad()
def score(refiner, rows):
    refiner.eval()
    sums = {prefix: {key: torch.zeros(21, dtype=torch.float64)
                    for key in ('squared', 'absolute', 'signed', 'count')}
            for prefix in ('full', 'tail')}
    for row in rows:
        output, error = suffix(refiner, row)
        error = error.double()
        hidden = ~row[-1] & output.sequence_validity
        for prefix, length in (('full', 96), ('tail', 32)):
            e, h = error[:, -length:], hidden[:, -length:]
            sums[prefix]['squared'] += (e.square()*h).sum((0, 1))
            sums[prefix]['absolute'] += (e.abs()*h).sum((0, 1))
            sums[prefix]['signed'] += (e*h).sum((0, 1))
            sums[prefix]['count'] += h.sum((0, 1))
    metrics = {}
    per_variable = {}
    for prefix, values in sums.items():
        for label, key in (('mse', 'squared'), ('mae', 'absolute'), ('signed_bias', 'signed')):
            metrics[prefix+'_'+label] = float(values[key].sum()/values['count'].sum())
            per_variable[prefix+'_'+label] = (values[key]/values['count']).tolist()
    return metrics, per_variable


def main():
    os.nice(10)
    torch.set_num_threads(4)
    torch.manual_seed(2023)
    started = perf_counter()
    device = torch.device('cpu')
    plan = json.loads((W/'plan.json').read_text())
    for path, expected in json.loads((W/'source_freeze.json').read_text())['files'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
    protocol = protocol_for(plan, plan['scenarios'][0])
    settings = load_e23_settings(W/'v97_weather.yaml', protocol=protocol, allow_variable_scaling=True)
    parent_settings = load_e23_settings(Path(plan['parent_settings']), protocol=protocol, allow_variable_scaling=True)
    payload = load_stage1_selection(Path(plan['parent_selection']), protocol, parent_settings, device=device)
    model = build_repair_model(protocol, settings, device=device)
    missing = model.load_state_dict(payload['model_state'], strict=False)
    assert not missing.unexpected_keys and all('.global_dense_refiner.' in name for name in missing.missing_keys)
    model.historical.retrieval.load_state_dict(payload['retrieval_state'])
    memory = _new_memory(protocol)
    memory.load_state_dict(payload['memory_state'])
    model.historical.refresh_snapshot(memory.snapshot())
    model.eval().requires_grad_(False)
    prefix_hash, memory_hash = coarse_hash(model), bank_content(memory)
    refiner = model.recent_mean_projector
    initial_state = copy.deepcopy(refiner.state_dict())
    captures = []

    def capture(module, args):
        explicit, router, points, mean, std = args
        captures.append((explicit_to(explicit, 'cpu'), router.latents.detach(),
                         points.detach(), mean.detach(), std.detach()))

    hook = refiner.register_forward_pre_hook(capture)
    train_rows, validation_rows, bank_identity = [], [], []
    try:
        with torch.no_grad():
            for epoch in (0, 1):
                rows, digest, ids = [], hashlib.sha256(), []
                batches = itertools.islice(iter_p4_batches(protocol, 'train', batch_size=128, epoch=epoch), BATCHES_PER_BANK)
                for i, batch in enumerate(batches, 1):
                    starts = _starts(batch, stride=model.stride, patches=model.patch_count, device=device)
                    full = model(batch.normalized_observed, batch.observed_mask,
                                 batch.query_mean, batch.query_std, starts)
                    assert len(captures) == 1
                    row = (*captures.pop(), batch.normalized_target, batch.observed_mask)
                    # Disable the capture hook while comparing the exact suffix invocation.
                    hook.remove()
                    cached, _ = suffix(refiner, row)
                    assert torch.equal(cached.normalized_sequence, full.explicit.normalized_sequence)
                    hook = refiner.register_forward_pre_hook(capture)
                    rows.append(row)
                    digest.update(batch.observed_mask.contiguous().numpy().tobytes())
                    ids.extend(ref.target_start for ref in batch.refs)
                    print(f'cache train bank={epoch} batch={i}/{BATCHES_PER_BANK}', flush=True)
                train_rows.append(rows)
                bank_identity.append(dict(epoch=epoch, windows=len(ids), sample_target_starts=ids,
                                          observed_mask_sha256=digest.hexdigest()))
            for i, batch in enumerate(iter_p4_batches(protocol, 'validation', batch_size=128), 1):
                starts = _starts(batch, stride=model.stride, patches=model.patch_count, device=device)
                model(batch.normalized_observed, batch.observed_mask, batch.query_mean, batch.query_std, starts)
                assert len(captures) == 1
                validation_rows.append((*captures.pop(), batch.normalized_target, batch.observed_mask))
                print(f'cache full fixed validation batch={i}/41', flush=True)
    finally:
        hook.remove()
    assert coarse_hash(model) == prefix_hash and bank_content(memory) == memory_hash
    preparation_seconds = perf_counter()-started
    initial, initial_variables = score(refiner, validation_rows)
    initial_rng = torch.get_rng_state().clone()
    metadata = dict(schema='v97_stage1_suffix_cpu_screen_v1',
                    source_lineage='ETTh1 v97/v101/v102; Weather parent coarse weights',
                    parent_selection=plan['parent_selection'], frozen_prefix_sha256=prefix_hash,
                    bank_content_sha256=memory_hash, validation_mask_sha256=protocol.mask_sha256['validation'],
                    dataset_manifest=plan['dataset_manifest'], scope='screen only: two repeated 1024-window training mask banks; '
                    'full fixed validation; no future targets and no test masks; requires fresh formal training',
                    training_banks=bank_identity, validation_windows=len(protocol.refs['validation']),
                    preparation_seconds=preparation_seconds, initial_metrics=initial,
                    initial_per_variable=initial_variables, candidates=CANDIDATES,
                    epochs_per_candidate=EPOCHS, batch_size=128, lr=1e-4, stage2_unchanged=True)
    torch.save(dict(train_rows=train_rows, validation_rows=validation_rows,
                    initial_state=initial_state, initial_rng=initial_rng, metadata=metadata),
               R/'v97_cpu_screen_inputs.pt')
    (R/'v104_screen_design.json').write_text(json.dumps(metadata, indent=2))
    results = []
    for name, full_mae_weight in CANDIDATES:
        refiner.load_state_dict(initial_state, strict=True)
        refiner.requires_grad_(True)
        torch.set_rng_state(initial_rng)
        optimizer = torch.optim.Adam(refiner.parameters(), lr=1e-4, weight_decay=0)
        best_mse, best_mae = float('inf'), float('inf')
        history = []
        candidate_started = perf_counter()
        for epoch in range(1, EPOCHS+1):
            refiner.train()
            for row in train_rows[(epoch-1) % 2]:
                optimizer.zero_grad(set_to_none=True)
                output, error = suffix(refiner, row)
                hidden = ~row[-1] & output.sequence_validity
                loss = (error[hidden].square().mean()+full_mae_weight*error[hidden].abs().mean()
                        +0.25*error[:, -32:][hidden[:, -32:]].square().mean()
                        +0.05*error[:, -32:][hidden[:, -32:]].abs().mean())
                assert torch.isfinite(loss)
                loss.backward()
                if epoch == 1:
                    assert refiner.global_dense_refiner.output.weight.grad.abs().sum() > 0
                optimizer.step()
            metrics, per_variable = score(refiner, validation_rows)
            history.append(dict(epoch=epoch, metrics=metrics))
            for criterion, value in (('mse', metrics['full_mse']), ('mae', metrics['full_mae'])):
                best = best_mse if criterion == 'mse' else best_mae
                if value < best:
                    torch.save(dict(refiner_state=copy.deepcopy(refiner.state_dict()), epoch=epoch,
                                    metrics=metrics, per_variable=per_variable,
                                    source='CPU limited-bank screening only; not a formal checkpoint'),
                               R/(name+'_'+criterion+'_screen.pt'))
                    if criterion == 'mse':
                        best_mse, selected_mse = value, dict(epoch=epoch, metrics=metrics, per_variable=per_variable)
                    else:
                        best_mae, selected_mae = value, dict(epoch=epoch, metrics=metrics, per_variable=per_variable)
            (R/(name+'_history.json')).write_text(json.dumps(history, indent=2))
            print(json.dumps(dict(candidate=name, epoch=epoch, **metrics)), flush=True)
        assert coarse_hash(model) == prefix_hash and bank_content(memory) == memory_hash
        results.append(dict(candidate=name, full_mae_weight=full_mae_weight,
                            mse_selected=selected_mse, mae_selected=selected_mae,
                            seconds=perf_counter()-candidate_started))
        (R/'v104_screen_results.json').write_text(json.dumps(dict(metadata=metadata, candidates=results,
                                                                  full_workflow_seconds=perf_counter()-started), indent=2))
    print('SCREEN COMPLETED', json.dumps(results), flush=True)


if __name__ == '__main__':
    main()
