"""Strict Stage 1 repair evaluation; no forecast model or future target loss."""
from time import perf_counter
ENTRY_STARTED = perf_counter()
from pathlib import Path
import argparse
import csv
import json
import torch

from weather_common import ROOT, data, protocol_for
from stage1_iteration import REPAIR_KEYS, repair_totals, summarize_draws
from integrations.e23.config import load_e23_settings
from integrations.e23.calibration_ensemble import build_calibration_repair
from integrations.e23.forecast_sensitive import parameter_hash
from integrations.e23.train import (load_stage1_selection, build_repair_model, _new_memory,
                                   _starts, _key_sha256, _atomic_json, _file_sha256)
from integrations.forecast_compare.protocol import iter_p4_batches
from integrations.forecast_compare.resources import selected_training_resources


@torch.no_grad()
def evaluate(criterion):
    torch.set_num_threads(4)
    plan = json.loads((ROOT/'plan.json').read_text())
    entry, scene = plan['models'][0], plan['scenarios'][0]
    base = ROOT/entry['label']/scene['name']
    selection = base/'stage1/runs'/('e23-stage1.json' if criterion == 'mse' else 'e23-stage1-mae.json')
    output = base/f'repair_evaluation_{criterion}'
    assert not output.exists(), output
    protocol = protocol_for(plan, scene)
    dataset_record = data.read_manifest(Path(plan['dataset_manifest']))
    settings = load_e23_settings(Path(entry['config']), protocol=protocol, allow_variable_scaling=True)
    selected = json.loads(selection.read_text())
    assert selected['selection_metric'] == criterion
    device = torch.device(plan['device'])
    payload = load_stage1_selection(selection, protocol, settings, device=device, allow_calibration_ensemble=True)
    model = build_calibration_repair(protocol, settings, selected['calibration_ensemble_spec'], device=device)
    model.load_state_dict(payload['model_state'], strict=True)
    model.historical.retrieval.load_state_dict(payload['retrieval_state'])
    memory = _new_memory(protocol)
    memory.load_state_dict(payload['memory_state'])
    model.historical.refresh_snapshot(memory.snapshot())
    model.eval().requires_grad_(False)
    del payload
    before = parameter_hash(model), memory.snapshot_hash(), _key_sha256(model.historical.retrieval.state_dict())
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    inference_started = perf_counter()
    validation = None
    draws = [item for item in dataset_record['masks'] if item['split'] == 'test']
    assert len(draws) == 20
    rows = []
    for item in [next(item for item in dataset_record['masks'] if item['split'] == 'validation'), *draws]:
        split = item['split']
        current = protocol if split == 'validation' else data.with_test_mask(protocol, dataset_record, item)
        sums = {key: 0. for key in ('repair96_squared', 'repair96_absolute', 'repair32_squared', 'repair32_absolute')}
        counts = dict(hidden96_coordinates=0, hidden32_coordinates=0, windows=0)
        for cpu_batch in iter_p4_batches(current, split, batch_size=settings.batch_size):
            batch = cpu_batch.to(device)
            starts = _starts(batch, stride=model.stride, patches=model.patch_count, device=device)
            repaired = model(batch.normalized_observed, batch.observed_mask, batch.query_mean,
                             batch.query_std, starts).explicit
            assert repaired.sequence_validity.all(), 'Sealed 96-point windows must be fully valid'
            assert torch.equal(repaired.normalized_sequence[batch.observed_mask],
                               batch.normalized_observed[batch.observed_mask])
            totals = repair_totals(repaired.normalized_sequence, batch.normalized_target,
                                   batch.observed_mask, repaired.sequence_validity, batch.query_std)
            for key in sums:
                sums[key] += totals[key]
            for key in ('hidden96_coordinates', 'hidden32_coordinates'):
                counts[key] += totals[key]
            counts['windows'] += len(batch.refs)
        assert counts['windows'] == len(current.refs[split])
        metrics = dict(seed=item['seed'], mask_sha256=item['sha256'], **counts,
                       window_missing_fraction=counts['hidden96_coordinates']/(counts['windows']*96*21))
        for length in (96, 32):
            metrics[f'repair{length}_mse'] = sums[f'repair{length}_squared']/counts[f'hidden{length}_coordinates']
            metrics[f'repair{length}_mae'] = sums[f'repair{length}_absolute']/counts[f'hidden{length}_coordinates']
        if split == 'validation':
            validation = metrics
            for source_key, key in (('global_mse','repair96_mse'), ('global_mae','repair96_mae'),
                                    ('global_last32_mse','repair32_mse'), ('global_last32_mae','repair32_mae')):
                if source_key in selected['validation_repair_metrics']:
                    assert abs(metrics[key]-selected['validation_repair_metrics'][source_key]) < 1e-6, key
        else:
            rows.append(metrics)
        print(json.dumps(dict(split=split, selection=criterion, **metrics)), flush=True)
    torch.cuda.synchronize(device)
    inference_seconds = perf_counter()-inference_started
    after = parameter_hash(model), memory.snapshot_hash(), _key_sha256(model.historical.retrieval.state_dict())
    assert before == after
    resources = selected_training_resources([Path(plan['parent_selection']),*[Path(s['selection']) for s in plan['calibration_sources']],selection])
    assert resources['resource_status'] == 'measured_all_stages', resources
    result = dict(model=entry['label'], dataset='Weather', protocol_id=dataset_record['protocol_id'],
                  metric_space='train_standardized', seq_len=96, pred_len=96, n_vars=21,
                  start_rate=.10, block_len=8, training_seed=2023,
                  selection_metric=criterion, primary=selected['primary'],
                  best_epoch=selected['best_epoch'], best_refinement_epoch=selected['best_refinement_epoch'],
                  selection_manifest=str(selection), selection_manifest_sha256=_file_sha256(selection),
                  checkpoint_sha256=selected['checkpoint_sha256'],
                  dataset_manifest=plan['dataset_manifest'], dataset_manifest_sha256=_file_sha256(Path(plan['dataset_manifest'])),
                  validation=validation, draws=rows, summary=summarize_draws(rows), training_resources=resources,
                  visible_values_exact=True, parameters_memory_and_keys_unchanged=True,
                  test_inference_seconds=inference_seconds, evaluation_full_seconds=perf_counter()-ENTRY_STARTED,
                  inference_peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
                  inference_peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30,
                  aggregation='equal mean over 20 development masks; coordinate-weighted mean also reported',
                  training_seed_std=None, test_role=plan['test_role'], forecast_evaluated=False)
    output.mkdir(parents=True)
    _atomic_json(result, output/'twenty-test-draws.json')
    with (output/'per_mask.csv').open('w', newline='') as stream:
        writer=csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = dict(model=entry['label'], selection_metric=criterion, best_refinement_epoch=selected['best_refinement_epoch'])
    for key in REPAIR_KEYS:
        summary[key] = result['summary'][key]['mean']
        summary[key+'_mask_std_ddof1'] = result['summary'][key]['mask_std_ddof1']
    summary.update({key: resources[key] for key in ('full_workflow_seconds','peak_allocated_gib','peak_reserved_gib')})
    with (output/'summary.csv').open('w', newline='') as stream:
        writer=csv.DictWriter(stream, fieldnames=list(summary)); writer.writeheader(); writer.writerow(summary)
    print(json.dumps(dict(status='completed', summary=summary)), flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--selection', choices=('mse','mae'), required=True)
    evaluate(parser.parse_args().selection)
