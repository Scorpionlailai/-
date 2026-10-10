"""Validation-only baseline diagnosis on CPU; does not share the training GPU."""
from pathlib import Path
from types import SimpleNamespace
from time import perf_counter
import csv
import hashlib
import importlib.util
import json
import os
import sys

R = Path(__file__).resolve().parent
WEATHER = Path('/home/xyw/paper/E23/artifacts/Weather/seq96_pred96/sr010_bl8/e23_v97_weather_cached_20261008')
sys.path.insert(0, str(WEATHER))
from weather_common import protocol_for
import torch
from integrations.e23.config import load_e23_settings
from integrations.e23.train import build_forecast_model
from integrations.e23.forecast_sensitive import parameter_hash
from integrations.forecast_compare.protocol import iter_p4_batches
from integrations.forecast_compare.t1_p4 import T1_SETTINGS, T1ForecastModel, t1_inputs, t1_clean_history
from malr.data.patching import patchify_preserving_values


def main():
    os.nice(10)
    torch.set_num_threads(4)
    device = torch.device('cpu')
    plan = json.loads((WEATHER/'plan.json').read_text())
    protocol = protocol_for(plan, plan['scenarios'][0])
    settings = load_e23_settings(Path(plan['parent_settings']), protocol=protocol, allow_variable_scaling=True)
    selection = Path('/home/xyw/paper/E24/artifacts/Weather/seq96_pred96/sr010_bl8/weather_author_comparison_20261008/T1/stage2/selection.json')
    selected = json.loads(selection.read_text())
    checkpoint = Path(selected['checkpoint'])
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == selected['checkpoint_sha256']
    spec = importlib.util.spec_from_file_location('diagnostic_official_t1', Path(selected['profile']['root'])/'models/T1.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    config = {**T1_SETTINGS, **selected['profile']['model'], 'enc_in': 21}
    imputer = module.Model(SimpleNamespace(**config))
    predictor = build_forecast_model(protocol, settings, device=device).predictor
    model = T1ForecastModel(imputer, predictor)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(payload['model_state'], strict=True)
    model.eval().requires_grad_(False)
    before = parameter_hash(model)
    conditions = ('repaired', 'oracle_all_hidden')
    keys = ('forecast_squared', 'forecast_absolute', 'forecast_count',
            'repair_squared', 'repair_absolute', 'repair_count',
            'tail_squared', 'tail_absolute', 'tail_count')
    stats = {name: {k: torch.zeros(21, dtype=torch.float64) for k in keys} for name in conditions}
    started = perf_counter()
    with torch.no_grad():
        for index, batch in enumerate(iter_p4_batches(protocol, 'validation', batch_size=128), 1):
            values, mask = t1_inputs(batch)
            candidate = model.imputer(values, mask=mask).float()
            original = torch.where(batch.observed_mask, values, candidate)
            truth = t1_clean_history(batch)
            hidden = ~batch.observed_mask
            for name in conditions:
                sequence = original if name == 'repaired' else truth
                normalized = (sequence-batch.query_mean[:, None])/batch.query_std[:, None]
                normalized = torch.where(batch.observed_mask, batch.normalized_observed, normalized)
                patch = patchify_preserving_values(normalized, batch.observed_mask, 16, 8)
                prediction = model.predictor(patch.values, patch.effective_mask, batch.query_mean,
                                            batch.query_std, patch_validity=patch.validity)
                forecast_error = (prediction-batch.forecast_target).double().transpose(1, 2)
                repair_error = (sequence-truth).double()
                s = stats[name]
                s['forecast_squared'] += forecast_error.square().sum((0, 1))
                s['forecast_absolute'] += forecast_error.abs().sum((0, 1))
                s['forecast_count'] += forecast_error.shape[0]*forecast_error.shape[1]
                for prefix, length in (('repair', 96), ('tail', 32)):
                    error, scope = repair_error[:, -length:], hidden[:, -length:]
                    s[prefix+'_squared'] += (error.square()*scope).sum((0, 1))
                    s[prefix+'_absolute'] += (error.abs()*scope).sum((0, 1))
                    s[prefix+'_count'] += scope.sum((0, 1))
            print(f'T1 CPU validation batch {index}/41 seconds={perf_counter()-started:.2f}', flush=True)
    assert before == parameter_hash(model)
    columns = json.loads(Path(plan['dataset_manifest']).read_text())['columns']
    metrics = [('forecast_mse', 'forecast_squared', 'forecast_count'),
               ('forecast_mae', 'forecast_absolute', 'forecast_count'),
               ('repair96_mse', 'repair_squared', 'repair_count'),
               ('repair96_mae', 'repair_absolute', 'repair_count'),
               ('repair32_mse', 'tail_squared', 'tail_count'),
               ('repair32_mae', 'tail_absolute', 'tail_count')]
    summary, rows = {}, []
    for name, s in stats.items():
        summary[name] = {label: float(s[num].sum()/s[den].sum()) for label, num, den in metrics}
        for c, column in enumerate(columns):
            rows.append(dict(condition=name, variable=column,
                             **{label: float(s[num][c]/s[den][c]) for label, num, den in metrics}))
    existing = json.loads((selection.parents[1]/'evaluation/mae/twenty-test-draws.json').read_text())['validation']
    replay_differences = {}
    for metric, old_name in (('forecast_mse', 'mse'), ('forecast_mae', 'mae'),
                             ('repair96_mse', 'repair96_mse'), ('repair96_mae', 'repair96_mae')):
        replay_differences[metric] = summary['repaired'][metric]-existing[old_name]
    result = dict(summary=summary, seconds=perf_counter()-started, device='cpu',
                  checkpoint=str(checkpoint), checkpoint_sha256=selected['checkpoint_sha256'],
                  validation_mask_sha256=protocol.mask_sha256['validation'], weights_unchanged=True,
                  cpu_minus_official_cuda_validation=replay_differences,
                  scope='fixed validation only; original trained head; oracle historical truth is diagnosis, '
                        'not deployable accuracy or a causal/irreducible ceiling', test_used=False)
    (R/'t1_diagnostic_validation.json').write_text(json.dumps(result, indent=2))
    with (R/'t1_diagnostic_per_variable.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # CPU and CUDA kernels need not be bitwise equal. Preserve the discrepancy
    # and reject changes large enough to invalidate a per-variable diagnosis.
    assert max(abs(v) for v in replay_differences.values()) < 5e-5, replay_differences
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
