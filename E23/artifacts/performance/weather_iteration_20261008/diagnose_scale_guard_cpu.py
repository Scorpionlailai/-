"""Compare normalized repair tails on the sealed fixed validation set."""
from pathlib import Path
from types import SimpleNamespace
from time import perf_counter
import hashlib
import importlib.util
import json
import os
import sys

R = Path(__file__).resolve().parent
CODE = Path('/home/xyw/.cache/codex-worktrees/e23-v105-weather-scale-guard-20261009')
sys.path[:0] = [str(CODE/'src'), str(CODE)]
import torch
from malr.models.e23.recent_mean import RecentMeanProjector
from integrations.forecast_compare.t1_p4 import T1_SETTINGS


def summarize_z(parts):
    values = torch.cat(parts).abs()
    return dict(coordinates=values.numel(), mean_squared_value=float(values.double().square().mean()),
                q99=float(torch.quantile(values, .99)), q999=float(torch.quantile(values, .999)),
                max=float(values.max()), fraction_above_16=float((values > 16).float().mean()),
                fraction_above_100=float((values > 100).float().mean()))


def main():
    os.nice(10)
    torch.set_num_threads(4)
    started = perf_counter()
    cache = torch.load(R/'v97_cpu_screen_inputs.pt', map_location='cpu', weights_only=False)
    refiner = RecentMeanProjector(21, mode='global_dense_guard_v5')
    refiner.load_state_dict(cache['initial_state'], strict=True)
    refiner.eval().requires_grad_(False)
    selection = Path('/home/xyw/paper/E24/artifacts/Weather/seq96_pred96/sr010_bl8/weather_author_comparison_20261008/T1/stage2/selection.json')
    selected = json.loads(selection.read_text())
    checkpoint = Path(selected['checkpoint'])
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == selected['checkpoint_sha256']
    spec = importlib.util.spec_from_file_location('guard_diagnostic_official_t1', Path(selected['profile']['root'])/'models/T1.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    imputer = module.Model(SimpleNamespace(**{**T1_SETTINGS, **selected['profile']['model'], 'enc_in': 21}))
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)['model_state']
    imputer.load_state_dict({name[len('imputer.'):]: value for name, value in state.items()
                             if name.startswith('imputer.')}, strict=True)
    imputer.eval().requires_grad_(False)
    names = ('Weather_parent_v79_on_v97_architecture', 'same_weights_with_v105_guard', 'T1')
    values = {name: [] for name in names}
    squared = {name: 0. for name in names}
    absolute = {name: 0. for name in names}
    count = 0
    with torch.no_grad():
        for i, row in enumerate(cache['validation_rows'], 1):
            explicit, latents, points, mean, std, target, mask = row
            hidden = ~mask & explicit.sequence_validity
            refiner.normalized_output_guard = False
            original, _ = refiner(explicit, SimpleNamespace(latents=latents), points, mean, std)
            refiner.normalized_output_guard = True
            guarded, _ = refiner(explicit, SimpleNamespace(latents=latents), points, mean, std)
            assert torch.equal(original.normalized_sequence[mask], guarded.normalized_sequence[mask])
            core = hidden & (original.normalized_sequence.abs() <= 16)
            assert torch.equal(original.normalized_sequence[core], guarded.normalized_sequence[core])
            observed_global = (explicit.observed_sequence*std[:, None]+mean[:, None]).masked_fill(~mask, 0)
            candidate = imputer(observed_global, mask=mask.float())
            t1_normalized = (candidate-mean[:, None])/std[:, None]
            for name, sequence in zip(names, (original.normalized_sequence, guarded.normalized_sequence, t1_normalized)):
                error = ((sequence-target)*std[:, None])[hidden].double()
                values[name].append(sequence[hidden])
                squared[name] += float(error.square().sum())
                absolute[name] += float(error.abs().sum())
            count += int(hidden.sum())
            print(f'guard/T1 validation batch={i}/41', flush=True)
    result = dict(scope='full fixed validation only; parent Weather v79 weights with zero-init v97 dense; '
                       'v105 deterministic missing-output tail compression; no optimizer or test data',
                  threshold=16., source_lineage='ETTh1 v97 architecture', windows=5175,
                  same_parameters_and_weights=True, visible_coordinates_exact=True,
                  central_hidden_range_exact=True, model_parameters=109719,
                  dataset_manifest=cache['metadata']['dataset_manifest'],
                  validation_mask_sha256=cache['metadata']['validation_mask_sha256'],
                  seconds=perf_counter()-started,
                  rows=[dict(model=name, repair96_mse=squared[name]/count,
                             repair96_mae=absolute[name]/count,
                             normalized_missing_values=summarize_z(values[name])) for name in names])
    (R/'scale_guard_validation_preflight.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
