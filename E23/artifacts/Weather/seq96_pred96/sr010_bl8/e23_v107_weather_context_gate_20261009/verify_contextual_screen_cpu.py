"""Check the formal implementation against the CPU screen, without training."""
from time import perf_counter
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import sys
import torch
from weather_common import ROOT, protocol_for
from integrations.e23.config import load_e23_settings
from integrations.e23.train import build_forecast_model
from integrations.e23.forecast_sensitive import parameter_hash
from malr.models.e23.recent_mean import RecentMeanProjector

SCREEN = Path('/home/xyw/paper/E23/artifacts/performance/weather_iteration_20261008')
sys.path.append(str(SCREEN))
from screen_v104_cpu import score


def state(status, **extra):
    record = dict(status=status, updated_at=datetime.now(timezone.utc).isoformat(), pid=os.getpid(), **extra)
    (ROOT/'verification_status.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)


def main():
    os.nice(10)
    torch.set_num_threads(4)
    state('running')
    started = perf_counter()
    data = torch.load(SCREEN/'v97_cpu_screen_inputs.pt', map_location='cpu', weights_only=False)
    refiner = RecentMeanProjector(21, mode='global_dense_context_gate_v6')
    incompatible = refiner.load_state_dict(data['initial_state'], strict=False)
    assert not incompatible.unexpected_keys and len(incompatible.missing_keys) == 2
    initial, _ = score(refiner, data['validation_rows'])
    assert initial == data['metadata']['initial_metrics']
    selected = torch.load(SCREEN/'v107_contextual_calibration_screen/v107_C_full_mae2_mse_screen.pt',
                          map_location='cpu', weights_only=False)
    refiner.load_state_dict(selected['refiner_state'], strict=True)
    canonical, _ = score(refiner, data['validation_rows'])
    differences = {k: abs(canonical[k]-selected['metrics'][k]) for k in canonical}
    assert max(differences.values()) < 1e-7, differences
    plan = json.loads((ROOT/'plan.json').read_text())
    protocol = protocol_for(plan, plan['scenarios'][0])
    settings = load_e23_settings(ROOT/'v107_weather.yaml', protocol=protocol, allow_variable_scaling=True)
    torch.manual_seed(2023)
    model = build_forecast_model(protocol, settings, device=torch.device('cpu'))
    predictor_hash = parameter_hash(model.predictor)
    assert predictor_hash == plan['predictor_initial_sha256']
    result = dict(full_validation_windows=len(protocol.refs['validation']),
                  initial_cached_validation_exact=True, screen_vs_canonical_absolute_difference=differences,
                  canonical_validation=canonical, screen_selected_epoch=selected['epoch'],
                  predictor_initial_sha256=predictor_hash, predictor_parameters=sum(p.numel() for p in model.predictor.parameters()),
                  all_screen_weights_not_reused_for_formal_training=True,
                  seconds=perf_counter()-started, scope='CPU implementation verification, not formal training')
    (ROOT/'canonical_preflight.json').write_text(json.dumps(result, indent=2))
    state('completed', exit_code=0, **result)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        state('failed', exit_code=1, error=repr(exc))
        raise
