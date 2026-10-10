from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch
import yaml

from integrations.e23.config import load_e23_settings
from integrations.e23 import train as training
from tests.integration.test_e23_stage1 import _protocol_settings, _batch


ROOT = Path(__file__).resolve().parents[2]


def _settings(tmp_path, protocol, rounds=3, changes=None):
    document = yaml.safe_load((ROOT / "configs/e23_multi_expert_p4_width64_global_stage1_v28.yaml").read_text())
    document["model"].update(recovery_schedule="iterative_tv_h_v1", recovery_rounds=rounds)
    for section, values in (changes or {}).items():
        document[section].update(values)
    path = tmp_path / "trial.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    return load_e23_settings(path, protocol=protocol)


@pytest.mark.parametrize("rounds", [1, 3])
def test_iterative_settings_build_same_parameter_shapes_and_original_predictor(tmp_path, rounds):
    protocol, _ = _protocol_settings()
    settings = _settings(tmp_path, protocol, rounds)
    model = training.build_forecast_model(protocol, settings, device=torch.device("cpu"))
    assert model.repair.recovery_rounds == rounds
    assert model.repair.recovery_schedule == "iterative_tv_h_v1"
    assert model.predictor.core.d_model == 16
    assert model.repair.d_model == 64


@pytest.mark.parametrize("rounds,changes", [
    (True, {}), (2, {}), (3, {"model": {"recovery_schedule": "parallel_v1"}}),
    (3, {"training": {"stage1_objective": "window_smooth_l1"}}),
    (3, {"model": {"multihead_router": "eight_full_width_gated_v1"}}),
    (3, {"model": {"post_fusion_refiner": "evidence_factorized_v1"}}),
])
def test_iterative_settings_reject_unsupported_trials(tmp_path, rounds, changes):
    protocol, _ = _protocol_settings()
    with pytest.raises(ValueError):
        _settings(tmp_path, protocol, rounds, changes)


def test_iterative_two_stage_training_preserves_schedule_and_global_selection(monkeypatch, tmp_path):
    # Catches dropping mode identity, evaluating an intermediate state, or using window MSE.
    protocol, _ = _protocol_settings()
    settings = _settings(tmp_path, protocol)
    def batches(_protocol, split, **kwargs):
        batch = _batch(split)
        return iter((replace(batch, query_std=torch.full_like(batch.query_std, .2)),))
    monkeypatch.setattr(training, "iter_p4_batches", batches)
    manifest = training.train_repair(protocol, settings, output=tmp_path / "stage1",
                                     device=torch.device("cpu"), epochs=2)
    record = json.loads(manifest.read_text())
    payload = training.load_stage1_selection(manifest, protocol, settings, device=torch.device("cpu"))
    assert record["recovery_schedule"] == payload["recovery_schedule"] == "iterative_tv_h_v1"
    assert record["recovery_rounds"] == payload["recovery_rounds"] == 3
    assert record["stage1_selection_metric_space"] == "train_standardized"
    model = training.build_repair_model(protocol, settings, device=torch.device("cpu"))
    model.load_state_dict(payload["model_state"], strict=True)
    model.historical.retrieval.load_state_dict(payload["retrieval_state"])
    memory = training._new_memory(protocol)
    memory.load_state_dict(payload["memory_state"])
    model.historical.refresh_snapshot(memory.snapshot())
    replay = training._validation_repair_metrics(model, protocol, settings, device=torch.device("cpu"))
    assert replay["global_mse"] == pytest.approx(record["best_val_repair_mse"])
    assert replay["global_mse"] != pytest.approx(replay["window_mse"])
    wrong = replace(settings, recovery_rounds=1)
    with pytest.raises(ValueError, match="recovery_rounds.*mismatch"):
        training.load_stage1_selection(manifest, protocol, wrong, device=torch.device("cpu"))
    forecast = training.train_forecast(protocol, settings, stage1_manifest=manifest,
        output=tmp_path / "stage2", device=torch.device("cpu"), epochs=1,
        selection_metric="mae", forecast_loss="mse_plus_0p2_mae")
    selected = training.load_stage2_selection(forecast, protocol, settings, device=torch.device("cpu"))
    assert selected["recovery_schedule"] == "iterative_tv_h_v1" and selected["recovery_rounds"] == 3
    with pytest.raises(ValueError, match="recovery_rounds.*mismatch"):
        training.load_stage2_selection(forecast, protocol, wrong, device=torch.device("cpu"))


def test_identical_shape_checkpoint_with_wrong_rounds_is_rejected(tmp_path):
    protocol, _ = _protocol_settings()
    settings = _settings(tmp_path, protocol)
    payload = dict(training._trial_identity(settings), recovery_rounds=1)
    with pytest.raises(ValueError, match="recovery_rounds.*mismatch"):
        training._verify_trial_payload(payload, settings)


def test_iterative_mode_rejects_history_adapter(tmp_path):
    protocol, _ = _protocol_settings()
    settings = _settings(tmp_path, protocol)
    with pytest.raises(ValueError, match="iterative"):
        training.build_repair_model(protocol, settings, device=torch.device("cpu"),
                                    history_adapter="aligned_v1")
