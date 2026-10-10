import json
from pathlib import Path

import pytest
import torch

from integrations.e23.config import load_e23_settings
from integrations.e23.train import (
    _file_sha256, load_stage2_selection, stage2_objective, train_forecast, train_repair,
)
from tests.integration.test_e23_stage1 import _batch, _protocol_settings


def _selected_stage1(monkeypatch, tmp_path):
    protocol, settings = _protocol_settings()
    import integrations.e23.train as train_module
    monkeypatch.setattr(train_module, "iter_p4_batches",
                        lambda _protocol, split, **kwargs: iter((_batch(split),)))
    selection = train_repair(protocol, settings, output=tmp_path / "stage1",
                             device=torch.device("cpu"), epochs=1)
    return protocol, settings, selection


def test_stage2_freezes_retrieval_but_trains_messages(monkeypatch, tmp_path):
    protocol, settings, stage1 = _selected_stage1(monkeypatch, tmp_path)
    stage2 = train_forecast(
        protocol, settings, stage1_manifest=stage1, output=tmp_path / "stage2",
        device=torch.device("cpu"), epochs=2,
    )
    record = json.loads(stage2.read_text())
    stage1_record = json.loads(stage1.read_text())
    assert record["stage"] == "forecast"
    assert record["epochs_completed"] == 2
    assert record["loss"] == "forecast_mse_only"
    assert record["optimizer_state_restored"] is False
    assert record["memory_sha256"] == stage1_record["memory_sha256"]
    assert record["key_sha256"] == stage1_record["key_sha256"]
    payload = load_stage2_selection(stage2, protocol, settings,
                                    device=torch.device("cpu"))
    assert payload["memory_sha256"] == stage1_record["memory_sha256"]
    assert payload["key_sha256"] == stage1_record["key_sha256"]
    stage1_payload = torch.load(stage1_record["checkpoint"], map_location="cpu", weights_only=False)
    stage2_state = payload["model_state"]
    assert torch.equal(stage1_payload["model_state"]["historical.key_encoder.network.0.weight"],
                       stage2_state["repair.historical.key_encoder.network.0.weight"])
    assert not torch.equal(stage1_payload["model_state"]["historical.message_projector.0.weight"],
                           stage2_state["repair.historical.message_projector.0.weight"])
    assert not torch.equal(stage1_payload["model_state"]["router.shared_score.0.weight"],
                           stage2_state["repair.router.shared_score.0.weight"])
    prediction = torch.tensor([1., 3.], requires_grad=True)
    truth = torch.tensor([0., 1.])
    torch.testing.assert_close(stage2_objective(prediction, truth), torch.tensor(2.5))


def test_mixed_forecast_loss_has_fixed_mae_term_and_finite_gradient():
    prediction = torch.tensor([1., 3.], requires_grad=True)
    truth = torch.tensor([0., 1.])
    loss = stage2_objective(prediction, truth, loss_mode="mse_plus_0p2_mae")
    torch.testing.assert_close(loss, torch.tensor(2.8))
    loss.backward()
    torch.testing.assert_close(prediction.grad, torch.tensor([1.1, 2.1]))
    with pytest.raises(ValueError, match="loss"):
        stage2_objective(prediction.detach(), truth, loss_mode="unknown")


def test_late_horizon_mixed_loss_has_fixed_weights_and_finite_gradient():
    prediction = torch.tensor([[[1., 2., 3., 4.]]], requires_grad=True)
    truth = torch.zeros_like(prediction)
    loss = stage2_objective(prediction, truth,
                            loss_mode="mse_plus_0p2_mae_late_horizon_v1")
    torch.testing.assert_close(loss, torch.tensor(8.65))
    loss.backward()
    torch.testing.assert_close(prediction.grad,
                               torch.tensor([[[0.4675, 0.9975, 1.6275, 2.3575]]]))
    with pytest.raises(ValueError, match="horizon"):
        stage2_objective(prediction[..., :3], truth[..., :3],
                         loss_mode="mse_plus_0p2_mae_late_horizon_v1")


def test_mixed_loss_manifest_and_checkpoint_are_paired(monkeypatch, tmp_path):
    protocol, settings, stage1 = _selected_stage1(monkeypatch, tmp_path)
    import integrations.e23.train as train_module
    selected = train_forecast(
        protocol, settings, stage1_manifest=stage1, output=tmp_path / "mixed-stage2",
        device=torch.device("cpu"), epochs=1, selection_metric="mae",
        forecast_loss="mse_plus_0p2_mae",
    )
    record = json.loads(selected.read_text())
    payload = torch.load(record["checkpoint"], map_location="cpu", weights_only=False)
    assert record["loss"] == payload["loss"] == "forecast_mse_plus_0p2_mae"
    assert load_stage2_selection(selected, protocol, settings,
                                 device=torch.device("cpu"))["epoch"] == 1
    tampered = tmp_path / "wrong-loss.json"
    tampered.write_text(json.dumps({**record, "loss": "forecast_mse_only"}))
    seen = []
    monkeypatch.setattr(train_module, "iter_p4_batches",
                        lambda *_args, **_kwargs: seen.append(True))
    with pytest.raises(ValueError, match="loss.*mismatch"):
        load_stage2_selection(tampered, protocol, settings, device=torch.device("cpu"))
    assert not seen


def test_late_horizon_loss_manifest_and_checkpoint_are_paired(monkeypatch, tmp_path):
    protocol, settings, stage1 = _selected_stage1(monkeypatch, tmp_path)
    import integrations.e23.train as train_module
    selected = train_forecast(
        protocol, settings, stage1_manifest=stage1, output=tmp_path / "late-horizon-stage2",
        device=torch.device("cpu"), epochs=1, selection_metric="mae",
        forecast_loss="mse_plus_0p2_mae_late_horizon_v1",
    )
    record = json.loads(selected.read_text())
    payload = torch.load(record["checkpoint"], map_location="cpu", weights_only=False)
    assert record["loss"] == payload["loss"] == "forecast_mse_plus_0p2_mae_late_horizon_v1"
    assert load_stage2_selection(selected, protocol, settings,
                                 device=torch.device("cpu"))["epoch"] == 1
    tampered = tmp_path / "wrong-late-loss.json"
    tampered.write_text(json.dumps({**record, "loss": "forecast_mse_plus_0p2_mae"}))
    seen = []
    monkeypatch.setattr(train_module, "iter_p4_batches",
                        lambda *_args, **_kwargs: seen.append(True))
    with pytest.raises(ValueError, match="loss checkpoint mismatch"):
        load_stage2_selection(tampered, protocol, settings, device=torch.device("cpu"))
    assert not seen


def test_width16_stage1_rejected_by_width64_stage2_before_batches(monkeypatch, tmp_path):
    protocol, _, stage1 = _selected_stage1(monkeypatch, tmp_path)
    root = Path(__file__).resolve().parents[2]
    wider = load_e23_settings(root / "configs/e23_multi_expert_p4_width64.yaml", protocol=protocol)
    seen = []
    import integrations.e23.train as train_module
    monkeypatch.setattr(train_module, "iter_p4_batches",
                        lambda *_args, **_kwargs: seen.append(True))
    with pytest.raises(ValueError, match="config.*mismatch"):
        train_forecast(protocol, wider, stage1_manifest=stage1,
                       output=tmp_path / "stage2", device=torch.device("cpu"), epochs=1)
    assert not seen
    assert not (tmp_path / "stage2").exists()


def test_stage2_mae_policy_selects_lower_mae_and_pairs_checkpoint(monkeypatch, tmp_path):
    protocol, settings, stage1 = _selected_stage1(monkeypatch, tmp_path)
    import integrations.e23.train as train_module
    rows = iter(({"mse": 0.1, "mae": 0.5}, {"mse": 0.2, "mae": 0.4}))
    monkeypatch.setattr(train_module, "_validation_forecast_metrics",
                        lambda *_args, **_kwargs: next(rows))
    stage2 = train_forecast(
        protocol, settings, stage1_manifest=stage1, output=tmp_path / "mae-stage2",
        device=torch.device("cpu"), epochs=2, selection_metric="mae",
    )
    record = json.loads(stage2.read_text())
    assert record["best_epoch"] == 2
    assert record["best_val_mse"] == 0.2
    assert record["best_val_mae"] == 0.4
    assert record["selection_metric"] == "mae"
    payload = load_stage2_selection(stage2, protocol, settings, device=torch.device("cpu"))
    assert payload["selection_metric"] == "mae"
    changed = {**record, "selection_metric": "mse"}
    tampered = tmp_path / "wrong-selection.json"
    tampered.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="selection.*mismatch"):
        load_stage2_selection(tampered, protocol, settings, device=torch.device("cpu"))
    with pytest.raises(ValueError, match="selection"):
        train_forecast(
            protocol, settings, stage1_manifest=stage1, output=tmp_path / "invalid",
            device=torch.device("cpu"), epochs=1, selection_metric="test",
        )
    assert not (tmp_path / "invalid").exists()


def test_stage2_variable_mixer_identity_is_paired_and_rejected_when_tampered(monkeypatch, tmp_path):
    protocol, settings, stage1 = _selected_stage1(monkeypatch, tmp_path)
    stage2 = train_forecast(
        protocol, settings, stage1_manifest=stage1, output=tmp_path / "mixer-stage2",
        device=torch.device("cpu"), epochs=1, selection_metric="mae",
        forecast_mixer="patch_attention_v1",
    )
    record = json.loads(stage2.read_text())
    assert record["forecast_mixer"] == "patch_attention_v1"
    payload = load_stage2_selection(stage2, protocol, settings, device=torch.device("cpu"))
    assert payload["forecast_mixer"] == "patch_attention_v1"
    assert "predictor.variable_mixer.residual_scale" in payload["model_state"]
    tampered = tmp_path / "wrong-mixer.json"
    tampered.write_text(json.dumps({**record, "forecast_mixer": "none"}))
    with pytest.raises(ValueError, match="forecast mixer.*mismatch"):
        load_stage2_selection(tampered, protocol, settings, device=torch.device("cpu"))
    with pytest.raises(ValueError, match="forecast mixer"):
        train_forecast(
            protocol, settings, stage1_manifest=stage1, output=tmp_path / "invalid-mixer",
            device=torch.device("cpu"), epochs=1, forecast_mixer="unknown",
        )
    assert not (tmp_path / "invalid-mixer").exists()


def test_stage2_old_checkpoint_without_selector_defaults_to_mse(monkeypatch, tmp_path):
    protocol, settings, stage1 = _selected_stage1(monkeypatch, tmp_path)
    stage2 = train_forecast(
        protocol, settings, stage1_manifest=stage1, output=tmp_path / "stage2",
        device=torch.device("cpu"), epochs=1,
    )
    record = json.loads(stage2.read_text())
    checkpoint = Path(record["checkpoint"])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    del payload["selection_metric"]
    del payload["forecast_mixer"]
    torch.save(payload, checkpoint)
    del record["selection_metric"]
    del record["forecast_mixer"]
    record["checkpoint_sha256"] = _file_sha256(checkpoint)
    stage2.write_text(json.dumps(record))
    assert load_stage2_selection(stage2, protocol, settings,
                                 device=torch.device("cpu"))["epoch"] == 1


@pytest.mark.parametrize("field,value", [
    ("e23_config_sha256", "corrupt"),
    ("protocol_config_sha256", "corrupt"),
    ("memory_sha256", "corrupt"),
    ("key_sha256", "corrupt"),
])
def test_stage2_rejects_mismatched_stage1_before_first_batch(monkeypatch, tmp_path, field, value):
    protocol, settings, stage1 = _selected_stage1(monkeypatch, tmp_path)
    record = json.loads(stage1.read_text())
    record[field] = value
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(record))
    seen = []
    import integrations.e23.train as train_module
    monkeypatch.setattr(train_module, "iter_p4_batches",
                        lambda *_args, **_kwargs: seen.append(True))
    with pytest.raises(ValueError, match="mismatch"):
        train_forecast(protocol, settings, stage1_manifest=tampered,
                       output=tmp_path / "stage2", device=torch.device("cpu"), epochs=1)
    assert not seen


def test_new_trial_stage2_manifest_and_checkpoint_carry_width_identity(monkeypatch, tmp_path):
    protocol, _ = _protocol_settings()
    root = Path(__file__).resolve().parents[2]
    settings = load_e23_settings(root / "configs/e23_multi_expert_p4_width128_topk.yaml",
                                 protocol=protocol)
    import integrations.e23.train as train_module
    monkeypatch.setattr(train_module, "iter_p4_batches",
                        lambda _protocol, split, **kwargs: iter((_batch(split),)))
    stage1 = train_repair(protocol, settings, output=tmp_path / "wide-stage1",
                          device=torch.device("cpu"), epochs=1)
    stage2 = train_forecast(protocol, settings, stage1_manifest=stage1,
                            output=tmp_path / "wide-stage2", device=torch.device("cpu"),
                            epochs=1, selection_metric="mae")
    record = json.loads(stage2.read_text())
    payload = torch.load(record["checkpoint"], map_location="cpu", weights_only=False)
    assert record["repair_width"] == payload["repair_width"] == 128
    assert record["candidate_selection"] == payload["candidate_selection"] == "top_k"
    assert load_stage2_selection(stage2, protocol, settings,
                                 device=torch.device("cpu"))["epoch"] == 1
    tampered = tmp_path / "wrong-width.json"
    tampered.write_text(json.dumps({**record, "repair_width": 64}))
    seen = []
    monkeypatch.setattr(train_module, "iter_p4_batches",
                        lambda *_args, **_kwargs: seen.append(True))
    with pytest.raises(ValueError, match="repair_width.*mismatch"):
        load_stage2_selection(tampered, protocol, settings, device=torch.device("cpu"))
    assert not seen


def test_contextual_v_stage2_pairs_selection_and_rejects_wrong_variant(monkeypatch, tmp_path):
    protocol, _ = _protocol_settings()
    root = Path(__file__).resolve().parents[2]
    settings = load_e23_settings(root / "configs/e23_multi_expert_p4_width64_window_v12.yaml",
                                 protocol=protocol)
    import integrations.e23.train as train_module
    monkeypatch.setattr(train_module, "iter_p4_batches",
                        lambda _protocol, split, **kwargs: iter((_batch(split),)))
    stage1 = train_repair(protocol, settings, output=tmp_path / "window-stage1",
                          device=torch.device("cpu"), epochs=1)
    stage2 = train_forecast(protocol, settings, stage1_manifest=stage1,
                            output=tmp_path / "window-stage2", device=torch.device("cpu"),
                            epochs=1, selection_metric="mae",
                            forecast_loss="mse_plus_0p2_mae")
    record = json.loads(stage2.read_text())
    payload = torch.load(record["checkpoint"], map_location="cpu", weights_only=False)
    assert record["variable_selection"] == payload["variable_selection"] == "window_context_topk_v1"
    assert record["loss"] == payload["loss"] == "forecast_mse_plus_0p2_mae"
    assert load_stage2_selection(stage2, protocol, settings,
                                 device=torch.device("cpu"))["epoch"] == 1
    tampered = tmp_path / "wrong-window-stage2.json"
    tampered.write_text(json.dumps({**record, "variable_selection": "global_topk"}))
    seen = []
    monkeypatch.setattr(train_module, "iter_p4_batches",
                        lambda *_args, **_kwargs: seen.append(True))
    with pytest.raises(ValueError, match="variable_selection.*mismatch"):
        load_stage2_selection(tampered, protocol, settings, device=torch.device("cpu"))
    assert not seen


def test_two_group_h_pairs_both_modes_and_rejects_v12_stage1(monkeypatch, tmp_path):
    protocol, _ = _protocol_settings()
    root = Path(__file__).resolve().parents[2]
    settings = load_e23_settings(root / "configs/e23_multi_expert_p4_width64_window_two_group_v13.yaml",
                                 protocol=protocol)
    v12 = load_e23_settings(root / "configs/e23_multi_expert_p4_width64_window_v12.yaml",
                            protocol=protocol)
    import integrations.e23.train as train_module
    monkeypatch.setattr(train_module, "iter_p4_batches",
                        lambda _protocol, split, **kwargs: iter((_batch(split),)))
    stage1 = train_repair(protocol, settings, output=tmp_path / "two-group-stage1",
                          device=torch.device("cpu"), epochs=1)
    stage1_record = json.loads(stage1.read_text())
    assert stage1_record["historical_retrieval"] == "two_group_visible_rerank_v1"
    assert stage1_record["variable_selection"] == "window_context_topk_v1"
    stage2 = train_forecast(protocol, settings, stage1_manifest=stage1,
                            output=tmp_path / "two-group-stage2", device=torch.device("cpu"),
                            epochs=1, selection_metric="mae",
                            forecast_loss="mse_plus_0p2_mae")
    record = json.loads(stage2.read_text())
    payload = torch.load(record["checkpoint"], map_location="cpu", weights_only=False)
    assert record["historical_retrieval"] == payload["historical_retrieval"] == "two_group_visible_rerank_v1"
    assert record["variable_selection"] == payload["variable_selection"] == "window_context_topk_v1"
    assert load_stage2_selection(stage2, protocol, settings,
                                 device=torch.device("cpu"))["epoch"] == 1
    seen = []
    monkeypatch.setattr(train_module, "iter_p4_batches",
                        lambda *_args, **_kwargs: seen.append(True))
    with pytest.raises(ValueError, match="config.*mismatch"):
        train_forecast(protocol, v12, stage1_manifest=stage1,
                       output=tmp_path / "wrong-stage2", device=torch.device("cpu"), epochs=1)
    assert not seen and not (tmp_path / "wrong-stage2").exists()
    tampered = tmp_path / "wrong-h-stage2.json"
    tampered.write_text(json.dumps({**record, "historical_retrieval": "one_group"}))
    with pytest.raises(ValueError, match="historical_retrieval.*mismatch"):
        load_stage2_selection(tampered, protocol, settings, device=torch.device("cpu"))
    assert not seen
