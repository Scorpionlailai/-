"""Repair scoring must not amplify equal errors in small-variance windows."""

from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from integrations.e23.config import load_e23_settings
from integrations.e23 import train as training
from integrations.forecast_compare.protocol import P4Batch
from malr.data.windows import WindowRef
from tests.integration.test_e23_stage1 import _protocol_settings, _batch


ROOT = Path(__file__).resolve().parents[2]


def _global_settings(tmp_path, protocol):
    path = tmp_path / "global.yaml"
    path.write_text(
        (ROOT / "configs/e23_multi_expert_p4_width64_window_v12.yaml").read_text()
        + "  stage1_objective: global_mse_v1\n"
    )
    return load_e23_settings(path, protocol=protocol)


def test_equal_global_errors_have_equal_loss_despite_different_window_scales():
    # Mutations caught: score normalized errors, retain Smooth L1, or include
    # observed/padded coordinates instead of hidden valid coordinates.
    candidate = torch.tensor([[[10., .1], [900., 800.], [700., 600.]]], requires_grad=True)
    target = torch.zeros_like(candidate)
    observed = torch.tensor([[[False, False], [True, True], [False, False]]])
    validity = torch.tensor([[[True, True], [True, True], [False, False]]])
    std = torch.tensor([[.01, 1.]], requires_grad=True)
    total, parts = training.stage1_objective(
        candidate, target, observed, torch.tensor(.5), retrieval_weight=.1,
        valid_mask=validity, loss_mode="global_mse_v1", query_std=std,
    )
    torch.testing.assert_close(total, torch.tensor(.06))
    torch.testing.assert_close(parts["repair"], torch.tensor(.01))
    total.backward()
    torch.testing.assert_close(candidate.grad, torch.tensor([[[.001, .1], [0., 0.], [0., 0.]]]))
    assert std.grad is None


def test_global_validation_scores_actual_errors_and_retains_window_diagnostic(monkeypatch):
    # Mutations caught: validation selects on normalized errors or silently
    # drops the small-std coordinates from the primary metric.
    protocol, settings = _protocol_settings()
    model = training.build_repair_model(protocol, settings, device=torch.device("cpu"))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    target = torch.zeros(1, 96, 7)
    target[0, 0, :2] = torch.tensor([100., .1])
    observed = torch.ones_like(target, dtype=torch.bool)
    observed[0, 0, :2] = False
    batch = P4Batch(
        target * observed, target, observed, torch.zeros(1, 7),
        torch.tensor([[.001, 1., 1., 1., 1., 1., 1.]]),
        torch.zeros(1, 7, 96), (WindowRef(0, 96, 96, 192, "validation"),),
    )
    monkeypatch.setattr(training, "iter_p4_batches", lambda *_args, **_kwargs: iter((batch,)))
    metrics = training._validation_repair_metrics(model, protocol, settings, device=torch.device("cpu"))
    assert metrics["global_mse"] == pytest.approx(.01)
    assert metrics["global_mae"] == pytest.approx(.1)
    assert metrics["window_mse"] == pytest.approx(5000.005)
    assert metrics["hidden_coordinates"] == 2
    assert metrics["small_std_hidden_fraction"] == pytest.approx(.5)
    assert metrics["small_std_global_squared_share"] == pytest.approx(.5)


def test_global_stage1_selection_and_stage2_keep_explicit_metric_identity(monkeypatch, tmp_path):
    # Mutations caught: new mode is ignored in training, the saved score is in
    # the wrong space, or Stage 2 loses the Stage 1 objective identity.
    protocol, _ = _protocol_settings()
    settings = _global_settings(tmp_path, protocol)

    def batches(_protocol, split, **_kwargs):
        batch = _batch(split)
        return iter((replace(batch, query_std=torch.full_like(batch.query_std, .2)),))

    monkeypatch.setattr(training, "iter_p4_batches", batches)
    manifest = training.train_repair(protocol, settings, output=tmp_path / "stage1",
                                     device=torch.device("cpu"), epochs=2)
    record = json.loads(manifest.read_text())
    payload = training.load_stage1_selection(manifest, protocol, settings, device=torch.device("cpu"))
    assert record["stage1_objective"] == payload["stage1_objective"] == "global_mse_v1"
    assert record["stage1_selection_metric_space"] == payload["stage1_selection_metric_space"] == "train_standardized"
    model = training.build_repair_model(protocol, settings, device=torch.device("cpu"))
    model.load_state_dict(payload["model_state"])
    model.historical.retrieval.load_state_dict(payload["retrieval_state"])
    memory = training._new_memory(protocol)
    memory.load_state_dict(payload["memory_state"])
    model.historical.refresh_snapshot(memory.snapshot())
    replay = training._validation_repair_metrics(model, protocol, settings, device=torch.device("cpu"))
    assert record["best_val_repair_mse"] == pytest.approx(replay["global_mse"])
    assert record["best_val_repair_mse"] != pytest.approx(replay["window_mse"])
    history = json.loads((manifest.parent / "e23-stage1-history.json").read_text())
    assert len(history["epochs"]) == 2
    assert all(row["metrics"]["global_mse"] >= 0 and row["metrics"]["window_mse"] >= 0
               for row in history["epochs"])
    assert record["validation_repair_metrics"] == history["epochs"][record["best_epoch"] - 1]["metrics"]

    forecast_manifest = training.train_forecast(
        protocol, settings, stage1_manifest=manifest, output=tmp_path / "stage2",
        device=torch.device("cpu"), epochs=1, selection_metric="mae", forecast_loss="mse_plus_0p2_mae",
    )
    selected = training.load_stage2_selection(forecast_manifest, protocol, settings, device=torch.device("cpu"))
    assert selected["stage1_objective"] == "global_mse_v1"

    wrong = manifest.parent / "wrong.json"
    wrong.write_text(json.dumps({**record, "stage1_selection_metric_space": "window_normalized"}))
    with pytest.raises(ValueError, match="stage1_selection_metric_space.*mismatch"):
        training.load_stage1_selection(wrong, protocol, settings, device=torch.device("cpu"))


def test_global_repair_mode_rejects_invalid_scale():
    candidate = torch.zeros(1, 2, 1)
    with pytest.raises(ValueError, match="query_std"):
        training.stage1_objective(
            candidate, candidate, torch.zeros_like(candidate, dtype=torch.bool),
            torch.tensor(0.), retrieval_weight=.1, loss_mode="global_mse_v1",
            query_std=torch.zeros(1, 1),
        )


def test_stage1_chooses_global_minimum_when_window_ranking_disagrees(monkeypatch, tmp_path):
    # Mutation caught: choose the window-score winner but label its saved
    # score as global. Both metrics were numerically verified above.
    protocol, _ = _protocol_settings()
    settings = _global_settings(tmp_path, protocol)
    monkeypatch.setattr(training, "iter_p4_batches",
                        lambda _protocol, split, **_kwargs: iter((_batch(split),)))
    rows = iter((
        {"window_mse": 1., "global_mse": 2., "global_mae": 1., "hidden_coordinates": 112},
        {"window_mse": 3., "global_mse": 1., "global_mae": .5, "hidden_coordinates": 112},
    ))

    def validation(model, *_args, **_kwargs):
        model.eval()
        return next(rows)

    monkeypatch.setattr(training, "_validation_repair_metrics", validation)
    manifest = training.train_repair(protocol, settings, output=tmp_path / "opposite-ranking",
                                     device=torch.device("cpu"), epochs=2)
    record = json.loads(manifest.read_text())
    assert record["best_epoch"] == 2
    assert record["best_val_repair_mse"] == 1.
    assert record["validation_repair_metrics"]["window_mse"] == 3.
    payload = training.load_stage1_selection(manifest, protocol, settings, device=torch.device("cpu"))
    assert payload["epoch"] == 2


def test_stage1_config_rejects_unknown_objective(tmp_path):
    protocol, _ = _protocol_settings()
    path = tmp_path / "unknown.yaml"
    path.write_text(
        (ROOT / "configs/e23_multi_expert_p4_width64_window_v12.yaml").read_text()
        + "  stage1_objective: unknown\n"
    )
    with pytest.raises(ValueError, match="Stage 1 objective"):
        load_e23_settings(path, protocol=protocol)
