from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from integrations.e23.train import (load_stage2_selection, train_forecast,
                                    train_history_adapter, train_repair)
from integrations.e23.run import evaluate_selected
from integrations.e23.profile_reference import profile_e23_selected
from integrations.forecast_compare.protocol import P4Batch
from malr.data.windows import WindowRef
from tests.integration.test_e23_stage1 import _protocol_settings


def _batch(protocol, split):
    start = 0 if split == "train" else (8640 if split == "validation" else 11520)
    starts = (start, start + 96)
    clean = torch.stack([torch.from_numpy(protocol.standardized[s:s + 96].copy()) for s in starts])
    mask = torch.ones_like(clean, dtype=torch.bool)
    mask[:, 8:16] = False
    refs = tuple(WindowRef(s, s + 96, s + 96, s + 192, split) for s in starts)
    return P4Batch(clean * mask, clean, mask, torch.zeros(2, 7),
                   torch.ones(2, 7), torch.zeros(2, 7, 96), refs)


def _parent(monkeypatch, tmp_path):
    protocol, settings = _protocol_settings()
    import integrations.e23.train as training
    monkeypatch.setattr(training, "iter_p4_batches",
                        lambda _protocol, split, **_kwargs: iter((_batch(protocol, split),)))
    stage1 = train_repair(protocol, settings, output=tmp_path / "stage1",
                          device=torch.device("cpu"), epochs=1)
    parent = train_forecast(protocol, settings, stage1_manifest=stage1,
                            output=tmp_path / "parent", device=torch.device("cpu"),
                            epochs=1, selection_metric="mae")
    return protocol, settings, parent


@pytest.mark.parametrize("improves", [False, True])
@pytest.mark.parametrize("variant", ["aligned_v1", "aligned_bounded_v2"])
def test_adapter_selects_epoch_zero_or_improved_epoch_and_freezes_parent(monkeypatch, tmp_path, improves, variant):
    protocol, settings, parent = _parent(monkeypatch, tmp_path)
    parent_record = json.loads(parent.read_text())
    parent_payload = load_stage2_selection(parent, protocol, settings, device=torch.device("cpu"))
    import integrations.e23.train as training
    baseline = {"mse": parent_record["best_val_mse"], "mae": parent_record["best_val_mae"]}
    next_metrics = {key: value + (-0.01 if improves else 0.01)
                    for key, value in baseline.items()}
    rows = iter((baseline, next_metrics))
    monkeypatch.setattr(training, "_validation_forecast_metrics",
                        lambda *_args, **_kwargs: next(rows))
    selected = train_history_adapter(
        protocol, settings, parent_selection=parent,
        output=tmp_path / "adapter", device=torch.device("cpu"), epochs=1,
        adapter_variant=variant,
    )
    record = json.loads(selected.read_text())
    assert record["history_adapter"] == variant
    assert record["epoch0_val_mse"] == baseline["mse"]
    assert record["epoch0_val_mae"] == baseline["mae"]
    assert record["best_epoch"] == (1 if improves else 0)
    assert record["best_val_mae"] == (next_metrics if improves else baseline)["mae"]
    payload = load_stage2_selection(selected, protocol, settings, device=torch.device("cpu"))
    assert payload["history_adapter"] == variant
    for name, value in parent_payload["model_state"].items():
        assert torch.equal(value, payload["model_state"][name]), name
    assert payload["memory_sha256"] == parent_payload["memory_sha256"]
    assert payload["key_sha256"] == parent_payload["key_sha256"]
    for field in ("history_adapter", "parent_stage2_checkpoint_sha256",
                  "historical_archive_sha256"):
        tampered = tmp_path / f"bad-{field}.json"
        tampered.write_text(json.dumps({**record, field: "corrupt"}))
        with pytest.raises(ValueError, match="mismatch|unsupported"):
            load_stage2_selection(tampered, protocol, settings, device=torch.device("cpu"))


def test_adapter_rejects_an_mse_selected_parent_before_output(monkeypatch, tmp_path):
    protocol, settings = _protocol_settings()
    import integrations.e23.train as training
    monkeypatch.setattr(training, "iter_p4_batches",
                        lambda _protocol, split, **_kwargs: iter((_batch(protocol, split),)))
    stage1 = train_repair(protocol, settings, output=tmp_path / "stage1",
                          device=torch.device("cpu"), epochs=1)
    parent = train_forecast(protocol, settings, stage1_manifest=stage1,
                            output=tmp_path / "parent", device=torch.device("cpu"),
                            epochs=1, selection_metric="mse")
    with pytest.raises(ValueError, match="MAE-selected parent"):
        train_history_adapter(protocol, settings, parent_selection=parent,
                              output=tmp_path / "adapter", device=torch.device("cpu"), epochs=1)
    assert not (tmp_path / "adapter").exists()


def test_train_adapter_cli_forwards_parent_selection_and_variant(monkeypatch, tmp_path):
    import integrations.e23.run as runner
    _, settings = _protocol_settings()
    monkeypatch.setattr(runner.MigrationCatalog, "load", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(runner, "load_e23_p4_protocol", lambda **_kwargs: object())
    monkeypatch.setattr(runner, "load_e23_settings", lambda *_args, **_kwargs: settings)
    received = []
    def fake_training(*_args, **kwargs):
        received.append((kwargs['parent_selection'], kwargs['adapter_variant']))
        manifest = tmp_path / 'selected.json'
        manifest.write_text(json.dumps({'epochs_completed': 100}))
        return manifest
    monkeypatch.setattr(runner, "train_history_adapter", fake_training)
    parent = tmp_path / "parent.json"
    runner.main([
        "train-adapter", "--protocol-config", "protocol.yaml",
        "--e23-config", "e23.yaml", "--migration-catalog", "catalog.jsonl",
        "--mask-artifacts", "masks", "--parent-selection", str(parent),
        "--output", str(tmp_path / "adapter"), "--device", "cpu",
        "--adapter-variant", "aligned_bounded_v2",
    ])
    assert received == [(parent, "aligned_bounded_v2")]


@pytest.mark.parametrize("variant", ["aligned_v1", "aligned_bounded_v2"])
def test_adapter_selected_evaluation_and_profile_reconstruct_paired_model(monkeypatch, tmp_path, variant):
    protocol, settings, parent = _parent(monkeypatch, tmp_path)
    selected = train_history_adapter(protocol, settings, parent_selection=parent,
                                      output=tmp_path / "adapter", device=torch.device("cpu"),
                                      epochs=1, adapter_variant=variant)
    sample = _batch(protocol, "train")
    simulated = replace(
        sample,
        normalized_observed=sample.normalized_observed.repeat(64, 1, 1),
        normalized_target=sample.normalized_target.repeat(64, 1, 1),
        observed_mask=sample.observed_mask.repeat(64, 1, 1),
        query_mean=sample.query_mean.repeat(64, 1),
        query_std=sample.query_std.repeat(64, 1),
        forecast_target=sample.forecast_target.repeat(64, 1, 1),
        refs=sample.refs * 64,
    )
    import integrations.e23.run as runner
    monkeypatch.setattr(runner, "iter_p4_batches",
                        lambda *_args, **_kwargs: iter((simulated for _ in range(21))))
    evaluation = evaluate_selected(protocol, settings, selection=selected,
                                   output=tmp_path / "evaluation.json", device=torch.device("cpu"))
    assert evaluation["history_adapter"] == variant
    assert evaluation["adapter_diagnostics"]["availability_rate"] >= 0
    assert len(evaluation["adapter_diagnostics"]["gates"]) == 7
    assert evaluation["h_state_unchanged"] is True
    import integrations.e23.profile_reference as profiler
    monkeypatch.setattr(profiler, "iter_p4_batches",
                        lambda *_args, **_kwargs: iter((simulated for _ in range(2))))
    profile = profile_e23_selected(protocol=protocol, settings=settings, selection=selected,
                                   device=torch.device("cpu"), warmup_batches=1, batches=1)
    assert profile["history_adapter"] == variant
    assert "history_adapter" in profile["segments_seconds"]
    record = json.loads(selected.read_text())
    tampered = tmp_path / "tampered-adapter-selection.json"
    tampered.write_text(json.dumps({**record, "parent_stage2_checkpoint_sha256": "corrupt"}))
    seen = []
    monkeypatch.setattr(runner, "iter_p4_batches",
                        lambda *_args, **_kwargs: seen.append(True))
    with pytest.raises(ValueError, match="parent Stage 2 checkpoint mismatch"):
        evaluate_selected(protocol, settings, selection=tampered,
                          output=tmp_path / "never-evaluate.json", device=torch.device("cpu"))
    assert not seen


def test_variant_tamper_is_rejected_before_test_batches(monkeypatch, tmp_path):
    protocol, settings, parent = _parent(monkeypatch, tmp_path)
    selected = train_history_adapter(protocol, settings, parent_selection=parent,
                                      output=tmp_path / "adapter", device=torch.device("cpu"),
                                      epochs=1, adapter_variant="aligned_bounded_v2")
    record = json.loads(selected.read_text())
    tampered = tmp_path / "tampered-variant.json"
    tampered.write_text(json.dumps({**record, "history_adapter": "aligned_v1"}))
    import integrations.e23.run as runner
    seen = []
    monkeypatch.setattr(runner, "iter_p4_batches", lambda *_a, **_k: seen.append(True))
    with pytest.raises(ValueError, match="history adapter mismatch"):
        evaluate_selected(protocol, settings, selection=tampered,
                          output=tmp_path / "never-evaluate.json", device=torch.device("cpu"))
    assert not seen
