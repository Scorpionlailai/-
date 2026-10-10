from dataclasses import replace
import json

import pytest
import torch

from integrations.e23.run import evaluate_selected, shared_drr
from integrations.e23.profile_reference import profile_e23_selected
from integrations.e23.train import train_forecast
from tests.integration.test_e23_stage1 import _batch, _protocol_settings
from tests.integration.test_e23_stage2 import _selected_stage1


def _selected_stage2(monkeypatch, tmp_path, *, selection_metric="mse", forecast_mixer="none"):
    protocol, settings, stage1 = _selected_stage1(monkeypatch, tmp_path)
    stage2 = train_forecast(protocol, settings, stage1_manifest=stage1,
                            output=tmp_path / "stage2", device=torch.device("cpu"), epochs=1,
                            selection_metric=selection_metric, forecast_mixer=forecast_mixer)
    return protocol, settings, stage2


def test_selected_test_rejects_mismatch_before_a_test_batch(monkeypatch, tmp_path):
    protocol, settings, selection = _selected_stage2(monkeypatch, tmp_path)
    original = json.loads(selection.read_text())
    import integrations.e23.run as run_module
    seen = []
    monkeypatch.setattr(run_module, "iter_p4_batches",
                        lambda *_args, **_kwargs: seen.append(True))
    for field in ("source_csv_sha256", "test_mask_sha256", "e23_config_sha256",
                  "checkpoint_sha256"):
        modified = dict(original)
        modified[field] = "corrupt"
        tampered = tmp_path / f"bad-{field}.json"
        tampered.write_text(json.dumps(modified))
        with pytest.raises(ValueError, match="mismatch"):
            evaluate_selected(protocol, settings, selection=tampered,
                              output=tmp_path / "evaluation.json", device=torch.device("cpu"))
    existing = tmp_path / "existing.json"
    existing.write_text("{}")
    with pytest.raises(FileExistsError):
        evaluate_selected(protocol, settings, selection=selection,
                          output=existing, device=torch.device("cpu"))
    assert not seen


@pytest.mark.parametrize("selection_metric,forecast_mixer", [
    ("mse", "none"), ("mae", "none"), ("mae", "patch_attention_v1"),
])
def test_selected_test_reports_all_windows_both_drrs_and_h_diagnostics(
    monkeypatch, tmp_path, selection_metric, forecast_mixer,
):
    protocol, settings, selection = _selected_stage2(
        monkeypatch, tmp_path, selection_metric=selection_metric,
        forecast_mixer=forecast_mixer,
    )
    base = _batch("test")
    batch = replace(
        base,
        normalized_observed=base.normalized_observed.repeat(64, 1, 1),
        normalized_target=base.normalized_target.repeat(64, 1, 1),
        observed_mask=base.observed_mask.repeat(64, 1, 1),
        query_mean=base.query_mean.repeat(64, 1),
        query_std=base.query_std.repeat(64, 1),
        forecast_target=base.forecast_target.repeat(64, 1, 1),
        refs=base.refs * 64,
    )
    import integrations.e23.run as run_module
    monkeypatch.setattr(run_module, "iter_p4_batches",
                        lambda *_args, **_kwargs: iter((batch for _ in range(21))))
    output = tmp_path / "selected-evaluation.json"
    result = evaluate_selected(protocol, settings, selection=selection,
                               output=output, device=torch.device("cpu"))
    assert result["windows"] == 2688
    assert result["mse"] >= 0 and result["mae"] >= 0
    assert result["drr_mse_percent"] is not None
    assert result["drr_mae_percent"] is not None
    assert set(result["expert_diagnostics"]["availability_rate"]) == {"none", "historical", "variable", "temporal"}
    assert set(result["expert_diagnostics"]["weight_mean"]) == {"none", "historical", "variable", "temporal"}
    assert result["h_state_unchanged"] is True
    assert result["e23_config_sha256"] == settings.config_sha256
    assert result["selection_metric"] == selection_metric
    assert result["forecast_mixer"] == forecast_mixer
    assert json.loads(output.read_text())["checkpoint_sha256"] == result["checkpoint_sha256"]


@pytest.mark.parametrize("flags,selection,mixer", [
    ((), "mse", "none"),
    (("--selection-metric", "mae"), "mae", "none"),
    (("--selection-metric", "mae", "--forecast-mixer", "patch_attention_v1"),
     "mae", "patch_attention_v1"),
])
def test_train_forecast_cli_forwards_selection_and_mixer(
    monkeypatch, tmp_path, flags, selection, mixer,
):
    import integrations.e23.run as run_module
    _, settings = _protocol_settings()
    monkeypatch.setattr(run_module.MigrationCatalog, "load", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(run_module, "load_e23_p4_protocol", lambda **_kwargs: object())
    monkeypatch.setattr(run_module, "load_e23_settings", lambda *_args, **_kwargs: settings)
    received = []
    def fake_training(*_args, **kwargs):
        received.append((kwargs['selection_metric'], kwargs['forecast_mixer']))
        manifest = tmp_path / 'selected.json'
        manifest.write_text(json.dumps({'epochs_completed': 100, 'training_seconds': 1.}))
        return manifest
    monkeypatch.setattr(run_module, "train_forecast", fake_training)
    args = [
        "train-forecast", "--protocol-config", "protocol.yaml",
        "--e23-config", "e23.yaml", "--migration-catalog", "catalog.jsonl",
        "--mask-artifacts", "masks", "--stage1-manifest", "stage1.json",
        "--output", str(tmp_path / "stage2"), "--device", "cpu",
    ]
    args.extend(flags)
    assert run_module.main(args) == 0
    assert received == [(selection, mixer)]
    resources = json.loads(next((tmp_path / 'stage2/resources').glob('*.json')).read_text())
    assert resources['status'] == 'completed'
    assert resources['batch_size'] == settings.batch_size
    assert resources['full_workflow_seconds'] >= resources['training_stages_seconds']
    assert resources['epochs_completed'] == 100


def test_zero_missing_clean_gap_has_undefined_drr():
    assert shared_drr(0.5, clean=0.4, missing=0.4) is None
    assert shared_drr(0.5, clean=0.3, missing=0.5) == pytest.approx(0.0)


@pytest.mark.parametrize("forecast_mixer", ["none", "patch_attention_v1"])
def test_e23_profile_reports_real_segments_without_h_state_change(
    monkeypatch, tmp_path, forecast_mixer,
):
    protocol, settings, selection = _selected_stage2(
        monkeypatch, tmp_path, forecast_mixer=forecast_mixer,
    )
    base = _batch("validation")
    import integrations.e23.profile_reference as profiler
    monkeypatch.setattr(profiler, "iter_p4_batches",
                        lambda *_args, **_kwargs: iter((base for _ in range(3))))
    report = profile_e23_selected(
        protocol=protocol, settings=settings, selection=selection,
        device=torch.device("cpu"), warmup_batches=1, batches=2,
    )
    assert report["target"] == "E23"
    assert report["forecast_mixer"] == forecast_mixer
    assert report["h_state_unchanged"] is True
    assert report["timed_batches"] == 2
    assert report["model_seconds"] >= 0
    assert report["pipeline_seconds"] >= report["model_seconds"]
    assert set(report["segments_seconds"]) >= {
        "historical", "variable", "temporal", "router", "decoder", "predictor",
    }
    assert all(value >= 0 for value in report["segments_seconds"].values())
