from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from integrations.e23.config import load_e23_p4_protocol, load_e23_settings
from integrations.e23.train import (_retrieval_loss, build_repair_model,
                                    load_stage1_selection, stage1_objective,
                                    train_repair)
from integrations.forecast_compare.protocol import P4Batch
from malr.data.windows import WindowRef
from malr.migration import MigrationCatalog
from malr.models.e23.model import sample_specialist_exposure


ROOT = Path(__file__).resolve().parents[2]
CATALOG = MigrationCatalog.load(ROOT.parent / "E22/migration/paper_catalog.jsonl", project="E23")
MASKS = ROOT.parent / "E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context"


def _protocol_settings():
    protocol = load_e23_p4_protocol(
        catalog=CATALOG,
        protocol_config=ROOT / "configs/stage1_seq96_dynamic_temporal_repair.yaml",
        mask_artifacts=MASKS,
    )
    settings = load_e23_settings(ROOT / "configs/e23_multi_expert_p4.yaml", protocol=protocol)
    return protocol, settings


def _batch(split):
    torch.manual_seed(18)
    target = torch.randn(2, 96, 7)
    mask = torch.ones_like(target, dtype=torch.bool)
    mask[:, 8:16, :] = False
    refs = tuple(WindowRef(i * 96, (i + 1) * 96, (i + 2) * 96,
                           (i + 3) * 96, split) for i in range(2))
    return P4Batch(
        target * mask, target, mask,
        torch.zeros(2, 7), torch.ones(2, 7),
        torch.zeros(2, 7, 96), refs,
    )


def test_stage1_loss_and_snapshot_pairing(monkeypatch, tmp_path):
    protocol, settings = _protocol_settings()
    import integrations.e23.train as train_module
    monkeypatch.setattr(train_module, "iter_p4_batches",
                        lambda _protocol, split, **kwargs: iter((_batch(split),)))
    manifest_path = train_repair(protocol, settings, output=tmp_path / "run",
                                 device=torch.device("cpu"), epochs=2)
    record = json.loads(manifest_path.read_text())
    assert record["stage"] == "repair"
    assert record["epochs_completed"] == 2
    assert record["best_epoch"] in (1, 2)
    assert record["source_csv_sha256"] == protocol.source_sha256
    assert record["e23_config_sha256"] == settings.config_sha256
    payload = load_stage1_selection(manifest_path, protocol, settings,
                                    device=torch.device("cpu"))
    assert payload["memory_state"]["snapshot"]["epoch"] == record["best_epoch"]
    assert payload["retrieval_state"]["frozen_epoch"] == record["best_epoch"]
    assert record["memory_sha256"] == payload["memory_sha256"]
    assert record["key_sha256"] == payload["key_sha256"]
    assert record["checkpoint_sha256"]

    # Formula uses only hidden, valid coordinates and the retrieval representation term.
    candidate = torch.tensor([[[1.], [2.]]], requires_grad=True)
    target = torch.tensor([[[0.], [100.]]])
    observed = torch.tensor([[[False], [True]]])
    total, parts = stage1_objective(candidate, target, observed, torch.tensor(0.5),
                                     retrieval_weight=0.1)
    torch.testing.assert_close(total, torch.tensor(0.55))
    torch.testing.assert_close(parts["repair"], torch.tensor(0.5))
    torch.testing.assert_close(parts["retrieval_weighted"], torch.tensor(0.05))


def test_specialist_exposure_is_reproducible_and_never_forces_an_unavailable_expert():
    damaged = torch.ones(1, 5, 10, dtype=torch.bool)
    available = torch.ones(1, 5, 10, 3, dtype=torch.bool)
    available[..., 1] = False
    available[:, 0, :5] = False
    first, selected = sample_specialist_exposure(damaged, available, fraction=0.2, seed=2021)
    second, selected_again = sample_specialist_exposure(damaged, available, fraction=0.2, seed=2021)
    assert torch.equal(first, second)
    assert torch.equal(selected, selected_again)
    assert selected.sum() == 10
    assert not (first & ~available).any()
    assert first[selected].sum(-1).le(1).all()
    assert torch.equal(first[~selected], available[~selected])


def test_retrieval_clean_key_ignores_final_patch_padding():
    protocol, settings = _protocol_settings()
    model = build_repair_model(protocol, settings, device=torch.device("cpu"))
    base = _batch("train")
    mask = torch.ones_like(base.observed_mask)
    mask[:, -1, 0] = False
    batch = replace(base, observed_mask=mask,
                    normalized_observed=base.normalized_target * mask)
    seen_masks = []
    handle = model.historical.key_encoder.register_forward_hook(
        lambda _module, args, _output: seen_masks.append(args[1].clone()),
    )
    try:
        loss = _retrieval_loss(model, batch)
    finally:
        handle.remove()
    assert torch.isfinite(loss)
    assert len(seen_masks) == 2
    clean_masks = seen_masks[1]
    assert (~clean_masks[..., -8:]).all(dim=-1).any()


def test_new_trial_stage1_identity_is_paired_and_checked_before_batches(monkeypatch, tmp_path):
    protocol, _ = _protocol_settings()
    settings = load_e23_settings(ROOT / "configs/e23_multi_expert_p4_width64_topp.yaml",
                                 protocol=protocol)
    import integrations.e23.train as train_module
    monkeypatch.setattr(train_module, "iter_p4_batches",
                        lambda _protocol, split, **kwargs: iter((_batch(split),)))
    manifest = train_repair(protocol, settings, output=tmp_path / "top-p-stage1",
                            device=torch.device("cpu"), epochs=1)
    record = json.loads(manifest.read_text())
    payload = torch.load(record["checkpoint"], map_location="cpu", weights_only=False)
    expected = {"repair_width": 64, "candidate_selection": "top_p",
                "historical_top_p": 0.9, "variable_top_p": 0.9,
                "historical_candidate_cap": 8}
    assert {key: record[key] for key in expected} == expected
    assert {key: payload[key] for key in expected} == expected
    assert load_stage1_selection(manifest, protocol, settings,
                                 device=torch.device("cpu"))["epoch"] == 1
    tampered = tmp_path / "wrong-trial.json"
    tampered.write_text(json.dumps({**record, "candidate_selection": "top_k"}))
    seen = []
    monkeypatch.setattr(train_module, "iter_p4_batches",
                        lambda *_args, **_kwargs: seen.append(True))
    with pytest.raises(ValueError, match="candidate_selection.*mismatch"):
        load_stage1_selection(tampered, protocol, settings, device=torch.device("cpu"))
    assert not seen


def test_contextual_v_stage1_identity_is_paired_and_checked_before_batches(monkeypatch, tmp_path):
    protocol, _ = _protocol_settings()
    settings = load_e23_settings(ROOT / "configs/e23_multi_expert_p4_width64_window_v12.yaml",
                                 protocol=protocol)
    import integrations.e23.train as train_module
    monkeypatch.setattr(train_module, "iter_p4_batches",
                        lambda _protocol, split, **kwargs: iter((_batch(split),)))
    manifest = train_repair(protocol, settings, output=tmp_path / "window-stage1",
                            device=torch.device("cpu"), epochs=1)
    record = json.loads(manifest.read_text())
    payload = torch.load(record["checkpoint"], map_location="cpu", weights_only=False)
    assert record["variable_selection"] == payload["variable_selection"] == "window_context_topk_v1"
    assert load_stage1_selection(manifest, protocol, settings,
                                 device=torch.device("cpu"))["epoch"] == 1
    tampered = tmp_path / "wrong-window.json"
    tampered.write_text(json.dumps({**record, "variable_selection": "global_topk"}))
    seen = []
    monkeypatch.setattr(train_module, "iter_p4_batches",
                        lambda *_args, **_kwargs: seen.append(True))
    with pytest.raises(ValueError, match="variable_selection.*mismatch"):
        load_stage1_selection(tampered, protocol, settings, device=torch.device("cpu"))
    assert not seen
