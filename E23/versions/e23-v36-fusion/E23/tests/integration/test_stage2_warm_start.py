from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
import torch

from malr import cli
from malr.config import load_config
from malr.data.splits import protocol_for
from malr.data.masks import generate_variable_missing_mask, save_fixed_mask
from malr.evaluation.report import file_sha256
from malr.training.online_checkpoint import load_online_checkpoint, save_online_checkpoint
from malr.training.online_trainer import OnlineTrainer
from tests.integration.test_dynamic_context_model import _dynamic_fixture
from tests.integration.test_online_trainer import _batch
from tests.integration.test_training_provenance import _series


def _forecast_fixture():
    model, data = _dynamic_fixture()
    model.begin_epoch(1)
    model.end_epoch(1)
    model.install_forecast_predictor()
    teacher_hash = model.install_frozen_route_teacher()
    model.install_forecast_hv_fusion()
    model.configure_forecast_finetuning(frozen_route_teacher=True, forecast_fusion="teacher_hv")
    return model, data, teacher_hash


def _checkpoint(path, model, teacher_hash, **overrides):
    state = model.online_state()
    metadata = {
        "dataset": "ETTh1", "training_stage": "forecast",
        "stage1_checkpoint_sha256": "s" * 64,
        "stage2_route_teacher": "frozen_stage1_hv_fusion",
        "stage2_route_teacher_sha256": teacher_hash,
        **overrides,
    }
    save_online_checkpoint(
        path, model.state_dict(), state["memory_state"], state["ema_state"],
        state["relation_prior_state"], metadata, epoch=state["epoch"],
        config_hash="c", data_hash="d", mask_hash="m", memory_hash=state["memory_hash"],
    )


def _load(model, path, teacher_hash):
    from malr.training.warm_start import load_forecast_warm_start

    return load_forecast_warm_start(
        model, path, source_epoch=22, dataset="ETTh1",
        expected_hashes={"config_hash": "c", "data_hash": "d", "mask_hash": "m"},
        stage1_checkpoint_sha256="s" * 64,
        teacher_mode="frozen_stage1_hv_fusion", teacher_sha256=teacher_hash,
    )


def test_warm_start_restores_full_student_and_preserves_stage1_teacher_and_memory(tmp_path):
    model, _, teacher_hash = _forecast_fixture()
    trained, _, _ = _forecast_fixture()
    trained.load_state_dict(model.state_dict())
    state = model.online_state()
    trained.load_online_state({
        **state,
        "hashes": {"memory_hash": state["memory_hash"]},
    })
    with torch.no_grad():
        for parameter in trained.parameters():
            if parameter.requires_grad:
                parameter.add_(0.0123)
    source = tmp_path / "stage2.pt"
    _checkpoint(source, trained, teacher_hash)
    before = model.online_state()
    _load(model, source, teacher_hash)
    for name, value in trained.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
    assert model.route_teacher.state_sha256() == teacher_hash
    assert model.online_state()["epoch"] == before["epoch"] == 1
    for key in ("memory_hash", "ema_hash", "relation_prior_hash"):
        assert model.online_state()[key] == before[key]


@pytest.mark.parametrize("overrides", [
    {"training_stage": "repair"}, {"dataset": "wrong"},
    {"stage1_checkpoint_sha256": "wrong"}, {"stage2_route_teacher": "frozen_stage1"},
    {"stage2_route_teacher_sha256": "wrong"}, {"training_epoch": 21},
])
def test_warm_start_rejects_wrong_lineage(tmp_path, overrides):
    model, _, teacher_hash = _forecast_fixture()
    source = tmp_path / "bad.pt"
    _checkpoint(source, model, teacher_hash, **overrides)
    with pytest.raises(ValueError):
        _load(model, source, teacher_hash)


@pytest.mark.parametrize("change", [None, "raw_patch", "clean_key"])
def test_nonempty_memory_allows_only_key_normalization_roundoff(tmp_path, change):
    model, _, teacher_hash = _forecast_fixture()
    rng = np.random.default_rng(17)
    for start in range(0, 113, 8):
        model.memory.stage_patch(0, start, np.arange(16, dtype=np.float64),
                                 rng.normal(size=8), observed=True)
    model.memory.publish_epoch(2)
    source = tmp_path / "nonempty.pt"
    _checkpoint(source, model, teacher_hash)
    original_hash = model.online_state()["memory_hash"]
    if change:
        payload = load_online_checkpoint(source)
        member = payload["online_state"]["memory_state"]["snapshot"]["groups"][0][0]["members"][0]
        member[change] = member[change].copy()
        member[change][0] += 1e-8 if change == "raw_patch" else .001
        torch.save(payload, source)
        with pytest.raises(ValueError, match="memory"):
            _load(model, source, teacher_hash)
    else:
        _load(model, source, teacher_hash)
        # Idempotent key normalization now keeps saved bank bytes unchanged.
        # The corruption cases above still reject altered raw values or keys.
        assert model.online_state()["memory_hash"] == original_hash


@pytest.mark.parametrize("improves", [False, True])
def test_warm_start_trainer_retains_initial_best_and_separates_epoch_clocks(
    tmp_path, monkeypatch, capsys, improves,
):
    model, data, _ = _forecast_fixture()
    initial_state = deepcopy(model.state_dict())
    trainer = OnlineTrainer(
        model, checkpoint_path=tmp_path / "best.pt", checkpoint_metadata={},
        max_epochs=2, patience=2, learning_rate=1e-4,
    )
    validation = iter([(2., 1.), (1., .9), (.5, .8)] if improves else [(0., 0.), (1., 1.), (2., 2.)])
    monkeypatch.setattr(trainer, "_validation", lambda batches: next(validation))
    epochs = []
    batch = _batch(data)

    def batches(epoch):
        epochs.append(epoch)
        return [batch]

    result = trainer.fit(
        batches, lambda: [batch], config_hash="c", data_hash="d", mask_hash="m",
        objective="forecast", epoch_offset=22, model_epoch_offset=23,
        stage_label="forecast", freeze_online_state=True, preserve_initial_best=True,
    )
    payload = load_online_checkpoint(result.best_checkpoint)
    assert epochs == [22, 23]
    assert result.epochs_completed == 24
    assert result.best_epoch == (24 if improves else 22)
    assert payload["metadata"]["training_epoch"] == result.best_epoch
    assert payload["online_state"]["epoch"] == payload["metadata"]["epoch"] == 1
    assert "epoch=23/24 phase=forecast start" in capsys.readouterr().out
    if not improves:
        for name, value in initial_state.items():
            torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)


def test_warm_start_cli_options_parse():
    args = cli.build_parser().parse_args([
        "train", "--config", "config.yaml", "--dataset", "ETTm2", "--artifacts", "out",
        "--variant", "ours", "--stage", "forecast", "--init-checkpoint", "stage1.pt",
        "--warm-start-checkpoint", "stage2.pt", "--warm-start-epoch", "22",
        "--additional-epochs", "78",
    ])
    assert args.warm_start_epoch == 22
    assert args.additional_epochs == 78
    assert args.warm_start_checkpoint == Path("stage2.pt")


@pytest.mark.parametrize("options", [
    {"additional_epochs": 78},
    {"warm_start_checkpoint_path": Path("missing.pt"), "warm_start_epoch": 22},
    {"warm_start_checkpoint_path": Path("missing.pt"), "warm_start_epoch": 0, "additional_epochs": 78},
    {"warm_start_checkpoint_path": Path("missing.pt"), "warm_start_epoch": 22, "additional_epochs": -1},
])
def test_invalid_warm_start_options_fail_before_artifact_loading(tmp_path, options):
    config = load_config(Path("configs/stage1_seq96_relational_recovery.yaml"))
    series = _series(tmp_path)
    with pytest.raises(ValueError, match="warm.start"):
        cli.run_training(
            config, series, protocol_for("ETTh1", len(series.values)),
            "ETTh1", "ours", tmp_path / "out", device="cpu", stage="forecast",
            init_checkpoint_path=Path("stage1.pt"), **options,
        )


def test_cli_warm_start_completes_with_real_training_and_records_lineage(tmp_path, capsys):
    config = load_config(Path("configs/stage1_seq96_dynamic_temporal_repair.yaml"))
    profile = replace(config.training.for_dataset("ETTh1"), batch_size=16,
                      max_epochs=1, early_stopping_patience=1, drop_last=False)
    dataset = replace(config.data.datasets["ETTh1"], train_end=200,
                      validation_end=300, test_end=400)
    config = replace(
        config, training=replace(config.training, profiles={"ETTh1": profile}),
        data=replace(config.data, datasets={"ETTh1": dataset}),
        relations=replace(config.relations, online=replace(
            config.relations.online, stage1_repair_epochs=1,
        )),
    )
    series = _series(tmp_path)
    protocol = cli._protocol_for_config(config, "ETTh1", len(series.values))
    old = tmp_path / "original"
    start, end, _ = cli._mask_identity(config, "ETTh1", protocol, "validation")
    mask, metadata = generate_variable_missing_mask(
        end - start, 7, .1, 8, config.training.seed, dataset="ETTh1",
        split="validation", support_start=start,
    )
    mask_path = cli._fixed_mask_path(old, "ETTh1", "validation")
    save_fixed_mask(mask_path, mask, metadata)
    repair = cli.run_training(config, series, protocol, "ETTh1", "ours", old,
                              device="cpu", stage="repair")
    stage1 = Path(repair["best_checkpoint"])
    forecast = cli.run_training(
        config, series, protocol, "ETTh1", "ours", old,
        device="cpu", stage="forecast", init_checkpoint_path=stage1,
    )
    source = Path(forecast["best_checkpoint"])
    original_digest = file_sha256(source)
    new = tmp_path / "continuation"
    new_mask = cli._fixed_mask_path(new, "ETTh1", "validation")
    new_mask.parent.mkdir(parents=True)
    shutil.copy2(mask_path, new_mask)
    result = cli.run_training(
        config, series, protocol, "ETTh1", "ours", new,
        device="cpu", stage="forecast", init_checkpoint_path=stage1,
        warm_start_checkpoint_path=source, warm_start_epoch=1, additional_epochs=2,
    )
    progress_output = capsys.readouterr().err
    assert "phase=train ETTh1 ours forecast epoch=2/3" in progress_output
    assert "phase=validation ETTh1 ours forecast epoch=2/3" in progress_output
    assert "phase=train ETTh1 ours forecast epoch=3/3" in progress_output
    assert result["epochs_completed"] == 3
    assert result["additional_epochs_completed"] == 2
    assert result["best_val_mse"] <= forecast["best_val_mse"] + 1e-10
    assert file_sha256(source) == original_digest
    manifest = json.loads(Path(result["run_manifest"]).read_text())
    lineage = manifest["metadata"]["stage2_warm_start"]
    assert lineage["checkpoint_sha256"] == original_digest
    assert lineage["planned_final_epoch"] == 3
    assert lineage["optimizer_state_restored"] is False
    assert manifest["metadata"]["stage2_route_teacher_invariance_verified"] is True
    assert manifest["metadata"]["config_hash"] == load_online_checkpoint(source)["metadata"]["config_hash"]
    with pytest.raises(ValueError, match="refusing to overwrite"):
        cli.run_training(
            config, series, protocol, "ETTh1", "ours", new, device="cpu", stage="forecast",
            init_checkpoint_path=stage1, warm_start_checkpoint_path=source,
            warm_start_epoch=1, additional_epochs=2,
        )
