from pathlib import Path

import pytest

from malr.config import config_sha256, load_config


def test_stage1_defaults_are_locked() -> None:
    cfg = load_config(Path("configs/stage1.yaml"))

    assert (cfg.data.seq_len, cfg.data.pred_len) == (96, 96)
    assert (cfg.patch.length, cfg.patch.stride, cfg.patch.count) == (16, 8, 12)
    assert cfg.missing.start_rate == pytest.approx(0.06)
    assert cfg.relations.topk == {
        "temporal": 6,
        "variable": 4,
        "historical": 4,
    }
    etth1_model = cfg.model.for_dataset("ETTh1")
    assert (
        etth1_model.d_model,
        etth1_model.n_heads,
        etth1_model.d_ff,
        etth1_model.encoder_layers,
        etth1_model.dropout,
    ) == (16, 4, 128, 3, 0.3)
    weather_model = cfg.model.for_dataset("Weather")
    assert (
        weather_model.d_model,
        weather_model.n_heads,
        weather_model.d_ff,
        weather_model.encoder_layers,
        weather_model.dropout,
    ) == (128, 16, 256, 3, 0.2)
    etth1_training = cfg.training.for_dataset("ETTh1")
    weather_training = cfg.training.for_dataset("Weather")
    assert (
        etth1_training.batch_size,
        etth1_training.max_epochs,
        etth1_training.early_stopping_patience,
        etth1_training.lr_schedule,
    ) == (128, 100, 100, "type3")
    assert (
        weather_training.batch_size,
        weather_training.max_epochs,
        weather_training.early_stopping_patience,
        weather_training.lr_schedule,
    ) == (128, 100, 20, "type3")
    assert cfg.training.seed == 2021


def test_invalid_multilayer_recovery_is_rejected(tmp_path: Path) -> None:
    text = Path("configs/stage1.yaml").read_text(encoding="utf-8")
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        text.replace(
            "num_message_passing_layers: 1",
            "num_message_passing_layers: 2",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="num_message_passing_layers must equal 1"):
        load_config(bad)


def test_config_hash_is_stable() -> None:
    cfg = load_config(Path("configs/stage1.yaml"))

    assert config_sha256(cfg) == config_sha256(cfg)
    assert len(config_sha256(cfg)) == 64


def test_sr010_no_recovery_loss_keeps_historical_relations_enabled() -> None:
    cfg = load_config(Path("configs/stage1_sr010_no_recloss.yaml"))

    assert cfg.missing.start_rate == pytest.approx(0.10)
    assert cfg.ablations.use_temporal is True
    assert cfg.ablations.use_variable is True
    assert cfg.ablations.use_historical is True
    assert cfg.ablations.use_recovery_loss is False
