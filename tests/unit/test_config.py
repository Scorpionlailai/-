from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import pytest
import yaml

from malr.config import RecoveryConfig, config_sha256, load_config


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


def test_legacy_config_defaults_to_additive_recovery() -> None:
    cfg = load_config(Path("configs/stage1.yaml"))

    assert cfg.recovery == RecoveryConfig()


def test_residual_experiment_config_is_explicit() -> None:
    cfg = load_config(Path("configs/stage1_sr010_residual_no_recloss.yaml"))

    assert cfg.missing.start_rate == pytest.approx(0.10)
    assert cfg.recovery.fusion == "residual_mlp"
    assert cfg.recovery.hidden_multiplier == 2
    assert cfg.recovery.dropout == pytest.approx(0.0)
    assert cfg.recovery.output_init == "zero"
    assert cfg.recovery.damage_scaling == "missing_fraction"
    assert cfg.ablations.use_recovery_loss is False
    assert cfg.ablations.use_temporal is True
    assert cfg.ablations.use_variable is True
    assert cfg.ablations.use_historical is True


@pytest.mark.parametrize(
    ("path", "refiner_enabled"),
    [
        ("configs/stage1_sr010_replacement_no_recloss.yaml", False),
        ("configs/stage1_sr010_replacement_global_no_recloss.yaml", True),
    ],
)
def test_replacement_experiment_configs_are_explicit(
    path: str, refiner_enabled: bool
) -> None:
    cfg = load_config(Path(path))

    assert cfg.missing.start_rate == pytest.approx(0.10)
    assert cfg.recovery.fusion == "replacement_mlp"
    assert cfg.recovery.refiner_enabled is refiner_enabled
    assert cfg.recovery.refiner_heads == 4
    assert cfg.recovery.refiner_ff_multiplier == 2
    assert cfg.recovery.refiner_dropout == pytest.approx(0.0)
    assert cfg.recovery.refiner_output_init == "zero"
    assert cfg.ablations.use_recovery_loss is False


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("fusion", "unknown"),
        ("hidden_multiplier", 0),
        ("hidden_multiplier", 1.5),
        ("hidden_multiplier", True),
        ("dropout", -0.1),
        ("dropout", 1.0),
        ("output_init", "default"),
        ("damage_scaling", "none"),
        ("refiner_heads", 0),
        ("refiner_heads", True),
        ("refiner_ff_multiplier", 0),
        ("refiner_ff_multiplier", True),
        ("refiner_dropout", -0.1),
        ("refiner_dropout", 1.0),
        ("refiner_output_init", "default"),
    ],
)
def test_invalid_recovery_config_is_rejected(
    tmp_path: Path, key: str, value: object
) -> None:
    raw = yaml.safe_load(Path("configs/stage1.yaml").read_text(encoding="utf-8"))
    raw["recovery"] = asdict(RecoveryConfig())
    raw["recovery"][key] = value
    bad = tmp_path / f"bad_recovery_{key}.yaml"
    bad.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="recovery"):
        load_config(bad)


def test_enabled_global_refiner_requires_replacement_fusion(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("configs/stage1.yaml").read_text(encoding="utf-8"))
    raw["recovery"] = asdict(RecoveryConfig())
    raw["recovery"]["refiner_enabled"] = True
    bad = tmp_path / "bad_refiner_fusion.yaml"
    bad.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="refiner.*replacement_mlp"):
        load_config(bad)


def test_enabled_global_refiner_heads_must_divide_model_width(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("configs/stage1.yaml").read_text(encoding="utf-8"))
    raw["recovery"] = asdict(RecoveryConfig())
    raw["recovery"].update(
        fusion="replacement_mlp",
        refiner_enabled=True,
        refiner_heads=3,
    )
    bad = tmp_path / "bad_refiner_heads.yaml"
    bad.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="refiner_heads"):
        load_config(bad)


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


def test_legacy_config_hash_omits_default_recovery() -> None:
    cfg = load_config(Path("configs/stage1.yaml"))
    payload_mapping = asdict(cfg)
    payload_mapping.pop("recovery")
    payload = json.dumps(
        payload_mapping,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    assert config_sha256(cfg) == sha256(payload).hexdigest()
    residual = load_config(Path("configs/stage1_sr010_residual_no_recloss.yaml"))
    assert config_sha256(residual) != config_sha256(cfg)


def test_sr010_no_recovery_loss_keeps_historical_relations_enabled() -> None:
    cfg = load_config(Path("configs/stage1_sr010_no_recloss.yaml"))

    assert cfg.missing.start_rate == pytest.approx(0.10)
    assert cfg.ablations.use_temporal is True
    assert cfg.ablations.use_variable is True
    assert cfg.ablations.use_historical is True
    assert cfg.ablations.use_recovery_loss is False
