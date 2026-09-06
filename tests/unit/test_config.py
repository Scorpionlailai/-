from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path

import pytest
import yaml

from malr.config import EdgeFilterConfig, RecoveryConfig, config_sha256, load_config


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
    assert cfg.training.lambda_imputation_mse == pytest.approx(0.0)
    assert cfg.training.lambda_imputation_mae == pytest.approx(0.0)


def test_imputation_supervision_config_sets_both_weights() -> None:
    cfg = load_config(
        Path("configs/stage1_sr010_replacement_filter_imputation.yaml")
    )

    assert cfg.training.lambda_imputation_mse == pytest.approx(0.1)
    assert cfg.training.lambda_imputation_mae == pytest.approx(0.1)
    assert cfg.ablations.use_recovery_loss is False
    assert config_sha256(cfg) != config_sha256(
        load_config(Path("configs/stage1_sr010_replacement_filter_fast_no_recloss.yaml"))
    )


def test_wide_strict_filter_config_changes_only_relation_capacity_and_thresholds() -> None:
    baseline = load_config(
        Path("configs/stage1_sr010_replacement_filter_imputation.yaml")
    )
    tuned = load_config(
        Path(
            "configs/"
            "stage1_sr010_replacement_filter_imputation_wide_strict.yaml"
        )
    )

    expected_filter = replace(
        baseline.relations.filter,
        temporal_threshold=0.7,
        variable_threshold=0.7,
        historical_threshold=0.7,
        historical_candidate_pool=64,
    )
    expected_relations = replace(
        baseline.relations,
        temporal_topk=10,
        variable_topk=6,
        historical_topk=8,
        filter=expected_filter,
    )

    assert tuned == replace(baseline, relations=expected_relations)
    assert config_sha256(tuned) != config_sha256(baseline)


@pytest.mark.parametrize(
    "key",
    ["lambda_imputation_mse", "lambda_imputation_mae"],
)
def test_negative_imputation_weight_is_rejected(tmp_path: Path, key: str) -> None:
    raw = yaml.safe_load(Path("configs/stage1.yaml").read_text(encoding="utf-8"))
    raw["training"][key] = -0.1
    bad = tmp_path / f"bad_{key}.yaml"
    bad.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="imputation"):
        load_config(bad)


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
    payload_mapping["relations"].pop("filter")
    payload_mapping["training"].pop("lambda_imputation_mse")
    payload_mapping["training"].pop("lambda_imputation_mae")
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


def test_missing_aware_filter_experiment_config_is_explicit() -> None:
    cfg = load_config(
        Path("configs/stage1_sr010_replacement_filter_no_recloss.yaml")
    )

    assert cfg.relations.selection == "missing_aware_filter"
    assert cfg.relations.filter == EdgeFilterConfig(
        temporal_threshold=0.5,
        variable_threshold=0.5,
        historical_threshold=0.5,
        historical_candidate_pool=16,
        hidden_multiplier=2,
        dropout=0.0,
        temperature=1.0,
        initial_keep_probability=0.9,
        lambda_keep=0.0001,
        straight_through=True,
    )
    assert cfg.missing.start_rate == pytest.approx(0.10)
    assert cfg.recovery.fusion == "replacement_mlp"
    assert cfg.recovery.refiner_enabled is False
    assert cfg.ablations.use_recovery_loss is False


def test_fast_filter_experiment_expands_history_and_uses_independent_optimization() -> None:
    cfg = load_config(
        Path("configs/stage1_sr010_replacement_filter_fast_no_recloss.yaml")
    )

    assert cfg.relations.filter.historical_candidate_pool == 32
    assert cfg.relations.filter.learning_rate == pytest.approx(1.0e-3)
    assert cfg.relations.filter.lambda_keep == pytest.approx(1.0e-3)
    assert cfg.relations.filter.initial_keep_probability == pytest.approx(0.9)
    assert cfg.ablations.use_recovery_loss is False


def test_filter_recovery_loss_experiment_changes_only_recovery_supervision() -> None:
    baseline = load_config(
        Path("configs/stage1_sr010_replacement_filter_no_recloss.yaml")
    )
    supervised = load_config(
        Path("configs/stage1_sr010_replacement_filter_recloss.yaml")
    )

    assert baseline.ablations.use_recovery_loss is False
    assert supervised.ablations.use_recovery_loss is True
    assert supervised == replace(
        baseline,
        ablations=replace(baseline.ablations, use_recovery_loss=True),
    )


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("temporal_threshold", 0.0, "threshold"),
        ("variable_threshold", 1.0, "threshold"),
        ("historical_candidate_pool", 0, "candidate_pool"),
        ("hidden_multiplier", True, "hidden_multiplier"),
        ("dropout", 1.0, "dropout"),
        ("temperature", 0.0, "temperature"),
        ("initial_keep_probability", 1.0, "initial_keep_probability"),
        ("learning_rate", 0.0, "learning_rate"),
        ("lambda_keep", -0.1, "lambda_keep"),
        ("straight_through", False, "straight_through"),
    ],
)
def test_invalid_missing_aware_filter_config_is_rejected(
    tmp_path: Path, key: str, value: object, match: str
) -> None:
    raw = yaml.safe_load(
        Path("configs/stage1_sr010_replacement_filter_no_recloss.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["relations"]["filter"][key] = value
    bad = tmp_path / f"bad_filter_{key}.yaml"
    bad.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        load_config(bad)
