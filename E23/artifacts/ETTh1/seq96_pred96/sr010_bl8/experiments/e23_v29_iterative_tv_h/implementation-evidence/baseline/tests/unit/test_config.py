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


def test_relation_v2_pressure_config_is_explicit() -> None:
    config = load_config(Path("configs/stage1_seq96_relational_recovery.yaml"))
    online = config.relations.online
    assert online.protocol == "relation_v2"
    assert online.temporal_distance_boundaries == (2, 4, 8, 16)
    assert online.route_improvement_threshold == pytest.approx(0.10)
    assert online.route_stability_min_targets == 128
    assert (online.bootstrap_epochs, online.joint_end_epoch) == (20, 80)
    assert online.stage1_repair_epochs == 100
    assert online.repair_loss_weight == pytest.approx(1.0)
    assert online.gain_loss_weight == pytest.approx(0.20)
    assert online.memory_loss_weight == pytest.approx(0.05)
    assert config.missing.start_rate == pytest.approx(0.10)
    assert config.missing.block_len == 8


def test_dynamic_context_v3_config_exposes_temporal_repair_controls() -> None:
    config = load_config(
        Path("configs/stage1_seq96_dynamic_temporal_repair.yaml")
    )
    assert config.relations.online.protocol == "dynamic_context_v3"
    assert config.relations.online.stage1_repair_epochs == 100
    assert config.relations.online.temporal_residual_limit == pytest.approx(2.0)
    assert config.relations.online.stage2_repair_lr_scale == pytest.approx(0.25)
    assert config.relations.temporal_topk == 6


def test_weather_e22_config_keeps_repair_protocol_with_weather_profile() -> None:
    source = load_config(Path("configs/stage1_seq96_dynamic_temporal_repair.yaml"))
    weather = load_config(
        Path("configs/stage1_seq96_dynamic_temporal_repair_weather.yaml")
    )

    assert set(weather.data.datasets) == {"Weather"}
    dataset = weather.data.datasets["Weather"]
    assert (dataset.path, dataset.n_vars, dataset.expected_length) == (
        "data/raw/weather.csv", 21, 52696
    )
    assert (
        dataset.train_start, dataset.train_end,
        dataset.validation_end, dataset.test_end,
    ) == (0, 36887, 42157, 52696)
    assert (weather.data.seq_len, weather.data.pred_len) == (96, 96)
    assert weather.missing == source.missing
    assert weather.patch == source.patch
    assert weather.prototype == source.prototype
    assert weather.relations == source.relations
    assert weather.recovery == source.recovery
    assert weather.ablations == source.ablations
    profile = weather.model.for_dataset("Weather")
    assert (
        profile.d_model, profile.n_heads, profile.d_ff,
        profile.encoder_layers, profile.dropout,
    ) == (128, 16, 256, 3, 0.2)
    training = weather.training.for_dataset("Weather")
    assert (
        training.batch_size, training.max_epochs,
        training.early_stopping_patience, training.lr_schedule,
    ) == (128, 100, 100, "type3")
    assert weather.training.seed == 2021


def test_weather_w02_changes_only_batch_size_from_w01() -> None:
    w01 = load_config(
        Path("configs/stage1_seq96_dynamic_temporal_repair_weather.yaml")
    )
    w02 = load_config(
        Path("configs/stage1_seq96_dynamic_temporal_repair_weather_bs16.yaml")
    )

    assert w02.training.for_dataset("Weather").batch_size == 16
    restored_training = replace(
        w02.training,
        profiles={
            "Weather": replace(w02.training.for_dataset("Weather"), batch_size=128)
        },
    )
    assert replace(w02, training=restored_training) == w01


def test_legacy_pressure_config_differs_only_by_online_protocol() -> None:
    legacy = load_config(Path("configs/stage1_seq96_online_memory_bl8_legacy.yaml"))
    v2 = load_config(Path("configs/stage1_seq96_relational_recovery.yaml"))
    assert legacy.relations.online.protocol == "legacy_v1"
    assert replace(
        legacy.relations.online,
        protocol="relation_v2",
        stage1_repair_epochs=100,
    ) == v2.relations.online
    assert legacy.missing == v2.missing
    assert legacy.data == v2.data
    assert legacy.patch == v2.patch
    assert legacy.training == v2.training


def test_stage1_accepts_patchtst_336_lookback(tmp_path: Path) -> None:
    raw = yaml.safe_load(
        Path("configs/stage1_sr010_grouped_memory_imputation.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["data"]["seq_len"] = 336
    raw["missing"]["start_rate"] = 0.15
    config_path = tmp_path / "stage1_sl336_sr015.yaml"
    config_path.write_text(
        yaml.safe_dump(raw, sort_keys=False),
        encoding="utf-8",
    )

    cfg = load_config(config_path)

    assert (cfg.data.seq_len, cfg.data.pred_len) == (336, 96)
    assert (cfg.patch.length, cfg.patch.stride, cfg.patch.count) == (16, 8, 42)
    assert cfg.missing.start_rate == pytest.approx(0.15)


def test_sl336_sr015_config_changes_only_lookback_and_start_rate() -> None:
    baseline = load_config(
        Path("configs/stage1_sr010_grouped_memory_imputation.yaml")
    )
    tuned = load_config(
        Path("configs/stage1_sl336_sr015_grouped_memory_imputation.yaml")
    )

    expected = replace(
        baseline,
        data=replace(baseline.data, seq_len=336),
        patch=replace(baseline.patch, seq_len=336),
        missing=replace(baseline.missing, start_rate=0.15),
    )

    assert tuned == expected
    assert tuned.patch.count == 42
    assert config_sha256(tuned) != config_sha256(baseline)


def test_node_gate_can_be_enabled_from_config(tmp_path: Path) -> None:
    raw = yaml.safe_load(
        Path("configs/stage1_sl336_sr015_grouped_memory_imputation.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["recovery"]["node_gate_enabled"] = True
    config_path = tmp_path / "node_gate.yaml"
    config_path.write_text(
        yaml.safe_dump(raw, sort_keys=False),
        encoding="utf-8",
    )

    cfg = load_config(config_path)

    assert cfg.recovery.node_gate_enabled is True


def test_gated_experiment_config_only_enables_the_node_gate() -> None:
    baseline = load_config(
        Path("configs/stage1_sl336_sr015_grouped_memory_imputation.yaml")
    )
    gated = load_config(
        Path("configs/stage1_sl336_sr015_grouped_memory_gated_imputation.yaml")
    )

    assert gated == replace(
        baseline,
        recovery=replace(baseline.recovery, node_gate_enabled=True),
    )
    assert config_sha256(gated) != config_sha256(baseline)


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


def test_huber_imputation_supervision_is_configurable(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("configs/stage1.yaml").read_text(encoding="utf-8"))
    raw["training"]["lambda_imputation_huber"] = 0.1
    raw["training"]["imputation_huber_delta"] = 1.0
    config_path = tmp_path / "huber.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    cfg = load_config(config_path)

    assert cfg.training.lambda_imputation_huber == pytest.approx(0.1)
    assert cfg.training.imputation_huber_delta == pytest.approx(1.0)


def test_stage1_sl336_sr010_huber_config_uses_only_huber_recovery_supervision() -> None:
    cfg = load_config(Path("configs/stage1_sl336_sr010_grouped_memory_huber.yaml"))

    assert (cfg.data.seq_len, cfg.missing.start_rate) == (336, pytest.approx(0.10))
    assert cfg.training.lambda_imputation_mse == pytest.approx(0.0)
    assert cfg.training.lambda_imputation_mae == pytest.approx(0.0)
    assert cfg.training.lambda_imputation_huber == pytest.approx(0.1)
    assert cfg.training.imputation_huber_delta == pytest.approx(1.0)
    assert cfg.relations.filter.lambda_keep == pytest.approx(0.0)


def test_stage1_multiscale_edge_config_loads() -> None:
    cfg = load_config(
        Path("configs/stage1_sl336_sr010_multiscale_edge_huber.yaml")
    )

    assert cfg.data.seq_len == 336
    assert cfg.relations.temporal_topk == 12
    assert cfg.relations.variable_topk == 6
    assert cfg.relations.temporal_fine_radius == 4
    assert cfg.relations.temporal_coarse_stride == 4
    assert cfg.relations.temporal_coarse_radius == 4
    assert cfg.relations.variable_global_summary is True
    assert cfg.training.lambda_edge_utility == pytest.approx(0.05)


def test_stage1_historical_only_config_disables_current_window_edges() -> None:
    cfg = load_config(
        Path("configs/stage1_sl336_sr010_historical_only_huber.yaml")
    )

    assert (cfg.data.seq_len, cfg.missing.start_rate) == (
        336,
        pytest.approx(0.10),
    )
    assert cfg.ablations.use_temporal is False
    assert cfg.ablations.use_variable is False
    assert cfg.ablations.use_historical is True
    assert cfg.training.lambda_imputation_huber == pytest.approx(0.1)
    assert cfg.training.lambda_edge_utility == pytest.approx(0.05)


def test_explicit_imputation_reconstruction_flag_is_independent_of_weights(
    tmp_path: Path,
) -> None:
    raw = yaml.safe_load(Path("configs/stage1.yaml").read_text(encoding="utf-8"))
    raw["training"]["imputation_reconstruction_enabled"] = True
    config_path = tmp_path / "reconstruction_without_loss.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    cfg = load_config(config_path)

    assert cfg.training.imputation_reconstruction_enabled is True
    assert cfg.training.imputation_enabled is True


def test_sl336_sr010_no_huber_is_a_loss_only_ablation() -> None:
    huber = load_config(
        Path("configs/stage1_sl336_sr010_grouped_memory_huber.yaml")
    )
    no_huber = load_config(
        Path("configs/stage1_sl336_sr010_grouped_memory_no_huber.yaml")
    )

    expected_training = replace(
        huber.training,
        imputation_reconstruction_enabled=True,
        lambda_imputation_huber=0.0,
    )
    assert no_huber == replace(huber, training=expected_training)
    assert no_huber.training.imputation_enabled is True
    assert config_sha256(no_huber) != config_sha256(huber)


def test_nonpositive_huber_delta_is_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("configs/stage1.yaml").read_text(encoding="utf-8"))
    raw["training"]["imputation_huber_delta"] = 0.0
    config_path = tmp_path / "bad_huber_delta.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="Huber delta"):
        load_config(config_path)


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
    payload_mapping["training"].pop("lambda_imputation_huber")
    payload_mapping["training"].pop("imputation_huber_delta")
    payload_mapping["training"].pop("imputation_reconstruction_enabled")
    payload_mapping["prototype"].pop("storage_mode")
    payload_mapping["prototype"].pop("online_observed_only")
    payload_mapping["relations"].pop("online")
    payload = json.dumps(
        payload_mapping,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    assert config_sha256(cfg) == sha256(payload).hexdigest()
    assert config_sha256(cfg) == "f9b44aa92803dc0f242819beceeea0a7f9d5a65659af306ae19173105f2403ed"
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


def test_online_memory_config_loads_with_explicit_observed_only() -> None:
    config = load_config(Path("configs/stage1_seq96_online_memory.yaml"))

    assert config.prototype.storage_mode == "online_grouped"
    assert config.prototype.online_observed_only is True
    assert config.data.seq_len == 96
    assert config.data.pred_len == 96
    assert (config.patch.length, config.patch.stride) == (16, 8)


def test_online_memory_mode_rejects_masked_patch_ingestion(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("configs/stage1.yaml").read_text(encoding="utf-8"))
    raw["prototype"]["storage_mode"] = "online_grouped"
    raw["prototype"]["online_observed_only"] = False
    path = tmp_path / "invalid_online_memory.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="online_observed_only"):
        load_config(path)
