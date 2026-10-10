from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DatasetConfig:
    path: str
    n_vars: int
    expected_length: int
    train_start: int
    train_end: int
    validation_end: int
    test_end: int
    allow_length_override: bool

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("dataset path cannot be empty")
        if self.n_vars <= 0 or self.expected_length <= 0:
            raise ValueError("dataset n_vars and expected_length must be positive")
        if not (
            0 <= self.train_start
            < self.train_end
            < self.validation_end
            < self.test_end
        ):
            raise ValueError(
                "dataset boundaries must satisfy "
                "train_start < train_end < validation_end < test_end"
            )
        if self.test_end > self.expected_length and not self.allow_length_override:
            raise ValueError("dataset test_end cannot exceed expected_length")


@dataclass(frozen=True)
class DataConfig:
    seq_len: int
    pred_len: int
    datasets: dict[str, DatasetConfig]


@dataclass(frozen=True)
class PatchConfig:
    seq_len: int
    length: int
    stride: int
    padding: str

    @property
    def count(self) -> int:
        padded = self.seq_len + self.stride
        return 1 + (padded - self.length) // self.stride


@dataclass(frozen=True)
class MissingConfig:
    type: str
    start_rate: float
    block_len: int
    allow_block_overlap: bool


@dataclass(frozen=True)
class PrototypeConfig:
    context_left: int
    target_span: int
    context_right: int
    scan_stride: int
    matrix_profile_window: int
    matrix_profile_exclusion: int
    motif_percentile: float
    min_occurrences: int
    max_per_variable: int
    retrieval_epochs: int = 20
    retrieval_hidden_multiplier: int = 2
    retrieval_dim: int = 0
    scan_stride_absolute: int = 1
    group_similarity_threshold: float = 0.8
    group_capacity: int = 16
    max_groups_per_variable: int = 128
    history_similarity_threshold: float = 0.7
    min_visible_points: int = 4
    retrieval_temperature: float = 1.0
    overlap_exclusion: int = 16
    storage_mode: str = "offline_grouped"
    online_observed_only: bool = True


@dataclass(frozen=True)
class EdgeFilterConfig:
    temporal_threshold: float = 0.5
    variable_threshold: float = 0.5
    historical_threshold: float = 0.5
    historical_candidate_pool: int = 16
    hidden_multiplier: int = 2
    dropout: float = 0.0
    temperature: float = 1.0
    initial_keep_probability: float = 0.9
    learning_rate: float | None = None
    lambda_keep: float = 0.0001
    straight_through: bool = True


@dataclass(frozen=True)
class OnlineRelationConfig:
    protocol: str = "legacy_v1"
    temporal_distance_boundaries: tuple[int, ...] = (2, 4, 8, 16)
    candidate_gain_threshold: float = 0.0
    route_improvement_threshold: float = 0.10
    route_stability_min_targets: int = 128
    bootstrap_epochs: int = 20
    joint_end_epoch: int = 80
    calibration_relation_lr_scale: float = 0.10
    stage1_repair_epochs: int = 0
    repair_loss_weight: float = 1.0
    gain_loss_weight: float = 0.20
    memory_loss_weight: float = 0.05
    temporal_residual_limit: float = 2.0
    stage2_repair_lr_scale: float = 0.25


@dataclass(frozen=True)
class RelationConfig:
    selection: str
    temporal_topk: int
    variable_topk: int
    historical_topk: int
    recovered_nodes_can_be_sources: bool
    num_message_passing_layers: int
    temporal_fine_radius: int = 0
    temporal_coarse_stride: int = 0
    temporal_coarse_radius: int = 0
    variable_global_summary: bool = False
    filter: EdgeFilterConfig = field(default_factory=EdgeFilterConfig)
    online: OnlineRelationConfig = field(default_factory=OnlineRelationConfig)

    @property
    def topk(self) -> dict[str, int]:
        return {
            "temporal": self.temporal_topk,
            "variable": self.variable_topk,
            "historical": self.historical_topk,
        }


@dataclass(frozen=True)
class RecoveryConfig:
    fusion: str = "additive"
    hidden_multiplier: int = 2
    dropout: float = 0.0
    output_init: str = "zero"
    damage_scaling: str = "missing_fraction"
    refiner_enabled: bool = False
    refiner_heads: int = 4
    refiner_ff_multiplier: int = 2
    refiner_dropout: float = 0.0
    refiner_output_init: str = "zero"
    node_gate_enabled: bool = False


@dataclass(frozen=True)
class ModelProfile:
    d_model: int
    n_heads: int
    d_ff: int
    encoder_layers: int
    dropout: float
    fc_dropout: float
    head_dropout: float
    attn_dropout: float


@dataclass(frozen=True)
class ModelConfig:
    normalization_eps: float
    profiles: dict[str, ModelProfile]

    def for_dataset(self, dataset: str) -> ModelProfile:
        try:
            return self.profiles[dataset]
        except KeyError as exc:
            raise ValueError(f"no model profile configured for {dataset!r}") from exc


@dataclass(frozen=True)
class TrainingProfile:
    learning_rate: float
    batch_size: int
    max_epochs: int
    early_stopping_patience: int
    lr_schedule: str
    drop_last: bool


@dataclass(frozen=True)
class TrainingConfig:
    optimizer: str
    weight_decay: float
    checkpoint_metric: str
    lambda_rec: float
    seed: int
    profiles: dict[str, TrainingProfile]
    lambda_imputation_mse: float = 0.0
    lambda_imputation_mae: float = 0.0
    lambda_imputation_huber: float = 0.0
    imputation_huber_delta: float = 1.0
    lambda_edge_utility: float = 0.0
    # ``None`` preserves the legacy behavior: enable the reconstruction
    # branch whenever at least one imputation loss has a positive weight.
    # An explicit value lets ablations keep the same branch architecture
    # while changing only the loss weights.
    imputation_reconstruction_enabled: bool | None = None

    @property
    def imputation_enabled(self) -> bool:
        if self.imputation_reconstruction_enabled is not None:
            return self.imputation_reconstruction_enabled
        return any(
            weight > 0.0
            for weight in (
                self.lambda_imputation_mse,
                self.lambda_imputation_mae,
                self.lambda_imputation_huber,
            )
        )

    def for_dataset(self, dataset: str) -> TrainingProfile:
        try:
            return self.profiles[dataset]
        except KeyError as exc:
            raise ValueError(f"no training profile configured for {dataset!r}") from exc


@dataclass(frozen=True)
class AblationConfig:
    use_temporal: bool
    use_variable: bool
    use_historical: bool
    use_recovery_loss: bool


@dataclass(frozen=True)
class Stage1Config:
    version: int
    data: DataConfig
    patch: PatchConfig
    missing: MissingConfig
    prototype: PrototypeConfig
    relations: RelationConfig
    recovery: RecoveryConfig
    model: ModelConfig
    training: TrainingConfig
    ablations: AblationConfig

    def validate(self) -> None:
        if self.version != 1:
            raise ValueError("stage1 config version must equal 1")
        if self.data.seq_len <= 0:
            raise ValueError("stage1 seq_len must be positive")
        if self.data.pred_len != 96:
            raise ValueError("stage1 pred_len must equal 96")
        if (self.patch.length, self.patch.stride, self.patch.padding) != (
            16,
            8,
            "end_replication",
        ):
            raise ValueError("stage1 patch geometry must be 16/8/end_replication")
        if self.patch.count <= 0:
            raise ValueError("stage1 seq_len must contain at least one patch")
        if self.missing.type != "variable_missing":
            raise ValueError("missing.type must equal variable_missing")
        if not 0.0 < self.missing.start_rate < 1.0:
            raise ValueError("missing.start_rate must be between 0 and 1")
        if self.missing.block_len <= 0:
            raise ValueError("missing.block_len must be positive")
        if not self.missing.allow_block_overlap:
            raise ValueError("stage1 requires overlapping blocks to be allowed")
        if self.relations.selection not in {"hard_topk", "missing_aware_filter"}:
            raise ValueError(
                "relations.selection must be hard_topk or missing_aware_filter"
            )
        if self.relations.num_message_passing_layers != 1:
            raise ValueError("num_message_passing_layers must equal 1")
        online = self.relations.online
        if online.protocol not in {"legacy_v1", "relation_v2", "dynamic_context_v3"}:
            raise ValueError(
                "relations.online.protocol must be legacy_v1, relation_v2, or dynamic_context_v3"
            )
        boundaries = online.temporal_distance_boundaries
        if (
            not isinstance(boundaries, tuple)
            or not boundaries
            or any(type(value) is not int or value <= 0 for value in boundaries)
            or any(left >= right for left, right in zip(boundaries, boundaries[1:]))
        ):
            raise ValueError(
                "relations.online.temporal_distance_boundaries must be strictly increasing positive integers"
            )
        for name, value in (
            ("candidate_gain_threshold", online.candidate_gain_threshold),
            ("route_improvement_threshold", online.route_improvement_threshold),
        ):
            if type(value) not in {int, float} or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"relations.online.{name} must lie in [0, 1]")
        if type(online.route_stability_min_targets) is not int or online.route_stability_min_targets <= 0:
            raise ValueError("relations.online.route_stability_min_targets must be a positive integer")
        if type(online.stage1_repair_epochs) is not int or online.stage1_repair_epochs < 0:
            raise ValueError("relations.online.stage1_repair_epochs must be a non-negative integer")
        for name, value in (
            ("repair_loss_weight", online.repair_loss_weight),
            ("gain_loss_weight", online.gain_loss_weight),
            ("memory_loss_weight", online.memory_loss_weight),
        ):
            if (
                type(value) not in {int, float}
                or not isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(
                    f"relations.online.{name} must be finite and non-negative"
                )
        if not float(online.repair_loss_weight) > 0.0:
            raise ValueError("relations.online.repair_loss_weight must be positive")
        if (
            type(online.temporal_residual_limit) not in {int, float}
            or not isfinite(float(online.temporal_residual_limit))
            or float(online.temporal_residual_limit) <= 0.0
        ):
            raise ValueError("relations.online.temporal_residual_limit must be finite and positive")
        if (
            type(online.stage2_repair_lr_scale) not in {int, float}
            or not isfinite(float(online.stage2_repair_lr_scale))
            or not 0.0 < float(online.stage2_repair_lr_scale) <= 1.0
        ):
            raise ValueError("relations.online.stage2_repair_lr_scale must lie in (0, 1]")
        if type(online.bootstrap_epochs) is not int or type(online.joint_end_epoch) is not int:
            raise ValueError("relations.online epoch boundaries must be integers")
        if not online.bootstrap_epochs < online.joint_end_epoch < max(
            profile.max_epochs for profile in self.training.profiles.values()
        ):
            raise ValueError(
                "relations.online must satisfy bootstrap_epochs < joint_end_epoch < max_epochs"
            )
        if (
            type(online.calibration_relation_lr_scale) not in {int, float}
            or not 0.0 < float(online.calibration_relation_lr_scale) <= 1.0
        ):
            raise ValueError(
                "relations.online.calibration_relation_lr_scale must lie in (0, 1]"
            )
        if self.relations.recovered_nodes_can_be_sources:
            raise ValueError("recovered nodes cannot be relation sources")
        if any(value <= 0 for value in self.relations.topk.values()):
            raise ValueError("all relation top-k values must be positive")
        if self.relations.temporal_fine_radius < 0:
            raise ValueError("temporal_fine_radius must be non-negative")
        if self.relations.temporal_coarse_stride < 0 or self.relations.temporal_coarse_radius < 0:
            raise ValueError("temporal coarse scope parameters must be non-negative")
        if (self.relations.temporal_coarse_stride == 0) != (
            self.relations.temporal_coarse_radius == 0
        ):
            raise ValueError("temporal coarse stride and radius must be set together")
        if type(self.relations.variable_global_summary) is not bool:
            raise ValueError("variable_global_summary must be boolean")
        filter_config = self.relations.filter
        thresholds = (
            filter_config.temporal_threshold,
            filter_config.variable_threshold,
            filter_config.historical_threshold,
        )
        if any(
            type(value) not in {int, float} or not 0.0 < float(value) < 1.0
            for value in thresholds
        ):
            raise ValueError("edge filter thresholds must lie in (0, 1)")
        if (
            type(filter_config.historical_candidate_pool) is not int
            or filter_config.historical_candidate_pool <= 0
        ):
            raise ValueError(
                "edge filter historical_candidate_pool must be a positive integer"
            )
        if (
            type(filter_config.hidden_multiplier) is not int
            or filter_config.hidden_multiplier <= 0
        ):
            raise ValueError(
                "edge filter hidden_multiplier must be a positive integer"
            )
        if (
            type(filter_config.dropout) not in {int, float}
            or not 0.0 <= float(filter_config.dropout) < 1.0
        ):
            raise ValueError("edge filter dropout must lie in [0, 1)")
        if (
            type(filter_config.temperature) not in {int, float}
            or filter_config.temperature <= 0.0
        ):
            raise ValueError("edge filter temperature must be positive")
        if (
            type(filter_config.initial_keep_probability) not in {int, float}
            or not 0.0 < float(filter_config.initial_keep_probability) < 1.0
        ):
            raise ValueError(
                "edge filter initial_keep_probability must lie in (0, 1)"
            )
        if (
            filter_config.learning_rate is not None
            and (
                type(filter_config.learning_rate) not in {int, float}
                or float(filter_config.learning_rate) <= 0.0
            )
        ):
            raise ValueError("edge filter learning_rate must be positive")
        if (
            type(filter_config.lambda_keep) not in {int, float}
            or filter_config.lambda_keep < 0.0
        ):
            raise ValueError("edge filter lambda_keep must be non-negative")
        if type(filter_config.straight_through) is not bool:
            raise ValueError("edge filter straight_through must be boolean")
        if (
            self.relations.selection == "missing_aware_filter"
            and not filter_config.straight_through
        ):
            raise ValueError(
                "missing_aware_filter requires straight_through to be true"
            )
        if self.recovery.fusion not in {
            "additive",
            "residual_mlp",
            "replacement_mlp",
        }:
            raise ValueError(
                "recovery.fusion must be additive, residual_mlp, or replacement_mlp"
            )
        if (
            type(self.recovery.hidden_multiplier) is not int
            or self.recovery.hidden_multiplier <= 0
        ):
            raise ValueError(
                "recovery.hidden_multiplier must be a positive non-boolean integer"
            )
        if not 0.0 <= self.recovery.dropout < 1.0:
            raise ValueError("recovery.dropout must be in [0, 1)")
        if self.recovery.output_init != "zero":
            raise ValueError("recovery.output_init must be zero")
        if self.recovery.damage_scaling != "missing_fraction":
            raise ValueError("recovery.damage_scaling must be missing_fraction")
        if type(self.recovery.refiner_enabled) is not bool:
            raise ValueError("recovery.refiner_enabled must be boolean")
        if (
            type(self.recovery.refiner_heads) is not int
            or self.recovery.refiner_heads <= 0
        ):
            raise ValueError("recovery.refiner_heads must be a positive integer")
        if (
            type(self.recovery.refiner_ff_multiplier) is not int
            or self.recovery.refiner_ff_multiplier <= 0
        ):
            raise ValueError(
                "recovery.refiner_ff_multiplier must be a positive integer"
            )
        if not 0.0 <= self.recovery.refiner_dropout < 1.0:
            raise ValueError("recovery.refiner_dropout must be in [0, 1)")
        if self.recovery.refiner_output_init != "zero":
            raise ValueError("recovery.refiner_output_init must be zero")
        if type(self.recovery.node_gate_enabled) is not bool:
            raise ValueError("recovery.node_gate_enabled must be boolean")
        if (
            self.recovery.refiner_enabled
            and self.recovery.fusion != "replacement_mlp"
        ):
            raise ValueError(
                "recovery refiner requires recovery.fusion=replacement_mlp"
            )
        if (
            self.recovery.node_gate_enabled
            and self.recovery.fusion != "replacement_mlp"
        ):
            raise ValueError(
                "recovery node gate requires recovery.fusion=replacement_mlp"
            )
        if self.prototype.matrix_profile_window != 64:
            raise ValueError("matrix_profile_window must equal 64")
        if self.prototype.max_per_variable > 256:
            raise ValueError("prototype max_per_variable cannot exceed 256")
        if self.prototype.retrieval_epochs <= 0:
            raise ValueError("prototype retrieval_epochs must be positive")
        if self.prototype.retrieval_hidden_multiplier <= 0:
            raise ValueError("prototype retrieval_hidden_multiplier must be positive")
        if self.prototype.retrieval_dim < 0:
            raise ValueError("prototype retrieval_dim cannot be negative")
        if self.prototype.scan_stride_absolute != 1:
            raise ValueError("stage1 grouped prototype scan_stride_absolute must equal 1")
        if not 0.0 < self.prototype.group_similarity_threshold <= 1.0:
            raise ValueError("prototype group_similarity_threshold must lie in (0, 1]")
        if self.prototype.group_capacity <= 0:
            raise ValueError("prototype group_capacity must be positive")
        if self.prototype.max_groups_per_variable <= 0:
            raise ValueError("prototype max_groups_per_variable must be positive")
        if not 0.0 < self.prototype.history_similarity_threshold <= 1.0:
            raise ValueError("prototype history_similarity_threshold must lie in (0, 1]")
        if self.prototype.min_visible_points <= 0:
            raise ValueError("prototype min_visible_points must be positive")
        if self.prototype.retrieval_temperature <= 0.0:
            raise ValueError("prototype retrieval_temperature must be positive")
        if self.prototype.overlap_exclusion < 0:
            raise ValueError("prototype overlap_exclusion cannot be negative")
        if self.prototype.storage_mode not in {"offline_grouped", "online_grouped"}:
            raise ValueError(
                "prototype storage_mode must be offline_grouped or online_grouped"
            )
        if self.prototype.storage_mode == "online_grouped" and self.data.seq_len != 96:
            raise ValueError("online_grouped stage1 requires seq_len=96")
        if type(self.prototype.online_observed_only) is not bool:
            raise ValueError("prototype online_observed_only must be boolean")
        if (
            self.prototype.storage_mode == "online_grouped"
            and not self.prototype.online_observed_only
        ):
            raise ValueError(
                "prototype online_observed_only must be true for online_grouped"
            )
        dataset_names = set(self.data.datasets)
        if set(self.model.profiles) != dataset_names:
            raise ValueError("model profiles must exactly match configured datasets")
        if set(self.training.profiles) != dataset_names:
            raise ValueError("training profiles must exactly match configured datasets")
        if self.model.normalization_eps <= 0:
            raise ValueError("model.normalization_eps must be positive")
        for name, profile in self.model.profiles.items():
            if profile.d_model % profile.n_heads:
                raise ValueError(
                    f"model profile {name!r} d_model must be divisible by n_heads"
                )
            if min(profile.d_model, profile.n_heads, profile.d_ff, profile.encoder_layers) <= 0:
                raise ValueError(f"model profile {name!r} dimensions must be positive")
            if (
                self.recovery.refiner_enabled
                and profile.d_model % self.recovery.refiner_heads
            ):
                raise ValueError(
                    f"recovery.refiner_heads must divide model profile {name!r} d_model"
                )
            if any(
                not 0.0 <= value < 1.0
                for value in (
                    profile.dropout,
                    profile.fc_dropout,
                    profile.head_dropout,
                    profile.attn_dropout,
                )
            ):
                raise ValueError(f"model profile {name!r} dropout must be in [0, 1)")
        if self.training.optimizer.lower() != "adam":
            raise ValueError("stage1 optimizer must equal adam")
        if self.training.checkpoint_metric != "val_mse":
            raise ValueError("checkpoint_metric must equal val_mse")
        if (
            type(self.training.lambda_imputation_mse) not in {int, float}
            or self.training.lambda_imputation_mse < 0.0
        ):
            raise ValueError("imputation MSE weight must be non-negative")
        if (
            type(self.training.lambda_imputation_mae) not in {int, float}
            or self.training.lambda_imputation_mae < 0.0
        ):
            raise ValueError("imputation MAE weight must be non-negative")
        if (
            type(self.training.lambda_imputation_huber) not in {int, float}
            or self.training.lambda_imputation_huber < 0.0
        ):
            raise ValueError("imputation Huber weight must be non-negative")
        if (
            type(self.training.lambda_edge_utility) not in {int, float}
            or self.training.lambda_edge_utility < 0.0
        ):
            raise ValueError("edge utility weight must be non-negative")
        if (
            type(self.training.imputation_huber_delta) not in {int, float}
            or self.training.imputation_huber_delta <= 0.0
        ):
            raise ValueError("imputation Huber delta must be positive")
        if (
            self.training.imputation_reconstruction_enabled is not None
            and type(self.training.imputation_reconstruction_enabled) is not bool
        ):
            raise ValueError("imputation_reconstruction_enabled must be boolean")
        for name, profile in self.training.profiles.items():
            if profile.learning_rate <= 0 or profile.batch_size <= 0:
                raise ValueError(f"training profile {name!r} has invalid optimizer values")
            if profile.max_epochs <= 0 or profile.early_stopping_patience <= 0:
                raise ValueError(f"training profile {name!r} has invalid epoch values")
            if profile.lr_schedule != "type3":
                raise ValueError(
                    f"training profile {name!r} must use PatchTST type3 schedule"
                )
            if not profile.drop_last:
                raise ValueError(
                    f"training profile {name!r} must drop incomplete batches"
                )


def _require_mapping(raw: Any, name: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"{name} must be a mapping")
    return raw


def _merge_config_mapping(
    base: dict[str, Any], override: dict[str, Any]
) -> dict[str, Any]:
    """Recursively merge a small YAML override into its parent config."""

    merged = dict(base)
    for key, value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            merged[key] = _merge_config_mapping(base_value, value)
        else:
            merged[key] = value
    return merged


def _load_config_mapping(path: Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved in stack:
        chain = " -> ".join(str(item) for item in (*stack, resolved))
        raise ValueError(f"config inheritance cycle: {chain}")
    raw = _require_mapping(
        yaml.safe_load(resolved.read_text(encoding="utf-8")), "config"
    )
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    if not isinstance(parent, str) or not parent.strip():
        raise ValueError("config extends must be a non-empty relative path")
    parent_path = Path(parent)
    if not parent_path.is_absolute():
        parent_path = resolved.parent / parent_path
    base = _load_config_mapping(parent_path, (*stack, resolved))
    return _merge_config_mapping(base, raw)


def load_config(path: Path) -> Stage1Config:
    raw = _load_config_mapping(path)
    data_raw = _require_mapping(raw["data"], "data")
    datasets_raw = _require_mapping(data_raw["datasets"], "data.datasets")
    data = DataConfig(
        seq_len=int(data_raw["seq_len"]),
        pred_len=int(data_raw["pred_len"]),
        datasets={
            name: DatasetConfig(**_require_mapping(value, f"dataset {name}"))
            for name, value in datasets_raw.items()
        },
    )
    patch_raw = _require_mapping(raw["patch"], "patch")
    recovery_raw = raw.get("recovery", {})
    model_raw = _require_mapping(raw["model"], "model")
    model_profiles_raw = _require_mapping(model_raw["profiles"], "model.profiles")
    training_raw = _require_mapping(raw["training"], "training")
    training_profiles_raw = _require_mapping(
        training_raw["profiles"], "training.profiles"
    )
    reconstruction_enabled_raw = training_raw.get(
        "imputation_reconstruction_enabled"
    )
    if (
        reconstruction_enabled_raw is not None
        and type(reconstruction_enabled_raw) is not bool
    ):
        raise ValueError("imputation_reconstruction_enabled must be boolean")
    relations_raw = dict(_require_mapping(raw["relations"], "relations"))
    filter_raw = relations_raw.pop("filter", {})
    online_raw = relations_raw.pop("online", {})
    online_mapping = _require_mapping(online_raw, "relations.online")
    if "temporal_distance_boundaries" in online_mapping:
        boundaries_raw = online_mapping["temporal_distance_boundaries"]
        if not isinstance(boundaries_raw, (list, tuple)):
            raise ValueError(
                "relations.online.temporal_distance_boundaries must be a list"
            )
        online_mapping = dict(online_mapping)
        online_mapping["temporal_distance_boundaries"] = tuple(boundaries_raw)
    config = Stage1Config(
        version=int(raw["version"]),
        data=data,
        patch=PatchConfig(seq_len=data.seq_len, **patch_raw),
        missing=MissingConfig(**_require_mapping(raw["missing"], "missing")),
        prototype=PrototypeConfig(
            **_require_mapping(raw["prototype"], "prototype")
        ),
        relations=RelationConfig(
            **relations_raw,
            filter=EdgeFilterConfig(
                **_require_mapping(filter_raw, "relations.filter")
            ),
            online=OnlineRelationConfig(**online_mapping),
        ),
        recovery=RecoveryConfig(**_require_mapping(recovery_raw, "recovery")),
        model=ModelConfig(
            normalization_eps=float(model_raw["normalization_eps"]),
            profiles={
                name: ModelProfile(**_require_mapping(value, f"model profile {name}"))
                for name, value in model_profiles_raw.items()
            },
        ),
        training=TrainingConfig(
            optimizer=str(training_raw["optimizer"]),
            weight_decay=float(training_raw["weight_decay"]),
            checkpoint_metric=str(training_raw["checkpoint_metric"]),
            lambda_rec=float(training_raw["lambda_rec"]),
            seed=int(training_raw["seed"]),
            lambda_imputation_mse=float(
                training_raw.get("lambda_imputation_mse", 0.0)
            ),
            lambda_imputation_mae=float(
                training_raw.get("lambda_imputation_mae", 0.0)
            ),
            lambda_imputation_huber=float(
                training_raw.get("lambda_imputation_huber", 0.0)
            ),
            lambda_edge_utility=float(training_raw.get("lambda_edge_utility", 0.0)),
            imputation_huber_delta=float(
                training_raw.get("imputation_huber_delta", 1.0)
            ),
            imputation_reconstruction_enabled=(
                reconstruction_enabled_raw
            ),
            profiles={
                name: TrainingProfile(
                    **_require_mapping(value, f"training profile {name}")
                )
                for name, value in training_profiles_raw.items()
            },
        ),
        ablations=AblationConfig(
            **_require_mapping(raw["ablations"], "ablations")
        ),
    )
    config.validate()
    return config


def config_sha256(config: Stage1Config) -> str:
    payload_mapping = asdict(config)
    training_payload = payload_mapping["training"]
    if config.training.lambda_imputation_mse == 0.0:
        training_payload.pop("lambda_imputation_mse")
    if config.training.lambda_imputation_mae == 0.0:
        training_payload.pop("lambda_imputation_mae")
    if config.training.lambda_imputation_huber == 0.0:
        training_payload.pop("lambda_imputation_huber")
        training_payload.pop("imputation_huber_delta")
    if config.training.imputation_reconstruction_enabled is None:
        training_payload.pop("imputation_reconstruction_enabled")
    if (
        config.prototype.storage_mode == "offline_grouped"
        and config.prototype.online_observed_only is True
    ):
        payload_mapping["prototype"].pop("storage_mode")
        payload_mapping["prototype"].pop("online_observed_only")
    if config.recovery == RecoveryConfig():
        payload_mapping.pop("recovery")
    if (
        config.relations.selection == "hard_topk"
        and config.relations.filter == EdgeFilterConfig()
    ):
        payload_mapping["relations"].pop("filter")
    elif config.relations.filter.learning_rate is None:
        # Preserve hashes for configurations created before an independent
        # filter learning rate was introduced.
        payload_mapping["relations"]["filter"].pop("learning_rate")
    if config.relations.online == OnlineRelationConfig():
        payload_mapping["relations"].pop("online")
    online_payload = payload_mapping["relations"].get("online")
    if isinstance(online_payload, dict) and online_payload.get("stage1_repair_epochs") == 0:
        online_payload.pop("stage1_repair_epochs")
    if isinstance(online_payload, dict) and config.relations.online.protocol != "relation_v2":
        online_payload.pop("repair_loss_weight")
        online_payload.pop("gain_loss_weight")
        online_payload.pop("memory_loss_weight")
    if isinstance(online_payload, dict) and config.relations.online.protocol != "dynamic_context_v3":
        online_payload.pop("temporal_residual_limit")
        online_payload.pop("stage2_repair_lr_scale")
    payload = json.dumps(
        payload_mapping,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload).hexdigest()
