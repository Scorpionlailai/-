from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
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
    lambda_keep: float = 0.0001
    straight_through: bool = True


@dataclass(frozen=True)
class RelationConfig:
    selection: str
    temporal_topk: int
    variable_topk: int
    historical_topk: int
    recovered_nodes_can_be_sources: bool
    num_message_passing_layers: int
    filter: EdgeFilterConfig = field(default_factory=EdgeFilterConfig)

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
        if (self.data.seq_len, self.data.pred_len) != (96, 96):
            raise ValueError("stage1 seq_len and pred_len must equal 96")
        if (self.patch.length, self.patch.stride, self.patch.padding) != (
            16,
            8,
            "end_replication",
        ):
            raise ValueError("stage1 patch geometry must be 16/8/end_replication")
        if self.patch.count != 12:
            raise ValueError("stage1 patch count must equal 12")
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
        if self.relations.recovered_nodes_can_be_sources:
            raise ValueError("recovered nodes cannot be relation sources")
        if any(value <= 0 for value in self.relations.topk.values()):
            raise ValueError("all relation top-k values must be positive")
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
        if (
            self.recovery.refiner_enabled
            and self.recovery.fusion != "replacement_mlp"
        ):
            raise ValueError(
                "recovery refiner requires recovery.fusion=replacement_mlp"
            )
        if self.prototype.matrix_profile_window != 64:
            raise ValueError("matrix_profile_window must equal 64")
        if self.prototype.max_per_variable > 256:
            raise ValueError("prototype max_per_variable cannot exceed 256")
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


def load_config(path: Path) -> Stage1Config:
    raw = _require_mapping(yaml.safe_load(path.read_text(encoding="utf-8")), "config")
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
    relations_raw = dict(_require_mapping(raw["relations"], "relations"))
    filter_raw = relations_raw.pop("filter", {})
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
    if config.recovery == RecoveryConfig():
        payload_mapping.pop("recovery")
    if (
        config.relations.selection == "hard_topk"
        and config.relations.filter == EdgeFilterConfig()
    ):
        payload_mapping["relations"].pop("filter")
    payload = json.dumps(
        payload_mapping,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload).hexdigest()
