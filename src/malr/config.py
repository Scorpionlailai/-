from __future__ import annotations

from dataclasses import asdict, dataclass
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
class RelationConfig:
    selection: str
    temporal_topk: int
    variable_topk: int
    historical_topk: int
    recovered_nodes_can_be_sources: bool
    num_message_passing_layers: int

    @property
    def topk(self) -> dict[str, int]:
        return {
            "temporal": self.temporal_topk,
            "variable": self.variable_topk,
            "historical": self.historical_topk,
        }


@dataclass(frozen=True)
class ModelConfig:
    d_model: int
    n_heads: int
    d_ff: int
    encoder_layers: int
    dropout: float
    head_dropout: float
    normalization_eps: float


@dataclass(frozen=True)
class TrainingConfig:
    optimizer: str
    learning_rate: float
    weight_decay: float
    batch_size: int
    max_epochs: int
    early_stopping_patience: int
    checkpoint_metric: str
    lambda_rec: float
    seed: int


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
        if self.relations.selection != "hard_topk":
            raise ValueError("relations.selection must equal hard_topk")
        if self.relations.num_message_passing_layers != 1:
            raise ValueError("num_message_passing_layers must equal 1")
        if self.relations.recovered_nodes_can_be_sources:
            raise ValueError("recovered nodes cannot be relation sources")
        if any(value <= 0 for value in self.relations.topk.values()):
            raise ValueError("all relation top-k values must be positive")
        if self.prototype.matrix_profile_window != 64:
            raise ValueError("matrix_profile_window must equal 64")
        if self.prototype.max_per_variable > 256:
            raise ValueError("prototype max_per_variable cannot exceed 256")
        if self.model.d_model % self.model.n_heads:
            raise ValueError("model.d_model must be divisible by model.n_heads")
        if self.training.optimizer.lower() != "adam":
            raise ValueError("stage1 optimizer must equal adam")
        if self.training.checkpoint_metric != "val_mse":
            raise ValueError("checkpoint_metric must equal val_mse")


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
    config = Stage1Config(
        version=int(raw["version"]),
        data=data,
        patch=PatchConfig(seq_len=data.seq_len, **patch_raw),
        missing=MissingConfig(**_require_mapping(raw["missing"], "missing")),
        prototype=PrototypeConfig(
            **_require_mapping(raw["prototype"], "prototype")
        ),
        relations=RelationConfig(
            **_require_mapping(raw["relations"], "relations")
        ),
        model=ModelConfig(**_require_mapping(raw["model"], "model")),
        training=TrainingConfig(
            **_require_mapping(raw["training"], "training")
        ),
        ablations=AblationConfig(
            **_require_mapping(raw["ablations"], "ablations")
        ),
    )
    config.validate()
    return config


def config_sha256(config: Stage1Config) -> str:
    payload = json.dumps(
        asdict(config),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload).hexdigest()

