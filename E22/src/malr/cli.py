from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import asdict, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys
from time import monotonic
from typing import Any

import numpy as np
import torch

from malr.config import Stage1Config, config_sha256, load_config
from malr.data.datasets import CleanSeries, load_clean_csv
from malr.data.repaired import (
    RepairedSplit,
    iter_repaired_batches,
    load_repaired_split,
    save_repaired_split,
)
from malr.data.masks import (
    EpochMaskManager,
    MaskExpectation,
    generate_variable_missing_mask,
    load_fixed_mask,
    save_fixed_mask,
)
from malr.data.normalization import (
    TrainVariableStats,
    fit_train_variable_stats,
    standardize_values,
)
from malr.data.splits import DatasetProtocol, protocol_from_bounds
from malr.data.windows import build_window_index
from malr.evaluation.evaluator import evaluate_model, evaluate_online_model
from malr.evaluation.metrics import degradation_recovery_rate, relative_degradation
from malr.evaluation.report import (
    ResultRecord,
    file_sha256,
    write_json_atomic,
    write_result_record,
)
from malr.models.filtering import MissingAwareEdgeFilter
from malr.models.patchtst import PatchTSTCore
from malr.models.online_stage1 import OnlineStage1Model
from malr.models.online_stage1_v2 import RelationV2OnlineStage1Model
from malr.training.warm_start import load_forecast_warm_start, verify_frozen_online_state
from malr.models.dynamic_context_model import DynamicContextOnlineModel
from malr.models.repaired_forecast import MaskAwareForecastModel
from malr.models.recovery import MissingAwareRecovery
from malr.models.relations import GroupedHistoricalRelation, HistoricalRelation
from malr.models.stage1 import ModelVariant, Stage1Model, initialize_from_clean_checkpoint
from malr.prototypes.bank import (
    GroupedPrototypeBank,
    PrototypeBank,
    build_grouped_prototype_bank,
    load_grouped_bank,
    load_bank,
    save_grouped_bank,
    source_checksum,
)
from malr.prototypes.retrieval import (
    MaskAwareRetrievalEncoder,
    RetrievalEncoderEMA,
    build_clean_patch_table,
    load_mask_aware_encoder,
    retrieval_state_tuple,
    train_mask_aware_encoder,
)
from malr.reproducibility import seed_everything
from malr.training.checkpoints import load_checkpoint
from malr.training.data import iter_prepared_batches
from malr.training.trainer import Trainer, validate_dependency_metadata
from malr.training.online_checkpoint import load_online_checkpoint
from malr.migration import MigrationCatalog, load_migrated_run
from malr.training.online_trainer import OnlineTrainer
from malr.training.repaired_export import reconstruct_split
from malr.training.repaired_forecast_trainer import (
    RepairedForecastTrainer,
    evaluate_repaired_forecast,
)
from malr.prototypes.online_bank import OnlinePrototypeMemory


def _progress_batch_count(samples: int, batch_size: int, *, drop_last: bool) -> int:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return samples // batch_size if drop_last else (samples + batch_size - 1) // batch_size


def _progress_batches(batches, *, total: int, label: str):
    """Log completed batches as line-based bars that remain readable via tail -f."""

    started = monotonic()
    last_report = started
    report_every = max(1, total // 20)

    def report(completed: int, now: float) -> None:
        nonlocal last_report
        elapsed = now - started
        fraction = min(completed / total, 1.0) if total else 0.0
        filled = round(fraction * 20)
        bar = "#" * filled + "-" * (20 - filled)
        eta = f"{elapsed / completed * (total - completed):.0f}s" if completed else "--"
        print(
            f"phase={label} [{bar}] {completed}/{total} "
            f"{fraction:.0%} elapsed={elapsed:.0f}s ETA={eta}",
            file=sys.stderr,
            flush=True,
        )
        last_report = now

    report(0, started)
    for completed, batch in enumerate(batches, start=1):
        yield batch
        now = monotonic()
        if completed == total or completed % report_every == 0 or now - last_report >= 10:
            report(completed, now)


def _project_root(config_path: Path) -> Path:
    resolved = config_path.resolve()
    for parent in resolved.parents:
        if parent.name == "configs":
            return parent.parent
    return Path.cwd()


def _protocol_for_config(
    config: Stage1Config, dataset: str, source_length: int | None = None
) -> DatasetProtocol:
    dataset_config = config.data.datasets[dataset]
    return protocol_from_bounds(
        dataset,
        dataset_config.expected_length if source_length is None else source_length,
        train_start=dataset_config.train_start,
        train_end=dataset_config.train_end,
        validation_end=dataset_config.validation_end,
        test_end=dataset_config.test_end,
    )


def _load_context(
    config_path: Path, dataset: str
) -> tuple[Stage1Config, CleanSeries, DatasetProtocol]:
    config = load_config(config_path)
    if dataset not in config.data.datasets:
        raise ValueError(f"dataset {dataset!r} is not configured")
    dataset_config = config.data.datasets[dataset]
    data_path = Path(dataset_config.path)
    if not data_path.is_absolute():
        data_path = _project_root(config_path) / data_path
    series = load_clean_csv(
        data_path,
        expected_n_vars=dataset_config.n_vars,
        expected_length=dataset_config.expected_length,
        allow_length_override=dataset_config.allow_length_override,
    )
    protocol = _protocol_for_config(config, dataset, len(series.values))
    return config, series, protocol


def validate_stress_config(source: Stage1Config, target: Stage1Config) -> None:
    """Keep a missingness probe to one parameter and the same model."""

    rate_changed = target.missing.start_rate != source.missing.start_rate
    block_changed = target.missing.block_len != source.missing.block_len
    if rate_changed and block_changed:
        raise ValueError("stress config may change only one missingness parameter")
    if rate_changed:
        if target.missing.start_rate <= source.missing.start_rate:
            raise ValueError("stress config requires a higher missing start_rate")
        normalized_missing = replace(
            target.missing, start_rate=source.missing.start_rate
        )
        mismatch_message = "stress config may change only missing start_rate"
    elif block_changed:
        if target.missing.block_len >= source.missing.block_len:
            raise ValueError("stress config requires a shorter missing block_len")
        normalized_missing = replace(target.missing, block_len=source.missing.block_len)
        mismatch_message = "stress config may change only missing block_len"
    else:
        raise ValueError("stress config requires a missingness parameter change")
    normalized_target = replace(target, missing=normalized_missing)
    if asdict(normalized_target) != asdict(source):
        raise ValueError(mismatch_message)


def _artifact_path(root: Path, category: str, name: str) -> Path:
    return root.resolve() / category / name


def _mask_identity(
    config: Stage1Config,
    dataset: str,
    protocol: DatasetProtocol,
    split: str,
) -> tuple[int, int, MaskExpectation]:
    bounds = protocol.bounds_for(split)  # type: ignore[arg-type]
    support_start = bounds.start - config.data.seq_len
    support_end = bounds.end
    length = support_end - support_start
    expected = MaskExpectation(
        dataset=dataset,
        split=split,
        seed=config.training.seed,
        start_rate=config.missing.start_rate,
        block_len=config.missing.block_len,
        support_start=support_start,
        support_end=support_end,
        shape=(length, config.data.datasets[dataset].n_vars),
    )
    return support_start, support_end, expected


def _baseline_fingerprint(
    config: Stage1Config,
    series: CleanSeries,
    protocol: DatasetProtocol,
    dataset: str,
    variant: str,
) -> str:
    """Hash only inputs that can change a clean or missing baseline."""

    if variant not in {ModelVariant.CLEAN.value, ModelVariant.MISSING.value}:
        raise ValueError("baseline fingerprint variant must be clean or missing")
    dataset_config = config.data.datasets[dataset]
    profile = config.training.for_dataset(dataset)
    payload: dict[str, Any] = {
        "format_version": 1,
        "variant": variant,
        "dataset": dataset,
        "source_csv_sha256": file_sha256(series.source_path),
        "protocol": {
            "train_start": protocol.train.start,
            "train_end": protocol.train.end,
            "validation_end": protocol.validation.end,
            "test_end": protocol.test.end,
        },
        "data": {
            "seq_len": config.data.seq_len,
            "pred_len": config.data.pred_len,
            "n_vars": dataset_config.n_vars,
        },
        "patch": asdict(config.patch),
        "model": {
            "normalization_eps": config.model.normalization_eps,
            "profile": asdict(config.model.for_dataset(dataset)),
        },
        "training": {
            "optimizer": config.training.optimizer,
            "weight_decay": config.training.weight_decay,
            "checkpoint_metric": config.training.checkpoint_metric,
            "lambda_rec": config.training.lambda_rec,
            "seed": config.training.seed,
            "profile": asdict(profile),
        },
    }
    if variant == ModelVariant.MISSING.value:
        payload["missing"] = asdict(config.missing)
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _base_metadata(
    config: Stage1Config,
    series: CleanSeries,
    protocol: DatasetProtocol,
    dataset: str,
) -> dict[str, Any]:
    train_values = series.values[protocol.train.start : protocol.train.end]
    return {
        "dataset": dataset,
        "config_hash": config_sha256(config),
        "source_csv_sha256": file_sha256(series.source_path),
        "seed": config.training.seed,
        "train_checksum": source_checksum(train_values),
        "train_start": protocol.train.start,
        "train_end": protocol.train.end,
        "validation_end": protocol.validation.end,
        "test_end": protocol.test.end,
        "metric_space": "train_standardized",
        "sentinel_handling": "preserve",
        "seq_len": config.data.seq_len,
        "pred_len": config.data.pred_len,
        "missing_type": config.missing.type,
        "missing_start_rate": config.missing.start_rate,
        "missing_block_len": config.missing.block_len,
        "clean_baseline_fingerprint": _baseline_fingerprint(
            config, series, protocol, dataset, ModelVariant.CLEAN.value
        ),
        "missing_baseline_fingerprint": _baseline_fingerprint(
            config, series, protocol, dataset, ModelVariant.MISSING.value
        ),
    }


def _train_stats(series: CleanSeries, protocol: DatasetProtocol) -> TrainVariableStats:
    return fit_train_variable_stats(
        torch.from_numpy(series.values[protocol.train.start : protocol.train.end]),
        eps=0.0,
    )


def _standardized_context(
    series: CleanSeries, protocol: DatasetProtocol
) -> tuple[np.ndarray, TrainVariableStats]:
    raw_stats = _train_stats(series, protocol)
    standardized = standardize_values(torch.from_numpy(series.values), raw_stats)
    values = np.ascontiguousarray(standardized.numpy(), dtype=np.float32)
    fallback = fit_train_variable_stats(
        torch.from_numpy(
            values[protocol.train.start : protocol.train.end]
        ),
        eps=0.0,
    )
    return values, fallback


def _build_core(config: Stage1Config, dataset: str, n_vars: int) -> PatchTSTCore:
    profile = config.model.for_dataset(dataset)
    return PatchTSTCore(
        c_in=n_vars,
        seq_len=config.data.seq_len,
        pred_len=config.data.pred_len,
        patch_len=config.patch.length,
        stride=config.patch.stride,
        d_model=profile.d_model,
        n_heads=profile.n_heads,
        d_ff=profile.d_ff,
        n_layers=profile.encoder_layers,
        dropout=profile.dropout,
        head_dropout=profile.head_dropout,
        attn_dropout=profile.attn_dropout,
        fc_dropout=profile.fc_dropout,
    )


def build_experiment_model(
    config: Stage1Config,
    dataset: str,
    variant: ModelVariant | str,
    *,
    clean_checkpoint: dict[str, Any] | None = None,
    bank: PrototypeBank | GroupedPrototypeBank | None = None,
    protocol: DatasetProtocol | None = None,
) -> Stage1Model | OnlineStage1Model:
    selected = ModelVariant(variant)
    if selected is ModelVariant.OURS and config.prototype.storage_mode == "online_grouped":
        online_protocol = protocol or _protocol_for_config(config, dataset)
        return build_online_experiment_model(config, dataset, online_protocol)
    core = _build_core(config, dataset, config.data.datasets[dataset].n_vars)
    if selected is ModelVariant.OURS:
        imputation_enabled = config.training.imputation_enabled
        edge_filter = None
        if config.relations.selection == "missing_aware_filter":
            filter_config = config.relations.filter
            edge_filter = MissingAwareEdgeFilter(
                d_model=core.d_model,
                temporal_topk=config.relations.temporal_topk,
                variable_topk=config.relations.variable_topk,
                historical_topk=config.relations.historical_topk,
                temporal_threshold=filter_config.temporal_threshold,
                variable_threshold=filter_config.variable_threshold,
                historical_threshold=filter_config.historical_threshold,
                hidden_multiplier=filter_config.hidden_multiplier,
                dropout=filter_config.dropout,
                temperature=filter_config.temperature,
                initial_keep_probability=filter_config.initial_keep_probability,
            )
        historical = None
        if config.ablations.use_historical:
            if bank is None:
                raise ValueError("ours with historical recovery requires a prototype bank")
            if isinstance(bank, GroupedPrototypeBank):
                retrieval_encoder = load_mask_aware_encoder(
                    bank.retrieval_state,
                    patch_len=bank.patch_len,
                    retrieval_dim=bank.retrieval_dim,
                )
                historical = GroupedHistoricalRelation(
                    bank,
                    retrieval_encoder=retrieval_encoder,
                    topk=config.relations.historical_topk,
                    similarity_threshold=config.prototype.history_similarity_threshold,
                    min_visible_points=config.prototype.min_visible_points,
                    overlap_exclusion=config.prototype.overlap_exclusion,
                    retrieval_temperature=config.prototype.retrieval_temperature,
                    patch_len=config.patch.length,
                    patch_stride=config.patch.stride,
                    d_model=core.d_model,
                )
            else:
                historical = HistoricalRelation(
                    bank,
                    topk=config.relations.historical_topk,
                    candidate_pool=(
                        config.relations.filter.historical_candidate_pool
                        if edge_filter is not None
                        else None
                    ),
                    patch_len=config.patch.length,
                    patch_stride=config.patch.stride,
                    d_model=core.d_model,
                )
        recovery = MissingAwareRecovery(
            c_in=core.c_in,
            patch_num=core.patch_num,
            patch_len=core.patch_len,
            d_model=core.d_model,
            temporal_topk=config.relations.temporal_topk,
            variable_topk=config.relations.variable_topk,
            historical_relation=historical,
            use_temporal=config.ablations.use_temporal,
            use_variable=config.ablations.use_variable,
            use_historical=config.ablations.use_historical,
            fusion=config.recovery.fusion,
            residual_hidden_multiplier=config.recovery.hidden_multiplier,
            residual_dropout=config.recovery.dropout,
            residual_output_init=config.recovery.output_init,
            damage_scaling=config.recovery.damage_scaling,
            global_refiner_enabled=config.recovery.refiner_enabled,
            global_refiner_heads=config.recovery.refiner_heads,
            global_refiner_ff_multiplier=config.recovery.refiner_ff_multiplier,
            global_refiner_dropout=config.recovery.refiner_dropout,
            global_refiner_output_init=config.recovery.refiner_output_init,
            node_gate_enabled=config.recovery.node_gate_enabled,
            edge_filter=edge_filter,
            relation_selection=config.relations.selection,
            temporal_fine_radius=config.relations.temporal_fine_radius,
            temporal_coarse_stride=config.relations.temporal_coarse_stride,
            temporal_coarse_radius=config.relations.temporal_coarse_radius,
            variable_global_summary=config.relations.variable_global_summary,
            edge_utility_delta=config.training.imputation_huber_delta,
            reconstruction_enabled=imputation_enabled,
            reconstruction_window_len=core.seq_len,
            reconstruction_patch_stride=core.stride,
        )
        model = Stage1Model(core, selected, recovery=recovery)
        return model
    model = Stage1Model(core, selected)
    if selected is ModelVariant.MISSING_PRETRAINED:
        if clean_checkpoint is None:
            raise ValueError("missing_pretrained requires a clean checkpoint")
        initialize_from_clean_checkpoint(model, clean_checkpoint)
    return model


def build_online_experiment_model(
    config: Stage1Config,
    dataset: str,
    protocol: DatasetProtocol,
    *,
    device: str | torch.device = "cpu",
) -> OnlineStage1Model | RelationV2OnlineStage1Model:
    """Build the configured online relation protocol without a bank."""

    n_vars = config.data.datasets[dataset].n_vars
    core = _build_core(config, dataset, n_vars)
    profile = config.model.for_dataset(dataset)
    retrieval = RetrievalEncoderEMA(
        MaskAwareRetrievalEncoder(
            config.patch.length,
            retrieval_dim=(config.prototype.retrieval_dim or profile.d_model),
            hidden_dim=config.prototype.retrieval_hidden_multiplier
            * (config.prototype.retrieval_dim or profile.d_model),
        )
    )
    memory = OnlinePrototypeMemory(
        patch_len=config.patch.length,
        patch_stride=config.patch.stride,
        train_end=protocol.train.end,
        variable_count=n_vars,
        group_similarity_threshold=config.prototype.group_similarity_threshold,
        group_capacity=config.prototype.group_capacity,
        max_groups_per_variable=config.prototype.max_groups_per_variable,
        observed_only=config.prototype.online_observed_only,
    )
    if config.relations.online.protocol == "dynamic_context_v3":
        online = config.relations.online
        return DynamicContextOnlineModel(
            core,
            memory=memory,
            retrieval=retrieval,
            variable_count=n_vars,
            temporal_distance_boundaries=online.temporal_distance_boundaries,
            historical_topk=config.relations.historical_topk,
            temporal_topk=config.relations.temporal_topk,
            variable_topk=config.relations.variable_topk,
            candidate_gain_threshold=online.candidate_gain_threshold,
            route_improvement_threshold=online.route_improvement_threshold,
            route_stability_min_targets=online.route_stability_min_targets,
            bootstrap_epochs=online.bootstrap_epochs,
            joint_end_epoch=online.joint_end_epoch,
            historical_similarity_threshold=config.prototype.history_similarity_threshold,
            repair_loss_weight=online.repair_loss_weight,
            gain_loss_weight=online.gain_loss_weight,
            memory_loss_weight=online.memory_loss_weight,
            temporal_residual_limit=online.temporal_residual_limit,
            stage2_repair_lr_scale=online.stage2_repair_lr_scale,
        ).to(device)
    if config.relations.online.protocol == "relation_v2":
        online = config.relations.online
        return RelationV2OnlineStage1Model(
            core,
            memory=memory,
            retrieval=retrieval,
            variable_count=n_vars,
            temporal_distance_boundaries=online.temporal_distance_boundaries,
            historical_topk=config.relations.historical_topk,
            temporal_topk=config.relations.temporal_topk,
            variable_topk=config.relations.variable_topk,
            candidate_gain_threshold=online.candidate_gain_threshold,
            route_improvement_threshold=online.route_improvement_threshold,
            route_stability_min_targets=online.route_stability_min_targets,
            bootstrap_epochs=online.bootstrap_epochs,
            joint_end_epoch=online.joint_end_epoch,
            historical_similarity_threshold=config.prototype.history_similarity_threshold,
            repair_loss_weight=online.repair_loss_weight,
            gain_loss_weight=online.gain_loss_weight,
            memory_loss_weight=online.memory_loss_weight,
        ).to(device)
    return OnlineStage1Model(
        core,
        memory=memory,
        retrieval=retrieval,
        variable_count=n_vars,
        historical_max_items=config.relations.historical_topk,
        edge_threshold=0.05,
        historical_similarity_threshold=config.prototype.history_similarity_threshold,
    ).to(device)


def _fixed_mask_path(artifacts: Path, dataset: str, split: str) -> Path:
    return _artifact_path(artifacts, "masks", f"{dataset}_{split}.npz")


def _load_dependency(
    path: Path,
    expected: dict[str, Any],
) -> dict[str, Any]:
    payload = load_checkpoint(path)
    validate_dependency_metadata(payload["metadata"], expected)
    return payload


def _repaired_split_path(
    artifacts: Path, dataset: str, checkpoint_digest: str, split: str
) -> Path:
    return _artifact_path(
        artifacts,
        "repaired",
        f"{dataset}-{checkpoint_digest[:12]}-{split}.npz",
    )


def _resolve_artifact_reference(manifest_path: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("repaired manifest contains an invalid path")
    path = Path(value)
    return path if path.is_absolute() else (manifest_path.parent / path).resolve()


def _load_repaired_handoff(
    manifest_path: Path,
    *,
    expected_metadata: Mapping[str, object],
) -> tuple[dict[str, Any], dict[str, RepairedSplit]]:
    manifest_path = manifest_path.resolve()
    manifest = _read_json(manifest_path)
    if manifest.get("format_version") != 1 or manifest.get("kind") != "repaired_windows":
        raise ValueError("unsupported repaired handoff manifest")
    validate_dependency_metadata(manifest.get("metadata", {}), expected_metadata)
    split_entries = manifest.get("splits")
    if not isinstance(split_entries, dict):
        raise ValueError("repaired handoff manifest lacks split metadata")
    splits: dict[str, RepairedSplit] = {}
    for split_name in ("train", "validation", "test"):
        entry = split_entries.get(split_name)
        if not isinstance(entry, dict):
            raise ValueError(f"repaired handoff lacks {split_name} split")
        path = _resolve_artifact_reference(manifest_path, entry.get("path"))
        if file_sha256(path) != entry.get("sha256"):
            raise ValueError(f"repaired {split_name} checksum mismatch")
        split = load_repaired_split(path)
        if split.window_count != entry.get("window_count"):
            raise ValueError(f"repaired {split_name} window count mismatch")
        if abs(split.missing_rate - float(entry.get("missing_rate", -1.0))) > 1e-12:
            raise ValueError(f"repaired {split_name} missing rate mismatch")
        splits[split_name] = split
    return manifest, splits


def _run_repaired_forecast_training(
    config: Stage1Config,
    dataset: str,
    artifacts: Path,
    *,
    device: str,
    metadata: dict[str, Any],
    run_metadata: dict[str, Any],
    repaired_data_path: Path,
) -> dict[str, Any]:
    repaired_manifest, splits = _load_repaired_handoff(
        repaired_data_path,
        expected_metadata={
            key: metadata[key]
            for key in ("dataset", "config_hash", "train_checksum")
        },
    )
    expected_shapes = {
        "train": len(splits["train"].values),
        "validation": len(splits["validation"].values),
        "test": len(splits["test"].values),
    }
    if any(count <= 0 for count in expected_shapes.values()):
        raise ValueError("repaired handoff contains an empty split")
    for split in splits.values():
        if split.values.shape[1:] != (
            config.data.seq_len,
            config.data.datasets[dataset].n_vars,
        ):
            raise ValueError("repaired handoff context shape does not match config")
        if split.targets.shape[1:] != (
            config.data.datasets[dataset].n_vars,
            config.data.pred_len,
        ):
            raise ValueError("repaired handoff target shape does not match config")

    profile = config.training.for_dataset(dataset)
    model = MaskAwareForecastModel(
        _build_core(config, dataset, config.data.datasets[dataset].n_vars)
    )
    repaired_manifest_path = repaired_data_path.resolve()
    repaired_manifest_sha256 = file_sha256(repaired_manifest_path)
    checkpoint_path = _artifact_path(
        artifacts, "checkpoints", f"{dataset}_ours_stage2.pt"
    )
    stage2_metadata = {
        **run_metadata,
        "online_mode": False,
        "training_stage": "forecast",
        "training_pipeline": "repaired_forecast",
        "stage1_repair_epochs": config.relations.online.stage1_repair_epochs,
        "repaired_manifest": str(repaired_manifest_path),
        "repaired_manifest_sha256": repaired_manifest_sha256,
        "repair_checkpoint": repaired_manifest["repair_checkpoint"],
        "repair_checkpoint_sha256": repaired_manifest["repair_checkpoint_sha256"],
        "stage2_loss": "forecast_mse_only",
        "stage1_parameters_loaded": False,
    }
    trainer = RepairedForecastTrainer(
        model,
        checkpoint_path=checkpoint_path,
        checkpoint_metadata=stage2_metadata,
        max_epochs=profile.max_epochs,
        patience=profile.early_stopping_patience,
        learning_rate=profile.learning_rate,
        weight_decay=config.training.weight_decay,
        device=device,
    )
    result = trainer.fit(
        lambda epoch: iter_repaired_batches(
            splits["train"],
            batch_size=profile.batch_size,
            patch_len=config.patch.length,
            patch_stride=config.patch.stride,
            shuffle_seed=config.training.seed + epoch,
            drop_last=profile.drop_last,
        ),
        lambda: iter_repaired_batches(
            splits["validation"],
            batch_size=profile.batch_size,
            patch_len=config.patch.length,
            patch_stride=config.patch.stride,
            drop_last=profile.drop_last,
        ),
    )
    output = asdict(result)
    output["best_checkpoint"] = str(result.best_checkpoint)
    output["training_stage"] = "forecast"
    output["repaired_manifest"] = str(repaired_manifest_path)
    checkpoint_digest = file_sha256(result.best_checkpoint)
    run_identity = json.dumps(
        {
            "dataset": dataset,
            "variant": ModelVariant.OURS.value,
            "config_hash": metadata["config_hash"],
            "checkpoint_sha256": checkpoint_digest,
            "repaired_manifest_sha256": repaired_manifest_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    run_id = f"{dataset}-ours-{sha256(run_identity).hexdigest()[:12]}"
    manifest = {
        "format_version": 1,
        "run_id": run_id,
        "dataset": dataset,
        "variant": ModelVariant.OURS.value,
        "online_mode": False,
        "training_pipeline": "repaired_forecast",
        "checkpoint": str(result.best_checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_digest,
        "clean_checkpoint": None,
        "bank": None,
        "repaired_manifest": str(repaired_manifest_path),
        "repaired_manifest_sha256": repaired_manifest_sha256,
        "metadata": stage2_metadata,
        "training": output,
        "ablations": asdict(config.ablations),
    }
    manifest_path = _artifact_path(artifacts, "runs", f"{run_id}.json")
    write_json_atomic(manifest_path, manifest)
    output["run_id"] = run_id
    output["run_manifest"] = str(manifest_path)
    return output


def run_repaired_export(
    config: Stage1Config,
    series: CleanSeries,
    protocol: DatasetProtocol,
    dataset: str,
    artifacts: Path,
    *,
    checkpoint_path: Path,
    device: str,
) -> dict[str, Any]:
    """Export fixed repaired windows from a completed repair checkpoint."""

    seed_everything(config.training.seed)
    metadata = _base_metadata(config, series, protocol, dataset)
    if (
        config.prototype.storage_mode != "online_grouped"
        or config.relations.online.protocol != "relation_v2"
    ):
        raise ValueError("repaired export requires online relation-v2 configuration")
    validation_support_start, _, validation_expected = _mask_identity(
        config, dataset, protocol, "validation"
    )
    validation_mask_path = _fixed_mask_path(artifacts, dataset, "validation")
    validation_mask, _ = load_fixed_mask(
        validation_mask_path, validation_expected
    )
    validation_mask_sha256 = file_sha256(validation_mask_path)
    checkpoint_path = checkpoint_path.resolve()
    checkpoint = load_online_checkpoint(
        checkpoint_path,
        expected_hashes={
            "config_hash": metadata["config_hash"],
            "data_hash": metadata["train_checksum"],
            "mask_hash": validation_mask_sha256,
        },
    )
    model = build_online_experiment_model(config, dataset, protocol, device=device)
    if not isinstance(model, RelationV2OnlineStage1Model):
        raise AssertionError("relation-v2 export constructed the wrong model")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.load_online_state(checkpoint["online_state"])
    restored_epoch = int(checkpoint["online_state"]["epoch"])
    inference_epoch = max(restored_epoch, config.relations.online.bootstrap_epochs + 1)
    export_mask_epoch = restored_epoch
    checkpoint_digest = file_sha256(checkpoint_path)
    standardized_series, stats = _standardized_context(series, protocol)
    profile = config.training.for_dataset(dataset)
    train_mask_manager = EpochMaskManager(
        length=protocol.train.end,
        n_vars=series.values.shape[1],
        start_rate=config.missing.start_rate,
        block_len=config.missing.block_len,
        base_seed=config.training.seed,
    )
    masks: dict[str, np.ndarray | Any] = {
        "train": lambda sample_id, ref: train_mask_manager.mask_for_sample(
            export_mask_epoch,
            sample_id,
            length=ref.input_end - ref.input_start,
        ),
        "validation": validation_mask,
    }
    mask_support_starts = {
        "train": 0,
        "validation": validation_support_start,
    }
    source_mask_hashes: dict[str, str | None] = {
        "train": None,
        "validation": validation_mask_sha256,
    }
    test_support_start, _, test_expected = _mask_identity(
        config, dataset, protocol, "test"
    )
    test_mask_path = _fixed_mask_path(artifacts, dataset, "test")
    test_mask, _ = load_fixed_mask(test_mask_path, test_expected)
    masks["test"] = test_mask
    mask_support_starts["test"] = test_support_start
    source_mask_hashes["test"] = file_sha256(test_mask_path)

    split_entries: dict[str, dict[str, Any]] = {}
    for split_name in ("train", "validation", "test"):
        refs = build_window_index(
            protocol, split_name, config.data.seq_len, config.data.pred_len
        )
        print(
            f"phase=repaired-export split={split_name} windows={len(refs)} start",
            flush=True,
        )
        batches = iter_prepared_batches(
            standardized_series,
            refs,
            masks[split_name],
            mask_support_start=mask_support_starts[split_name],
            train_stats=stats,
            variant=ModelVariant.OURS,
            batch_size=profile.batch_size,
            patch_len=config.patch.length,
            patch_stride=config.patch.stride,
            normalization_eps=config.model.normalization_eps,
            drop_last=False,
        )
        repaired = reconstruct_split(
            model,
            batches,
            seq_len=config.data.seq_len,
            patch_len=config.patch.length,
            patch_stride=config.patch.stride,
            epoch_index=inference_epoch,
            device=device,
        )
        split_path = _repaired_split_path(
            artifacts, dataset, checkpoint_digest, split_name
        )
        save_repaired_split(split_path, repaired)
        split_entries[split_name] = {
            "path": str(split_path.resolve()),
            "sha256": file_sha256(split_path),
            "window_count": repaired.window_count,
            "missing_rate": repaired.missing_rate,
            "source_mask_sha256": source_mask_hashes[split_name],
        }
        print(
            f"phase=repaired-export split={split_name} complete "
            f"missing_rate={repaired.missing_rate:.6f} path={split_path}",
            flush=True,
        )
    handoff = {
        "format_version": 1,
        "kind": "repaired_windows",
        "dataset": dataset,
        "metadata": metadata,
        "repair_checkpoint": str(checkpoint_path),
        "repair_checkpoint_sha256": checkpoint_digest,
        "repair_checkpoint_epoch": restored_epoch,
        "inference_epoch": inference_epoch,
        "train_mask_epoch": export_mask_epoch,
        "value_space": "query_normalized",
        "observed_values_preserved": True,
        "stage2_interface": ["repaired_values", "original_observation_mask"],
        "confidence_included": False,
        "splits": split_entries,
    }
    manifest_path = _artifact_path(
        artifacts,
        "repaired",
        f"{dataset}-{checkpoint_digest[:12]}-manifest.json",
    )
    write_json_atomic(manifest_path, handoff)
    return {
        "repaired_manifest": str(manifest_path),
        "repaired_manifest_sha256": file_sha256(manifest_path),
        "repair_checkpoint": str(checkpoint_path),
        "splits": split_entries,
    }


@torch.no_grad()
def _capture_frozen_route_teacher_probe(
    model: RelationV2OnlineStage1Model,
    batch: Any,
    *,
    device: str | torch.device,
) -> dict[str, torch.Tensor]:
    """Capture the complete teacher-owned routing decision for one fixed batch."""

    if model.route_teacher is None:
        raise RuntimeError("frozen route teacher probe requires an installed teacher")
    evaluation_device = torch.device(device)
    current = batch.to(evaluation_device)
    model.to(evaluation_device).eval()
    output = model(
        current.patch_values,
        current.patch_mask,
        current.rho,
        current.state,
        current.query_mean,
        current.query_std,
        patch_starts=current.patch_starts,
        epoch_index=model.memory.snapshot().epoch,
        patch_valid=current.patch_valid,
        patch_validity=current.patch_validity,
    )
    if isinstance(model, DynamicContextOnlineModel):
        temporal_gain = torch.zeros_like(output.historical.predicted_gain)
        temporal_uncertainty = torch.ones_like(output.historical.uncertainty)
    else:
        temporal_gain = output.temporal.predicted_gain
        temporal_uncertainty = output.temporal.uncertainty
    return {
        "gain": torch.stack(
            (
                output.historical.predicted_gain,
                temporal_gain,
                output.variable.predicted_gain,
            ),
            dim=-1,
        ).detach().cpu().clone(),
        "uncertainty": torch.stack(
            (
                output.historical.uncertainty,
                temporal_uncertainty,
                output.variable.uncertainty,
            ),
            dim=-1,
        ).detach().cpu().clone(),
        "route_available": output.route_available.detach().cpu().clone(),
        "route_index": output.route_index.detach().cpu().clone(),
    }


def _verify_frozen_route_teacher_probe(
    before: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor]
) -> None:
    """Require bit-exact teacher scores, availability, and routes after fitting."""

    required = {"gain", "uncertainty", "route_available", "route_index"}
    if set(before) != required or set(after) != required:
        raise RuntimeError("frozen route teacher probe fields are incomplete")
    changed = [name for name in sorted(required) if not torch.equal(before[name], after[name])]
    if changed:
        raise RuntimeError(
            "frozen route teacher invariance check failed for: " + ", ".join(changed)
        )


def _json_safe_finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite numeric value")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite numeric value") from exc
    if not np.isfinite(result):
        raise ValueError(f"{field} must be a finite numeric value")
    return result


def _serialize_stage2_validation_evaluation(
    evaluation: Any,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Convert validation-only evaluator output into trusted JSON primitives."""

    diagnostics = {
        str(name): _json_safe_finite_float(
            value, field=f"stage2_validation_diagnostics.{name}"
        )
        for name, value in evaluation.diagnostics.items()
    }
    for name in ("route_repair_regret", "route_repair_regret_positive_rate"):
        if name not in diagnostics:
            raise ValueError(f"stage2_validation_diagnostics lacks {name}")

    ablations = {
        str(name): {
            "mse": _json_safe_finite_float(
                metrics.mse,
                field=f"stage2_validation_inference_ablations.{name}.mse",
            ),
            "mae": _json_safe_finite_float(
                metrics.mae,
                field=f"stage2_validation_inference_ablations.{name}.mae",
            ),
        }
        for name, metrics in evaluation.inference_ablations.items()
    }
    for name in (
        "without_historical",
        "without_temporal",
        "without_variable",
        "historical_only",
    ):
        if name not in ablations:
            raise ValueError(f"stage2_validation_inference_ablations lacks {name}")
    return diagnostics, ablations


def run_training(
    config: Stage1Config,
    series: CleanSeries,
    protocol: DatasetProtocol,
    dataset: str,
    variant: ModelVariant | str,
    artifacts: Path,
    *,
    device: str,
    clean_checkpoint_path: Path | None = None,
    bank_path: Path | None = None,
    stage: str | None = None,
    init_checkpoint_path: Path | None = None,
    repaired_data_path: Path | None = None,
    stage2_route_teacher: str | None = None,
    repair_batch_size: int | None = None,
    repair_matmul_precision: str | None = None,
    warm_start_checkpoint_path: Path | None = None,
    warm_start_epoch: int | None = None,
    additional_epochs: int | None = None,
) -> dict[str, Any]:
    if stage not in {None, "repair", "forecast"}:
        raise ValueError("stage must be one of: repair, forecast")
    if stage2_route_teacher not in {
        None,
        "frozen_stage1",
        "frozen_stage1_hv_fusion",
    }:
        raise ValueError(
            "stage2_route_teacher must be 'frozen_stage1' or "
            "'frozen_stage1_hv_fusion' when provided"
        )
    selected = ModelVariant(variant)
    online_mode = (
        selected is ModelVariant.OURS
        and config.prototype.storage_mode == "online_grouped"
    )
    relation_v2_mode = online_mode and config.relations.online.protocol in {
        "relation_v2",
        "dynamic_context_v3",
    }
    warm_start_requested = any(value is not None for value in (
        warm_start_checkpoint_path, warm_start_epoch, additional_epochs,
    ))
    if warm_start_requested:
        if not relation_v2_mode or stage != "forecast" or init_checkpoint_path is None:
            raise ValueError("warm-start requires ours --stage forecast and --init-checkpoint")
        if (
            warm_start_checkpoint_path is None
            or type(warm_start_epoch) is not int or warm_start_epoch <= 0
            or type(additional_epochs) is not int or additional_epochs <= 0
        ):
            raise ValueError("warm-start requires checkpoint, positive source epoch and additional epochs")
        target = artifacts / "checkpoints" / f"{dataset}_{selected.value}_stage2.pt"
        if target.exists() or target.resolve() == warm_start_checkpoint_path.resolve():
            raise ValueError("warm-start must use a new experiment; refusing to overwrite checkpoint")
    if repair_batch_size is not None or repair_matmul_precision is not None:
        if not relation_v2_mode or stage != "repair":
            raise ValueError("repair runtime options are only supported for ours --stage repair")
    if repair_batch_size is not None and (type(repair_batch_size) is not int or repair_batch_size <= 0):
        raise ValueError("repair_batch_size must be a positive integer")
    if repair_matmul_precision is not None and repair_matmul_precision not in {"highest", "high", "medium"}:
        raise ValueError("repair_matmul_precision must be highest, high, or medium")
    stage1_epochs = config.relations.online.stage1_repair_epochs if relation_v2_mode else 0
    if stage2_route_teacher is not None:
        if not relation_v2_mode or selected is not ModelVariant.OURS or stage != "forecast":
            raise ValueError(
                "--stage2-route-teacher requires relation-v2 ours --stage forecast"
            )
        if init_checkpoint_path is None:
            raise ValueError("--stage2-route-teacher requires --init-checkpoint")
    if init_checkpoint_path is not None and stage != "forecast":
        raise ValueError("init_checkpoint_path requires stage='forecast'")
    if repaired_data_path is not None:
        raise ValueError(
            "--repaired-data is no longer a Stage-2 input; supervised forecast "
            "fine-tuning consumes --init-checkpoint plus the original masked data"
        )
    if stage is not None and not relation_v2_mode:
        raise ValueError("--stage is only supported for online relation-v2 ours training")
    if relation_v2_mode and stage1_epochs > 0 and stage is None:
        raise ValueError(
            "relation-v2 two-stage training requires --stage repair or --stage forecast"
        )
    if stage == "repair" and stage1_epochs <= 0:
        raise ValueError("stage='repair' requires relations.online.stage1_repair_epochs > 0")
    if stage == "forecast" and init_checkpoint_path is None:
        raise ValueError("stage='forecast' requires the Stage-1 --init-checkpoint")
    seed_everything(config.training.seed)
    metadata = _base_metadata(config, series, protocol, dataset)
    standardized_series, stats = _standardized_context(series, protocol)
    training_profile = config.training.for_dataset(dataset)
    train_batch_size = repair_batch_size or training_profile.batch_size
    displayed_total_epochs = (
        warm_start_epoch + additional_epochs
        if warm_start_requested
        else stage1_epochs if stage == "repair" else training_profile.max_epochs
    )
    displayed_epoch = warm_start_epoch or 0
    progress_variant = f"{selected.value} {stage}" if stage else selected.value
    dependency_expected = {
        key: metadata[key] for key in ("dataset", "config_hash", "train_checksum")
    }

    clean_payload = None
    if selected is ModelVariant.MISSING_PRETRAINED:
        if clean_checkpoint_path is None:
            raise ValueError(f"{selected.value} requires --clean-checkpoint")
        # Dependency validation deliberately precedes model allocation.
        clean_payload = _load_dependency(clean_checkpoint_path, dependency_expected)

    bank = None
    if selected is ModelVariant.OURS and config.ablations.use_historical and not online_mode:
        if bank_path is None:
            raise ValueError("ours with historical recovery requires --bank")
        clean_for_bank = clean_checkpoint_path or _artifact_path(
            artifacts, "checkpoints", f"{dataset}_clean.pt"
        )
        try:
            bank = load_grouped_bank(
                bank_path,
                dataset=dataset,
                train_end=protocol.train.end,
                source_checksum=source_checksum(
                    series.values[protocol.train.start : protocol.train.end]
                ),
                config_sha256=metadata["config_hash"],
                clean_checkpoint_sha256=(
                    file_sha256(clean_for_bank) if clean_for_bank.exists() else ""
                ),
            )
        except ValueError as exc:
            if "unsupported grouped prototype bank version" not in str(exc):
                raise
            bank = load_bank(
                bank_path,
                source_train=standardized_series[: protocol.train.end],
                dataset=dataset,
                config_sha256=metadata["config_hash"],
            )

    model = (
        build_online_experiment_model(config, dataset, protocol, device=device)
        if online_mode
        else build_experiment_model(
            config,
            dataset,
            selected,
            clean_checkpoint=clean_payload,
            bank=bank,
        )
    )
    train_refs = build_window_index(
        protocol, "train", config.data.seq_len, config.data.pred_len
    )
    validation_refs = build_window_index(
        protocol, "validation", config.data.seq_len, config.data.pred_len
    )
    mask_manager = EpochMaskManager(
        length=protocol.train.end,
        n_vars=series.values.shape[1],
        start_rate=config.missing.start_rate,
        block_len=config.missing.block_len,
        base_seed=config.training.seed,
    )
    validation_support_start, validation_support_end, validation_expected = _mask_identity(
        config, dataset, protocol, "validation"
    )
    validation_mask_path = _fixed_mask_path(artifacts, dataset, "validation")
    if selected is ModelVariant.CLEAN:
        validation_mask = np.ones(
            (validation_support_end - validation_support_start, series.values.shape[1]),
            dtype=bool,
        )
        validation_mask_metadata = None
    else:
        validation_mask, validation_mask_metadata = load_fixed_mask(
            validation_mask_path, validation_expected
        )

    def training_batches(epoch: int):
        nonlocal displayed_epoch
        displayed_epoch = epoch + 1
        timeline_mask = (
            np.ones((protocol.train.end, series.values.shape[1]), dtype=bool)
            if selected is ModelVariant.CLEAN
            else lambda sample_id, ref: mask_manager.mask_for_sample(
                epoch,
                sample_id,
                length=ref.input_end - ref.input_start,
            )
        )
        batches = iter_prepared_batches(
            standardized_series,
            train_refs,
            timeline_mask,
            mask_support_start=0,
            train_stats=stats,
            variant=selected,
            batch_size=train_batch_size,
            patch_len=config.patch.length,
            patch_stride=config.patch.stride,
            shuffle_seed=config.training.seed + epoch,
            normalization_eps=config.model.normalization_eps,
            drop_last=training_profile.drop_last,
            include_imputation_target=(
                selected is ModelVariant.OURS
                and (
                    config.training.imputation_enabled
                    or config.training.lambda_edge_utility > 0.0
                )
            ),
        )
        return _progress_batches(
            batches,
            total=_progress_batch_count(len(train_refs), train_batch_size, drop_last=training_profile.drop_last),
            label=f"train {dataset} {progress_variant} epoch={displayed_epoch}/{displayed_total_epochs}",
        )

    def validation_batches():
        batches = iter_prepared_batches(
            standardized_series,
            validation_refs,
            validation_mask,
            mask_support_start=validation_support_start,
            train_stats=stats,
            variant=selected,
            batch_size=training_profile.batch_size,
            patch_len=config.patch.length,
            patch_stride=config.patch.stride,
            normalization_eps=config.model.normalization_eps,
            drop_last=training_profile.drop_last,
        )
        epoch_label = f" epoch={displayed_epoch}/{displayed_total_epochs}" if displayed_epoch else " probe"
        return _progress_batches(
            batches,
            total=_progress_batch_count(len(validation_refs), training_profile.batch_size, drop_last=training_profile.drop_last),
            label=f"validation {dataset} {progress_variant}{epoch_label}",
        )

    checkpoint_path = _artifact_path(
        artifacts, "checkpoints", f"{dataset}_{selected.value}.pt"
    )
    training_checkpoint_path = checkpoint_path
    stage1_checkpoint_sha256: str | None = None
    run_metadata = {
        **metadata,
        "role": "clean" if selected is ModelVariant.CLEAN else "missing_experiment",
        "variant": selected.value,
        "online_mode": online_mode,
        "validation_mask_sha256": (
            None
            if validation_mask_metadata is None
            else file_sha256(validation_mask_path)
        ),
        "validation_final_rate": (
            0.0
            if validation_mask_metadata is None
            else validation_mask_metadata.final_rate
        ),
        "imputation_loss_weights": {
            "reconstruction_enabled": config.training.imputation_enabled,
            "mse": float(config.training.lambda_imputation_mse),
            "mae": float(config.training.lambda_imputation_mae),
            "huber": float(config.training.lambda_imputation_huber),
            "huber_delta": float(config.training.imputation_huber_delta),
        },
        "edge_utility_loss": {
            "lambda": float(config.training.lambda_edge_utility),
            "target": "selected_edge_clean_patch_latent_huber_error",
            "delta": float(config.training.imputation_huber_delta),
        },
    }
    if relation_v2_mode:
        online_loss = config.relations.online
        dynamic_context_mode = online_loss.protocol == "dynamic_context_v3"
        run_metadata["stage1_repair_loss"] = {
            "formula": (
                "sequence_missing_smooth_l1 + lambda_gain*gain_HV + lambda_memory*memory"
                if dynamic_context_mode
                else "lambda_repair*mean_available(H,T,V) + lambda_gain*gain + lambda_memory*memory"
            ),
            "lambda_repair": float(online_loss.repair_loss_weight),
            "lambda_gain": float(online_loss.gain_loss_weight),
            "lambda_memory": float(online_loss.memory_loss_weight),
            "lambda_route": 0.0,
            "repair_target": (
                "unique_explicit_repaired_sequence_missing_coordinate_huber"
                if dynamic_context_mode
                else "direct_branch_missing_coordinate_huber"
            ),
            "route_policy": "argmax_positive_branch_gain_else_none",
        }
    if stage == "repair":
        run_metadata["stage1_runtime"] = {
            "train_batch_size": train_batch_size,
            "validation_batch_size": training_profile.batch_size,
            "matmul_precision": repair_matmul_precision or torch.get_float32_matmul_precision(),
        }
    if stage is not None:
        run_metadata["training_stage"] = stage
    assert model is not None
    if online_mode:
        mask_hash = (
            "clean"
            if validation_mask_metadata is None
            else file_sha256(validation_mask_path)
        )
        if isinstance(model, RelationV2OnlineStage1Model) and stage == "repair":
            training_checkpoint_path = checkpoint_path.with_name(
                f"{dataset}_{selected.value}_stage1.pt"
            )
            run_metadata["stage1_repair_epochs"] = stage1_epochs
            trainer = OnlineTrainer(
                model,
                checkpoint_path=training_checkpoint_path,
                checkpoint_metadata=run_metadata,
                max_epochs=stage1_epochs,
                patience=stage1_epochs,
                learning_rate=training_profile.learning_rate,
                calibration_relation_lr_scale=config.relations.online.calibration_relation_lr_scale,
                weight_decay=config.training.weight_decay,
                device=device,
            )
            previous_matmul_precision = torch.get_float32_matmul_precision()
            if repair_matmul_precision is not None:
                torch.set_float32_matmul_precision(repair_matmul_precision)
            try:
                result = trainer.fit(
                    training_batches,
                    validation_batches,
                    config_hash=metadata["config_hash"],
                    data_hash=metadata["train_checksum"],
                    mask_hash=mask_hash,
                    objective="repair",
                    validation_objective="repair",
                    stage_label="repair",
                )
            finally:
                if repair_matmul_precision is not None:
                    torch.set_float32_matmul_precision(previous_matmul_precision)
        elif isinstance(model, RelationV2OnlineStage1Model) and stage == "forecast":
            assert init_checkpoint_path is not None
            init_checkpoint_path = init_checkpoint_path.resolve()
            initial = load_online_checkpoint(
                init_checkpoint_path,
                expected_hashes={
                    "config_hash": metadata["config_hash"],
                    "data_hash": metadata["train_checksum"],
                    "mask_hash": mask_hash,
                },
            )
            initial_metadata = initial.get("metadata", {})
            if initial_metadata.get("training_stage") != "repair":
                raise ValueError(
                    "Stage-2 initialization must be a finalized Stage-1 repair checkpoint"
                )
            model.load_state_dict(initial["model_state"], strict=True)
            model.load_online_state(initial["online_state"])
            stage1_checkpoint_sha256 = file_sha256(init_checkpoint_path)
            teacher_sha256 = None
            if isinstance(model, DynamicContextOnlineModel):
                model.install_forecast_predictor()
                if stage2_route_teacher is None:
                    stage2_route_teacher = "frozen_stage1_hv_fusion"
            if stage2_route_teacher == "frozen_stage1":
                teacher_sha256 = model.install_frozen_route_teacher()
                model.configure_forecast_finetuning(frozen_route_teacher=True)
            elif stage2_route_teacher == "frozen_stage1_hv_fusion":
                teacher_sha256 = model.install_frozen_route_teacher()
                model.install_forecast_hv_fusion()
                model.configure_forecast_finetuning(
                    frozen_route_teacher=True,
                    forecast_fusion="teacher_hv",
                )
            else:
                model.configure_forecast_finetuning()
            initial_epoch = int(initial["online_state"]["epoch"])
            training_checkpoint_path = checkpoint_path.with_name(
                f"{dataset}_{selected.value}_stage2.pt"
            )
            run_metadata.update(
                {
                    "online_mode": True,
                    "training_pipeline": "supervised_relation_finetune",
                    "stage1_checkpoint": str(init_checkpoint_path),
                    "stage1_checkpoint_sha256": stage1_checkpoint_sha256,
                    "stage1_checkpoint_epoch": initial_epoch,
                    "stage1_parameters_loaded": True,
                    "stage2_input": "original_masked_values_plus_observation_mask",
                    "stage2_loss": "forecast_mse_only",
                    "stage2_updates": "patchtst_plus_H_T_V_messages_plus_graph_recovery_plus_mask_projection",
                    "stage2_frozen": "gain_uncertainty_heads_plus_retrieval_encoder_plus_patch_decoder_plus_online_state",
                    "stage2_route_score_gradient": "detached",
                    "stage2_memory_updates": False,
                    "stage2_epoch_display": "independent_1_to_max_epochs",
                }
            )
            if stage2_route_teacher == "frozen_stage1":
                assert teacher_sha256 is not None
                run_metadata.update(
                    {
                        "stage2_route_teacher": "frozen_stage1",
                        "stage2_route_teacher_sha256": teacher_sha256,
                        "stage2_route_formula": (
                            "argmax_positive_teacher_branch_gain_else_none"
                        ),
                        "stage2_route_teacher_gradient": "disabled",
                        "stage2_updates": (
                            "patchtst_plus_H_T_V_student_messages_plus_graph_recovery_plus_mask_projection"
                        ),
                        "stage2_frozen": (
                            "frozen_stage1_route_teacher_plus_gain_uncertainty_heads_plus_"
                            "retrieval_encoder_plus_patch_decoder_plus_online_state"
                        ),
                        "stage2_trainable_parameters": sum(
                            parameter.numel()
                            for parameter in model.parameters()
                            if parameter.requires_grad
                        ),
                        "stage2_frozen_parameters": sum(
                            parameter.numel()
                            for parameter in model.parameters()
                            if not parameter.requires_grad
                        ),
                    }
                )
            elif stage2_route_teacher == "frozen_stage1_hv_fusion":
                assert teacher_sha256 is not None
                assert model.forecast_fusion is not None
                run_metadata.update(
                    {
                        "stage2_route_teacher": "frozen_stage1_hv_fusion",
                        "stage2_route_teacher_sha256": teacher_sha256,
                        "stage2_route_formula": (
                            "argmax_positive_teacher_branch_gain_else_none"
                        ),
                        "stage2_route_teacher_gradient": "disabled",
                        "stage2_forecast_fusion": (
                            "learned_hv_none_from_forecast_mse"
                        ),
                        "stage2_temporal_forecast_weight": 0,
                        "stage2_updates": (
                            "patchtst_plus_H_V_student_messages_plus_graph_recovery_plus_"
                            "mask_projection_plus_forecast_fusion"
                        ),
                        "stage2_frozen": (
                            "frozen_stage1_route_teacher_plus_temporal_student_plus_"
                            "gain_uncertainty_heads_plus_retrieval_encoder_plus_"
                            "patch_decoder_plus_online_state"
                        ),
                        "stage2_trainable_parameters": sum(
                            parameter.numel()
                            for parameter in model.parameters()
                            if parameter.requires_grad
                        ),
                        "stage2_frozen_parameters": sum(
                            parameter.numel()
                            for parameter in model.parameters()
                            if not parameter.requires_grad
                        ),
                        "stage2_forecast_fusion_trainable_parameters": sum(
                            parameter.numel()
                            for parameter in model.forecast_fusion.parameters()
                            if parameter.requires_grad
                        ),
                        "stage2_forecast_fusion_frozen_parameters": sum(
                            parameter.numel()
                            for parameter in model.forecast_fusion.parameters()
                            if not parameter.requires_grad
                        ),
                    }
                )
            if warm_start_requested:
                assert warm_start_checkpoint_path is not None
                assert warm_start_epoch is not None and additional_epochs is not None
                warm_start_checkpoint_path = warm_start_checkpoint_path.resolve()
                warm_start_verification = load_forecast_warm_start(
                    model, warm_start_checkpoint_path, source_epoch=warm_start_epoch,
                    dataset=dataset,
                    expected_hashes={
                        "config_hash": metadata["config_hash"],
                        "data_hash": metadata["train_checksum"], "mask_hash": mask_hash,
                    },
                    stage1_checkpoint_sha256=stage1_checkpoint_sha256,
                    teacher_mode=stage2_route_teacher, teacher_sha256=teacher_sha256,
                )
                run_metadata["stage2_epoch_display"] = "warm_start_source_epoch_plus_additional"
                run_metadata["stage2_warm_start"] = {
                    "kind": "weights_only_new_optimizer",
                    "checkpoint": str(warm_start_checkpoint_path),
                    "checkpoint_sha256": file_sha256(warm_start_checkpoint_path),
                    "source_training_epoch": warm_start_epoch,
                    "additional_epochs": additional_epochs,
                    "planned_final_epoch": warm_start_epoch + additional_epochs,
                    "optimizer_state_restored": False,
                    "rng_state_restored": False,
                    "initial_best_retained": True,
                    "frozen_state_verification": warm_start_verification,
                }
                print(
                    f"Stage-2 warm-start: source_epoch={warm_start_epoch} "
                    f"additional_epochs={additional_epochs} new_optimizer=True; "
                    "frozen teacher and online state verified",
                    flush=True,
                )
            warm_start_online_state = model.online_state() if warm_start_requested else None
            teacher_probe_batch = None
            teacher_probe_before = None
            if stage2_route_teacher in {
                "frozen_stage1",
                "frozen_stage1_hv_fusion",
            }:
                teacher_probe_batch = next(iter(validation_batches()), None)
                if teacher_probe_batch is None:
                    raise ValueError(
                        "frozen route teacher verification requires a validation batch"
                    )
                teacher_probe_before = _capture_frozen_route_teacher_probe(
                    model, teacher_probe_batch, device=device
                )
            trainer = OnlineTrainer(
                model,
                checkpoint_path=training_checkpoint_path,
                checkpoint_metadata=run_metadata,
                max_epochs=additional_epochs if warm_start_requested else training_profile.max_epochs,
                patience=additional_epochs if warm_start_requested else training_profile.early_stopping_patience,
                learning_rate=training_profile.learning_rate,
                calibration_relation_lr_scale=config.relations.online.calibration_relation_lr_scale,
                weight_decay=config.training.weight_decay,
                device=device,
            )
            result = trainer.fit(
                training_batches,
                validation_batches,
                config_hash=metadata["config_hash"],
                data_hash=metadata["train_checksum"],
                mask_hash=mask_hash,
                objective="forecast",
                validation_objective="forecast",
                epoch_offset=warm_start_epoch or 0,
                model_epoch_offset=initial_epoch + (warm_start_epoch or 0),
                stage_label="forecast",
                freeze_online_state=True,
                preserve_initial_best=warm_start_requested,
            )
            if warm_start_online_state is not None:
                run_metadata["stage2_warm_start_final_online_state_verification"] = (
                    verify_frozen_online_state(warm_start_online_state, model.online_state())
                )
            if stage2_route_teacher in {
                "frozen_stage1",
                "frozen_stage1_hv_fusion",
            }:
                assert teacher_sha256 is not None
                assert model.route_teacher is not None
                assert teacher_probe_batch is not None
                assert teacher_probe_before is not None
                if model.route_teacher.state_sha256() != teacher_sha256:
                    raise RuntimeError(
                        "frozen route teacher checksum changed during Stage-2 training"
                    )
                teacher_probe_after = _capture_frozen_route_teacher_probe(
                    model, teacher_probe_batch, device=device
                )
                _verify_frozen_route_teacher_probe(
                    teacher_probe_before, teacher_probe_after
                )
                validation_evaluation = evaluate_online_model(
                    model, validation_batches(), device=device
                )
                validation_diagnostics, validation_ablations = (
                    _serialize_stage2_validation_evaluation(validation_evaluation)
                )
                run_metadata.update(
                    {
                        "stage2_route_teacher_checksum_verified": True,
                        "stage2_route_teacher_invariance_verified": True,
                        "stage2_validation_diagnostics": validation_diagnostics,
                        "stage2_validation_inference_ablations": validation_ablations,
                    }
                )
        else:
            trainer = OnlineTrainer(
                model,
                checkpoint_path=training_checkpoint_path,
                checkpoint_metadata=run_metadata,
                max_epochs=training_profile.max_epochs,
                patience=training_profile.early_stopping_patience,
                learning_rate=training_profile.learning_rate,
                calibration_relation_lr_scale=config.relations.online.calibration_relation_lr_scale,
                weight_decay=config.training.weight_decay,
                device=device,
            )
            result = trainer.fit(
                training_batches,
                validation_batches,
                config_hash=metadata["config_hash"],
                data_hash=metadata["train_checksum"],
                mask_hash=mask_hash,
            )
    else:
        trainer = Trainer(
            model,
            checkpoint_path=checkpoint_path,
            checkpoint_metadata=run_metadata,
            max_epochs=training_profile.max_epochs,
            patience=training_profile.early_stopping_patience,
            learning_rate=training_profile.learning_rate,
            filter_learning_rate=(
                config.relations.filter.learning_rate
                if selected is ModelVariant.OURS
                and config.relations.selection == "missing_aware_filter"
                else None
            ),
            weight_decay=config.training.weight_decay,
            lr_schedule=training_profile.lr_schedule,
            lambda_rec=config.training.lambda_rec,
            lambda_keep=(
                config.relations.filter.lambda_keep
                if selected is ModelVariant.OURS
                and config.relations.selection == "missing_aware_filter"
                else 0.0
            ),
            lambda_imputation_mse=(
                config.training.lambda_imputation_mse
                if selected is ModelVariant.OURS
                else 0.0
            ),
            lambda_imputation_mae=(
                config.training.lambda_imputation_mae
                if selected is ModelVariant.OURS
                else 0.0
            ),
            lambda_imputation_huber=(
                config.training.lambda_imputation_huber
                if selected is ModelVariant.OURS
                else 0.0
            ),
            lambda_edge_utility=(
                config.training.lambda_edge_utility
                if selected is ModelVariant.OURS
                else 0.0
            ),
            imputation_huber_delta=config.training.imputation_huber_delta,
            use_recovery_loss=(
                selected is ModelVariant.OURS and config.ablations.use_recovery_loss
            ),
            device=device,
        )
        result = trainer.fit(training_batches, validation_batches)
    output = asdict(result)
    if warm_start_requested:
        output["stage2_warm_start"] = run_metadata["stage2_warm_start"]
        output["additional_epochs_completed"] = result.epochs_completed - warm_start_epoch
    if online_mode and isinstance(model, RelationV2OnlineStage1Model):
        output["stage1_repair_epochs"] = config.relations.online.stage1_repair_epochs
    if stage is not None:
        output["training_stage"] = stage
    output["best_checkpoint"] = str(result.best_checkpoint)
    checkpoint_digest = file_sha256(result.best_checkpoint)
    run_identity = json.dumps(
        {
            "dataset": dataset,
            "variant": selected.value,
            "config_hash": metadata["config_hash"],
            "checkpoint_sha256": checkpoint_digest,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    run_id = f"{dataset}-{selected.value}-{sha256(run_identity).hexdigest()[:12]}"
    manifest = {
        "format_version": 1,
        "run_id": run_id,
        "dataset": dataset,
        "variant": selected.value,
        "online_mode": online_mode,
        "training_pipeline": (
            "supervised_relation_finetune" if stage == "forecast" else None
        ),
        "checkpoint": str(result.best_checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_digest,
        "clean_checkpoint": (
            str(
                (
                    clean_checkpoint_path
                    or _artifact_path(artifacts, "checkpoints", f"{dataset}_clean.pt")
                ).resolve()
            )
            if (
                selected is ModelVariant.MISSING_PRETRAINED
                or (
                    selected is ModelVariant.OURS
                    and config.ablations.use_historical
                    and not online_mode
                )
            )
            and (
                clean_checkpoint_path is not None
                or _artifact_path(artifacts, "checkpoints", f"{dataset}_clean.pt").exists()
            )
            else None
        ),
        "bank": (
            str(bank_path.resolve())
            if selected is ModelVariant.OURS
            and config.ablations.use_historical
            and not online_mode
            and bank_path is not None
            else None
        ),
        "stage1_checkpoint": (
            str(init_checkpoint_path.resolve())
            if stage == "forecast" and init_checkpoint_path is not None
            else None
        ),
        "stage1_checkpoint_sha256": (
            stage1_checkpoint_sha256
            if stage == "forecast" and init_checkpoint_path is not None
            else None
        ),
        "metadata": run_metadata,
        "training": output,
        "ablations": asdict(config.ablations),
    }
    manifest_path = _artifact_path(artifacts, "runs", f"{run_id}.json")
    write_json_atomic(manifest_path, manifest)
    output["run_id"] = run_id
    output["run_manifest"] = str(manifest_path)
    return output


def _command_validate_data(args: argparse.Namespace) -> None:
    config, series, protocol = _load_context(args.config, args.dataset)
    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "shape": list(series.values.shape),
                "columns": list(series.columns),
                "train": asdict(protocol.train),
                "validation": asdict(protocol.validation),
                "test": asdict(protocol.test),
                "metric_space": "train_standardized",
                "sentinel_handling": "preserve",
                "model_profile": asdict(config.model.for_dataset(args.dataset)),
                "training_profile": asdict(
                    config.training.for_dataset(args.dataset)
                ),
                "config_hash": config_sha256(config),
            },
            ensure_ascii=True,
            indent=2,
        )
    )


def _command_make_eval_masks(args: argparse.Namespace) -> None:
    config, series, protocol = _load_context(args.config, args.dataset)
    outputs = {}
    for split in ("validation", "test"):
        support_start, support_end, _ = _mask_identity(
            config, args.dataset, protocol, split
        )
        observed, metadata = generate_variable_missing_mask(
            length=support_end - support_start,
            n_vars=series.values.shape[1],
            start_rate=config.missing.start_rate,
            block_len=config.missing.block_len,
            seed=config.training.seed,
            dataset=args.dataset,
            split=split,
            support_start=support_start,
        )
        path = _fixed_mask_path(args.artifacts, args.dataset, split)
        checksum = save_fixed_mask(path, observed, metadata)
        outputs[split] = {
            "path": str(path),
            "checksum": checksum,
            "final_rate": metadata.final_rate,
        }
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


def _command_train(args: argparse.Namespace, variant: ModelVariant) -> None:
    config, series, protocol = _load_context(args.config, args.dataset)
    clean_checkpoint = getattr(args, "clean_checkpoint", None)
    if clean_checkpoint is None and variant is ModelVariant.MISSING_PRETRAINED:
        clean_checkpoint = _artifact_path(
            args.artifacts, "checkpoints", f"{args.dataset}_clean.pt"
        )
    bank = getattr(args, "bank", None)
    if (
        bank is None
        and variant is ModelVariant.OURS
        and config.ablations.use_historical
        and config.prototype.storage_mode != "online_grouped"
    ):
        bank = _artifact_path(args.artifacts, "banks", f"{args.dataset}.npz")
    result = run_training(
        config,
        series,
        protocol,
        args.dataset,
        variant,
        args.artifacts,
        device=args.device,
        clean_checkpoint_path=clean_checkpoint,
        bank_path=bank,
        stage=getattr(args, "stage", None),
        init_checkpoint_path=getattr(args, "init_checkpoint", None),
        repaired_data_path=getattr(args, "repaired_data", None),
        stage2_route_teacher=getattr(args, "stage2_route_teacher", None),
        repair_batch_size=getattr(args, "repair_batch_size", None),
        repair_matmul_precision=getattr(args, "repair_matmul_precision", None),
        warm_start_checkpoint_path=getattr(args, "warm_start_checkpoint", None),
        warm_start_epoch=getattr(args, "warm_start_epoch", None),
        additional_epochs=getattr(args, "additional_epochs", None),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _command_export_repaired(args: argparse.Namespace) -> None:
    config, series, protocol = _load_context(args.config, args.dataset)
    checkpoint = args.checkpoint or _artifact_path(
        args.artifacts, "checkpoints", f"{args.dataset}_ours_stage1.pt"
    )
    result = run_repaired_export(
        config,
        series,
        protocol,
        args.dataset,
        args.artifacts,
        checkpoint_path=checkpoint,
        device=args.device,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _command_build_bank(args: argparse.Namespace) -> None:
    config, series, protocol = _load_context(args.config, args.dataset)
    metadata = _base_metadata(config, series, protocol, args.dataset)
    clean_checkpoint = args.clean_checkpoint or _artifact_path(
        args.artifacts, "checkpoints", f"{args.dataset}_clean.pt"
    )
    expected = {
        key: metadata[key] for key in ("dataset", "config_hash", "train_checksum")
    }
    clean_payload = _load_dependency(clean_checkpoint, expected)
    core = _build_core(config, args.dataset, config.data.datasets[args.dataset].n_vars)
    clean_model = Stage1Model(core, ModelVariant.CLEAN)
    initialize_from_clean_checkpoint(clean_model, clean_payload)
    standardized_series, _ = _standardized_context(series, protocol)
    prototype = config.prototype
    clean_patches, starts = build_clean_patch_table(
        standardized_series,
        train_end=protocol.train.end,
        patch_len=config.patch.length,
        scan_stride=prototype.scan_stride_absolute,
    )
    retrieval = train_mask_aware_encoder(
        core.W_P,
        clean_patches,
        n_vars=clean_patches.shape[0],
        start_rate=config.missing.start_rate,
        block_len=config.missing.block_len,
        seed=config.training.seed,
        epochs=prototype.retrieval_epochs,
        hidden_multiplier=prototype.retrieval_hidden_multiplier,
        retrieval_dim=(prototype.retrieval_dim or core.d_model),
        temperature=prototype.retrieval_temperature,
        device=args.device,
    )
    raw_stats = _train_stats(series, protocol)
    bank = build_grouped_prototype_bank(
        clean_patches,
        retrieval.clean_keys,
        starts,
        dataset=args.dataset,
        train_end=protocol.train.end,
        source_checksum=source_checksum(
            series.values[protocol.train.start : protocol.train.end]
        ),
        config_sha256=metadata["config_hash"],
        clean_checkpoint_sha256=file_sha256(clean_checkpoint),
        patch_stride=config.patch.stride,
        group_similarity_threshold=prototype.group_similarity_threshold,
        group_capacity=prototype.group_capacity,
        max_groups_per_variable=prototype.max_groups_per_variable,
        train_mean=raw_stats.mean.numpy(),
        train_std=raw_stats.std.numpy(),
        retrieval_state=retrieval_state_tuple(retrieval.encoder),
    )
    output = args.bank or _artifact_path(
        args.artifacts, "banks", f"{args.dataset}.npz"
    )
    save_grouped_bank(bank, output)
    print(
        json.dumps(
            {
                "path": str(output),
                "format_version": bank.format_version,
                "group_counts": [len(groups) for groups in bank.groups],
                "member_counts": [
                    sum(len(group.members) for group in groups)
                    for groups in bank.groups
                ],
                "retrieval_training_loss": retrieval.mean_loss,
                "retrieval_training_epochs": retrieval.epochs,
                "diagnostics": bank.diagnostics,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return payload


def _baseline_cache_path(
    artifacts: Path,
    dataset: str,
    variant: str,
    fingerprint: str,
    metadata: Mapping[str, Any],
) -> Path:
    resolved = artifacts.resolve()
    collection_root = next(
        (path for path in (resolved, *resolved.parents) if path.name == "artifacts"),
        resolved.parent,
    )
    sequence_group = f"seq{metadata['seq_len']}_pred{metadata['pred_len']}"
    root = collection_root / dataset / sequence_group
    if variant == ModelVariant.MISSING.value:
        start_rate = int(round(float(metadata["missing_start_rate"]) * 100.0))
        root = root / f"sr{start_rate:03d}_bl{metadata['missing_block_len']}" / "missing"
    else:
        root = root / "clean"
    return root / fingerprint / "baseline.json"


def _cache_baseline_result(
    artifacts: Path,
    manifest: dict[str, Any],
    record: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    variant = str(manifest.get("variant", ""))
    if variant not in {ModelVariant.CLEAN.value, ModelVariant.MISSING.value}:
        return
    fingerprint_key = f"{variant}_baseline_fingerprint"
    fingerprint = metadata.get(fingerprint_key)
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError(f"missing {fingerprint_key} for baseline cache")
    cached_manifest = dict(manifest)
    cached_manifest["metadata"] = {
        **dict(manifest.get("metadata", {})),
        "clean_baseline_fingerprint": metadata["clean_baseline_fingerprint"],
        "missing_baseline_fingerprint": metadata["missing_baseline_fingerprint"],
    }
    path = _baseline_cache_path(
        artifacts, str(manifest["dataset"]), variant, fingerprint, metadata
    )
    source_checkpoint = Path(str(manifest["checkpoint"])).resolve()
    expected_checkpoint_sha256 = str(manifest["checkpoint_sha256"])
    if file_sha256(source_checkpoint) != expected_checkpoint_sha256:
        raise ValueError("cannot cache baseline with a mismatched checkpoint")
    cached_checkpoint = path.parent / "checkpoint.pt"
    cached_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if not cached_checkpoint.exists():
        shutil.copy2(source_checkpoint, cached_checkpoint)
    if file_sha256(cached_checkpoint) != expected_checkpoint_sha256:
        raise ValueError("cached baseline checkpoint checksum mismatch")
    cached_manifest["source_checkpoint"] = str(source_checkpoint)
    cached_manifest["checkpoint"] = str(cached_checkpoint.resolve())
    cached_record = {
        **record,
        "clean_baseline_fingerprint": metadata["clean_baseline_fingerprint"],
        "missing_baseline_fingerprint": metadata["missing_baseline_fingerprint"],
    }
    if path.exists():
        existing = _read_json(path)
        if existing.get("fingerprint") != fingerprint:
            raise ValueError("baseline cache fingerprint mismatch")
        return
    write_json_atomic(
        path,
        {
            "format_version": 1,
            "kind": "forecast_baseline_cache",
            "dataset": manifest["dataset"],
            "variant": variant,
            "fingerprint": fingerprint,
            "manifest": cached_manifest,
            "result": cached_record,
        },
    )


def _load_cached_baseline(
    artifacts: Path,
    dataset: str,
    variant: str,
    metadata: Mapping[str, Any],
    *,
    expected_mask_sha256: str | None,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    fingerprint_key = f"{variant}_baseline_fingerprint"
    fingerprint = metadata.get(fingerprint_key)
    if not isinstance(fingerprint, str) or not fingerprint:
        return None
    path = _baseline_cache_path(
        artifacts, dataset, variant, fingerprint, metadata
    )
    if not path.exists():
        return None
    payload = _read_json(path)
    if (
        payload.get("format_version") != 1
        or payload.get("kind") != "forecast_baseline_cache"
        or payload.get("dataset") != dataset
        or payload.get("variant") != variant
        or payload.get("fingerprint") != fingerprint
    ):
        raise ValueError(f"invalid baseline cache entry: {path}")
    manifest = payload.get("manifest")
    record = payload.get("result")
    if not isinstance(manifest, dict) or not isinstance(record, dict):
        raise ValueError(f"baseline cache entry lacks manifest/result: {path}")
    manifest_metadata = manifest.get("metadata")
    if not isinstance(manifest_metadata, dict):
        raise ValueError(f"baseline cache entry lacks metadata: {path}")
    if manifest_metadata.get(fingerprint_key) != fingerprint:
        raise ValueError(f"baseline cache metadata fingerprint mismatch: {path}")
    for key in ("dataset", "source_csv_sha256", "seed", "seq_len", "pred_len"):
        if manifest_metadata.get(key) != metadata.get(key):
            raise ValueError(f"baseline cache provenance mismatch for {key}: {path}")
    if record.get("source_csv_sha256") != metadata.get("source_csv_sha256"):
        raise ValueError(f"baseline cache source checksum mismatch: {path}")
    checkpoint = Path(str(manifest.get("checkpoint", "")))
    if (
        not checkpoint.is_file()
        or file_sha256(checkpoint) != manifest.get("checkpoint_sha256")
    ):
        raise ValueError(f"baseline cache checkpoint checksum mismatch: {path}")
    if record.get("mask_sha256") != expected_mask_sha256:
        raise ValueError(f"baseline cache test mask mismatch: {path}")
    return manifest, record


def _find_variant_result(
    artifacts: Path,
    dataset: str,
    variant: str,
    metadata: Mapping[str, Any],
    *,
    expected_mask_sha256: str | None,
) -> dict[str, Any] | None:
    result_directory = _artifact_path(artifacts, "evaluations", "")
    matches = []
    for path in result_directory.glob("*.json"):
        payload = _read_json(path)
        if (
            payload.get("dataset") == dataset
            and payload.get("variant") == variant
            and payload.get("config_hash") == metadata.get("config_hash")
            and payload.get("mask_sha256") == expected_mask_sha256
        ):
            matches.append(payload)
    if matches:
        return sorted(matches, key=lambda item: str(item.get("timestamp_utc", "")))[-1]
    cached = _load_cached_baseline(
        artifacts,
        dataset,
        variant,
        metadata,
        expected_mask_sha256=expected_mask_sha256,
    )
    return None if cached is None else cached[1]


def _restore_online_model_for_evaluation(
    config: Stage1Config,
    dataset: str,
    protocol: DatasetProtocol,
    checkpoint: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    device: str,
) -> tuple[OnlineStage1Model | RelationV2OnlineStage1Model, dict[str, Any]]:
    """Recreate optional frozen-teacher structure before strict restoration."""

    manifest_metadata = manifest.get("metadata", {})
    if not isinstance(manifest_metadata, Mapping):
        raise ValueError("run manifest metadata must be a mapping")
    model_state = checkpoint.get("model_state")
    if not isinstance(model_state, Mapping):
        raise ValueError("online checkpoint model_state must be a mapping")
    contains_teacher_state = any(
        isinstance(name, str) and name.startswith("route_teacher.")
        for name in model_state
    )
    contains_fusion_state = any(
        isinstance(name, str) and name.startswith("forecast_fusion.")
        for name in model_state
    )
    # E22 keeps the H/V fusion in both Stage-1 and Stage-2 checkpoints; only
    # the explicit teacher metadata distinguishes the Stage-2 container.
    if config.relations.online.protocol == "dynamic_context_v3" and manifest_metadata.get(
        "stage2_route_teacher"
    ) is None:
        contains_fusion_state = False
    teacher_mode = manifest_metadata.get("stage2_route_teacher")
    teacher_modes = {"frozen_stage1", "frozen_stage1_hv_fusion"}
    if contains_teacher_state or contains_fusion_state or teacher_mode is not None:
        if teacher_mode not in teacher_modes:
            raise ValueError(
                "frozen route teacher metadata contract mismatch: "
                "stage2_route_teacher"
            )
        fixed_contract = {
            "stage2_route_formula": (
                "argmax_positive_teacher_branch_gain_else_none"
            ),
            "stage2_route_teacher_gradient": "disabled",
            "stage2_loss": "forecast_mse_only",
        }
        for field, expected_value in fixed_contract.items():
            if manifest_metadata.get(field) != expected_value:
                raise ValueError(
                    f"frozen route teacher metadata contract mismatch: {field}"
                )
        teacher_sha256 = manifest_metadata.get("stage2_route_teacher_sha256")
        if (
            not isinstance(teacher_sha256, str)
            or len(teacher_sha256) != 64
            or any(character not in "0123456789abcdef" for character in teacher_sha256)
        ):
            raise ValueError(
                "frozen route teacher metadata contract mismatch: "
                "stage2_route_teacher_sha256"
            )
        if teacher_mode == "frozen_stage1_hv_fusion":
            fusion_contract = {
                "stage2_forecast_fusion": "learned_hv_none_from_forecast_mse",
                "stage2_temporal_forecast_weight": 0,
            }
            for field, expected_value in fusion_contract.items():
                value = manifest_metadata.get(field)
                if field == "stage2_temporal_forecast_weight":
                    matches = (
                        type(value) in {int, float}
                        and np.isfinite(value)
                        and value == expected_value
                    )
                else:
                    matches = value == expected_value
                if not matches:
                    raise ValueError(
                        f"frozen route teacher metadata contract mismatch: {field}"
                    )
            if not contains_fusion_state:
                raise ValueError(
                    "frozen route teacher metadata contract mismatch: "
                    "forecast_fusion_state"
                )
        elif contains_fusion_state:
            raise ValueError(
                "frozen route teacher metadata contract mismatch: "
                "forecast_fusion_state"
            )

    model = build_online_experiment_model(config, dataset, protocol, device=device)
    if isinstance(model, DynamicContextOnlineModel) and (
        teacher_mode in teacher_modes or any(
            isinstance(name, str) and name.startswith("forecast_core.")
            for name in model_state
        )
    ):
        model.install_forecast_predictor()
    if teacher_mode in teacher_modes:
        if not isinstance(model, RelationV2OnlineStage1Model):
            raise ValueError("frozen Stage-1 route teacher requires relation-v2 model")
        expected_teacher_sha256 = teacher_sha256
        # Register a shape-compatible container so strict loading can restore
        # the immutable teacher tensors embedded in the Stage-2 checkpoint.
        model.install_frozen_route_teacher()
        if teacher_mode == "frozen_stage1_hv_fusion":
            model.install_forecast_hv_fusion()
            model.configure_forecast_finetuning(
                frozen_route_teacher=True,
                forecast_fusion="teacher_hv",
            )

    model.load_state_dict(model_state, strict=True)
    if teacher_mode in teacher_modes:
        assert isinstance(model, RelationV2OnlineStage1Model)
        assert model.route_teacher is not None
        model.route_teacher.requires_grad_(False)
        model.route_teacher.eval()
        if any(parameter.requires_grad for parameter in model.route_teacher.parameters()):
            raise ValueError("frozen route teacher contains trainable parameters")
        if model.route_teacher.state_sha256() != expected_teacher_sha256:
            raise ValueError("frozen route teacher checksum mismatch")

        source_value = manifest_metadata.get(
            "stage1_checkpoint", manifest.get("stage1_checkpoint")
        )
        source_sha256 = manifest_metadata.get(
            "stage1_checkpoint_sha256", manifest.get("stage1_checkpoint_sha256")
        )
        if not isinstance(source_value, str) or not isinstance(source_sha256, str):
            raise ValueError("frozen route teacher source provenance is incomplete")
        source_path = Path(source_value)
        if source_path.is_file() and file_sha256(source_path) != source_sha256:
            raise ValueError("Stage-1 source checkpoint checksum mismatch")

    model.load_online_state(checkpoint["online_state"])
    return model, model.online_state()


def _evaluate_manifest(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    *,
    allow_existing: bool,
) -> dict[str, Any]:
    run_id = str(manifest["run_id"])
    result_path = _artifact_path(args.artifacts, "evaluations", f"{run_id}.json")
    existing_record = None
    if result_path.exists():
        if allow_existing:
            existing_record = _read_json(result_path)
        else:
            raise ValueError(f"run {run_id!r} has already been evaluated")

    config, series, protocol = _load_context(args.config, args.dataset)
    if manifest.get("dataset") != args.dataset:
        raise ValueError("run manifest dataset does not match evaluation dataset")
    metadata = _base_metadata(config, series, protocol, args.dataset)
    expected = {key: metadata[key] for key in ("dataset", "config_hash", "train_checksum")}
    manifest_metadata = manifest.get("metadata", {})
    validate_dependency_metadata(manifest_metadata, expected)
    if (
        isinstance(manifest_metadata, Mapping)
        and manifest_metadata.get("stage2_route_teacher")
        in {"frozen_stage1", "frozen_stage1_hv_fusion"}
    ):
        _validate_e20_trusted_metadata(manifest_metadata)
    checkpoint_path = Path(manifest["checkpoint"])
    if file_sha256(checkpoint_path) != manifest.get("checkpoint_sha256"):
        raise ValueError("final checkpoint checksum mismatch")
    if existing_record is not None:
        for key in ("run_id", "dataset", "variant", "config_hash", "checkpoint_sha256"):
            expected_value = (
                manifest.get("checkpoint_sha256")
                if key == "checkpoint_sha256"
                else manifest.get(key)
                if key != "config_hash"
                else metadata["config_hash"]
            )
            if existing_record.get(key) != expected_value:
                raise ValueError(f"stored evaluation provenance mismatch for {key}")
        _cache_baseline_result(args.artifacts, manifest, existing_record, metadata)
        return existing_record
    selected = ModelVariant(manifest["variant"])
    repaired_forecast = manifest.get("training_pipeline") == "repaired_forecast"
    online_mode = (
        False
        if repaired_forecast
        else bool(
            manifest.get("online_mode")
            or manifest.get("metadata", {}).get("online_mode")
        )
    )
    checkpoint = (
        load_online_checkpoint(checkpoint_path, expected_metadata=expected)
        if online_mode
        else _load_dependency(checkpoint_path, expected)
    )
    standardized_series, stats = _standardized_context(series, protocol)
    training_profile = config.training.for_dataset(args.dataset)

    clean_payload = None
    if selected is ModelVariant.MISSING_PRETRAINED:
        clean_path_value = manifest.get("clean_checkpoint")
        if not clean_path_value:
            raise ValueError("run manifest lacks its clean checkpoint dependency")
        clean_payload = _load_dependency(Path(clean_path_value), expected)
    bank = None
    bank_path = None
    if (
        selected is ModelVariant.OURS
        and config.ablations.use_historical
        and not online_mode
        and not repaired_forecast
    ):
        bank_value = manifest.get("bank")
        if not bank_value:
            raise ValueError("ours run manifest lacks its prototype bank")
        bank_path = Path(bank_value)
        clean_value = manifest.get("clean_checkpoint")
        if not clean_value:
            clean_value = str(
                _artifact_path(args.artifacts, "checkpoints", f"{args.dataset}_clean.pt")
            )
        try:
            bank = load_grouped_bank(
                bank_path,
                dataset=args.dataset,
                train_end=protocol.train.end,
                source_checksum=source_checksum(
                    series.values[protocol.train.start : protocol.train.end]
                ),
                config_sha256=metadata["config_hash"],
                clean_checkpoint_sha256=(
                    file_sha256(Path(clean_value))
                    if Path(clean_value).exists()
                    else ""
                ),
            )
        except ValueError as exc:
            if "unsupported grouped prototype bank version" not in str(exc):
                raise
            bank = load_bank(
                bank_path,
                source_train=standardized_series[: protocol.train.end],
                dataset=args.dataset,
                config_sha256=metadata["config_hash"],
            )
    online_state = None
    online_hashes = None
    repaired_splits = None
    repaired_manifest = None
    if repaired_forecast:
        repaired_manifest_value = manifest.get("repaired_manifest")
        repaired_manifest_sha256 = manifest.get("repaired_manifest_sha256")
        repaired_manifest_path = _resolve_artifact_reference(
            _artifact_path(args.artifacts, "runs", f"{run_id}.json"),
            repaired_manifest_value,
        )
        if file_sha256(repaired_manifest_path) != repaired_manifest_sha256:
            raise ValueError("repaired handoff manifest checksum mismatch")
        repaired_manifest, repaired_splits = _load_repaired_handoff(
            repaired_manifest_path,
            expected_metadata=expected,
        )
        model = MaskAwareForecastModel(
            _build_core(config, args.dataset, config.data.datasets[args.dataset].n_vars)
        )
        model.load_state_dict(checkpoint["model_state"], strict=True)
    elif online_mode:
        model, online_hashes = _restore_online_model_for_evaluation(
            config,
            args.dataset,
            protocol,
            checkpoint,
            manifest,
            device=args.device,
        )
        online_state = checkpoint["online_state"]
    else:
        model = build_experiment_model(
            config,
            args.dataset,
            selected,
            clean_checkpoint=clean_payload,
            bank=bank,
        )
        model.load_state_dict(checkpoint["model_state"], strict=True)

    support_start, support_end, mask_expected = _mask_identity(
        config, args.dataset, protocol, "test"
    )
    mask_path = _fixed_mask_path(args.artifacts, args.dataset, "test")
    if selected is ModelVariant.CLEAN:
        test_mask = np.ones(
            (support_end - support_start, series.values.shape[1]), dtype=bool
        )
        mask_metadata = None
        mask_digest = None
    else:
        test_mask, mask_metadata = load_fixed_mask(mask_path, mask_expected)
        mask_digest = file_sha256(mask_path)
    if repaired_forecast:
        assert repaired_splits is not None and repaired_manifest is not None
        repaired_test_metadata = repaired_manifest["splits"]["test"]
        if repaired_test_metadata.get("source_mask_sha256") != mask_digest:
            raise ValueError("repaired test split does not use the current fixed mask")
        batches = iter_repaired_batches(
            repaired_splits["test"],
            batch_size=training_profile.batch_size,
            patch_len=config.patch.length,
            patch_stride=config.patch.stride,
            drop_last=training_profile.drop_last,
        )
        test_window_count = len(repaired_splits["test"].values)
    else:
        refs = build_window_index(
            protocol, "test", config.data.seq_len, config.data.pred_len
        )
        batches = iter_prepared_batches(
            standardized_series,
            refs,
            test_mask,
            mask_support_start=support_start,
            train_stats=stats,
            variant=selected,
            batch_size=training_profile.batch_size,
            patch_len=config.patch.length,
            patch_stride=config.patch.stride,
            normalization_eps=config.model.normalization_eps,
            drop_last=training_profile.drop_last,
        )
        test_window_count = len(refs)
    batches = _progress_batches(
        batches,
        total=_progress_batch_count(test_window_count, training_profile.batch_size, drop_last=training_profile.drop_last),
        label=f"test {args.dataset} {selected.value}",
    )
    online_evaluation = None
    if repaired_forecast:
        assert isinstance(model, MaskAwareForecastModel)
        metrics = evaluate_repaired_forecast(model, batches, device=args.device)
    elif isinstance(model, RelationV2OnlineStage1Model):
        online_evaluation = evaluate_online_model(model, batches, device=args.device)
        metrics = online_evaluation.metrics
    else:
        metrics = evaluate_model(model, batches, device=args.device)

    if selected is ModelVariant.CLEAN:
        missing_mask_path = _fixed_mask_path(args.artifacts, args.dataset, "test")
        _, _, missing_mask_expected = _mask_identity(
            config, args.dataset, protocol, "test"
        )
        load_fixed_mask(missing_mask_path, missing_mask_expected)
        missing_mask_digest = file_sha256(missing_mask_path)
    else:
        missing_mask_digest = mask_digest
    clean_result = (
        {"mse": metrics.mse, "mae": metrics.mae}
        if selected is ModelVariant.CLEAN
        else _find_variant_result(
            args.artifacts,
            args.dataset,
            ModelVariant.CLEAN.value,
            metadata,
            expected_mask_sha256=None,
        )
    )
    naive_result = (
        {"mse": metrics.mse, "mae": metrics.mae}
        if selected is ModelVariant.MISSING
        else _find_variant_result(
            args.artifacts,
            args.dataset,
            ModelVariant.MISSING.value,
            metadata,
            expected_mask_sha256=missing_mask_digest,
        )
    )
    drr_mse = drr_mae = None
    degradation_mse_absolute = degradation_mae_absolute = None
    degradation_mse_relative_percent = degradation_mae_relative_percent = None
    if clean_result is not None:
        clean_mse = float(clean_result["mse"])
        clean_mae = float(clean_result["mae"])
        degradation_mse_absolute = metrics.mse - clean_mse
        degradation_mae_absolute = metrics.mae - clean_mae
        degradation_mse_relative_percent = relative_degradation(
            clean=clean_mse, value=metrics.mse
        )
        degradation_mae_relative_percent = relative_degradation(
            clean=clean_mae, value=metrics.mae
        )
    if clean_result is not None and naive_result is not None:
        drr_mse = degradation_recovery_rate(
            clean=float(clean_result["mse"]),
            naive=float(naive_result["mse"]),
            ours=metrics.mse,
        )
        drr_mae = degradation_recovery_rate(
            clean=float(clean_result["mae"]),
            naive=float(naive_result["mae"]),
            ours=metrics.mae,
        )
    training = manifest["training"]
    record = ResultRecord(
        run_id=run_id,
        dataset=args.dataset,
        variant=selected.value,
        ablations={key: bool(value) for key, value in manifest["ablations"].items()},
        mse=metrics.mse,
        mae=metrics.mae,
        drr_mse=drr_mse,
        drr_mae=drr_mae,
        final_rate=(
            0.0 if mask_metadata is None else mask_metadata.final_rate
        ),
        trainable_parameters=int(training["trainable_parameters"]),
        training_seconds=float(training["training_seconds"]),
        metric_space="train_standardized",
        config_hash=metadata["config_hash"],
        checkpoint_sha256=file_sha256(checkpoint_path),
        mask_sha256=mask_digest,
        bank_sha256=file_sha256(bank_path) if bank_path is not None else None,
        source_csv_sha256=file_sha256(series.source_path),
        seed=config.training.seed,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        degradation_mse_absolute=degradation_mse_absolute,
        degradation_mae_absolute=degradation_mae_absolute,
        degradation_mse_relative_percent=degradation_mse_relative_percent,
        degradation_mae_relative_percent=degradation_mae_relative_percent,
        clean_baseline_fingerprint=metadata["clean_baseline_fingerprint"],
        missing_baseline_fingerprint=metadata["missing_baseline_fingerprint"],
        online_memory_sha256=(
            None if online_hashes is None else str(online_hashes["memory_hash"])
        ),
        online_ema_sha256=(
            None if online_hashes is None else str(online_hashes["ema_hash"])
        ),
        online_relation_prior_sha256=(
            None
            if online_hashes is None
            else str(online_hashes["relation_prior_hash"])
        ),
        online_diagnostics=(
            {} if online_evaluation is None else online_evaluation.diagnostics
        ),
        inference_ablations=(
            {}
            if online_evaluation is None
            else {
                name: {"mse": value.mse, "mae": value.mae}
                for name, value in online_evaluation.inference_ablations.items()
            }
        ),
    )
    write_result_record(record, result_path)
    output = asdict(record)
    _cache_baseline_result(args.artifacts, manifest, output, metadata)
    return output


def _latest_run_manifest(
    artifacts: Path, dataset: str, variant: str
) -> dict[str, Any]:
    matches = []
    for path in _artifact_path(artifacts, "runs", "").glob("*.json"):
        payload = _read_json(path)
        if payload.get("dataset") == dataset and payload.get("variant") == variant:
            matches.append((path.stat().st_mtime_ns, payload))
    if not matches:
        raise ValueError(f"no finalized {variant} run found for {dataset}")
    return sorted(matches, key=lambda item: item[0])[-1][1]


def _resolve_run_manifest(
    artifacts: Path, dataset: str, run_id: str
) -> dict[str, Any]:
    """Resolve an explicit run id or the delivery-friendly latest-* aliases."""

    symbolic = {
        "latest-clean": ModelVariant.CLEAN.value,
        "latest-missing": ModelVariant.MISSING.value,
        "latest-ours": ModelVariant.OURS.value,
    }
    if run_id in symbolic:
        return _latest_run_manifest(artifacts, dataset, symbolic[run_id])
    manifest = _read_json(_artifact_path(artifacts, "runs", f"{run_id}.json"))
    if manifest.get("run_id") != run_id:
        raise ValueError("run manifest identity mismatch")
    return manifest


_VALIDATION_GATE_METADATA_FIELDS = (
    "dataset",
    "config_hash",
    "source_csv_sha256",
    "seed",
    "validation_mask_sha256",
    "validation_final_rate",
    "train_start",
    "train_end",
    "validation_end",
    "test_end",
    "seq_len",
    "pred_len",
    "metric_space",
)
_MISSING_METADATA = object()
_E20_VALIDATION_MSE_LIMIT = 0.8150254474
_E20_DIAGNOSTIC_FIELDS = (
    "route_repair_regret",
    "route_repair_regret_positive_rate",
)
_E20_ABLATION_FIELDS = (
    "without_historical",
    "without_temporal",
    "without_variable",
    "historical_only",
)


def _validate_e20_trusted_metadata(metadata: Mapping[str, Any]) -> None:
    if metadata.get("stage2_route_teacher") == "frozen_stage1_hv_fusion":
        if metadata.get("stage2_forecast_fusion") != (
            "learned_hv_none_from_forecast_mse"
        ):
            raise ValueError(
                "frozen route teacher metadata contract mismatch: "
                "stage2_forecast_fusion"
            )
        temporal_weight = metadata.get("stage2_temporal_forecast_weight")
        if not (
            type(temporal_weight) in {int, float}
            and np.isfinite(temporal_weight)
            and temporal_weight == 0
        ):
            raise ValueError(
                "frozen route teacher metadata contract mismatch: "
                "stage2_temporal_forecast_weight"
            )
    for field in (
        "stage2_route_teacher_checksum_verified",
        "stage2_route_teacher_invariance_verified",
    ):
        if metadata.get(field) is not True:
            raise ValueError(
                f"frozen route teacher metadata contract mismatch: {field}"
            )

    diagnostics = metadata.get("stage2_validation_diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise ValueError(
            "frozen route teacher metadata contract mismatch: "
            "stage2_validation_diagnostics"
        )
    for field in _E20_DIAGNOSTIC_FIELDS:
        if field not in diagnostics:
            raise ValueError(
                "frozen route teacher metadata contract mismatch: " + field
            )
        _json_safe_finite_float(
            diagnostics[field], field=f"stage2_validation_diagnostics.{field}"
        )

    ablations = metadata.get("stage2_validation_inference_ablations")
    if not isinstance(ablations, Mapping):
        raise ValueError(
            "frozen route teacher metadata contract mismatch: "
            "stage2_validation_inference_ablations"
        )
    for name in _E20_ABLATION_FIELDS:
        metrics = ablations.get(name)
        if not isinstance(metrics, Mapping):
            raise ValueError(
                "frozen route teacher metadata contract mismatch: " + name
            )
        for metric_name in ("mse", "mae"):
            if metric_name not in metrics:
                raise ValueError(
                    "frozen route teacher metadata contract mismatch: "
                    f"{name}.{metric_name}"
                )
            _json_safe_finite_float(
                metrics[metric_name],
                field=f"stage2_validation_inference_ablations.{name}.{metric_name}",
            )


def _validate_ours_evaluation_gate(
    ours_manifest: dict[str, Any],
    missing_manifest: dict[str, Any],
    *,
    expected_missing_fingerprint: str | None = None,
) -> None:
    ours_metadata = ours_manifest.get("metadata")
    missing_metadata = missing_manifest.get("metadata")
    ours_metadata = ours_metadata if isinstance(ours_metadata, dict) else {}
    missing_metadata = missing_metadata if isinstance(missing_metadata, dict) else {}
    mismatches = []
    for field in _VALIDATION_GATE_METADATA_FIELDS:
        ours_value = ours_metadata.get(field, _MISSING_METADATA)
        missing_value = missing_metadata.get(field, _MISSING_METADATA)
        if (
            field == "config_hash"
            and expected_missing_fingerprint is not None
            and missing_metadata.get("missing_baseline_fingerprint")
            == expected_missing_fingerprint
        ):
            continue
        if (
            ours_value is _MISSING_METADATA
            or missing_value is _MISSING_METADATA
            or ours_value != missing_value
        ):
            ours_display = (
                "<missing>" if ours_value is _MISSING_METADATA else repr(ours_value)
            )
            missing_display = (
                "<missing>"
                if missing_value is _MISSING_METADATA
                else repr(missing_value)
            )
            mismatches.append(
                f"{field}: missing={missing_display}, ours={ours_display}"
            )
    for field in ("missing_start_rate", "missing_block_len"):
        ours_value = ours_metadata.get(field, _MISSING_METADATA)
        missing_value = missing_metadata.get(field, _MISSING_METADATA)
        if (
            (ours_value is _MISSING_METADATA) != (missing_value is _MISSING_METADATA)
            or (
                ours_value is not _MISSING_METADATA
                and ours_value != missing_value
            )
        ):
            mismatches.append(
                f"{field}: missing={missing_value!r}, ours={ours_value!r}"
            )
    if mismatches:
        raise ValueError(
            "validation comparison provenance mismatch: "
            + "; ".join(mismatches)
            + "; test evaluation is blocked"
        )

    if ours_metadata.get("stage2_route_teacher") in {
        "frozen_stage1",
        "frozen_stage1_hv_fusion",
    }:
        _validate_e20_trusted_metadata(ours_metadata)
        legacy_pressure = ours_metadata.get("missing_start_rate") in (None, 0.10)
        legacy_block = ours_metadata.get("missing_block_len") in (None, 8)
        if legacy_pressure and legacy_block:
            try:
                raw_ours_val_mse = ours_manifest["training"]["best_val_mse"]
            except (KeyError, TypeError) as exc:
                raise ValueError(
                    "E20 validation gate requires a real finite best_val_mse; "
                    "test evaluation is blocked"
                ) from exc
            if type(raw_ours_val_mse) not in {int, float} or not np.isfinite(
                raw_ours_val_mse
            ):
                raise ValueError(
                    "E20 validation gate requires a real finite best_val_mse; "
                    "test evaluation is blocked"
                )
            ours_val_mse = float(raw_ours_val_mse)
            if not ours_val_mse < _E20_VALIDATION_MSE_LIMIT:
                raise ValueError(
                    f"E20 validation gate failed: best_val_mse={ours_val_mse} must be "
                    f"lower than {_E20_VALIDATION_MSE_LIMIT}; "
                    "test evaluation is blocked"
                )
            return

    try:
        ours_val_mse = float(ours_manifest["training"]["best_val_mse"])
        missing_val_mse = float(missing_manifest["training"]["best_val_mse"])
        ours_val_mae = float(ours_manifest["training"]["best_val_mae"])
        missing_val_mae = float(missing_manifest["training"]["best_val_mae"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "validation comparison requires numeric best_val_mse and best_val_mae values; "
            "test evaluation is blocked"
        ) from exc
    if not ours_val_mse < missing_val_mse:
        raise ValueError(
            f"validation gate failed: ours best_val_mse={ours_val_mse} must be lower "
            f"than missing best_val_mse={missing_val_mse}; test evaluation is blocked"
        )
    if not ours_val_mae < missing_val_mae:
        raise ValueError(
            f"validation gate failed: ours best_val_mae={ours_val_mae} must be lower "
            f"than missing best_val_mae={missing_val_mae}; test evaluation is blocked"
        )


def _compatible_missing_baseline(
    artifacts: Path,
    dataset: str,
    metadata: Mapping[str, Any],
    *,
    validation_mask_sha256: str,
    test_mask_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    fingerprint = metadata["missing_baseline_fingerprint"]
    matches: list[tuple[int, dict[str, Any]]] = []
    for path in _artifact_path(artifacts, "runs", "").glob("*.json"):
        payload = _read_json(path)
        payload_metadata = payload.get("metadata")
        payload_metadata = payload_metadata if isinstance(payload_metadata, dict) else {}
        compatible = (
            payload_metadata.get("config_hash") == metadata.get("config_hash")
            or payload_metadata.get("missing_baseline_fingerprint") == fingerprint
        )
        if (
            payload.get("dataset") == dataset
            and payload.get("variant") == ModelVariant.MISSING.value
            and compatible
            and payload_metadata.get("validation_mask_sha256")
            == validation_mask_sha256
        ):
            matches.append((path.stat().st_mtime_ns, payload))
    if matches:
        return sorted(matches, key=lambda item: item[0])[-1][1], None
    cached = _load_cached_baseline(
        artifacts,
        dataset,
        ModelVariant.MISSING.value,
        metadata,
        expected_mask_sha256=test_mask_sha256,
    )
    if cached is None:
        raise ValueError(
            "no compatible cached Missing baseline found; train and evaluate Missing "
            "once for this data/model/missing protocol"
        )
    cached_manifest, cached_record = cached
    cached_metadata = cached_manifest.get("metadata", {})
    if cached_metadata.get("validation_mask_sha256") != validation_mask_sha256:
        raise ValueError("cached Missing baseline validation mask mismatch")
    return cached_manifest, cached_record


def _command_evaluate(args: argparse.Namespace) -> None:
    manifest = _resolve_run_manifest(args.artifacts, args.dataset, args.run_id)
    missing_record = None
    if manifest.get("variant") == ModelVariant.OURS.value:
        if args.config.exists():
            config, series, protocol = _load_context(args.config, args.dataset)
            metadata = _base_metadata(config, series, protocol, args.dataset)
            _, _, validation_expected = _mask_identity(
                config, args.dataset, protocol, "validation"
            )
            validation_mask_path = _fixed_mask_path(
                args.artifacts, args.dataset, "validation"
            )
            load_fixed_mask(validation_mask_path, validation_expected)
            _, _, test_expected = _mask_identity(
                config, args.dataset, protocol, "test"
            )
            test_mask_path = _fixed_mask_path(args.artifacts, args.dataset, "test")
            load_fixed_mask(test_mask_path, test_expected)
            missing_manifest, cached_missing_record = _compatible_missing_baseline(
                args.artifacts,
                args.dataset,
                metadata,
                validation_mask_sha256=file_sha256(validation_mask_path),
                test_mask_sha256=file_sha256(test_mask_path),
            )
            _validate_ours_evaluation_gate(
                manifest,
                missing_manifest,
                expected_missing_fingerprint=metadata["missing_baseline_fingerprint"],
            )
        else:
            # Test doubles and legacy callers without a readable config retain
            # the strict same-manifest comparison path.
            missing_manifest = _latest_run_manifest(
                args.artifacts, args.dataset, ModelVariant.MISSING.value
            )
            cached_missing_record = None
            _validate_ours_evaluation_gate(manifest, missing_manifest)
        missing_record = (
            cached_missing_record
            if cached_missing_record is not None
            else _evaluate_manifest(args, missing_manifest, allow_existing=True)
        )
    record = _evaluate_manifest(
        args,
        manifest,
        allow_existing=manifest.get("variant")
        in {ModelVariant.CLEAN.value, ModelVariant.MISSING.value},
    )
    if missing_record is not None:
        missing_mask_sha256 = missing_record.get("mask_sha256")
        ours_mask_sha256 = record.get("mask_sha256")
        if (
            not isinstance(missing_mask_sha256, str)
            or not isinstance(ours_mask_sha256, str)
            or missing_mask_sha256 != ours_mask_sha256
        ):
            raise ValueError(
                "test mask SHA-256 mismatch after gated evaluation: "
                f"missing={missing_mask_sha256!r}, ours={ours_mask_sha256!r}"
            )
    print(json.dumps(record, ensure_ascii=False, indent=2))


def _command_stress_evaluate(args: argparse.Namespace) -> None:
    """Score a frozen online model under one changed missingness parameter."""

    source_config, source_series, source_protocol = _load_context(
        args.source_config, args.dataset
    )
    target_config, target_series, target_protocol = _load_context(
        args.config, args.dataset
    )
    validate_stress_config(source_config, target_config)
    if (
        source_config.relations.online.protocol != "dynamic_context_v3"
        or source_protocol != target_protocol
        or file_sha256(source_series.source_path)
        != file_sha256(target_series.source_path)
    ):
        raise ValueError("stress evaluation requires the same E22 data and protocol")

    manifest = _read_json(args.source_run_manifest)
    if (
        manifest.get("dataset") != args.dataset
        or manifest.get("variant") != ModelVariant.OURS.value
        or manifest.get("metadata", {}).get("training_stage") != "forecast"
    ):
        raise ValueError("stress source must be a finalized E22 forecast run")
    source_metadata = _base_metadata(
        source_config, source_series, source_protocol, args.dataset
    )
    expected_source = {
        key: source_metadata[key] for key in ("dataset", "config_hash", "train_checksum")
    }
    manifest_metadata = manifest["metadata"]
    validate_dependency_metadata(manifest_metadata, expected_source)
    source_mask_sha256 = manifest_metadata.get("validation_mask_sha256")
    if not isinstance(source_mask_sha256, str) or len(source_mask_sha256) != 64:
        raise ValueError("stress source lacks a fixed validation mask hash")
    checkpoint_path = Path(str(manifest["checkpoint"]))
    checkpoint_sha256 = file_sha256(checkpoint_path)
    if checkpoint_sha256 != manifest.get("checkpoint_sha256"):
        raise ValueError("stress source checkpoint checksum mismatch")

    split = args.split
    support_start, _, mask_expected = _mask_identity(
        target_config, args.dataset, target_protocol, split
    )
    mask_path = _fixed_mask_path(args.artifacts, args.dataset, split)
    target_mask, mask_metadata = load_fixed_mask(mask_path, mask_expected)
    mask_sha256 = file_sha256(mask_path)
    result_path = _artifact_path(
        args.artifacts,
        "stress_evaluations",
        f"{manifest['run_id']}_{split}.json",
    )
    if result_path.exists():
        raise ValueError(f"stress evaluation already exists: {result_path}")

    checkpoint = load_online_checkpoint(
        checkpoint_path,
        expected_metadata=expected_source,
        expected_hashes={
            "config_hash": source_metadata["config_hash"],
            "data_hash": source_metadata["train_checksum"],
            "mask_hash": source_mask_sha256,
        },
    )
    model, _ = _restore_online_model_for_evaluation(
        source_config,
        args.dataset,
        source_protocol,
        checkpoint,
        manifest,
        device=args.device,
    )
    standardized_series, stats = _standardized_context(target_series, target_protocol)
    profile = target_config.training.for_dataset(args.dataset)
    refs = build_window_index(
        target_protocol,
        split,
        target_config.data.seq_len,
        target_config.data.pred_len,
    )
    batches = iter_prepared_batches(
        standardized_series,
        refs,
        target_mask,
        mask_support_start=support_start,
        train_stats=stats,
        variant=ModelVariant.OURS,
        batch_size=profile.batch_size,
        patch_len=target_config.patch.length,
        patch_stride=target_config.patch.stride,
        normalization_eps=target_config.model.normalization_eps,
        drop_last=profile.drop_last,
    )
    batches = _progress_batches(
        batches,
        total=_progress_batch_count(len(refs), profile.batch_size, drop_last=profile.drop_last),
        label=f"{split} {args.dataset} ours zero-shot",
    )
    evaluation = evaluate_online_model(model, batches, device=args.device)
    record = {
        "kind": "zero_shot_missing_stress",
        "dataset": args.dataset,
        "split": split,
        "source_run_id": manifest["run_id"],
        "source_checkpoint_sha256": checkpoint_sha256,
        "source_config_hash": source_metadata["config_hash"],
        "source_validation_mask_sha256": source_mask_sha256,
        "target_config_hash": config_sha256(target_config),
        "target_missing_start_rate": target_config.missing.start_rate,
        "target_missing_block_len": target_config.missing.block_len,
        "target_mask_sha256": mask_sha256,
        "final_missing_rate": mask_metadata.final_rate,
        "metric_space": "train_standardized",
        "mse": evaluation.metrics.mse,
        "mae": evaluation.metrics.mae,
        "imputation_mse": evaluation.imputation_mse,
        "imputation_mae": evaluation.imputation_mae,
        "online_diagnostics": evaluation.diagnostics,
        "inference_ablations": {
            name: {"mse": value.mse, "mae": value.mae}
            for name, value in evaluation.inference_ablations.items()
        },
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json_atomic(result_path, record)
    print(json.dumps({**record, "record_path": str(result_path)}, ensure_ascii=False, indent=2))


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=Path("configs/stage1.yaml"))
    parser.add_argument(
        "--dataset",
        required=True,
        help="dataset key declared under data.datasets in the selected config",
    )
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))


def _command_inspect_migrated(args: argparse.Namespace) -> None:
    """Check a copied run without evaluation or writes to its artifacts."""
    catalog = MigrationCatalog.load(args.migration_catalog)
    manifest, paths = load_migrated_run(args.run_manifest, catalog)
    checkpoints = {}
    for name in ("stage1_checkpoint", "checkpoint"):
        if name in paths:
            payload = load_online_checkpoint(paths[name])
            checkpoints[name] = int(payload["online_state"]["epoch"])
    output = {
        "run_id": manifest["run_id"],
        "manifest": str(catalog.resolve(args.run_manifest)),
        "checkpoint": str(paths["checkpoint"]),
        "checkpoint_epoch": checkpoints["checkpoint"],
        "stage1_checkpoint_epoch": checkpoints.get("stage1_checkpoint"),
    }
    if args.mask_split:
        mask = paths["checkpoint"].parent.parent / "masks" / f"{manifest['dataset']}_{args.mask_split}.npz"
        mask = catalog.resolve(mask)
        output["mask"] = str(mask)
        output["mask_sha256"] = file_sha256(mask)
    print(json.dumps(output, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="malr")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate-data")
    _add_common(validate)
    validate.set_defaults(handler=_command_validate_data)

    masks = commands.add_parser("make-eval-masks")
    _add_common(masks)
    masks.set_defaults(handler=_command_make_eval_masks)

    clean = commands.add_parser("train-clean")
    _add_common(clean)
    clean.add_argument("--device", default="cpu")
    clean.set_defaults(handler=lambda args: _command_train(args, ModelVariant.CLEAN))

    bank = commands.add_parser("build-bank")
    _add_common(bank)
    bank.add_argument("--clean-checkpoint", type=Path)
    bank.add_argument("--bank", type=Path)
    bank.add_argument("--device", default="cpu")
    bank.set_defaults(handler=_command_build_bank)

    train = commands.add_parser("train")
    _add_common(train)
    train.add_argument(
        "--variant",
        choices=("missing", "missing_pretrained", "ours"),
        required=True,
    )
    train.add_argument("--clean-checkpoint", type=Path)
    train.add_argument("--bank", type=Path)
    train.add_argument("--stage", choices=("repair", "forecast"))
    train.add_argument("--repair-batch-size", type=int)
    train.add_argument(
        "--repair-matmul-precision",
        choices=("highest", "high", "medium"),
    )
    train.add_argument("--init-checkpoint", type=Path)
    train.add_argument("--warm-start-checkpoint", type=Path)
    train.add_argument("--warm-start-epoch", type=int)
    train.add_argument("--additional-epochs", type=int)
    train.add_argument(
        "--stage2-route-teacher",
        choices=("frozen_stage1", "frozen_stage1_hv_fusion"),
    )
    train.add_argument("--repaired-data", type=Path)
    train.add_argument("--device", default="cpu")
    train.set_defaults(
        handler=lambda args: _command_train(args, ModelVariant(args.variant))
    )

    export_repaired = commands.add_parser("export-repaired")
    _add_common(export_repaired)
    export_repaired.add_argument("--checkpoint", type=Path)
    export_repaired.add_argument("--device", default="cpu")
    export_repaired.set_defaults(handler=_command_export_repaired)

    evaluate = commands.add_parser("evaluate")
    _add_common(evaluate)
    evaluate.add_argument("--run-id", required=True)
    evaluate.add_argument("--device", default="cpu")
    evaluate.set_defaults(handler=_command_evaluate)

    stress = commands.add_parser("stress-evaluate")
    _add_common(stress)
    stress.add_argument("--source-config", type=Path, required=True)
    stress.add_argument("--source-run-manifest", type=Path, required=True)
    stress.add_argument("--split", choices=("validation", "test"), required=True)
    stress.add_argument("--device", default="cpu")
    stress.set_defaults(handler=_command_stress_evaluate)
    inspect = commands.add_parser("inspect-migrated")
    inspect.add_argument("--migration-catalog", type=Path, required=True)
    inspect.add_argument("--run-manifest", type=Path, required=True)
    inspect.add_argument("--mask-split", choices=("validation", "test"))
    inspect.set_defaults(handler=_command_inspect_migrated)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.handler(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
