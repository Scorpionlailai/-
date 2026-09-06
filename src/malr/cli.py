from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from malr.config import Stage1Config, config_sha256, load_config
from malr.data.datasets import CleanSeries, load_clean_csv
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
from malr.data.splits import DatasetProtocol, protocol_for
from malr.data.windows import build_window_index
from malr.evaluation.evaluator import evaluate_model
from malr.evaluation.metrics import degradation_recovery_rate
from malr.evaluation.report import (
    ResultRecord,
    file_sha256,
    write_json_atomic,
    write_result_record,
)
from malr.models.filtering import MissingAwareEdgeFilter
from malr.models.patchtst import PatchTSTCore
from malr.models.recovery import MissingAwareRecovery
from malr.models.relations import HistoricalRelation
from malr.models.stage1 import ModelVariant, Stage1Model, initialize_from_clean_checkpoint
from malr.prototypes.bank import (
    PrototypeBank,
    build_prototype_bank,
    load_bank,
    save_bank,
    source_checksum,
)
from malr.reproducibility import seed_everything
from malr.training.checkpoints import load_checkpoint
from malr.training.data import iter_prepared_batches
from malr.training.trainer import Trainer, validate_dependency_metadata


def _project_root(config_path: Path) -> Path:
    resolved = config_path.resolve()
    return resolved.parent.parent if resolved.parent.name == "configs" else Path.cwd()


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
    protocol = protocol_for(dataset, len(series.values))
    return config, series, protocol


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
    bank: PrototypeBank | None = None,
) -> Stage1Model:
    selected = ModelVariant(variant)
    core = _build_core(config, dataset, config.data.datasets[dataset].n_vars)
    if selected is ModelVariant.OURS:
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
            edge_filter=edge_filter,
            relation_selection=config.relations.selection,
        )
        model = Stage1Model(core, selected, recovery=recovery)
        return model
    model = Stage1Model(core, selected)
    if selected is ModelVariant.MISSING_PRETRAINED:
        if clean_checkpoint is None:
            raise ValueError("missing_pretrained requires a clean checkpoint")
        initialize_from_clean_checkpoint(model, clean_checkpoint)
    return model


def _fixed_mask_path(artifacts: Path, dataset: str, split: str) -> Path:
    return _artifact_path(artifacts, "masks", f"{dataset}_{split}.npz")


def _load_dependency(
    path: Path,
    expected: dict[str, Any],
) -> dict[str, Any]:
    payload = load_checkpoint(path)
    validate_dependency_metadata(payload["metadata"], expected)
    return payload


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
) -> dict[str, Any]:
    selected = ModelVariant(variant)
    seed_everything(config.training.seed)
    metadata = _base_metadata(config, series, protocol, dataset)
    standardized_series, stats = _standardized_context(series, protocol)
    training_profile = config.training.for_dataset(dataset)
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
    if selected is ModelVariant.OURS and config.ablations.use_historical:
        if bank_path is None:
            raise ValueError("ours with historical recovery requires --bank")
        bank = load_bank(
            bank_path,
            source_train=standardized_series[: protocol.train.end],
            dataset=dataset,
            config_sha256=metadata["config_hash"],
        )

    model = build_experiment_model(
        config,
        dataset,
        selected,
        clean_checkpoint=clean_payload,
        bank=bank,
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
        timeline_mask = (
            np.ones((protocol.train.end, series.values.shape[1]), dtype=bool)
            if selected is ModelVariant.CLEAN
            else mask_manager.mask_for_epoch(epoch)
        )
        return iter_prepared_batches(
            standardized_series,
            train_refs,
            timeline_mask,
            mask_support_start=0,
            train_stats=stats,
            variant=selected,
            batch_size=training_profile.batch_size,
            patch_len=config.patch.length,
            patch_stride=config.patch.stride,
            shuffle_seed=config.training.seed + epoch,
            normalization_eps=config.model.normalization_eps,
            drop_last=training_profile.drop_last,
        )

    def validation_batches():
        return iter_prepared_batches(
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

    checkpoint_path = _artifact_path(
        artifacts, "checkpoints", f"{dataset}_{selected.value}.pt"
    )
    run_metadata = {
        **metadata,
        "role": "clean" if selected is ModelVariant.CLEAN else "missing_experiment",
        "variant": selected.value,
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
    }
    trainer = Trainer(
        model,
        checkpoint_path=checkpoint_path,
        checkpoint_metadata=run_metadata,
        max_epochs=training_profile.max_epochs,
        patience=training_profile.early_stopping_patience,
        learning_rate=training_profile.learning_rate,
        weight_decay=config.training.weight_decay,
        lr_schedule=training_profile.lr_schedule,
        lambda_rec=config.training.lambda_rec,
        lambda_keep=(
            config.relations.filter.lambda_keep
            if selected is ModelVariant.OURS
            and config.relations.selection == "missing_aware_filter"
            else 0.0
        ),
        use_recovery_loss=(
            selected is ModelVariant.OURS and config.ablations.use_recovery_loss
        ),
        device=device,
    )
    result = trainer.fit(training_batches, validation_batches)
    output = asdict(result)
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
        "checkpoint": str(result.best_checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_digest,
        "clean_checkpoint": (
            str(clean_checkpoint_path.resolve())
            if selected is ModelVariant.MISSING_PRETRAINED
            and clean_checkpoint_path is not None
            else None
        ),
        "bank": (
            str(bank_path.resolve())
            if selected is ModelVariant.OURS
            and config.ablations.use_historical
            and bank_path is not None
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
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _command_build_bank(args: argparse.Namespace) -> None:
    config, series, protocol = _load_context(args.config, args.dataset)
    metadata = _base_metadata(config, series, protocol, args.dataset)
    if args.clean_checkpoint is not None:
        expected = {
            key: metadata[key] for key in ("dataset", "config_hash", "train_checksum")
        }
        _load_dependency(args.clean_checkpoint, expected)
    prototype = config.prototype
    standardized_series, _ = _standardized_context(series, protocol)
    bank = build_prototype_bank(
        standardized_series[: protocol.train.end],
        train_end=protocol.train.end,
        dataset=args.dataset,
        config_sha256=metadata["config_hash"],
        allowed_train_end=protocol.train.end,
        window=prototype.matrix_profile_window,
        scan_stride=prototype.scan_stride * config.patch.stride,
        exclusion=prototype.matrix_profile_exclusion,
        motif_percentile=prototype.motif_percentile,
        min_occurrences=prototype.min_occurrences,
        max_per_variable=prototype.max_per_variable,
    )
    output = args.bank or _artifact_path(
        args.artifacts, "banks", f"{args.dataset}.npz"
    )
    save_bank(bank, output)
    print(
        json.dumps(
            {
                "path": str(output),
                "prototype_counts": [len(records) for records in bank.records],
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


def _find_variant_result(
    artifacts: Path, dataset: str, variant: str, config_hash: str
) -> dict[str, Any] | None:
    result_directory = _artifact_path(artifacts, "evaluations", "")
    matches = []
    for path in result_directory.glob("*.json"):
        payload = _read_json(path)
        if (
            payload.get("dataset") == dataset
            and payload.get("variant") == variant
            and payload.get("config_hash") == config_hash
        ):
            matches.append(payload)
    if not matches:
        return None
    return sorted(matches, key=lambda item: str(item.get("timestamp_utc", "")))[-1]


def _evaluate_manifest(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    *,
    allow_existing: bool,
) -> dict[str, Any]:
    run_id = str(manifest["run_id"])
    result_path = _artifact_path(args.artifacts, "evaluations", f"{run_id}.json")
    if result_path.exists():
        if allow_existing:
            return _read_json(result_path)
        raise ValueError(f"run {run_id!r} has already been evaluated")

    config, series, protocol = _load_context(args.config, args.dataset)
    if manifest.get("dataset") != args.dataset:
        raise ValueError("run manifest dataset does not match evaluation dataset")
    metadata = _base_metadata(config, series, protocol, args.dataset)
    expected = {key: metadata[key] for key in ("dataset", "config_hash", "train_checksum")}
    validate_dependency_metadata(manifest.get("metadata", {}), expected)
    checkpoint_path = Path(manifest["checkpoint"])
    if file_sha256(checkpoint_path) != manifest.get("checkpoint_sha256"):
        raise ValueError("final checkpoint checksum mismatch")
    checkpoint = _load_dependency(checkpoint_path, expected)
    selected = ModelVariant(manifest["variant"])
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
    if selected is ModelVariant.OURS and config.ablations.use_historical:
        bank_value = manifest.get("bank")
        if not bank_value:
            raise ValueError("ours run manifest lacks its prototype bank")
        bank_path = Path(bank_value)
        bank = load_bank(
            bank_path,
            source_train=standardized_series[: protocol.train.end],
            dataset=args.dataset,
            config_sha256=metadata["config_hash"],
        )
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
    metrics = evaluate_model(model, batches, device=args.device)

    clean_result = _find_variant_result(
        args.artifacts, args.dataset, "clean", metadata["config_hash"]
    )
    naive_result = _find_variant_result(
        args.artifacts, args.dataset, "missing", metadata["config_hash"]
    )
    drr_mse = drr_mae = None
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
    )
    write_result_record(record, result_path)
    return asdict(record)


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


def _validate_ours_evaluation_gate(
    ours_manifest: dict[str, Any], missing_manifest: dict[str, Any]
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
    if mismatches:
        raise ValueError(
            "validation comparison provenance mismatch: "
            + "; ".join(mismatches)
            + "; test evaluation is blocked"
        )

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


def _command_evaluate(args: argparse.Namespace) -> None:
    manifest_path = _artifact_path(args.artifacts, "runs", f"{args.run_id}.json")
    manifest = _read_json(manifest_path)
    if manifest.get("run_id") != args.run_id:
        raise ValueError("run manifest identity mismatch")
    missing_record = None
    if manifest.get("variant") == ModelVariant.OURS.value:
        missing_manifest = _latest_run_manifest(
            args.artifacts, args.dataset, ModelVariant.MISSING.value
        )
        _validate_ours_evaluation_gate(manifest, missing_manifest)
        missing_record = _evaluate_manifest(args, missing_manifest, allow_existing=True)
    record = _evaluate_manifest(args, manifest, allow_existing=False)
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


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=Path("configs/stage1.yaml"))
    parser.add_argument("--dataset", choices=("ETTh1", "Weather"), required=True)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))


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
    train.add_argument("--device", default="cpu")
    train.set_defaults(
        handler=lambda args: _command_train(args, ModelVariant(args.variant))
    )

    evaluate = commands.add_parser("evaluate")
    _add_common(evaluate)
    evaluate.add_argument("--run-id", required=True)
    evaluate.add_argument("--device", default="cpu")
    evaluate.set_defaults(handler=_command_evaluate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.handler(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
