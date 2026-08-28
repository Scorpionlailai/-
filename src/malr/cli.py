from __future__ import annotations

import argparse
from copy import deepcopy
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
from malr.data.normalization import TrainVariableStats, fit_train_variable_stats
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
        "train_checksum": source_checksum(train_values),
        "train_start": protocol.train.start,
        "train_end": protocol.train.end,
        "validation_end": protocol.validation.end,
        "test_end": protocol.test.end,
    }


def _train_stats(series: CleanSeries, protocol: DatasetProtocol) -> TrainVariableStats:
    return fit_train_variable_stats(
        torch.from_numpy(series.values[protocol.train.start : protocol.train.end])
    )


def _build_core(config: Stage1Config, n_vars: int) -> PatchTSTCore:
    return PatchTSTCore(
        c_in=n_vars,
        seq_len=config.data.seq_len,
        pred_len=config.data.pred_len,
        patch_len=config.patch.length,
        stride=config.patch.stride,
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        d_ff=config.model.d_ff,
        n_layers=config.model.encoder_layers,
        dropout=config.model.dropout,
        head_dropout=config.model.head_dropout,
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
    core = _build_core(config, config.data.datasets[dataset].n_vars)
    if selected is ModelVariant.OURS:
        if clean_checkpoint is None or bank is None:
            raise ValueError("ours requires a clean checkpoint and prototype bank")
        teacher = deepcopy(core.W_P)
        historical = HistoricalRelation(
            bank,
            teacher,
            topk=config.relations.historical_topk,
            patch_len=config.patch.length,
            patch_stride=config.patch.stride,
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
        )
        model = Stage1Model(core, selected, recovery=recovery)
        initialize_from_clean_checkpoint(model, clean_checkpoint)
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
    dependency_expected = {
        key: metadata[key] for key in ("dataset", "config_hash", "train_checksum")
    }

    clean_payload = None
    if selected in {ModelVariant.MISSING_PRETRAINED, ModelVariant.OURS}:
        if clean_checkpoint_path is None:
            raise ValueError(f"{selected.value} requires --clean-checkpoint")
        # Dependency validation deliberately precedes model allocation.
        clean_payload = _load_dependency(clean_checkpoint_path, dependency_expected)

    bank = None
    if selected is ModelVariant.OURS:
        if bank_path is None:
            raise ValueError("ours requires --bank")
        bank = load_bank(
            bank_path,
            source_train=series.values[: protocol.train.end],
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
    stats = _train_stats(series, protocol)
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
    if selected is ModelVariant.CLEAN:
        validation_mask = np.ones(
            (validation_support_end - validation_support_start, series.values.shape[1]),
            dtype=bool,
        )
    else:
        validation_mask, _ = load_fixed_mask(
            _fixed_mask_path(artifacts, dataset, "validation"), validation_expected
        )

    def training_batches(epoch: int):
        timeline_mask = (
            np.ones((protocol.train.end, series.values.shape[1]), dtype=bool)
            if selected is ModelVariant.CLEAN
            else mask_manager.mask_for_epoch(epoch)
        )
        return iter_prepared_batches(
            series.values,
            train_refs,
            timeline_mask,
            mask_support_start=0,
            train_stats=stats,
            variant=selected,
            batch_size=config.training.batch_size,
            patch_len=config.patch.length,
            patch_stride=config.patch.stride,
            shuffle_seed=config.training.seed + epoch,
        )

    def validation_batches():
        return iter_prepared_batches(
            series.values,
            validation_refs,
            validation_mask,
            mask_support_start=validation_support_start,
            train_stats=stats,
            variant=selected,
            batch_size=config.training.batch_size,
            patch_len=config.patch.length,
            patch_stride=config.patch.stride,
        )

    checkpoint_path = _artifact_path(
        artifacts, "checkpoints", f"{dataset}_{selected.value}.pt"
    )
    run_metadata = {
        **metadata,
        "role": "clean" if selected is ModelVariant.CLEAN else "missing_experiment",
        "variant": selected.value,
    }
    trainer = Trainer(
        model,
        checkpoint_path=checkpoint_path,
        checkpoint_metadata=run_metadata,
        max_epochs=config.training.max_epochs,
        patience=config.training.early_stopping_patience,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        lambda_rec=config.training.lambda_rec,
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
            if clean_checkpoint_path is not None
            else None
        ),
        "bank": str(bank_path.resolve()) if bank_path is not None else None,
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
                "config_hash": config_sha256(config),
            },
            ensure_ascii=False,
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
    if clean_checkpoint is None and variant in {
        ModelVariant.MISSING_PRETRAINED,
        ModelVariant.OURS,
    }:
        clean_checkpoint = _artifact_path(
            args.artifacts, "checkpoints", f"{args.dataset}_clean.pt"
        )
    bank = getattr(args, "bank", None)
    if bank is None and variant is ModelVariant.OURS:
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
    expected = {key: metadata[key] for key in ("dataset", "config_hash", "train_checksum")}
    clean_checkpoint = args.clean_checkpoint or _artifact_path(
        args.artifacts, "checkpoints", f"{args.dataset}_clean.pt"
    )
    _load_dependency(clean_checkpoint, expected)
    prototype = config.prototype
    bank = build_prototype_bank(
        series.values[: protocol.train.end],
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
    artifacts: Path, dataset: str, variant: str
) -> dict[str, Any] | None:
    result_directory = _artifact_path(artifacts, "evaluations", "")
    matches = []
    for path in result_directory.glob("*.json"):
        payload = _read_json(path)
        if payload.get("dataset") == dataset and payload.get("variant") == variant:
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

    clean_payload = None
    if selected in {ModelVariant.MISSING_PRETRAINED, ModelVariant.OURS}:
        clean_path_value = manifest.get("clean_checkpoint")
        if not clean_path_value:
            raise ValueError("run manifest lacks its clean checkpoint dependency")
        clean_payload = _load_dependency(Path(clean_path_value), expected)
    bank = None
    bank_path = None
    if selected is ModelVariant.OURS:
        bank_value = manifest.get("bank")
        if not bank_value:
            raise ValueError("ours run manifest lacks its prototype bank")
        bank_path = Path(bank_value)
        bank = load_bank(
            bank_path,
            source_train=series.values[: protocol.train.end],
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

    support_start, _, mask_expected = _mask_identity(
        config, args.dataset, protocol, "test"
    )
    mask_path = _fixed_mask_path(args.artifacts, args.dataset, "test")
    test_mask, mask_metadata = load_fixed_mask(mask_path, mask_expected)
    stats = _train_stats(series, protocol)
    refs = build_window_index(
        protocol, "test", config.data.seq_len, config.data.pred_len
    )
    batches = iter_prepared_batches(
        series.values,
        refs,
        test_mask,
        mask_support_start=support_start,
        train_stats=stats,
        variant=selected,
        batch_size=config.training.batch_size,
        patch_len=config.patch.length,
        patch_stride=config.patch.stride,
    )
    metrics = evaluate_model(model, batches, device=args.device)

    clean_result = _find_variant_result(args.artifacts, args.dataset, "clean")
    naive_result = _find_variant_result(args.artifacts, args.dataset, "missing")
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
            0.0 if selected is ModelVariant.CLEAN else mask_metadata.final_rate
        ),
        trainable_parameters=int(training["trainable_parameters"]),
        training_seconds=float(training["training_seconds"]),
        config_hash=metadata["config_hash"],
        checkpoint_sha256=file_sha256(checkpoint_path),
        mask_sha256=file_sha256(mask_path),
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


def _command_evaluate(args: argparse.Namespace) -> None:
    manifest_path = _artifact_path(args.artifacts, "runs", f"{args.run_id}.json")
    manifest = _read_json(manifest_path)
    if manifest.get("run_id") != args.run_id:
        raise ValueError("run manifest identity mismatch")
    if manifest.get("variant") == ModelVariant.OURS.value:
        # A single final command evaluates the selected clean and naive baselines
        # first when needed, enabling both DRRs without any model touching test twice.
        for baseline_variant in (ModelVariant.CLEAN.value, ModelVariant.MISSING.value):
            baseline = _latest_run_manifest(
                args.artifacts, args.dataset, baseline_variant
            )
            _evaluate_manifest(args, baseline, allow_existing=True)
    record = _evaluate_manifest(args, manifest, allow_existing=False)
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
