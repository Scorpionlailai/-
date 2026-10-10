"""Locked ETTh1 P4 E23 training and one-shot selected evaluation."""

from __future__ import annotations

import argparse
import json
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import Sequence

_PROCESS_STARTED = perf_counter()

import torch

from integrations.forecast_compare.protocol import P4Protocol, iter_p4_batches
from integrations.forecast_compare.resources import TrainingResources, selected_training_resources
from integrations.forecast_compare.report import CLEAN_FINGERPRINT, MISSING_FINGERPRINT
from malr.data.patching import patchify_with_state
from malr.evaluation.report import file_sha256
from malr.migration import MigrationCatalog
from malr.prototypes.online_bank import OnlinePrototypeMemory

from .config import E23Settings, load_e23_p4_protocol, load_e23_settings
from .train import (
    _atomic_json, _file_sha256, _key_sha256, _memory_state_sha256, _new_memory,
    _starts, _trial_identity, build_forecast_model, load_stage2_selection, train_forecast,
    train_repair, train_history_adapter,
)


def shared_drr(model: float, *, clean: float, missing: float) -> float | None:
    """Percentage recovered from the same cached PatchTST Missing−Clean gap."""
    if not all(isfinite(value) and value >= 0 for value in (model, clean, missing)):
        raise ValueError("shared DRR inputs must be finite and nonnegative")
    gap = missing - clean
    return None if gap <= 1e-12 else (missing - model) / gap * 100.0


def _shared_references(protocol: P4Protocol) -> dict[str, object]:
    root = protocol.artifacts.parents[2]
    paths = {
        "clean": root / "clean" / CLEAN_FINGERPRINT / "baseline.json",
        "missing": root / "sr010_bl8" / "missing" / MISSING_FINGERPRINT / "baseline.json",
    }
    result: dict[str, object] = {}
    for name, path in paths.items():
        record = json.loads(path.read_text(encoding="utf-8"))
        values = record.get("result", {})
        expected = CLEAN_FINGERPRINT if name == "clean" else MISSING_FINGERPRINT
        if record.get("fingerprint") != expected or record.get("variant") != name:
            raise ValueError(f"cached {name} baseline fingerprint mismatch")
        if (values.get("source_csv_sha256") != protocol.source_sha256
                or values.get("clean_baseline_fingerprint") != CLEAN_FINGERPRINT
                or values.get("missing_baseline_fingerprint") != MISSING_FINGERPRINT
                or values.get("metric_space") != "train_standardized"
                or values.get("seed") != 2021):
            raise ValueError(f"cached {name} baseline identity mismatch")
        expected_mask = None if name == "clean" else protocol.mask_sha256["test"]
        if values.get("mask_sha256") != expected_mask:
            raise ValueError(f"cached {name} baseline test mask mismatch")
        result[name] = {
            "mse": float(values["mse"]), "mae": float(values["mae"]),
            "path": str(path.resolve()), "sha256": file_sha256(path),
            "fingerprint": expected,
        }
    return result


def evaluate_selected(
    protocol: P4Protocol, settings: E23Settings, *, selection: Path,
    output: Path, device: torch.device,
) -> dict[str, object]:
    selection = Path(selection).resolve(strict=True)
    payload = load_stage2_selection(selection, protocol, settings, device=device)
    record = json.loads(selection.read_text(encoding="utf-8"))
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"E23 selected evaluation already exists: {output}")
    references = _shared_references(protocol)
    memory: OnlinePrototypeMemory = _new_memory(protocol)
    memory.load_state_dict(payload["memory_state"])
    model = build_forecast_model(
        protocol, settings, device=device,
        forecast_mixer=record.get("forecast_mixer", "none"),
        history_adapter=record.get("history_adapter", "none"),
        forecast_input=record.get("forecast_input", "explicit_v1"),
    )
    model.load_state_dict(payload["model_state"], strict=True)
    model.repair.historical.retrieval.load_state_dict(payload["retrieval_state"])
    model.repair.historical.refresh_snapshot(memory.snapshot())
    model.eval()
    memory_before = memory.snapshot_hash()
    key_before = _key_sha256(model.repair.historical.retrieval.state_dict())
    squared = absolute = repair_squared = repair_absolute = 0.0
    target_count = hidden_count = windows = damaged_count = 0
    availability = torch.zeros(4, dtype=torch.float64)
    weight = torch.zeros(4, dtype=torch.float64)
    suppressed = torch.zeros(4, dtype=torch.float64)
    candidate_count = torch.zeros(3, dtype=torch.float64)
    adapter_available = 0
    adapter_candidates = 0
    empty_fallback = 0
    total = len(protocol.refs["test"]) // settings.batch_size
    names = ("none", "historical", "variable", "temporal")
    with torch.inference_mode():
        for index, cpu_batch in enumerate(iter_p4_batches(
            protocol, "test", batch_size=settings.batch_size,
        ), 1):
            batch = cpu_batch.to(device)
            starts = _starts(batch, stride=model.repair.stride,
                             patches=model.repair.patch_count, device=device)
            prediction, output_batch = model(
                batch.normalized_observed, batch.observed_mask,
                batch.query_mean, batch.query_std, starts,
            )
            if prediction.shape != batch.forecast_target.shape:
                raise ValueError("E23 selected forecast shape mismatch")
            error = prediction - batch.forecast_target
            squared += error.square().sum().item()
            absolute += error.abs().sum().item()
            target_count += error.numel()
            windows += error.shape[0]
            hidden = ~batch.observed_mask
            repair_error = output_batch.explicit.normalized_sequence - batch.normalized_target
            repair_squared += repair_error.square()[hidden].sum().item()
            repair_absolute += repair_error.abs()[hidden].sum().item()
            hidden_count += hidden.sum().item()

            patches = patchify_with_state(
                batch.normalized_observed, batch.observed_mask,
                model.repair.patch_length, model.repair.stride,
            )
            damaged = patches.rho.lt(1.0)
            damaged_count += damaged.sum().item()
            available = torch.stack((
                torch.ones_like(damaged),
                *(output_batch.evidence[name].available for name in names[1:]),
            ), dim=-1)
            availability += (available & damaged[..., None]).sum(dim=(0, 1, 2)).cpu()
            routed_weight = output_batch.router.weights
            weight += (routed_weight * damaged[..., None]).sum(dim=(0, 1, 2)).double().cpu()
            suppressed += ((routed_weight < 1e-3) & available & damaged[..., None]).sum(
                dim=(0, 1, 2),
            ).cpu()
            empty_fallback += ((~available[..., 1:].any(-1)) & damaged).sum().item()
            for expert_index, name in enumerate(names[1:]):
                candidate_count[expert_index] += output_batch.evidence[name].provenance[
                    "candidate_count"
                ][damaged].sum().item()
            if model.repair.history_adapter is not None:
                adapter_evidence = output_batch.evidence["historical_variable"]
                adapter_available += (adapter_evidence.available & damaged).sum().item()
                adapter_candidates += adapter_evidence.provenance["candidate_count"][damaged].sum().item()
            if index == 1 or index % 5 == 0 or index == total:
                print(f"stage=evaluate split=test batch={index}/{total} windows={windows}", flush=True)
    if windows != 2688 or target_count == 0 or hidden_count == 0 or damaged_count == 0:
        raise ValueError("E23 selected test does not contain the locked 2688 windows/targets")
    memory_after = memory.snapshot_hash()
    key_after = _key_sha256(model.repair.historical.retrieval.state_dict())
    if memory_after != memory_before or key_after != key_before:
        raise RuntimeError("E23 selected test changed read-only H state")
    mse, mae = squared / target_count, absolute / target_count
    clean, missing = references["clean"], references["missing"]
    result: dict[str, object] = {
        "model": "E23 explicit H/V/T repair + mask-aware PatchTST",
        "stage": "forecast", "split": "test", "metric_space": "train_standardized",
        "selection_manifest": str(selection),
        "selection_manifest_sha256": _file_sha256(selection),
        "checkpoint_sha256": record["checkpoint_sha256"],
        "stage1_manifest_sha256": record["stage1_manifest_sha256"],
        "source_csv_sha256": protocol.source_sha256,
        "validation_mask_sha256": protocol.mask_sha256["validation"],
        "test_mask_sha256": protocol.mask_sha256["test"],
        "protocol_config_sha256": record["protocol_config_sha256"],
        "e23_config_sha256": settings.config_sha256,
        **_trial_identity(settings),
        "memory_sha256": record["memory_sha256"],
        "key_sha256": record["key_sha256"],
        "h_state_unchanged": True,
        "seed": settings.seed, "best_epoch": record["best_epoch"],
        "best_val_mse": record["best_val_mse"],
        "selection_metric": record.get("selection_metric", "mse"),
        "forecast_mixer": record.get("forecast_mixer", "none"),
        "forecast_input": record.get("forecast_input", "explicit_v1"),
        "history_adapter": record.get("history_adapter", "none"),
        **({"historical_archive_sha256": record["historical_archive_sha256"],
            "parent_stage2_manifest_sha256": record["parent_stage2_manifest_sha256"],
            "parent_stage2_checkpoint_sha256": record["parent_stage2_checkpoint_sha256"]}
           if model.repair.history_adapter is not None else {}),
        "windows": windows, "target_elements": target_count,
        "mse": mse, "mae": mae,
        "repair_mse": repair_squared / hidden_count,
        "repair_mae": repair_absolute / hidden_count,
        "cached_references": references,
        "drr_mse_percent": shared_drr(mse, clean=clean["mse"], missing=missing["mse"]),
        "drr_mae_percent": shared_drr(mae, clean=clean["mae"], missing=missing["mae"]),
        "expert_diagnostics": {
            "damaged_patch_targets": damaged_count,
            "availability_rate": {
                name: availability[i].item() / damaged_count for i, name in enumerate(names)
            },
            "weight_mean": {
                name: weight[i].item() / damaged_count for i, name in enumerate(names)
            },
            "available_but_suppressed_rate": {
                name: suppressed[i].item() / damaged_count for i, name in enumerate(names)
            },
            "mean_candidate_count": {
                name: candidate_count[i].item() / damaged_count for i, name in enumerate(names[1:])
            },
            "no_evidence_fallback_rate": empty_fallback / damaged_count,
        },
        **({"adapter_diagnostics": {
            "availability_rate": adapter_available / damaged_count,
            "mean_candidate_count": adapter_candidates / damaged_count,
            "gates": model.repair.history_adapter.gates.detach().cpu().tolist(),
            "effective_scales": (
                0.1 * torch.tanh(model.repair.history_adapter.gates)
            ).detach().cpu().tolist(),
        }} if model.repair.history_adapter is not None else {}),
    }
    result['training_resources'] = selected_training_resources([
        Path(record['stage1_manifest']), selection,
    ])
    _atomic_json(result, output)
    print(json.dumps(result, indent=2), flush=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train-repair", "train-forecast", "train-adapter", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--protocol-config", type=Path, required=True)
        command.add_argument("--e23-config", type=Path, required=True)
        command.add_argument("--migration-catalog", type=Path, required=True)
        command.add_argument("--mask-artifacts", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--device", default="cuda")
        if name == "train-forecast":
            command.add_argument("--stage1-manifest", type=Path, required=True)
            command.add_argument("--selection-metric", choices=("mse", "mae"), default="mse")
            command.add_argument("--repair-retention", choices=("none", "uniform_v1", "forecast_gain_v1"), default="none")
            command.add_argument("--forecast-loss", choices=(
                "mse", "mse_plus_0p2_mae", "mse_plus_0p2_mae_late_horizon_v1",
                "mse_plus_0p2_mae_utility_router_v1"),
                                 default="mse")
            command.add_argument("--forecast-mixer", choices=("none", "patch_attention_v1"),
                                 default="none")
            command.add_argument("--forecast-input", choices=("explicit_v1", "implicit_latent_v1"),
                                 default="explicit_v1")
        if name == "train-adapter":
            command.add_argument("--parent-selection", type=Path, required=True)
            command.add_argument("--adapter-variant",
                                 choices=("aligned_v1", "aligned_bounded_v2"),
                                 default="aligned_v1")
        if name == "evaluate":
            command.add_argument("--selection", type=Path, required=True)
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    if args.command == "evaluate":
        catalog = MigrationCatalog.load(args.migration_catalog, project="E23")
        protocol = load_e23_p4_protocol(
            catalog=catalog, protocol_config=args.protocol_config,
            mask_artifacts=args.mask_artifacts,
        )
        settings = load_e23_settings(args.e23_config, protocol=protocol)
        evaluate_selected(protocol, settings, selection=args.selection,
                          output=args.output, device=device)
        return 0
    stage = {"train-repair": "repair", "train-forecast": "forecast",
             "train-adapter": "adapter"}[args.command]
    with TrainingResources(args.output, device=device, batch_size=128, model="E23",
                           started_at=_PROCESS_STARTED) as resources:
        catalog = MigrationCatalog.load(args.migration_catalog, project="E23")
        protocol = load_e23_p4_protocol(
            catalog=catalog, protocol_config=args.protocol_config,
            mask_artifacts=args.mask_artifacts,
        )
        settings = load_e23_settings(args.e23_config, protocol=protocol)
        resources.batch_size = settings.batch_size
        with resources.stage(stage):
            if args.command == "train-repair":
                selected = train_repair(protocol, settings, output=args.output, device=device)
            elif args.command == "train-forecast":
                selected = train_forecast(protocol, settings, stage1_manifest=args.stage1_manifest,
                                          output=args.output, device=device,
                                          selection_metric=args.selection_metric,
                                          forecast_loss=args.forecast_loss,
                                          repair_retention=args.repair_retention,
                                          forecast_mixer=args.forecast_mixer,
                                          forecast_input=args.forecast_input)
            else:
                selected = train_history_adapter(
                    protocol, settings, parent_selection=args.parent_selection,
                    output=args.output, device=device, adapter_variant=args.adapter_variant,
                )
        resources.selection_manifest = selected
        print(json.dumps({"manifest": str(selected)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
