"""Read-only reference timing over identity-checked P4 inputs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch import Tensor

from integrations.forecast_compare.protocol import P4Batch, P4Protocol, iter_p4_batches
from malr.config import config_sha256
from malr.evaluation.report import file_sha256
from malr.migration import MigrationCatalog

from .config import E23Settings
from .train import (_file_sha256, _key_sha256, _new_memory, _starts,
                    build_forecast_model, load_stage2_selection)


@dataclass(frozen=True)
class ProfileStep:
    output: Tensor
    segment_seconds: Mapping[str, float]


@dataclass(frozen=True)
class ProfileTarget:
    name: str
    run_batch: Callable[[P4Batch], Tensor | ProfileStep]
    memory_state: Callable[[], Any]
    checkpoint_sha256: str | None = None


def _same_state(before: Any, after: Any) -> bool:
    if isinstance(before, Tensor) and isinstance(after, Tensor):
        return torch.equal(before, after)
    if isinstance(before, Mapping) and isinstance(after, Mapping):
        return before.keys() == after.keys() and all(
            _same_state(before[key], after[key]) for key in before
        )
    if isinstance(before, (tuple, list)) and isinstance(after, type(before)):
        return len(before) == len(after) and all(
            _same_state(left, right) for left, right in zip(before, after)
        )
    return before == after


def _selection_identity(
    protocol: P4Protocol, selected_manifest: Path, catalog: MigrationCatalog,
    target: ProfileTarget,
) -> dict[str, str]:
    path = catalog.resolve(selected_manifest)
    owner = path.relative_to(catalog.paper_root).parts[0]
    if owner not in {"E22", "SAITS"}:
        raise ValueError("profile selection must be a reviewed E22 or SAITS manifest")
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError("profile selection must be a JSON object")
    if owner == "SAITS" and (record.get("stage") != "forecast" or record.get("best_epoch") is None):
        raise ValueError("profile selection must be a selected forecast Stage 2 run")
    if owner == "E22" and (
        record.get("training", {}).get("training_stage") != "forecast"
        or record.get("training", {}).get("best_epoch") is None
    ):
        raise ValueError("profile selection must be a selected forecast Stage 2 run")
    metadata = record.get("metadata", {})
    source = record.get("source_csv_sha256", metadata.get("source_csv_sha256"))
    config = record.get("model_config_sha256", metadata.get("config_hash"))
    if source != protocol.source_sha256 or config != config_sha256(protocol.config):
        raise ValueError("profile selection source/config identity mismatch")
    if record.get("validation_mask_sha256", metadata.get("validation_mask_sha256")) != protocol.mask_sha256["validation"]:
        raise ValueError("profile selection validation mask mismatch")
    if record.get("test_mask_sha256", protocol.mask_sha256["test"]) != protocol.mask_sha256["test"]:
        raise ValueError("profile selection test mask mismatch")
    checkpoint = catalog.resolve(record["checkpoint"], expected_sha256=record["checkpoint_sha256"])
    if target.name in {"E22", "SAITS"} and target.checkpoint_sha256 != record["checkpoint_sha256"]:
        raise ValueError("profile target checkpoint mismatch")
    return {
        "selection_sha256": file_sha256(path),
        "checkpoint_sha256": file_sha256(checkpoint),
        "source_sha256": protocol.source_sha256,
        "validation_mask_sha256": protocol.mask_sha256["validation"],
        "test_mask_sha256": protocol.mask_sha256["test"],
    }


def profile_selected_reference(
    *, protocol: P4Protocol, selected_manifest: Path, catalog: MigrationCatalog,
    target: ProfileTarget, device: torch.device, warmup_batches: int = 5,
    batches: int = 10, split: str = "validation",
) -> dict[str, float | int | str | dict[str, float]]:
    """Measure a supplied reference runner without training or changing memory."""
    if warmup_batches < 0 or batches <= 0:
        raise ValueError("profile batch counts must be nonnegative and timed batches positive")
    if split not in {"validation", "test"}:
        raise ValueError("profile split must be validation or test")
    identity = _selection_identity(protocol, selected_manifest, catalog, target)
    before = target.memory_state()
    iterator = iter(iter_p4_batches(protocol, split, batch_size=128))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def run_once() -> tuple[float, float, int, dict[str, float]]:
        synchronize()
        started = perf_counter()
        batch = next(iterator).to(device)
        synchronize()
        model_started = perf_counter()
        with torch.inference_mode():
            result = target.run_batch(batch)
        synchronize()
        finished = perf_counter()
        if isinstance(result, ProfileStep):
            segments = dict(result.segment_seconds)
            output = result.output
        else:
            segments, output = {}, result
        if not isinstance(output, Tensor) or not torch.isfinite(output).all():
            raise ValueError("profile target returned a nonfinite or invalid output")
        if any(value < 0 or not isinstance(value, (int, float)) for value in segments.values()):
            raise ValueError("profile target returned invalid segment times")
        return finished - started, finished - model_started, batch.normalized_observed.shape[0], segments

    first = run_once()
    cold_start = first[0]
    if warmup_batches == 0:
        measured = [first]
        remaining = batches - 1
    else:
        for _ in range(warmup_batches - 1):
            run_once()
        measured = []
        remaining = batches
    for _ in range(remaining):
        measured.append(run_once())
    if not _same_state(before, target.memory_state()):
        raise ValueError("reference profiling changed online memory")
    pipeline = sum(row[0] for row in measured)
    model = sum(row[1] for row in measured)
    windows = sum(row[2] for row in measured)
    segments: dict[str, float] = {}
    for row in measured:
        for name, value in row[3].items():
            segments[name] = segments.get(name, 0.0) + float(value)
    return {
        **identity,
        "target": target.name,
        "split": split,
        "warmup_batches": warmup_batches,
        "timed_batches": batches,
        "cold_start_seconds": cold_start,
        "pipeline_seconds": pipeline,
        "model_seconds": model,
        "throughput_windows_per_second": windows / pipeline,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "segments_seconds": segments,
    }


def profile_e23_selected(
    *, protocol: P4Protocol, settings: E23Settings, selection: Path,
    device: torch.device, warmup_batches: int = 5, batches: int = 10,
    split: str = "validation",
) -> dict[str, float | int | str | bool | dict[str, float]]:
    """Profile selected E23 on a fixed P4 input window order and boundary."""
    if warmup_batches < 0 or batches < 1:
        raise ValueError("profile batch counts are invalid")
    if split not in {"validation", "test"}:
        raise ValueError("profile split must be validation or test")
    selection = Path(selection).resolve(strict=True)
    load_started = perf_counter()
    payload = load_stage2_selection(selection, protocol, settings, device=device)
    record = json.loads(selection.read_text(encoding="utf-8"))
    forecast_mixer = record.get("forecast_mixer", "none")
    history_adapter = record.get("history_adapter", "none")
    model = build_forecast_model(protocol, settings, device=device,
                                 forecast_mixer=forecast_mixer,
                                 history_adapter=history_adapter)
    model.load_state_dict(payload["model_state"], strict=True)
    model.repair.historical.retrieval.load_state_dict(payload["retrieval_state"])
    model_load_seconds = perf_counter() - load_started
    memory = _new_memory(protocol)
    memory.load_state_dict(payload["memory_state"])
    index_started = perf_counter()
    model.repair.historical.refresh_snapshot(memory.snapshot())
    index_build_seconds = perf_counter() - index_started
    model.eval()
    memory_before = memory.snapshot_hash()
    key_before = _key_sha256(model.repair.historical.retrieval.state_dict())
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    modules = {
        "historical": model.repair.historical,
        "variable": model.repair.variable,
        "temporal": model.repair.temporal,
        "router": model.repair.router,
        "decoder": model.repair.decoder,
        "predictor": model.predictor,
    }
    if model.repair.history_adapter is not None:
        modules["history_adapter"] = model.repair.history_adapter
    active: dict[str, float] = {}
    segments: dict[str, float] = {}
    iterator = iter(iter_p4_batches(protocol, split, batch_size=settings.batch_size))

    def run_once() -> tuple[float, float, int, dict[str, float]]:
        segments.clear()
        synchronize()
        started = perf_counter()
        batch = next(iterator).to(device)
        starts = _starts(batch, stride=model.repair.stride,
                         patches=model.repair.patch_count, device=device)
        synchronize()
        model_started = perf_counter()
        with torch.inference_mode():
            prediction, _ = model(batch.normalized_observed, batch.observed_mask,
                                  batch.query_mean, batch.query_std, starts)
        synchronize()
        finished = perf_counter()
        if not torch.isfinite(prediction).all():
            raise ValueError("E23 profile produced nonfinite prediction")
        return finished - started, finished - model_started, prediction.shape[0], dict(segments)

    def measure_pass() -> tuple[float, list[tuple[float, float, int, dict[str, float]]]]:
        first = run_once()
        if warmup_batches == 0:
            measured = [first]
            remaining = batches - 1
        else:
            for _ in range(warmup_batches - 1):
                run_once()
            measured = []
            remaining = batches
        for _ in range(remaining):
            measured.append(run_once())
        return first[0], measured

    # The speed verdict uses the same uninstrumented timing boundary as E22/SAITS.
    cold, measured = measure_pass()
    handles = []
    for name, module in modules.items():
        def before(_module, _args, *, label=name):
            synchronize()
            active[label] = perf_counter()

        def after(_module, _args, _output, *, label=name):
            synchronize()
            segments[label] = segments.get(label, 0.0) + perf_counter() - active.pop(label)

        handles.append(module.register_forward_pre_hook(before))
        handles.append(module.register_forward_hook(after))
    iterator = iter(iter_p4_batches(protocol, split, batch_size=settings.batch_size))
    try:
        _, segmented = measure_pass()
    finally:
        for handle in handles:
            handle.remove()
    if memory.snapshot_hash() != memory_before or _key_sha256(model.repair.historical.retrieval.state_dict()) != key_before:
        raise RuntimeError("E23 profile changed H bank/key state")
    elapsed_pipeline = sum(row[0] for row in measured)
    elapsed_model = sum(row[1] for row in measured)
    windows = sum(row[2] for row in measured)
    totals = {name: sum(row[3].get(name, 0.0) for row in segmented) for name in modules}
    segmented_model = sum(row[1] for row in segmented)
    return {
        "target": "E23", "split": split, "selection_sha256": _file_sha256(selection),
        "checkpoint_sha256": record["checkpoint_sha256"],
        "forecast_mixer": forecast_mixer,
        "history_adapter": history_adapter,
        "e23_config_sha256": settings.config_sha256,
        "source_sha256": protocol.source_sha256,
        "validation_mask_sha256": protocol.mask_sha256["validation"],
        "test_mask_sha256": protocol.mask_sha256["test"],
        "h_state_unchanged": True,
        "warmup_batches": warmup_batches, "timed_batches": batches,
        "batch_size": settings.batch_size,
        "windows": windows,
        "model_load_seconds": model_load_seconds,
        "index_build_seconds": index_build_seconds,
        "cold_start_seconds": cold,
        "pipeline_seconds": elapsed_pipeline,
        "model_seconds": elapsed_model,
        "throughput_windows_per_second": windows / elapsed_pipeline,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "segments_seconds": totals,
        "segment_pass_model_seconds": segmented_model,
        "unattributed_model_seconds": max(segmented_model - sum(totals.values()), 0.0),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "none",
    }


__all__ = ["ProfileStep", "ProfileTarget", "profile_selected_reference",
           "profile_e23_selected"]
