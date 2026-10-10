"""Locked-P4 E23 training and paired online H state."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import tempfile
from time import monotonic
from collections.abc import Mapping
from typing import Literal

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from integrations.forecast_compare.protocol import P4Batch, P4Protocol, iter_p4_batches
from malr.config import config_sha256
from malr.data.patching import patchify_preserving_values, patchify_with_state
from malr.models.e23.model import E23ForecastModel, E23RepairOutput, MultiExpertRepairEncoder
from malr.models.explicit_repair import build_explicit_repair
from malr.models.patchtst import PatchTSTCore
from malr.models.repaired_forecast import MaskAwareForecastModel
from malr.prototypes.online_bank import OnlinePrototypeMemory, OnlinePrototypeMember
from malr.prototypes.retrieval import masked_contrastive_loss, masked_pair_loss

from .config import E23Settings


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _historical_archive_sha256(protocol: P4Protocol) -> str:
    train_end = protocol.config.data.datasets["ETTh1"].train_end
    return sha256(np.ascontiguousarray(protocol.standardized[:train_end]).tobytes()).hexdigest()


def _key_sha256(state: dict[str, object]) -> str:
    digest = sha256()
    for field in ("format_version", "patch_len", "retrieval_dim", "decay", "frozen_epoch"):
        digest.update(f"{field}={state[field]}\n".encode())
    for role in ("ema_state", "frozen_state"):
        for name, array in state[role]:
            value = np.ascontiguousarray(array)
            digest.update(f"{role}:{name}:{value.dtype}:{value.shape}\n".encode())
            digest.update(value.tobytes())
    return digest.hexdigest()


def _memory_state_sha256(state: Mapping[str, object]) -> str:
    """Hash serialized state, avoiding legacy re-normalization on object reload."""
    digest = sha256()

    def update(value: object) -> None:
        if isinstance(value, Mapping):
            digest.update(b"{")
            for key in sorted(value):
                update(key)
                update(value[key])
            digest.update(b"}")
        elif isinstance(value, (tuple, list)):
            digest.update(b"[")
            for item in value:
                update(item)
            digest.update(b"]")
        elif isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            digest.update(f"array:{array.dtype}:{array.shape}:".encode())
            digest.update(array.tobytes())
        else:
            digest.update(f"{type(value).__name__}:{value!r};".encode())

    update(state)
    return digest.hexdigest()


def _atomic_torch_save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(value: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _revision() -> str:
    try:
        return subprocess.check_output(
            ["/home/xyw/miniforge3/envs/malr/bin/git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _identity(protocol: P4Protocol, settings: E23Settings) -> dict[str, object]:
    return {
        "source_csv_sha256": protocol.source_sha256,
        "validation_mask_sha256": protocol.mask_sha256["validation"],
        "test_mask_sha256": protocol.mask_sha256["test"],
        "protocol_config_sha256": config_sha256(protocol.config),
        "protocol_config_path": str(protocol.config_path.resolve()),
        "e23_config_sha256": settings.config_sha256,
        "batch_size": settings.batch_size,
        "seed": settings.seed,
        **_trial_identity(settings),
    }


def _trial_identity(settings: E23Settings) -> dict[str, object]:
    # Old v3 manifests/checkpoints predate these fields and remain loadable.
    if (settings.d_model != 128 and settings.historical_top_p is None
            and settings.variable_selection == "global_topk"
            and settings.historical_retrieval == "one_group"
            and settings.stage1_objective == "window_smooth_l1"
            and settings.recovery_schedule == "parallel_v1"):
        return {}
    identity: dict[str, object] = {
        "repair_width": settings.d_model,
        "candidate_selection": "top_k" if settings.historical_top_p is None else "top_p",
    }
    if settings.historical_top_p is not None:
        identity.update(
            historical_top_p=settings.historical_top_p,
            variable_top_p=settings.variable_top_p,
            historical_candidate_cap=settings.historical_candidate_cap,
        )
    if settings.variable_selection != "global_topk":
        identity["variable_selection"] = settings.variable_selection
    if settings.variable_aggregation != "scalar_v1":
        identity["variable_aggregation"] = settings.variable_aggregation
    if settings.historical_retrieval != "one_group":
        identity["historical_retrieval"] = settings.historical_retrieval
    if settings.multihead_router != "none":
        identity["multihead_router"] = settings.multihead_router
    if settings.stage1_objective != "window_smooth_l1":
        identity["stage1_objective"] = settings.stage1_objective
        identity["stage1_selection_metric_space"] = "train_standardized"
    if settings.recovery_schedule != "parallel_v1":
        identity["recovery_schedule"] = settings.recovery_schedule
        identity["recovery_rounds"] = settings.recovery_rounds
    if settings.recovery_schedule == "stacked_parallel_tv_h_v1":
        identity.update(recovery_layers=settings.recovery_layers,expert_groups=settings.expert_groups,
                        group_width=settings.d_model,group_fusion="gated_concat_v1",
                        activation_checkpointing=settings.activation_checkpointing,
                        code_freeze_sha256=settings.code_freeze_sha256)
    return identity


def _verify_trial_payload(payload: dict, settings: E23Settings) -> None:
    for key, expected in _trial_identity(settings).items():
        if payload.get(key) != expected:
            raise ValueError(f"selected {key} checkpoint mismatch")


def _verify_identity(record: dict, protocol: P4Protocol, settings: E23Settings) -> None:
    for key, expected in _identity(protocol, settings).items():
        if record.get(key) != expected:
            raise ValueError(f"selected {key} mismatch")


@torch.no_grad()
def _train_observed_pairwise_affine(protocol: P4Protocol) -> tuple[Tensor, Tensor]:
    """Fit directed source-to-target maps without hidden or future values."""
    channels = protocol.config.data.datasets["ETTh1"].n_vars
    shape = (channels, channels)
    count = torch.zeros(shape, dtype=torch.float64)
    target_sum = torch.zeros_like(count)
    source_sum = torch.zeros_like(count)
    source_square = torch.zeros_like(count)
    cross = torch.zeros_like(count)
    for batch in iter_p4_batches(protocol, "train", batch_size=128, epoch=0):
        values = batch.normalized_target.reshape(-1, channels).double()
        observed = batch.observed_mask.reshape(-1, channels).double()
        visible_values = values * observed
        count += observed.T @ observed
        target_sum += visible_values.T @ observed
        source_sum += observed.T @ visible_values
        source_square += observed.T @ visible_values.square()
        cross += visible_values.T @ visible_values
    if (count[~torch.eye(channels, dtype=torch.bool)] < 2).any():
        raise ValueError("insufficient jointly observed training coordinates for V map")
    denominator = source_square - source_sum.square() / count.clamp_min(1)
    if (denominator[~torch.eye(channels, dtype=torch.bool)] <= 1e-8).any():
        raise ValueError("degenerate source variance in train-observed V map")
    slope = (cross - target_sum * source_sum / count.clamp_min(1)) / denominator.clamp_min(1e-8)
    intercept = (target_sum - slope * source_sum) / count.clamp_min(1)
    diagonal = torch.eye(channels, dtype=torch.bool)
    slope[diagonal] = 1
    intercept[diagonal] = 0
    if not torch.isfinite(slope).all() or not torch.isfinite(intercept).all():
        raise ValueError("train-observed V map is nonfinite")
    return slope.float(), intercept.float()


def build_repair_model(
    protocol: P4Protocol, settings: E23Settings, *, device: torch.device,
    history_adapter: Literal["none", "aligned_v1", "aligned_bounded_v2"] = "none",
) -> MultiExpertRepairEncoder:
    if history_adapter not in ("none", "aligned_v1", "aligned_bounded_v2"):
        raise ValueError("unsupported E23 history adapter")
    if settings.recovery_schedule != "parallel_v1" and history_adapter != "none":
        raise ValueError(f"{settings.recovery_schedule} recovery cannot use a history adapter")
    archive = None
    if history_adapter != "none":
        train_end = protocol.config.data.datasets["ETTh1"].train_end
        archive = torch.as_tensor(np.ascontiguousarray(protocol.standardized[:train_end]),
                                  dtype=torch.float32)
    model = MultiExpertRepairEncoder(
        variable_count=protocol.config.data.datasets["ETTh1"].n_vars,
        window_length=protocol.config.data.seq_len,
        patch_length=protocol.config.patch.length,
        stride=protocol.config.patch.stride,
        d_model=settings.d_model, retrieval_dim=settings.d_model,
        historical_topk=settings.historical_topk,
        historical_top_p=settings.historical_top_p,
        variable_top_p=settings.variable_top_p,
        variable_selection=settings.variable_selection,
        variable_aggregation=settings.variable_aggregation,
        historical_candidate_cap=settings.historical_candidate_cap,
        historical_retrieval=settings.historical_retrieval,
        historical_threshold=settings.historical_threshold,
        variable_topk=settings.variable_topk,
        variable_offsets=settings.variable_offsets,
        temporal_local_radius=settings.temporal_local_radius,
        temporal_boundaries=settings.temporal_boundaries,
        temporal_selection=settings.temporal_selection,
        post_fusion_refiner=settings.post_fusion_refiner,
        multihead_router=settings.multihead_router,
        historical_archive=archive,
        bounded_history=history_adapter == "aligned_bounded_v2",
        recovery_schedule=settings.recovery_schedule,
        recovery_rounds=settings.recovery_rounds,
        recovery_layers=settings.recovery_layers,expert_groups=settings.expert_groups,
        activation_checkpointing=settings.activation_checkpointing,
    )
    if settings.variable_selection == "window_affine_v1":
        slope, intercept = _train_observed_pairwise_affine(protocol)
        model.variable.set_affine_map(slope, intercept)
    return model.to(device)


def _new_memory(protocol: P4Protocol) -> OnlinePrototypeMemory:
    return OnlinePrototypeMemory(
        patch_len=protocol.config.patch.length,
        patch_stride=protocol.config.patch.stride,
        train_end=protocol.config.data.datasets["ETTh1"].train_end,
        variable_count=protocol.config.data.datasets["ETTh1"].n_vars,
    )


def _starts(batch: P4Batch, *, stride: int, patches: int, device: torch.device) -> Tensor:
    if len(batch.refs) != batch.normalized_observed.shape[0]:
        raise ValueError("P4 batch window references are required for causal H retrieval")
    first = torch.tensor([ref.input_start for ref in batch.refs], device=device, dtype=torch.long)
    return first[:, None] + torch.arange(patches, device=device)[None, :] * stride


def stage1_objective(
    candidate: Tensor, target: Tensor, observed_mask: Tensor,
    retrieval_loss: Tensor, *, retrieval_weight: float,
    valid_mask: Tensor | None = None,
    loss_mode: str = "window_smooth_l1", query_std: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    if candidate.shape != target.shape or observed_mask.shape != candidate.shape:
        raise ValueError("Stage 1 candidate/target/mask shape mismatch")
    valid = torch.ones_like(observed_mask, dtype=torch.bool) if valid_mask is None else valid_mask.to(torch.bool)
    if valid.shape != observed_mask.shape:
        raise ValueError("Stage 1 validity shape mismatch")
    hidden = ~observed_mask.to(torch.bool) & valid
    if loss_mode == "window_smooth_l1":
        repair = F.smooth_l1_loss(candidate[hidden], target[hidden]) if hidden.any() else candidate.sum() * 0
    elif loss_mode == "global_mse_v1":
        if (query_std is None or query_std.shape != (candidate.shape[0], candidate.shape[-1])
                or not torch.isfinite(query_std).all() or torch.any(query_std <= 0)):
            raise ValueError("global repair MSE requires positive finite query_std [B,C]")
        # Query means cancel in the error. Detached visible-only scales restore
        # train-standardized coordinates without changing the model's inputs.
        scales = query_std.detach()[:, None, :].expand_as(candidate)[hidden]
        error = (candidate[hidden] - target[hidden]) * scales
        repair = error.square().mean() if hidden.any() else candidate.sum() * 0
    else:
        raise ValueError("unsupported Stage 1 objective")
    weighted = retrieval_loss * retrieval_weight
    return repair + weighted, {"repair": repair, "retrieval": retrieval_loss,
                               "retrieval_weighted": weighted}


def _retrieval_loss(model: MultiExpertRepairEncoder, batch: P4Batch) -> Tensor:
    patch_len, stride = model.patch_length, model.stride
    query = patchify_with_state(batch.normalized_observed, batch.observed_mask,
                                patch_len, stride)
    clean = patchify_preserving_values(batch.normalized_target,
                                       torch.ones_like(batch.observed_mask),
                                       patch_len, stride)
    damaged = query.rho.lt(1.0)
    if not damaged.any():
        return model.historical.key_encoder.network[0].weight.sum() * 0
    values = query.values[damaged]
    mask = query.effective_mask[damaged]
    target = clean.values[damaged]
    target_validity = clean.validity[damaged]
    encoder = model.historical.key_encoder
    query_keys = encoder(values, mask)
    clean_keys = encoder(target, target_validity)
    count = query_keys.shape[0]
    return masked_pair_loss(query_keys, clean_keys) + (
        0.5 * masked_contrastive_loss(query_keys, clean_keys) /
        torch.log(torch.tensor(float(max(count, 2)), device=values.device))
    )


@torch.no_grad()
def _stage_observed(
    memory: OnlinePrototypeMemory, model: MultiExpertRepairEncoder,
    batch: P4Batch, starts: Tensor,
    known_identities: set[tuple[int, int]] | None = None,
) -> int:
    raw = batch.normalized_target * batch.query_std[:, None, :] + batch.query_mean[:, None, :]
    patches = patchify_with_state(raw, batch.observed_mask,
                                  model.patch_length, model.stride)
    eligible = patches.rho.eq(1.0) & patches.validity.all(-1)
    selected = eligible.nonzero(as_tuple=False)
    if selected.numel() == 0:
        return 0
    identities = selected.detach().cpu().numpy()
    starts_cpu = starts.detach().cpu().numpy()
    if known_identities is not None:
        pending = set(known_identities)
        keep = []
        for index, row in enumerate(identities):
            sample, variable, patch_index = map(int, row)
            identity = (variable, int(starts_cpu[sample, patch_index]))
            if identity not in pending:
                pending.add(identity)
                keep.append(index)
        if not keep:
            return 0
        selected = selected[torch.tensor(keep, device=selected.device)]
        identities = identities[keep]
    raw_selected = patches.values[selected[:, 0], selected[:, 1], selected[:, 2]]
    keys = model.historical.retrieval.encode_frozen(
        raw_selected, torch.ones_like(raw_selected, dtype=torch.bool),
    )
    raw_cpu = raw_selected.detach().cpu().numpy()
    keys_cpu = keys.detach().cpu().numpy()
    staged = 0
    for row, patch, key in zip(identities, raw_cpu, keys_cpu, strict=True):
        sample, variable, patch_index = map(int, row)
        start = int(starts_cpu[sample, patch_index])
        accepted = memory.stage_patch(variable, start, patch, key, observed=True)
        staged += int(accepted)
        if accepted and known_identities is not None:
            known_identities.add((variable, start))
    return staged


@torch.no_grad()
def _publish_epoch(
    memory: OnlinePrototypeMemory, model: MultiExpertRepairEncoder, epoch: int,
) -> None:
    retrieval = model.historical.retrieval
    retrieval.update_ema()
    encoder = retrieval.ema_encoder
    device = next(encoder.parameters()).device

    def encode(members: tuple[OnlinePrototypeMember, ...]) -> np.ndarray:
        raw = torch.as_tensor(
            np.stack([member.raw_patch for member in members]),
            dtype=torch.float32, device=device,
        )
        return encoder(raw, torch.ones_like(raw, dtype=torch.bool)).cpu().numpy()

    memory.rekey_batch(encode)
    memory.publish_epoch(epoch)
    retrieval.frozen_encoder.load_state_dict(retrieval.ema_encoder.state_dict())
    model.historical.refresh_snapshot(memory.snapshot())


@torch.no_grad()
def _validation_repair_metrics(
    model: MultiExpertRepairEncoder, protocol: P4Protocol, settings: E23Settings,
    *, device: torch.device,
) -> dict[str, float | int]:
    model.eval()
    window_squared = global_squared = global_absolute = 0.0
    small_window_squared = small_global_squared = 0.0
    count = small_count = 0
    total = len(protocol.refs["validation"]) // settings.batch_size
    for index, cpu_batch in enumerate(iter_p4_batches(protocol, "validation",
                                                        batch_size=settings.batch_size), 1):
        batch = cpu_batch.to(device)
        starts = _starts(batch, stride=model.stride, patches=model.patch_count, device=device)
        output = model(batch.normalized_observed, batch.observed_mask,
                       batch.query_mean, batch.query_std, starts)
        hidden = ~batch.observed_mask
        error = output.explicit.normalized_sequence - batch.normalized_target
        global_error = error * batch.query_std[:, None, :]
        small = hidden & (batch.query_std[:, None, :] < 0.01)
        window_squared += error.square()[hidden].sum().item()
        global_squared += global_error.square()[hidden].sum().item()
        global_absolute += global_error.abs()[hidden].sum().item()
        small_window_squared += error.square()[small].sum().item()
        small_global_squared += global_error.square()[small].sum().item()
        small_count += small.sum().item()
        count += hidden.sum().item()
        if index == 1 or index % 5 == 0 or index == total:
            print(f"stage=repair validation batch={index}/{total}", flush=True)
    if count == 0:
        raise ValueError("fixed validation contains no hidden coordinates")
    return {
        "window_mse": window_squared / count,
        "global_mse": global_squared / count,
        "global_mae": global_absolute / count,
        "hidden_coordinates": count,
        "small_std_hidden_fraction": small_count / count,
        "small_std_window_squared_share": small_window_squared / window_squared if window_squared else 0.0,
        "small_std_global_squared_share": small_global_squared / global_squared if global_squared else 0.0,
    }


def _validation_repair_mse(
    model: MultiExpertRepairEncoder, protocol: P4Protocol, settings: E23Settings,
    *, device: torch.device,
) -> float:
    """Legacy window-coordinate score; old trial selection stays reproducible."""
    return _validation_repair_metrics(model, protocol, settings, device=device)["window_mse"]


def _paired_payload(
    model: MultiExpertRepairEncoder, memory: OnlinePrototypeMemory,
    *, epoch: int, validation_mse: float, settings: E23Settings,
    validation_metrics: dict[str, float | int] | None = None,
) -> dict[str, object]:
    retrieval_state = model.historical.retrieval.state_dict()
    memory_state = memory.state_dict()
    return {
        "format_version": 1, "model_state": model.state_dict(),
        "memory_state": memory_state, "retrieval_state": retrieval_state,
        "memory_sha256": _memory_state_sha256(memory_state),
        "key_sha256": _key_sha256(retrieval_state),
        "epoch": epoch, "validation_mse": validation_mse,
        **({"validation_repair_metrics": validation_metrics} if validation_metrics is not None else {}),
        **_trial_identity(settings),
    }


def train_repair(
    protocol: P4Protocol, settings: E23Settings, *, output: Path,
    device: torch.device, epochs: int = 100,
) -> Path:
    if not 0 < epochs <= settings.stage1_epochs:
        raise ValueError("Stage 1 epochs exceed the locked first-run setting")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"E23 Stage 1 output already exists: {output}")
    output.mkdir(parents=True)
    torch.manual_seed(settings.seed)
    model = build_repair_model(protocol, settings, device=device)
    memory = _new_memory(protocol)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.stage1_lr,
                                 weight_decay=settings.weight_decay)
    checkpoint = output / "checkpoints/e23-stage1.pt"
    manifest_path = output / "runs/e23-stage1.json"
    best_mse = float("inf")
    best_epoch = 0
    start_time = monotonic()
    train_total = len(protocol.refs["train"]) // settings.batch_size
    staged_identities: set[tuple[int, int]] = set()
    history: list[dict[str, object]] = []
    global_objective = settings.stage1_objective == "global_mse_v1"
    for epoch in range(1, epochs + 1):
        model.historical.retrieval.begin_epoch(epoch)
        model.train()
        for index, cpu_batch in enumerate(iter_p4_batches(
            protocol, "train", batch_size=settings.batch_size, epoch=epoch - 1,
        ), 1):
            batch = cpu_batch.to(device)
            starts = _starts(batch, stride=model.stride, patches=model.patch_count, device=device)
            optimizer.zero_grad(set_to_none=True)
            output_batch = model(
                batch.normalized_observed, batch.observed_mask,
                batch.query_mean, batch.query_std, starts,
                epoch=epoch, explore_variable=True,
                specialist_exposure=settings.specialist_exposure,
                specialist_seed=settings.seed + epoch * 100_000 + index,
            )
            retrieval_loss = _retrieval_loss(model, batch)
            loss, parts = stage1_objective(
                output_batch.explicit.normalized_sequence,
                batch.normalized_target, batch.observed_mask,
                retrieval_loss, retrieval_weight=settings.retrieval_loss_weight,
                loss_mode=settings.stage1_objective, query_std=batch.query_std,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite E23 Stage 1 loss")
            loss.backward()
            optimizer.step()
            staged = _stage_observed(memory, model, batch, starts, staged_identities)
            if index == 1 or index % 5 == 0 or index == train_total:
                print(f"stage=repair epoch={epoch}/{epochs} batch={index}/{train_total} "
                      f"loss={loss.item():.6f} repair={parts['repair'].item():.6f} "
                      f"retrieval={parts['retrieval'].item():.6f} staged={staged}", flush=True)
        _publish_epoch(memory, model, epoch)
        validation_metrics = None
        if global_objective:
            validation_metrics = _validation_repair_metrics(model, protocol, settings, device=device)
            validation_mse = validation_metrics["global_mse"]
            history.append({"epoch": epoch, "metrics": validation_metrics})
            _atomic_json({
                "selection_metric_space": "train_standardized",
                "diagnostic_metric_space": "window_normalized",
                "small_std_threshold": 0.01, "epochs": history,
            }, output / "runs/e23-stage1-history.json")
        else:
            validation_mse = _validation_repair_mse(model, protocol, settings, device=device)
        if validation_mse < best_mse:
            best_mse = validation_mse
            best_epoch = epoch
            _atomic_torch_save(_paired_payload(
                model, memory, epoch=epoch, validation_mse=validation_mse,
                settings=settings,
                validation_metrics=validation_metrics,
            ), checkpoint)
        print(f"stage=repair epoch={epoch}/{epochs} val_repair_mse={validation_mse:.8f} "
              f"best_epoch={best_epoch} bank_members={memory.snapshot().member_count}"
              + (f" metric_space=train_standardized val_window_mse={validation_metrics['window_mse']:.8f}"
                 if validation_metrics is not None else ""), flush=True)
    if best_epoch == 0:
        raise RuntimeError("no E23 Stage 1 checkpoint selected")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    record = {
        "stage": "repair", "best_epoch": best_epoch,
        "epochs_completed": epochs, "best_val_repair_mse": best_mse,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _file_sha256(checkpoint),
        "memory_sha256": payload["memory_sha256"],
        "key_sha256": payload["key_sha256"],
        "code_revision": _revision(),
        "optimizer_state_restored": False,
        "learning_rate": settings.stage1_lr,
        "retrieval_loss_weight": settings.retrieval_loss_weight,
        "specialist_exposure": settings.specialist_exposure,
        "training_seconds": monotonic() - start_time,
        "selection": "lowest fixed-validation hidden-coordinate repair MSE",
        **({"validation_repair_metrics": payload["validation_repair_metrics"]}
           if global_objective else {}),
        **_identity(protocol, settings),
    }
    _atomic_json(record, manifest_path)
    return manifest_path


def load_stage1_selection(
    manifest_path: Path, protocol: P4Protocol, settings: E23Settings,
    *, device: torch.device,
) -> dict[str, object]:
    manifest_path = Path(manifest_path).resolve(strict=True)
    record = json.loads(manifest_path.read_text(encoding="utf-8"))
    if record.get("stage") != "repair" or int(record.get("best_epoch", 0)) < 1:
        raise ValueError("expected validation-selected E23 Stage 1 manifest")
    _verify_identity(record, protocol, settings)
    checkpoint = Path(record["checkpoint"])
    if not checkpoint.is_file() or _file_sha256(checkpoint) != record.get("checkpoint_sha256"):
        raise ValueError("Stage 1 checkpoint SHA-256 mismatch")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    _verify_trial_payload(payload, settings)
    if payload.get("epoch") != record["best_epoch"]:
        raise ValueError("Stage 1 model/online epoch mismatch")
    if _memory_state_sha256(payload["memory_state"]) != record.get("memory_sha256"):
        raise ValueError("Stage 1 serialized memory SHA-256 mismatch")
    memory = _new_memory(protocol)
    memory.load_state_dict(payload["memory_state"])
    if memory.snapshot().epoch != payload["epoch"]:
        raise ValueError("Stage 1 memory epoch mismatch")
    model = build_repair_model(protocol, settings, device=device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.historical.retrieval.load_state_dict(payload["retrieval_state"])
    if (payload["retrieval_state"]["frozen_epoch"] != payload["epoch"]
            or _key_sha256(payload["retrieval_state"]) != record.get("key_sha256")):
        raise ValueError("Stage 1 paired key SHA-256 or epoch mismatch")
    if payload.get("memory_sha256") != record.get("memory_sha256") or payload.get("key_sha256") != record.get("key_sha256"):
        raise ValueError("Stage 1 payload state hashes mismatch")
    _verify_bank_key_pair(memory, model)
    model.historical.refresh_snapshot(memory.snapshot())
    return payload


@torch.no_grad()
def _verify_bank_key_pair(
    memory: OnlinePrototypeMemory, model: MultiExpertRepairEncoder,
) -> None:
    members = [member for groups in memory.snapshot().groups for group in groups
               for member in group.members]
    if not members:
        return
    device = next(model.parameters()).device
    encoder = model.historical.retrieval.frozen_encoder
    for start in range(0, len(members), 4096):
        selected = members[start:start + 4096]
        raw = torch.as_tensor(np.stack([item.raw_patch for item in selected]),
                              dtype=torch.float32, device=device)
        stored = torch.as_tensor(np.stack([item.clean_key for item in selected]),
                                 dtype=torch.float32, device=device)
        rebuilt = encoder(raw, torch.ones_like(raw, dtype=torch.bool))
        if not torch.allclose(rebuilt, stored, atol=2e-6, rtol=2e-6):
            raise ValueError("Stage 1 frozen key encoder does not match published bank")


def build_forecast_model(
    protocol: P4Protocol, settings: E23Settings, *, device: torch.device,
    forecast_mixer: Literal["none", "patch_attention_v1"] = "none",
    history_adapter: Literal["none", "aligned_v1", "aligned_bounded_v2"] = "none",
    forecast_input: Literal["explicit_v1", "implicit_latent_v1"] = "explicit_v1",
) -> E23ForecastModel:
    if settings.recovery_schedule == "stacked_parallel_tv_h_v1" and (forecast_mixer != "none" or forecast_input != "explicit_v1" or history_adapter != "none"):
        raise ValueError("stacked recovery requires original explicit predictor and no adapter")
    if forecast_mixer not in ("none", "patch_attention_v1"):
        raise ValueError("unsupported E23 forecast mixer")
    if forecast_input not in ("explicit_v1", "implicit_latent_v1"):
        raise ValueError("unsupported E23 forecast input")
    if forecast_input == "implicit_latent_v1" and forecast_mixer != "none":
        raise ValueError("implicit E23 input requires the original predictor")
    profile = protocol.config.model.for_dataset("ETTh1")
    core = PatchTSTCore(
        c_in=protocol.config.data.datasets["ETTh1"].n_vars,
        seq_len=protocol.config.data.seq_len,
        pred_len=protocol.config.data.pred_len,
        patch_len=protocol.config.patch.length,
        stride=protocol.config.patch.stride,
        d_model=profile.d_model, n_heads=profile.n_heads,
        d_ff=profile.d_ff, n_layers=profile.encoder_layers,
        dropout=profile.dropout, fc_dropout=profile.fc_dropout,
        head_dropout=profile.head_dropout, attn_dropout=profile.attn_dropout,
    )
    repair = build_repair_model(protocol, settings, device=device,
                                history_adapter=history_adapter)
    predictor = MaskAwareForecastModel(
        core, variable_mixer=forecast_mixer == "patch_attention_v1",
    )
    return E23ForecastModel(repair, predictor, forecast_input=forecast_input).to(device)


def stage2_objective(
    forecast: Tensor, target: Tensor,
    *, loss_mode: Literal["mse", "mse_plus_0p2_mae",
                         "mse_plus_0p2_mae_late_horizon_v1",
                         "mse_plus_0p2_mae_utility_router_v1"] = "mse",
) -> Tensor:
    if forecast.shape != target.shape:
        raise ValueError("Stage 2 forecast/target shape mismatch")
    if loss_mode == "mse":
        return F.mse_loss(forecast, target)
    if loss_mode in ("mse_plus_0p2_mae", "mse_plus_0p2_mae_utility_router_v1"):
        return F.mse_loss(forecast, target) + 0.2 * F.l1_loss(forecast, target)
    if loss_mode == "mse_plus_0p2_mae_late_horizon_v1":
        if forecast.ndim != 3 or forecast.shape[-1] % 4:
            raise ValueError("late-horizon loss requires [B,C,T] with four equal horizon bands")
        weights = forecast.new_tensor((0.85, 0.95, 1.05, 1.15))
        weights = weights.repeat_interleave(forecast.shape[-1] // 4)
        error = forecast - target
        return ((error.square() + 0.2 * error.abs()) * weights).mean()
    raise ValueError("unsupported Stage 2 forecast loss")


def _forecast_utility_router_loss(
    model: E23ForecastModel, batch: P4Batch, output: E23RepairOutput,
    *, subset: int = 32, margin: float = 0.05,
) -> tuple[Tensor, float]:
    """Supervise existing route weights from train-only single-expert forecasts."""
    size = min(subset, batch.normalized_observed.shape[0])
    repair = model.repair
    patches = patchify_with_state(
        batch.normalized_observed[:size], batch.observed_mask[:size],
        repair.patch_length, repair.stride,
    )
    values = patches.values.masked_fill(~patches.validity, 0)
    damaged = patches.rho.lt(1.0)
    option_available = output.router.available[:size].any(dim=2)
    losses = []
    was_training = model.predictor.training
    model.predictor.eval()
    try:
        with torch.no_grad():
            for index, name in enumerate(("none", "historical", "variable", "temporal")):
                latents = output.base_latents[:size]
                if name != "none":
                    latents = latents + output.router.deltas[name][:size] * (
                        output.router.available[:size, ..., index, None]
                    )
                explicit = build_explicit_repair(
                    repair.decoder(latents), values, patches.effective_mask,
                    patches.validity, batch.query_mean[:size], batch.query_std[:size],
                    window_len=repair.window_length,
                    patch_len=repair.patch_length, stride=repair.stride,
                )
                forecast = model.predictor(
                    explicit.repaired_patches.masked_fill(~explicit.validity_patches, 0),
                    explicit.observation_patches,
                    batch.query_mean[:size], batch.query_std[:size],
                    patch_validity=explicit.validity_patches,
                )
                error = forecast - batch.forecast_target[:size]
                losses.append((error.square() + 0.2 * error.abs()).mean(dim=-1))
    finally:
        model.predictor.train(was_training)
    scored = torch.stack(losses, dim=-1).masked_fill(~option_available, torch.inf)
    ranking = scored.sort(dim=-1).values
    confident = (
        damaged.any(dim=-1)
        & option_available.sum(dim=-1).ge(2)
        & (ranking[..., 1] - ranking[..., 0]).ge(margin)
    )
    weights = output.router.weights[:size]
    average_weights = (weights * damaged[..., None]).sum(dim=2) / damaged.sum(dim=2).clamp_min(1)[..., None]
    if not confident.any():
        return average_weights.sum() * 0, 0.0
    winner = scored.argmin(dim=-1)
    chosen = average_weights.gather(-1, winner[..., None]).squeeze(-1)
    auxiliary = -chosen.clamp_min(1e-8).log()[confident].mean()
    return auxiliary, confident.float().mean().item()


@torch.no_grad()
def _validation_forecast_metrics(
    model: E23ForecastModel, protocol: P4Protocol, settings: E23Settings,
    *, device: torch.device,
) -> dict[str, float]:
    model.eval()
    squared = absolute = 0.0
    count = 0
    total = len(protocol.refs["validation"]) // settings.batch_size
    for index, cpu_batch in enumerate(iter_p4_batches(protocol, "validation",
                                                        batch_size=settings.batch_size), 1):
        batch = cpu_batch.to(device)
        starts = _starts(batch, stride=model.repair.stride,
                         patches=model.repair.patch_count, device=device)
        prediction, _ = model(batch.normalized_observed, batch.observed_mask,
                              batch.query_mean, batch.query_std, starts)
        error = prediction - batch.forecast_target
        squared += error.square().sum().item()
        absolute += error.abs().sum().item()
        count += error.numel()
        if index == 1 or index % 5 == 0 or index == total:
            print(f"stage=forecast validation batch={index}/{total}", flush=True)
    if count == 0:
        raise ValueError("fixed validation has no forecast targets")
    return {"mse": squared / count, "mae": absolute / count}


def train_forecast(
    protocol: P4Protocol, settings: E23Settings, *, stage1_manifest: Path,
    output: Path, device: torch.device, epochs: int = 100,
    selection_metric: Literal["mse", "mae"] = "mse",
    forecast_mixer: Literal["none", "patch_attention_v1"] = "none",
    forecast_input: Literal["explicit_v1", "implicit_latent_v1"] = "explicit_v1",
    forecast_loss: Literal["mse", "mse_plus_0p2_mae",
                           "mse_plus_0p2_mae_late_horizon_v1",
                           "mse_plus_0p2_mae_utility_router_v1"] = "mse",
) -> Path:
    if settings.recovery_schedule == "stacked_parallel_tv_h_v1" and (selection_metric != "mae" or forecast_loss != "mse_plus_0p2_mae" or forecast_mixer != "none" or forecast_input != "explicit_v1"):
        raise ValueError("stacked Stage 2 requires MAE selection and the fixed mixed loss/explicit predictor")
    if not 0 < epochs <= settings.stage2_epochs:
        raise ValueError("Stage 2 epochs exceed the locked first-run setting")
    if selection_metric not in ("mse", "mae"):
        raise ValueError("unsupported Stage 2 selection metric")
    if forecast_mixer not in ("none", "patch_attention_v1"):
        raise ValueError("unsupported Stage 2 forecast mixer")
    if forecast_input not in ("explicit_v1", "implicit_latent_v1"):
        raise ValueError("unsupported Stage 2 forecast input")
    if forecast_input == "implicit_latent_v1" and forecast_mixer != "none":
        raise ValueError("implicit Stage 2 input requires the original predictor")
    if forecast_loss not in ("mse", "mse_plus_0p2_mae",
                             "mse_plus_0p2_mae_late_horizon_v1",
                             "mse_plus_0p2_mae_utility_router_v1"):
        raise ValueError("unsupported Stage 2 forecast loss")
    loss_identity = {
        "mse": "forecast_mse_only",
        "mse_plus_0p2_mae": "forecast_mse_plus_0p2_mae",
        "mse_plus_0p2_mae_late_horizon_v1": "forecast_mse_plus_0p2_mae_late_horizon_v1",
        "mse_plus_0p2_mae_utility_router_v1": "forecast_mse_plus_0p2_mae_utility_router_v1",
    }[forecast_loss]
    utility_identity = ({"utility_router_subset": 32, "utility_router_margin": 0.05,
                         "utility_router_weight": 0.05}
                        if forecast_loss == "mse_plus_0p2_mae_utility_router_v1" else {})
    stage1_manifest = Path(stage1_manifest).resolve(strict=True)
    stage1_payload = load_stage1_selection(stage1_manifest, protocol, settings, device=device)
    stage1_record = json.loads(stage1_manifest.read_text(encoding="utf-8"))
    stage1_manifest_hash = _file_sha256(stage1_manifest)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"E23 Stage 2 output already exists: {output}")
    output.mkdir(parents=True)

    torch.manual_seed(settings.seed)
    model = build_forecast_model(protocol, settings, device=device,
                                 forecast_mixer=forecast_mixer,
                                 forecast_input=forecast_input)
    model.repair.load_state_dict(stage1_payload["model_state"], strict=True)
    historical = model.repair.historical
    historical.retrieval.load_state_dict(stage1_payload["retrieval_state"])
    for parameter in historical.key_encoder.parameters():
        parameter.requires_grad_(False)
    memory = _new_memory(protocol)
    memory.load_state_dict(stage1_payload["memory_state"])
    historical.refresh_snapshot(memory.snapshot())
    read_only_snapshot_hash = memory.snapshot_hash()
    read_only_key_hash = _key_sha256(historical.retrieval.state_dict())
    if read_only_key_hash != stage1_record["key_sha256"]:
        raise ValueError("Stage 2 frozen key SHA-256 mismatch")
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=settings.stage2_lr, weight_decay=settings.weight_decay,
    )
    checkpoint = output / "checkpoints/e23-stage2.pt"
    manifest_path = output / "runs/e23-stage2.json"
    best_mse = float("inf")
    best_mae = float("inf")
    best_selection = float("inf")
    best_epoch = 0
    start_time = monotonic()
    train_total = len(protocol.refs["train"]) // settings.batch_size
    for epoch in range(1, epochs + 1):
        model.train()
        for index, cpu_batch in enumerate(iter_p4_batches(
            protocol, "train", batch_size=settings.batch_size, epoch=epoch - 1,
        ), 1):
            batch = cpu_batch.to(device)
            starts = _starts(batch, stride=model.repair.stride,
                             patches=model.repair.patch_count, device=device)
            optimizer.zero_grad(set_to_none=True)
            prediction, repair_output = model(batch.normalized_observed, batch.observed_mask,
                                              batch.query_mean, batch.query_std, starts)
            loss = stage2_objective(prediction, batch.forecast_target,
                                    loss_mode=forecast_loss)
            if utility_identity:
                utility, coverage = _forecast_utility_router_loss(
                    model, batch, repair_output, subset=32, margin=0.05,
                )
                loss = loss + 0.05 * utility
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite E23 Stage 2 forecast loss")
            loss.backward()
            optimizer.step()
            if index == 1 or index % 5 == 0 or index == train_total:
                print(f"stage=forecast epoch={epoch}/{epochs} batch={index}/{train_total} "
                      f"mse={loss.item():.6f}"
                      + (f" utility={utility.item():.6f} coverage={coverage:.3f}"
                         if utility_identity else ""), flush=True)
        validation = _validation_forecast_metrics(model, protocol, settings, device=device)
        if validation[selection_metric] < best_selection:
            best_selection = validation[selection_metric]
            best_mse, best_mae, best_epoch = validation["mse"], validation["mae"], epoch
            _atomic_torch_save({
                "format_version": 1, "epoch": epoch,
                "model_state": model.state_dict(),
                "memory_state": stage1_payload["memory_state"],
                "retrieval_state": stage1_payload["retrieval_state"],
                "memory_sha256": stage1_record["memory_sha256"],
                "key_sha256": stage1_record["key_sha256"],
                "stage1_manifest_sha256": stage1_manifest_hash,
                "stage1_checkpoint_sha256": stage1_record["checkpoint_sha256"],
                "validation_mse": best_mse, "validation_mae": best_mae,
                "selection_metric": selection_metric,
                "forecast_mixer": forecast_mixer,
                "forecast_input": forecast_input,
                **({"loss": loss_identity} if forecast_loss != "mse" else {}),
                **utility_identity,
                **_trial_identity(settings),
            }, checkpoint)
        if memory.snapshot_hash() != read_only_snapshot_hash or _key_sha256(historical.retrieval.state_dict()) != read_only_key_hash:
            raise RuntimeError("Stage 2 changed the read-only H bank/key state")
        print(f"stage=forecast epoch={epoch}/{epochs} val_mse={validation['mse']:.8f} "
              f"val_mae={validation['mae']:.8f} best_epoch={best_epoch}", flush=True)
    if best_epoch == 0:
        raise RuntimeError("no E23 Stage 2 checkpoint selected")
    record = {
        "stage": "forecast", "best_epoch": best_epoch,
        "epochs_completed": epochs, "best_val_mse": best_mse,
        "best_val_mae": best_mae,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _file_sha256(checkpoint),
        "stage1_manifest": str(stage1_manifest),
        "stage1_manifest_sha256": stage1_manifest_hash,
        "stage1_checkpoint_sha256": stage1_record["checkpoint_sha256"],
        "memory_sha256": stage1_record["memory_sha256"],
        "key_sha256": stage1_record["key_sha256"],
        "loss": loss_identity, "optimizer_state_restored": False,
        **utility_identity,
        "rng_reset_seed": settings.seed,
        "learning_rate": settings.stage2_lr,
        "code_revision": _revision(),
        "training_seconds": monotonic() - start_time,
        "selection_metric": selection_metric,
        "forecast_mixer": forecast_mixer,
        "forecast_input": forecast_input,
        "selection": f"lowest fixed-validation forecast {selection_metric.upper()}",
        **_identity(protocol, settings),
    }
    _atomic_json(record, manifest_path)
    return manifest_path


def train_history_adapter(
    protocol: P4Protocol, settings: E23Settings, *, parent_selection: Path,
    output: Path, device: torch.device, epochs: int = 100,
    adapter_variant: Literal["aligned_v1", "aligned_bounded_v2"] = "aligned_v1",
) -> Path:
    """Fit only the zero-gated aligned-history adapter from a selected flat Stage 2."""
    if not 0 < epochs <= settings.stage2_epochs:
        raise ValueError("adapter epochs exceed the locked Stage 2 setting")
    if adapter_variant not in ("aligned_v1", "aligned_bounded_v2"):
        raise ValueError("unsupported E23 history adapter")
    parent_selection = Path(parent_selection).resolve(strict=True)
    parent_payload = load_stage2_selection(parent_selection, protocol, settings, device=device)
    parent_record = json.loads(parent_selection.read_text(encoding="utf-8"))
    if (parent_record.get("selection_metric") != "mae"
            or parent_record.get("forecast_mixer", "none") != "none"
            or parent_record.get("history_adapter", "none") != "none"):
        raise ValueError("adapter requires a flat MAE-selected parent without a forecast mixer")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"E23 adapter output already exists: {output}")
    output.mkdir(parents=True)
    torch.manual_seed(settings.seed)
    model = build_forecast_model(protocol, settings, device=device,
                                 history_adapter=adapter_variant)
    missing, unexpected = model.load_state_dict(parent_payload["model_state"], strict=False)
    expected = {name for name in model.state_dict() if name.startswith("repair.history_adapter.")}
    if set(missing) != expected or unexpected:
        raise ValueError("adapter parent model state differs beyond the new adapter")
    historical = model.repair.historical
    historical.retrieval.load_state_dict(parent_payload["retrieval_state"])
    memory = _new_memory(protocol)
    memory.load_state_dict(parent_payload["memory_state"])
    historical.refresh_snapshot(memory.snapshot())
    memory_before = memory.snapshot_hash()
    key_before = _key_sha256(historical.retrieval.state_dict())
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    adapter = model.repair.history_adapter
    for parameter in adapter.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=settings.stage1_lr,
                                 weight_decay=settings.weight_decay)
    model.eval()
    baseline = _validation_forecast_metrics(model, protocol, settings, device=device)
    if (abs(baseline["mse"] - parent_record["best_val_mse"]) > 1e-9
            or abs(baseline["mae"] - parent_record["best_val_mae"]) > 1e-9):
        raise ValueError("zero-gated adapter does not reproduce selected parent validation")
    parent_manifest_hash = _file_sha256(parent_selection)
    archive_hash = _historical_archive_sha256(protocol)
    checkpoint = output / "checkpoints/e23-stage2-history-adapter.pt"
    manifest_path = output / "runs/e23-stage2-history-adapter.json"
    fixed = {
        "format_version": 1,
        "memory_state": parent_payload["memory_state"],
        "retrieval_state": parent_payload["retrieval_state"],
        "memory_sha256": parent_record["memory_sha256"],
        "key_sha256": parent_record["key_sha256"],
        "stage1_manifest_sha256": parent_record["stage1_manifest_sha256"],
        "stage1_checkpoint_sha256": parent_record["stage1_checkpoint_sha256"],
        "selection_metric": "mae", "forecast_mixer": "none",
        "history_adapter": adapter_variant,
        "parent_stage2_manifest_sha256": parent_manifest_hash,
        "parent_stage2_checkpoint_sha256": parent_record["checkpoint_sha256"],
        "historical_archive_sha256": archive_hash,
    }
    def save(epoch: int, validation: dict[str, float]) -> None:
        _atomic_torch_save({**fixed, "model_state": model.state_dict(),
                            "epoch": epoch, "validation_mse": validation["mse"],
                            "validation_mae": validation["mae"]}, checkpoint)
    save(0, baseline)
    best_epoch, best_mse, best_mae = 0, baseline["mse"], baseline["mae"]
    start_time = monotonic()
    train_total = len(protocol.refs["train"]) // settings.batch_size
    for epoch in range(1, epochs + 1):
        model.eval()  # Frozen predictor and repair dropout must retain selected behavior.
        for index, cpu_batch in enumerate(iter_p4_batches(
            protocol, "train", batch_size=settings.batch_size, epoch=epoch - 1,
        ), 1):
            batch = cpu_batch.to(device)
            starts = _starts(batch, stride=model.repair.stride,
                             patches=model.repair.patch_count, device=device)
            optimizer.zero_grad(set_to_none=True)
            prediction, _ = model(batch.normalized_observed, batch.observed_mask,
                                  batch.query_mean, batch.query_std, starts)
            loss = stage2_objective(prediction, batch.forecast_target)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite E23 history adapter loss")
            loss.backward()
            optimizer.step()
            if index == 1 or index % 5 == 0 or index == train_total:
                print(f"stage=adapter epoch={epoch}/{epochs} batch={index}/{train_total} "
                      f"mse={loss.item():.6f}", flush=True)
        validation = _validation_forecast_metrics(model, protocol, settings, device=device)
        if validation["mae"] < best_mae:
            best_epoch, best_mse, best_mae = epoch, validation["mse"], validation["mae"]
            save(epoch, validation)
        if memory.snapshot_hash() != memory_before or _key_sha256(historical.retrieval.state_dict()) != key_before:
            raise RuntimeError("adapter training changed read-only H state")
        for name, value in parent_payload["model_state"].items():
            if not torch.equal(model.state_dict()[name], value):
                raise RuntimeError(f"adapter training changed frozen parent weight {name}")
        print(f"stage=adapter epoch={epoch}/{epochs} val_mse={validation['mse']:.8f} "
              f"val_mae={validation['mae']:.8f} best_epoch={best_epoch}", flush=True)
    record = {
        "stage": "forecast", "best_epoch": best_epoch,
        "epochs_completed": epochs, "best_val_mse": best_mse, "best_val_mae": best_mae,
        "epoch0_val_mse": baseline["mse"], "epoch0_val_mae": baseline["mae"],
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _file_sha256(checkpoint),
        "stage1_manifest": parent_record["stage1_manifest"],
        "stage1_manifest_sha256": parent_record["stage1_manifest_sha256"],
        "stage1_checkpoint_sha256": parent_record["stage1_checkpoint_sha256"],
        "parent_stage2_manifest": str(parent_selection),
        "parent_stage2_manifest_sha256": parent_manifest_hash,
        "parent_stage2_checkpoint_sha256": parent_record["checkpoint_sha256"],
        "historical_archive_sha256": archive_hash,
        "memory_sha256": parent_record["memory_sha256"],
        "key_sha256": parent_record["key_sha256"],
        "history_adapter": adapter_variant, "forecast_mixer": "none",
        "loss": "forecast_mse_only", "selection_metric": "mae",
        "optimizer_state_restored": False, "rng_reset_seed": settings.seed,
        "learning_rate": settings.stage1_lr, "training_seconds": monotonic() - start_time,
        "trainable_parameters": sum(p.numel() for p in adapter.parameters()),
        "code_revision": _revision(),
        "selection": "earliest lowest fixed-validation MAE over epochs 0..100",
        **_identity(protocol, settings),
    }
    _atomic_json(record, manifest_path)
    return manifest_path


def load_stage2_selection(
    manifest_path: Path, protocol: P4Protocol, settings: E23Settings,
    *, device: torch.device,
) -> dict[str, object]:
    manifest_path = Path(manifest_path).resolve(strict=True)
    record = json.loads(manifest_path.read_text(encoding="utf-8"))
    history_adapter = record.get("history_adapter", "none")
    if history_adapter not in ("none", "aligned_v1", "aligned_bounded_v2"):
        raise ValueError("unsupported Stage 2 history adapter")
    minimum_epoch = 0 if history_adapter != "none" else 1
    if record.get("stage") != "forecast" or int(record.get("best_epoch", -1)) < minimum_epoch:
        raise ValueError("expected validation-selected E23 Stage 2 manifest")
    loss_identity = record.get("loss")
    if loss_identity not in ("forecast_mse_only", "forecast_mse_plus_0p2_mae",
                             "forecast_mse_plus_0p2_mae_late_horizon_v1",
                             "forecast_mse_plus_0p2_mae_utility_router_v1"):
        raise ValueError("Stage 2 loss identity mismatch")
    if loss_identity == "forecast_mse_plus_0p2_mae_utility_router_v1":
        expected_utility = {"utility_router_subset": 32, "utility_router_margin": 0.05,
                            "utility_router_weight": 0.05}
        if any(record.get(key) != value for key, value in expected_utility.items()):
            raise ValueError("Stage 2 utility router identity mismatch")
    selection_metric = record.get("selection_metric", "mse")
    if selection_metric not in ("mse", "mae"):
        raise ValueError("Stage 2 selection metric mismatch")
    forecast_mixer = record.get("forecast_mixer", "none")
    if forecast_mixer not in ("none", "patch_attention_v1"):
        raise ValueError("Stage 2 forecast mixer mismatch")
    forecast_input = record.get("forecast_input", "explicit_v1")
    if forecast_input not in ("explicit_v1", "implicit_latent_v1"):
        raise ValueError("Stage 2 forecast input mismatch")
    if forecast_input == "implicit_latent_v1" and forecast_mixer != "none":
        raise ValueError("implicit Stage 2 input requires the original predictor")
    if settings.recovery_schedule == "stacked_parallel_tv_h_v1" and (selection_metric != "mae" or loss_identity != "forecast_mse_plus_0p2_mae" or forecast_mixer != "none" or forecast_input != "explicit_v1" or history_adapter != "none"):
        raise ValueError("stacked Stage 2 selection identity mismatch")
    parent_payload = None
    if history_adapter != "none":
        if selection_metric != "mae" or forecast_mixer != "none":
            raise ValueError("adapter selection or mixer mismatch")
        if record.get("historical_archive_sha256") != _historical_archive_sha256(protocol):
            raise ValueError("adapter historical archive mismatch")
        parent_path = Path(record["parent_stage2_manifest"]).resolve(strict=True)
        if parent_path == manifest_path or _file_sha256(parent_path) != record.get("parent_stage2_manifest_sha256"):
            raise ValueError("adapter parent Stage 2 manifest mismatch")
        parent_record = json.loads(parent_path.read_text(encoding="utf-8"))
        if parent_record.get("history_adapter", "none") != "none":
            raise ValueError("adapter parent architecture mismatch")
        parent_payload = load_stage2_selection(parent_path, protocol, settings, device=device)
        if parent_record.get("checkpoint_sha256") != record.get("parent_stage2_checkpoint_sha256"):
            raise ValueError("adapter parent Stage 2 checkpoint mismatch")
    _verify_identity(record, protocol, settings)
    stage1_manifest = Path(record["stage1_manifest"])
    if not stage1_manifest.is_file() or _file_sha256(stage1_manifest) != record.get("stage1_manifest_sha256"):
        raise ValueError("Stage 1 manifest SHA-256 mismatch")
    stage1_payload = load_stage1_selection(stage1_manifest, protocol, settings, device=device)
    stage1_record = json.loads(stage1_manifest.read_text(encoding="utf-8"))
    if (record.get("stage1_checkpoint_sha256") != stage1_record["checkpoint_sha256"]
            or record.get("memory_sha256") != stage1_record["memory_sha256"]
            or record.get("key_sha256") != stage1_record["key_sha256"]):
        raise ValueError("Stage 2 handoff state hash mismatch")
    checkpoint = Path(record["checkpoint"])
    if not checkpoint.is_file() or _file_sha256(checkpoint) != record.get("checkpoint_sha256"):
        raise ValueError("Stage 2 checkpoint SHA-256 mismatch")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    _verify_trial_payload(payload, settings)
    if payload.get("loss", "forecast_mse_only") != loss_identity:
        raise ValueError("Stage 2 loss checkpoint mismatch")
    if loss_identity == "forecast_mse_plus_0p2_mae_utility_router_v1":
        if any(payload.get(key) != value for key, value in expected_utility.items()):
            raise ValueError("Stage 2 utility router checkpoint mismatch")
    if payload.get("selection_metric", "mse") != selection_metric:
        raise ValueError("Stage 2 selection metric mismatch")
    if payload.get("forecast_mixer", "none") != forecast_mixer:
        raise ValueError("Stage 2 forecast mixer identity mismatch")
    if payload.get("forecast_input", "explicit_v1") != forecast_input:
        raise ValueError("Stage 2 forecast input identity mismatch")
    if payload.get("history_adapter", "none") != history_adapter:
        raise ValueError("Stage 2 history adapter mismatch")
    if history_adapter != "none":
        for key in ("parent_stage2_manifest_sha256", "parent_stage2_checkpoint_sha256",
                    "historical_archive_sha256"):
            if payload.get(key) != record.get(key):
                raise ValueError(f"Stage 2 adapter {key} mismatch")
        for key, value in parent_payload["model_state"].items():
            if key not in payload["model_state"] or not torch.equal(value, payload["model_state"][key]):
                raise ValueError("Stage 2 adapter changed frozen parent state")
    if (payload.get("epoch") != record["best_epoch"]
            or payload.get("memory_sha256") != record["memory_sha256"]
            or payload.get("key_sha256") != record["key_sha256"]
            or payload.get("stage1_manifest_sha256") != record["stage1_manifest_sha256"]
            or payload.get("stage1_checkpoint_sha256") != record["stage1_checkpoint_sha256"]):
        raise ValueError("Stage 2 paired checkpoint identity mismatch")
    if (_memory_state_sha256(payload["memory_state"]) != record["memory_sha256"]
            or _key_sha256(payload["retrieval_state"]) != record["key_sha256"]):
        raise ValueError("Stage 2 H bank/key state SHA-256 mismatch")
    for key, value in stage1_payload["model_state"].items():
        if key.startswith("historical.key_encoder."):
            if not torch.equal(value, payload["model_state"]["repair." + key]):
                raise ValueError("Stage 2 changed frozen H query encoder")
    return payload


__all__ = ["build_repair_model", "build_forecast_model", "stage1_objective",
           "stage2_objective", "train_repair", "train_forecast", "train_history_adapter",
           "load_stage1_selection", "load_stage2_selection"]
