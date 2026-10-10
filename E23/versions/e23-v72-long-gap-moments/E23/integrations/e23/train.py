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
    if settings.stage1_gap_structure != "none":
        from .long_gap import MIN_LENGTH, LEVEL_WEIGHT, TREND_WEIGHT
        identity.update(stage1_gap_structure=settings.stage1_gap_structure,
                        gap_min_length=MIN_LENGTH, gap_level_weight=LEVEL_WEIGHT,
                        gap_trend_weight=TREND_WEIGHT, gap_weighting="coordinate_count",
                        gap_metric_space="train_standardized",
                        gap_coordinates="per_run_linspace_minus1_plus1",
                        gap_scope="original_lookback_hidden_runs")
    if settings.recovery_schedule != "parallel_v1":
        identity["recovery_schedule"] = settings.recovery_schedule
        identity["recovery_rounds"] = settings.recovery_rounds
    if settings.recovery_schedule == "feature_tv_h_v1":
        identity.update(recovery_layers=settings.recovery_layers,
                        feature_transfer="shared_point_channel_variable_v1")
    if settings.front_context != "none":
        identity.update(front_context=settings.front_context, front_context_width=64,
                        front_context_blocks=2, front_context_kernels=[7,31],
                        front_context_dilations=[1,2], front_context_block_residual_scale=0.1,
                        front_context_scale="train_standardized", front_context_features=5,
                        front_context_pool="valid_mean", front_context_cross_variable=False,
                        front_context_scope=96, front_context_output="latent_only")
    if settings.point_refiner_span:
        identity.update(point_refiner="point_residual_v1", point_refiner_span=settings.point_refiner_span,
                        point_refiner_context=96, point_refiner_width=64)
    if settings.point_router != "none":
        identity.update(point_router=settings.point_router, point_router_width=64,
                        point_router_logits_bound=2.0, point_router_features=94,
                        point_router_experts=["none", "historical", "variable", "temporal"])
    if settings.point_candidate_readout != "none":
        identity.update(point_candidate_readout=settings.point_candidate_readout,
                        point_candidate_heads=8, point_candidate_width=64,
                        point_candidate_query_features=143, point_candidate_scale="train_standardized",
                        point_candidate_scope=96, point_candidate_null_value=0)
    if settings.point_context_readout != "none":
        identity.update(point_context_readout=settings.point_context_readout,point_context_features=141,
                        point_context_extra_parameters=9153,point_context_scope=32,
                        point_context_relative_delta_bound=2.,point_context_sources="visible_cnn_and_routed_hvt")
    if settings.repair_input_scaling != "legacy_v1":
        identity.update(repair_input_scaling=settings.repair_input_scaling,repair_input_scale_floor=.1,
                        repair_output_statistics="original_visible_stats",repair_key_pair_scaling="full_robust_repair_scale")
    if settings.temporal_evidence_readout != "none":
        identity.update(temporal_evidence_readout=settings.temporal_evidence_readout,temporal_evidence_heads=8,
                        temporal_evidence_width=64,temporal_evidence_layers=2,temporal_evidence_scope=32,
                        temporal_evidence_sources="true_visible_same_variable_all96",temporal_evidence_null_value=0,
                        temporal_evidence_relative_bound=2.,temporal_evidence_query_features=141,temporal_evidence_source_features=76)
    if settings.recent_mean_projection != "none":
        identity.update(recent_mean_projection=settings.recent_mean_projection,recent_mean_features=205,
                        recent_mean_extra_parameters=13249,recent_mean_relative_delta_bound=2.,
                        recent_mean_share_cap=4.,recent_mean_scope=32,recent_mean_target="clean_lookback_mean")
    if settings.historical_key_policy != "legacy_v1":
        identity.update(historical_key_policy=settings.historical_key_policy,historical_key_coordinates="train_standardized",
                        historical_key_temperature=.1,historical_key_variance_target=.1,
                        historical_key_variance_weight=.5,historical_key_covariance_weight=.05,
                        historical_key_loss_queries="damaged_visible_full_validity",historical_key_ema_update="optimizer_step")
    if settings.point_variable_readout != "none":
        identity.update(point_variable_readout=settings.point_variable_readout,repair_readout_width=64,
                        repair_readout_scale_bounds=[.5,1.5],repair_readout_bias_bound=.5,
                        repair_readout_sites=["point_candidate","tail32"],repair_readout_extra_parameters=1792)
    if settings.repair_coordinates != "global_v1":
        identity.update(repair_coordinates=settings.repair_coordinates,repair_scale_floor=.1,
                        front_context_scale="robust_local_visible",point_candidate_scale="robust_local_visible",
                        repair_loss_scale="train_standardized",predictor_statistics_unchanged=True)
    if settings.variable_calibration != "none":
        identity.update(variable_calibration=settings.variable_calibration,
                        variable_calibration_width=64,variable_calibration_features=164,
                        variable_calibration_global_values=True,variable_calibration_source_target_identity=True,
                        variable_calibration_statistics="visible_only_mean_logstd",variable_calibration_std_log_floor=1e-6)
    if settings.front_variable_context != "none":
        identity.update(front_variable_context=settings.front_variable_context,front_context_cross_variable=True,
                        front_signature_width=64,front_signature_blocks=2,front_signature_heads=64,
                        front_signature_head_dim=96,front_signature_kernels=[5,31],front_signature_dropout=.1,
                        front_signature_norm_axes=[64,96],front_signature_statistics="visible_only_mean_logstd",
                        front_signature_stat_width=32,front_signature_scale_bounds=[.25,4.],
                        front_signature_offset_bound=2.,front_signature_block_scale_init=.001)
        if settings.front_variable_context in ("signature_plain_v2", "signature_plain_pos_v3"):
            identity.update(front_signature_statistics="none",front_signature_stat_width=0,
                            front_signature_extra_parameters=105088)
            identity.pop("front_signature_scale_bounds")
            identity.pop("front_signature_offset_bound")
        if settings.front_variable_context == "signature_plain_pos_v3":
            identity.pop("front_signature_extra_parameters")
            identity.update(front_signature_position="variable_time_learned_zero_v1",
                            front_signature_position_width=64,front_signature_position_length=96,
                            front_signature_position_parameters_per_variable=6144,
                            front_signature_attention_parameters=105088)
    if settings.overlap_fusion != "none":
        identity.update(overlap_fusion=settings.overlap_fusion, overlap_width=32,
                        overlap_features=84, overlap_log_weight_bound=2.0)
    if settings.stage1_structure != "none":
        identity.update(stage1_structure=settings.stage1_structure, recent_state_span=32,
                        recent_level_weight=1.0 if settings.stage1_structure == "recent_mean_v2" else .2,
                        recent_trend_weight=0.0 if settings.stage1_structure == "recent_mean_v2" else .2,
                        recent_trend_coordinates="linspace_minus1_plus1")
    if settings.repair_stride != 8 or settings.point_refiner_arch != "legacy":
        identity.update(repair_stride=settings.repair_stride, repair_patch_length=settings.repair_patch_length,
                        repair_patch_count=96//settings.repair_stride, forecast_patch_stride=8,
                        history_sampling_stride=8, temporal_local_radius=settings.temporal_local_radius,
                        temporal_boundaries=list(settings.temporal_boundaries),
                        variable_offset_points=[v*settings.repair_stride for v in settings.variable_offsets])
    if settings.point_refiner_arch != "legacy":
        identity.update(point_refiner_arch=settings.point_refiner_arch,
                        historical_query_length=16, historical_candidates="parent16_crop_v1")
        if settings.point_refiner_arch == "none_v1":
            identity.update(point_refiner="none", point_refiner_span=0)
        else:
            identity.update(point_feature_scale="train_standardized",
                            hybrid_blocks=2, hybrid_width=64, hybrid_kernels=[31,5],
                            hybrid_specialist_residual="disabled_at_singleton_targets")
    if settings.point_feature_width:
        identity.update(point_feature_width=settings.point_feature_width,
                        point_feature_expansion="nonoverlap_position_features_v1")
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
        patch_length=settings.repair_patch_length,
        stride=settings.repair_stride,
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
        point_refiner_span=settings.point_refiner_span,
        point_feature_width=settings.point_feature_width,
        point_refiner_arch=settings.point_refiner_arch,
        historical_query_length=settings.historical_query_length,
        front_context=settings.front_context,
        point_router=settings.point_router, overlap_fusion=settings.overlap_fusion,
        point_candidate_readout=settings.point_candidate_readout,
        variable_calibration=settings.variable_calibration,
        front_variable_context=settings.front_variable_context,
        repair_coordinates=settings.repair_coordinates,
        point_variable_readout=settings.point_variable_readout,
        historical_key_policy=settings.historical_key_policy,
        recent_mean_projection=settings.recent_mean_projection,
        temporal_evidence_readout=settings.temporal_evidence_readout,
        repair_input_scaling=settings.repair_input_scaling,
        point_context_readout=settings.point_context_readout,
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
    structure_mode: str = "none",
    gap_structure_mode: str = "none",
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
    total = repair + weighted
    parts = {"repair": repair, "retrieval": retrieval_loss, "retrieval_weighted": weighted}
    if structure_mode != "none":
        if structure_mode not in ("recent_state_v1", "recent_mean_v2") or loss_mode != "global_mse_v1":
            raise ValueError("recent state loss requires global MSE")
        from .recent_state import recent_state_loss
        if structure_mode == "recent_mean_v2":
            structural, diagnostics = recent_state_loss(candidate,target,hidden,query_std,level_weight=1.0,trend_weight=0.0)
        else:
            structural, diagnostics = recent_state_loss(candidate, target, hidden, query_std)
        total = total+structural
        parts.update(diagnostics)
    if gap_structure_mode != "none":
        if gap_structure_mode != "long_gap_moments_v1" or loss_mode != "global_mse_v1":
            raise ValueError("long-gap supervision requires global Stage1 MSE")
        from .long_gap import long_gap_moments, LEVEL_WEIGHT, TREND_WEIGHT
        level, trend, count, segments = long_gap_moments(candidate, target, hidden, query_std)
        gap_loss = LEVEL_WEIGHT*level + TREND_WEIGHT*trend
        total = total + gap_loss
        parts.update(gap_level=level, gap_trend=trend, gap_structure_weighted=gap_loss,
                     gap_hidden_coordinates=count, gap_segments=segments)
    return total, parts


def _retrieval_loss(model: MultiExpertRepairEncoder, batch: P4Batch) -> Tensor:
    if getattr(model,"repair_input_scaling","legacy_v1") == "all_floor_v1":
        from dataclasses import replace as replace_batch
        from malr.models.e23.coordinates import repair_scale
        scale=repair_scale(batch.query_std);ratio=(batch.query_std/scale)[:,None]
        batch=replace_batch(batch,normalized_observed=torch.where(batch.observed_mask,batch.normalized_observed,0)*ratio,
                            normalized_target=batch.normalized_target*ratio,query_std=scale)
    if getattr(model.historical,"key_policy","legacy_v1") == "global_diverse_v1":
        from .diverse_keys import global_diverse_key_loss
        return global_diverse_key_loss(model,batch)
    patch_len = getattr(model.historical, "query_length", model.patch_length)
    stride = getattr(model.historical, "query_stride", model.stride)
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
                                  memory.patch_len, memory.patch_stride)
    if model.stride != memory.patch_stride:
        starts = starts[:, :1] + torch.arange(patches.values.shape[2], device=starts.device)[None] * memory.patch_stride
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
    if getattr(model.historical,"key_policy","legacy_v1") == "legacy_v1":
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
    *, device: torch.device, auxiliary_predictor=None,
) -> dict[str, float | int]:
    model.eval()
    window_squared = global_squared = global_absolute = 0.0
    small_window_squared = small_global_squared = 0.0
    count = small_count = 0
    auxiliary_squared = auxiliary_absolute = auxiliary_count = 0
    if auxiliary_predictor is not None:
        auxiliary_predictor.eval()
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
        if auxiliary_predictor is not None:
            from integrations.forecast_compare.auxiliary_forecast import predict_repaired
            prediction = predict_repaired(auxiliary_predictor, output.explicit.normalized_sequence, batch)
            future_error = prediction.float() - batch.forecast_target.float()
            auxiliary_squared += future_error.square().sum().item()
            auxiliary_absolute += future_error.abs().sum().item()
            auxiliary_count += future_error.numel()
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
        **({"auxiliary_forecast_mse": auxiliary_squared / auxiliary_count,
            "auxiliary_forecast_mae": auxiliary_absolute / auxiliary_count}
           if auxiliary_predictor is not None else {}),
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
    device: torch.device, epochs: int = 100, auxiliary_forecast: str = "none",
    training_batch_size: int | None = None,
) -> Path:
    from integrations.forecast_compare.auxiliary_forecast import (
        auxiliary_identity, build_auxiliary_predictor, auxiliary_forecast_loss,
    )
    from .forecast_sensitive import parameter_hash
    auxiliary_metadata = auxiliary_identity(auxiliary_forecast)
    if not 0 < epochs <= settings.stage1_epochs:
        raise ValueError("Stage 1 epochs exceed the locked first-run setting")
    batch_size = settings.batch_size if training_batch_size is None else training_batch_size
    if type(batch_size) is not int or batch_size not in (32, 128):
        raise ValueError("Stage 1 batch trial supports 32 or the original 128")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"E23 Stage 1 output already exists: {output}")
    output.mkdir(parents=True)
    torch.manual_seed(settings.seed)
    model = build_repair_model(protocol, settings, device=device)
    auxiliary_predictor = build_auxiliary_predictor(protocol, device=device) if auxiliary_metadata else None
    if auxiliary_predictor is not None:
        auxiliary_metadata.update(
            stage1_aux_predictor_initial_sha256=parameter_hash(auxiliary_predictor),
            stage1_aux_predictor_parameters=sum(p.numel() for p in auxiliary_predictor.parameters()),
        )
    memory = _new_memory(protocol)
    optimizer = torch.optim.Adam(list(model.parameters()) + (list(auxiliary_predictor.parameters()) if auxiliary_predictor is not None else []), lr=settings.stage1_lr,
                                 weight_decay=settings.weight_decay)
    checkpoint = output / "checkpoints/e23-stage1.pt"
    manifest_path = output / "runs/e23-stage1.json"
    best_mse = float("inf")
    best_epoch = 0
    start_time = monotonic()
    train_total = len(protocol.refs["train"]) // batch_size
    staged_identities: set[tuple[int, int]] = set()
    history: list[dict[str, object]] = []
    global_objective = settings.stage1_objective == "global_mse_v1"
    auxiliary_gradient_batches = auxiliary_windows = 0
    for epoch in range(1, epochs + 1):
        model.historical.retrieval.begin_epoch(epoch)
        model.train()
        if auxiliary_predictor is not None:
            auxiliary_predictor.train()
        for index, cpu_batch in enumerate(iter_p4_batches(
            protocol, "train", batch_size=batch_size, epoch=epoch - 1,
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
                structure_mode=settings.stage1_structure,
                gap_structure_mode=settings.stage1_gap_structure,
            )
            if auxiliary_predictor is not None:
                auxiliary_loss = auxiliary_forecast_loss(
                    auxiliary_predictor, output_batch.explicit.normalized_sequence, batch,
                    epoch=epoch, batch_index=index,
                )
                loss = loss + 0.05 * auxiliary_loss
                auxiliary_windows += len(batch.normalized_observed)
                auxiliary_gradient_batches += int(epoch > 10)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite E23 Stage 1 loss")
            loss.backward()
            optimizer.step()
            if getattr(model.historical,"key_policy","legacy_v1") == "global_diverse_v1":
                model.historical.retrieval.update_ema()
            staged = _stage_observed(memory, model, batch, starts, staged_identities)
            if index == 1 or index % 5 == 0 or index == train_total:
                gap_log = (f" gap_level={parts['gap_level'].item():.6f} gap_trend={parts['gap_trend'].item():.6f}"
                           f" gap_segments={int(parts['gap_segments'].item())}"
                           if settings.stage1_gap_structure != "none" else "")
                print(f"stage=repair epoch={epoch}/{epochs} batch={index}/{train_total} "
                      f"loss={loss.item():.6f} repair={parts['repair'].item():.6f} "
                      f"retrieval={parts['retrieval'].item():.6f} staged={staged}{gap_log}", flush=True)
        _publish_epoch(memory, model, epoch)
        validation_metrics = None
        if global_objective:
            validation_metrics = _validation_repair_metrics(model, protocol, settings, device=device,
                                                            auxiliary_predictor=auxiliary_predictor)
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
            paired = _paired_payload(
                model, memory, epoch=epoch, validation_mse=validation_mse,
                settings=settings,
                validation_metrics=validation_metrics,
            )
            if auxiliary_predictor is not None:
                paired.update(**auxiliary_metadata, auxiliary_predictor_state=auxiliary_predictor.state_dict())
            _atomic_torch_save(paired, checkpoint)
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
        "stage1_training_batch_size": batch_size,
        "validation_batch_size": settings.batch_size,
        "optimizer_updates": train_total * epochs,
        "training_windows_per_epoch": train_total * batch_size,
        **auxiliary_metadata,
        **({"stage1_aux_repair_gradient_batches": auxiliary_gradient_batches,
            "stage1_aux_training_windows": auxiliary_windows} if auxiliary_metadata else {}),
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
    from integrations.forecast_compare.auxiliary_forecast import verify_auxiliary_identity
    verify_auxiliary_identity(record, payload)
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
    repair_retention: Literal["none", "uniform_v1", "forecast_gain_v1"] = "none",
    repair_update_policy: Literal["joint", "frozen_v1", "slow_lr_0p1_v1", "freeze_front_v1"] = "joint",
    common_stage2: bool = False,
    forecast_padding: Literal["zero_invalid_v1", "replicate_last_value_v1"] = "zero_invalid_v1",
    extra_mse_checkpoint: bool = False,
) -> Path:
    from .aligned_stage2 import aligned_identity, reset_training_rng
    common_identity = aligned_identity(common_stage2)
    if type(extra_mse_checkpoint) is not bool or (extra_mse_checkpoint and selection_metric != "mae"):
        raise ValueError("extra MSE checkpoint requires boolean flag and primary MAE selection")
    tracking_identity = {"checkpoint_tracking": "mae_and_mse_v1"} if extra_mse_checkpoint else {}
    if forecast_padding not in ("zero_invalid_v1", "replicate_last_value_v1"):
        raise ValueError("unsupported forecast padding")
    if forecast_padding != "zero_invalid_v1" and (common_stage2 or forecast_input != "explicit_v1" or forecast_mixer != "none"):
        raise ValueError("independent padding alignment requires original explicit head and legacy recipe")
    padding_identity = {"forecast_padding": forecast_padding, "forecast_padding_alignment_only": True} if forecast_padding != "zero_invalid_v1" else {}
    if common_stage2 and (forecast_loss != "mse" or repair_retention != "none"
            or selection_metric != "mae" or forecast_input != "explicit_v1"
            or forecast_mixer != "none" or repair_update_policy not in ("joint", "frozen_v1", "slow_lr_0p1_v1")
            or settings.recovery_schedule != "iterative_tv_h_v1" or settings.recovery_rounds != 1):
        raise ValueError("common Stage2 requires pure MSE, MAE selection and joint original-head training")
    from .repair_updates import repair_update_identity, build_stage2_optimizer, stage2_training_mode
    update_identity = repair_update_identity(repair_update_policy)
    if repair_update_policy == "freeze_front_v1" and settings.front_context != "window_cnn_v1":
        raise ValueError("freeze_front_v1 requires v46 window context")
    if update_identity and (settings.recovery_schedule != "iterative_tv_h_v1"
            or settings.recovery_rounds != 1 or settings.point_refiner_span not in (0, 32)
            or forecast_input != "explicit_v1" or forecast_mixer != "none"
            or selection_metric != "mae"
            or (not common_stage2 and not (repair_update_policy in ("freeze_front_v1", "frozen_v1")
                    and forecast_loss == "mse" and repair_retention == "none")
                and (forecast_loss != "mse_plus_0p2_mae" or repair_retention != "uniform_v1"))):
        raise ValueError("repair update controls require the original v31-B or tail32 setup")
    from .forecast_sensitive import (retention_identity, retention_active, snapshot_teacher,
                                      region_oracle_gains, gains_to_weights, weighted_repair_loss, parameter_hash, RegionTrainingDiagnostics)
    retention = retention_identity(repair_retention)
    if retention and (forecast_loss != "mse_plus_0p2_mae" or selection_metric != "mae" or forecast_input != "explicit_v1" or forecast_mixer != "none"):
        raise ValueError("repair retention requires original explicit predictor and MAE/mixed loss")
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
    predictor_initial_sha256 = parameter_hash(model.predictor)
    if common_stage2 or forecast_padding != "zero_invalid_v1":
        model.forecast_padding = "replicate_last_value_v1"
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
    optimizer = build_stage2_optimizer(model, mode=repair_update_policy,
        learning_rate=settings.stage2_lr, weight_decay=settings.weight_decay)
    update_metadata = ({**update_identity,
        "repair_initial_sha256": parameter_hash(model.repair),
        "repair_trainable_parameters": sum(p.numel() for p in model.repair.parameters() if p.requires_grad),
        "repair_learning_rate": settings.stage2_lr * update_identity["repair_learning_rate_scale"],
        "predictor_learning_rate": settings.stage2_lr} if update_identity else {})
    if common_stage2:
        reset_training_rng(settings.seed)
    checkpoint = output / "checkpoints/e23-stage2.pt"
    manifest_path = output / "runs/e23-stage2.json"
    best_mse = float("inf")
    best_mae = float("inf")
    best_selection = float("inf")
    best_epoch = 0
    start_time = monotonic()
    train_total = len(protocol.refs["train"]) // settings.batch_size
    teacher_seconds = 0.0
    dual_selection = extra_mse_checkpoint or common_stage2 or (repair_update_policy == "freeze_front_v1"
        and forecast_loss == "mse" and repair_retention == "none")
    mse_choice = {"mse": float("inf"), "mae": float("inf"), "epoch": 0}
    mse_checkpoint = output / "checkpoints/e23-stage2-mse.pt"
    def save_forecast_choice(path, epoch, validation, metric):
        _atomic_torch_save({
            "format_version": 1, "epoch": epoch, "model_state": model.state_dict(),
            "memory_state": stage1_payload["memory_state"], "retrieval_state": stage1_payload["retrieval_state"],
            "memory_sha256": stage1_record["memory_sha256"], "key_sha256": stage1_record["key_sha256"],
            "stage1_manifest_sha256": stage1_manifest_hash, "stage1_checkpoint_sha256": stage1_record["checkpoint_sha256"],
            "validation_mse": validation["mse"], "validation_mae": validation["mae"],
            "selection_metric": metric, "forecast_mixer": forecast_mixer, "forecast_input": forecast_input,
            **({"loss": loss_identity} if forecast_loss != "mse" else {}),
            **utility_identity, **retention, **update_metadata, **common_identity, **padding_identity, **tracking_identity, **_trial_identity(settings),
        }, path)
    region_diagnostics = RegionTrainingDiagnostics()
    active_batches = weighted_windows = training_windows = 0
    for epoch in range(1, epochs + 1):
        teacher = None
        if repair_retention == "forecast_gain_v1" and retention_active(repair_retention, epoch):
            if device.type == "cuda": torch.cuda.synchronize(device)
            teacher_start = monotonic()
            teacher = snapshot_teacher(model)
            if device.type == "cuda": torch.cuda.synchronize(device)
            teacher_seconds += monotonic() - teacher_start
        stage2_training_mode(model, repair_update_policy)
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
            if retention_active(repair_retention, epoch):
                hidden = ~batch.observed_mask
                weights = torch.ones_like(batch.normalized_observed)
                if teacher is not None:
                    if device.type == "cuda": torch.cuda.synchronize(device)
                    teacher_start = monotonic()
                    gains = region_oracle_gains(teacher, batch, subset=16)
                    teacher_output = teacher.repair(batch.normalized_observed[:len(gains)],
                        batch.observed_mask[:len(gains)], batch.query_mean[:len(gains)],
                        batch.query_std[:len(gains)], starts[:len(gains)])
                    region_diagnostics.update(gains, teacher_output.explicit.normalized_sequence, batch)
                    weights = gains_to_weights(gains, hidden)
                    if device.type == "cuda": torch.cuda.synchronize(device)
                    teacher_seconds += monotonic() - teacher_start
                    weighted_windows += min(16, len(batch.normalized_observed))
                retention_loss = weighted_repair_loss(repair_output.explicit.normalized_sequence,
                    batch.normalized_target, hidden, batch.query_std, weights)
                loss = loss + 0.05 * retention_loss
                active_batches += 1
            training_windows += len(batch.normalized_observed)
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
        del teacher
        validation = _validation_forecast_metrics(model, protocol, settings, device=device)
        if validation[selection_metric] < best_selection:
            best_selection = validation[selection_metric]
            best_mse, best_mae, best_epoch = validation["mse"], validation["mae"], epoch
            save_forecast_choice(checkpoint, epoch, validation, selection_metric)
        if dual_selection and validation["mse"] < mse_choice["mse"]:
            mse_choice = {**validation, "epoch": epoch}
            save_forecast_choice(mse_checkpoint, epoch, validation, "mse")
        if memory.snapshot_hash() != read_only_snapshot_hash or _key_sha256(historical.retrieval.state_dict()) != read_only_key_hash:
            raise RuntimeError("Stage 2 changed the read-only H bank/key state")
        if repair_update_policy == "frozen_v1" and parameter_hash(model.repair) != update_metadata["repair_initial_sha256"]:
            raise RuntimeError("Stage 2 changed the frozen repair state")
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
        **retention,
        **update_metadata,
        **common_identity, **padding_identity, **tracking_identity,
        **({"repair_retention_teacher_seconds": teacher_seconds,
            "repair_retention_active_batches": active_batches,
            "repair_retention_weighted_windows": weighted_windows,
            "repair_retention_training_windows": training_windows} if retention else {}),
        **({"repair_retention_region_diagnostics": region_diagnostics.result()} if repair_retention == "forecast_gain_v1" else {}),
        "predictor_initial_sha256": predictor_initial_sha256,
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
    if dual_selection:
        mse_record = {**record, "best_epoch": mse_choice["epoch"],
            "best_val_mse": mse_choice["mse"], "best_val_mae": mse_choice["mae"],
            "checkpoint": str(mse_checkpoint.resolve()), "checkpoint_sha256": _file_sha256(mse_checkpoint),
            "selection_metric": "mse", "selection": "lowest fixed-validation forecast MSE",
            "training_resource_parent_manifest": str(manifest_path.resolve()),
            "training_resource_parent_sha256": _file_sha256(manifest_path)}
        _atomic_json(mse_record, output / "runs/e23-stage2-mse.json")
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
    tracking = record.get("checkpoint_tracking")
    if tracking not in (None, "mae_and_mse_v1"):
        raise ValueError("checkpoint tracking identity mismatch")
    history_adapter = record.get("history_adapter", "none")
    from .aligned_stage2 import aligned_identity
    downstream = record.get("downstream_protocol", "legacy")
    if downstream not in ("legacy", "t1_mse_v1"):
        raise ValueError("unknown downstream protocol")
    common_identity = aligned_identity(downstream == "t1_mse_v1")
    padding_only = record.get("forecast_padding_alignment_only", False)
    if type(padding_only) is not bool:
        raise ValueError("padding alignment flag must be boolean")
    if padding_only and (downstream != "legacy" or record.get("forecast_padding") != "replicate_last_value_v1"
            or record.get("forecast_input") != "explicit_v1" or record.get("forecast_mixer") != "none"):
        raise ValueError("padding alignment identity mismatch")
    if downstream == "legacy" and not padding_only and record.get("forecast_padding", "zero_invalid_v1") != "zero_invalid_v1":
        raise ValueError("unexpected padding without alignment identity")
    if common_identity and (any(record.get(k) != v for k, v in common_identity.items())
            or record.get("loss") != "forecast_mse_only" or record.get("repair_retention", "none") != "none"
            or record.get("repair_update_policy", "joint") not in ("joint", "frozen_v1", "slow_lr_0p1_v1")
            or record.get("selection_metric") not in ("mae", "mse")
            or record.get("forecast_input") != "explicit_v1" or record.get("forecast_mixer") != "none"
            or record.get("learning_rate") != settings.stage2_lr):
        raise ValueError("common Stage2 identity mismatch")
    from .repair_updates import repair_update_identity
    update_identity = repair_update_identity(record.get("repair_update_policy", "joint"))
    if record.get("repair_update_policy") == "freeze_front_v1" and settings.front_context != "window_cnn_v1":
        raise ValueError("freeze_front_v1 requires v46 window context")
    if any(record.get(k) != v for k, v in update_identity.items()):
        raise ValueError("Stage 2 repair update identity mismatch")
    if update_identity and (record.get("repair_learning_rate") != settings.stage2_lr * update_identity["repair_learning_rate_scale"]
            or record.get("predictor_learning_rate") != settings.stage2_lr):
        raise ValueError("Stage 2 repair/predictor learning rate mismatch")
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
    from .forecast_sensitive import retention_identity
    retention = retention_identity(record.get("repair_retention", "none"))
    if any(record.get(k) != v for k,v in retention.items()):
        raise ValueError("Stage 2 repair retention identity mismatch")
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
    if payload.get("checkpoint_tracking") != tracking:
        raise ValueError("checkpoint tracking payload mismatch")
    if payload.get("forecast_padding_alignment_only", False) != padding_only or (padding_only and payload.get("forecast_padding") != "replicate_last_value_v1"):
        raise ValueError("padding alignment checkpoint mismatch")
    if (payload.get("downstream_protocol", "legacy") != downstream
            or any(payload.get(k) != v for k, v in common_identity.items())):
        raise ValueError("common Stage2 checkpoint identity mismatch")
    if (payload.get("repair_update_policy", "joint") != record.get("repair_update_policy", "joint")
            or any(payload.get(k) != v for k, v in update_identity.items())
            or (update_identity and any(payload.get(k) != record.get(k) for k in (
                "repair_initial_sha256", "repair_trainable_parameters", "repair_learning_rate", "predictor_learning_rate")))):
        raise ValueError("Stage 2 repair update checkpoint mismatch")
    if record.get("repair_update_policy") == "frozen_v1":
        for key, value in stage1_payload["model_state"].items():
            if "repair." + key not in payload["model_state"] or not torch.equal(value, payload["model_state"]["repair." + key]):
                raise ValueError("Stage 2 changed frozen Stage 1 repair state")
    if record.get("repair_update_policy") == "freeze_front_v1":
        keys = [k for k in stage1_payload["model_state"] if k.startswith("front_context.")]
        if not keys:
            raise ValueError("Stage 2 frozen front state missing")
        for key in keys:
            if "repair."+key not in payload["model_state"] or not torch.equal(stage1_payload["model_state"][key], payload["model_state"]["repair."+key]):
                raise ValueError("Stage 2 changed frozen Stage 1 front context state")
    if payload.get("repair_retention", "none") != record.get("repair_retention", "none") or any(payload.get(k) != v for k,v in retention.items()):
        raise ValueError("Stage 2 repair retention checkpoint mismatch")
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
