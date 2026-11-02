"""Shared edge-utility labels and admission rules for relation candidates."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


EDGE_RELATION_TYPES: tuple[str, ...] = ("historical", "temporal", "variable")


def relative_edge_gain(
    baseline_error: Tensor,
    edge_error: Tensor,
    *,
    eps: float = 1e-8,
) -> Tensor:
    """Compute a detached relative error reduction for one candidate edge."""

    if baseline_error.shape != edge_error.shape:
        raise ValueError("baseline_error and edge_error must have the same shape")
    if not baseline_error.is_floating_point() or not edge_error.is_floating_point():
        raise ValueError("edge errors must be floating-point tensors")
    if not torch.isfinite(baseline_error).all() or not torch.isfinite(edge_error).all():
        raise ValueError("edge errors must contain only finite values")
    if torch.any(baseline_error < 0.0) or torch.any(edge_error < 0.0):
        raise ValueError("edge errors must be non-negative")
    if not torch.isfinite(torch.as_tensor(eps)) or eps <= 0.0:
        raise ValueError("eps must be positive and finite")
    return ((baseline_error - edge_error) / (baseline_error + float(eps))).detach()


def _bounded_relative_edge_gain(
    baseline_error: Tensor,
    edge_error: Tensor,
    *,
    eps: float = 1e-8,
) -> Tensor:
    """Return a finite [-1, 1] utility for train-time candidate labels."""

    return relative_edge_gain(
        baseline_error,
        edge_error,
        eps=eps,
    ).nan_to_num(nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)


def masked_patch_huber_error(
    prediction: Tensor,
    target: Tensor,
    missing_mask: Tensor,
    *,
    delta: float = 1.0,
) -> Tensor:
    """Average patch Huber error over missing coordinates, preserving route axes.

    ``prediction`` may insert candidate or route dimensions immediately before
    the patch-coordinate axis.  The target and mask are expanded across those
    dimensions, so the returned tensor keeps every non-coordinate prediction
    dimension.
    """

    if target.shape != missing_mask.shape:
        raise ValueError("target and missing_mask must have the same shape")
    if prediction.ndim < target.ndim:
        raise ValueError("prediction must include every target dimension")
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise ValueError("prediction and target must be floating-point")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("prediction and target must contain only finite values")
    if missing_mask.dtype != torch.bool:
        raise ValueError("missing_mask must be bool")
    if not torch.isfinite(torch.as_tensor(delta)) or delta <= 0.0:
        raise ValueError("delta must be positive and finite")

    extra = prediction.ndim - target.ndim
    expanded_target = target
    expanded_mask = missing_mask
    for _ in range(extra):
        expanded_target = expanded_target.unsqueeze(-2)
        expanded_mask = expanded_mask.unsqueeze(-2)
    try:
        expanded_target = expanded_target.expand_as(prediction)
        expanded_mask = expanded_mask.expand_as(prediction)
    except RuntimeError as error:
        raise ValueError("prediction is not broadcast-compatible with target") from error
    error = F.huber_loss(prediction, expanded_target, delta=float(delta), reduction="none")
    selected = error.masked_fill(~expanded_mask, 0.0)
    denominator = expanded_mask.sum(dim=-1).clamp_min(1)
    return selected.sum(dim=-1) / denominator.to(prediction.dtype)


def counterfactual_relative_gain(
    baseline_prediction: Tensor,
    candidate_prediction: Tensor,
    target: Tensor,
    missing_mask: Tensor,
) -> Tensor:
    """Label each transformed candidate by its decoded masked-patch gain."""

    baseline_error = masked_patch_huber_error(
        baseline_prediction, target, missing_mask
    )
    candidate_error = masked_patch_huber_error(
        candidate_prediction, target, missing_mask
    )
    try:
        baseline_error = baseline_error.unsqueeze(-1).expand_as(candidate_error)
    except RuntimeError as error:
        raise ValueError("candidate_prediction must add only route/candidate axes") from error
    # Candidate labels train the relation predictors.  A nearly perfect
    # baseline should not create an arbitrarily large negative outlier when a
    # candidate is poor; keep the utility bounded while preserving its sign.
    return _bounded_relative_edge_gain(baseline_error, candidate_error)


def edge_gain_admission(
    predicted_gain: Tensor,
    eligible: Tensor,
    *,
    threshold: float = 0.05,
) -> Tensor:
    """Return the hard mask of eligible edges predicted to clear the gain gate."""

    if predicted_gain.shape != eligible.shape:
        raise ValueError("predicted_gain and eligible must have the same shape")
    if not predicted_gain.is_floating_point():
        raise ValueError("predicted_gain must be floating-point")
    if not torch.isfinite(predicted_gain).all():
        raise ValueError("predicted_gain must contain only finite values")
    if not torch.isfinite(torch.as_tensor(threshold)):
        raise ValueError("threshold must be finite")
    return eligible.to(device=predicted_gain.device, dtype=torch.bool) & (
        predicted_gain >= float(threshold)
    )


def balanced_edge_gain_loss(
    predicted_gain: Tensor,
    target_gain: Tensor,
    eligible: Tensor,
    relation_type: Tensor,
    *,
    relation_type_count: int = len(EDGE_RELATION_TYPES),
    beta: float = 1.0,
) -> Tensor:
    """Regress continuous gains with equal weight across available relation types."""

    if predicted_gain.shape != target_gain.shape:
        raise ValueError("predicted_gain and target_gain must have the same shape")
    if predicted_gain.shape != eligible.shape or predicted_gain.shape != relation_type.shape:
        raise ValueError("edge-gain metadata must match predicted_gain")
    if not predicted_gain.is_floating_point() or not target_gain.is_floating_point():
        raise ValueError("edge gains must be floating-point")
    if not torch.isfinite(predicted_gain).all() or not torch.isfinite(target_gain).all():
        raise ValueError("edge gains must contain only finite values")
    if (
        type(relation_type_count) is not int
        or relation_type_count <= 0
        or relation_type_count > len(EDGE_RELATION_TYPES)
    ):
        raise ValueError("relation_type_count must select at least one known relation type")
    if not torch.isfinite(torch.as_tensor(beta)) or beta <= 0.0:
        raise ValueError("beta must be positive and finite")
    if relation_type.dtype not in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise ValueError("relation_type must be an integer tensor")
    if torch.any((relation_type < 0) | (relation_type >= relation_type_count)):
        raise ValueError("relation_type contains an unknown relation index")

    valid = eligible.to(device=predicted_gain.device, dtype=torch.bool)
    per_edge = F.smooth_l1_loss(
        predicted_gain,
        target_gain.detach(),
        beta=float(beta),
        reduction="none",
    )
    type_losses: list[Tensor] = []
    for type_index in range(relation_type_count):
        selected = valid & relation_type.eq(type_index)
        if selected.any():
            type_losses.append(per_edge[selected].mean())
    if not type_losses:
        return predicted_gain.sum() * 0.0
    return torch.stack(type_losses).mean()


__all__ = [
    "EDGE_RELATION_TYPES",
    "balanced_edge_gain_loss",
    "counterfactual_relative_gain",
    "edge_gain_admission",
    "masked_patch_huber_error",
    "relative_edge_gain",
]
