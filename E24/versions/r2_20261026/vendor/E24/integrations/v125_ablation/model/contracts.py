"""Typed patch-level evidence shared by the E23 experts and router."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from torch import Tensor


@dataclass(frozen=True)
class ExpertInputs:
    base_latents: Tensor  # [B,C,N,D]
    patch_values: Tensor  # [B,C,N,P], observed-only values
    patch_mask: Tensor  # [B,C,N,P], original observations
    patch_validity: Tensor  # [B,C,N,P]
    rho: Tensor  # [B,C,N]
    patch_starts: Tensor  # [B,N] or [B,C,N]
    query_mean: Tensor  # [B,C]
    query_std: Tensor  # [B,C]
    damaged: Tensor  # [B,C,N]


@dataclass(frozen=True)
class CandidateEvidence:
    messages: Tensor  # [B,C,N,K,D], differentiable
    quality: Tensor  # [B,C,N,K,Q], differentiable where applicable
    weights: Tensor  # [B,C,N,K], original expert attention
    valid: Tensor  # [B,C,N,K]


@dataclass(frozen=True)
class ExpertEvidence:
    message: Tensor  # [B,C,N,D]
    quality: Tensor  # [B,C,N,D]
    available: Tensor  # [B,C,N]
    provenance: Mapping[str, Any]
    candidates: CandidateEvidence | None = None


@dataclass(frozen=True)
class RouterOutput:
    latents: Tensor  # [B,C,N,D]
    weights: Tensor  # [B,C,N,4] ordered none,H,V,T
    deltas: Mapping[str, Tensor]
    available: Tensor  # [B,C,N,4], same order
    head_weights: Tensor | None = None  # [B,C,N,H,4]
    candidate_attention: Mapping[str, Tensor] | None = None  # [B,C,N,H,K]
    head_gate_weights: Tensor | None = None  # [B,C,N,H+1] none, then heads
    head_deltas: Tensor | None = None  # [B,C,N,H,D] before head gate
    point_weights: Tensor | None = None  # [B,C,N,P,4], numerical repair only


__all__ = ["ExpertInputs", "CandidateEvidence", "ExpertEvidence", "RouterOutput"]
