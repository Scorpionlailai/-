"""Common output protocol for historical, temporal, and variable branches."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import torch
from torch import Tensor


def _all_finite(values: Sequence[Tensor]) -> bool:
    """Validate same-device tensors with one host read on the normal path."""

    if not values:
        return True
    if any(value.device != values[0].device for value in values[1:]):
        return all(bool(torch.isfinite(value).all()) for value in values)
    return bool(torch.stack(tuple(torch.isfinite(value).all() for value in values)).all())


@dataclass(frozen=True)
class WeightedAggregationPlan:
    """Reusable live weights and membership for a weighted relation message."""

    coefficients: Tensor
    selected: Tensor
    available: Tensor

    def __post_init__(self) -> None:
        if self.coefficients.shape != self.selected.shape:
            raise ValueError("weighted plan coefficients and selection must match")
        if self.selected.dtype != torch.bool or self.available.dtype != torch.bool:
            raise TypeError("weighted plan masks must be bool")
        if self.available.shape != self.selected.shape[:-1]:
            raise ValueError("weighted plan availability has invalid shape")
        if not self.coefficients.is_floating_point() or not torch.isfinite(self.coefficients).all():
            raise ValueError("weighted plan coefficients must be finite floating point")

    def detached(self) -> WeightedAggregationPlan:
        """Return this plan with tensors detached for teacher/student replay."""

        return WeightedAggregationPlan(
            coefficients=self.coefficients.detach(),
            selected=self.selected.detach(),
            available=self.available.detach(),
        )


@dataclass(frozen=True)
class TemporalAggregationPlan:
    """Reusable local/global temporal aggregation and teacher summaries."""

    local_coefficients: Tensor
    global_coefficients: Tensor
    local_selected: Tensor
    global_selected: Tensor
    local_mode: Tensor
    local_available: Tensor
    global_available: Tensor
    local_gain: Tensor
    global_gain: Tensor
    local_uncertainty: Tensor
    global_uncertainty: Tensor
    available: Tensor

    def __post_init__(self) -> None:
        coefficient_shape = self.local_coefficients.shape
        if self.global_coefficients.shape != coefficient_shape:
            raise ValueError("temporal plan coefficient shapes must match")
        if self.local_selected.shape != coefficient_shape or self.global_selected.shape != coefficient_shape:
            raise ValueError("temporal plan selection shapes must match coefficients")
        if not self.local_coefficients.is_floating_point() or not self.global_coefficients.is_floating_point():
            raise TypeError("temporal plan coefficients must be floating point")
        if self.local_selected.dtype != torch.bool or self.global_selected.dtype != torch.bool:
            raise TypeError("temporal plan selection masks must be bool")
        leading_shape = coefficient_shape[:-1]
        masks = (self.local_available, self.global_available, self.available)
        if any(value.shape != leading_shape for value in masks):
            raise ValueError("temporal plan availability has invalid shape")
        if any(value.dtype != torch.bool for value in masks):
            raise TypeError("temporal plan availability masks must be bool")
        if self.local_mode.shape != leading_shape:
            raise ValueError("temporal plan local mode has invalid shape")
        if self.local_mode.is_floating_point() or self.local_mode.dtype == torch.bool:
            raise TypeError("temporal plan local mode must be integer")
        summaries = (
            self.local_gain,
            self.global_gain,
            self.local_uncertainty,
            self.global_uncertainty,
        )
        if any(value.shape != leading_shape for value in summaries):
            raise ValueError("temporal plan summaries have invalid shape")
        if any(not value.is_floating_point() for value in summaries):
            raise ValueError("temporal plan summaries must be finite floating point")
        coefficients = (self.local_coefficients, self.global_coefficients)
        if not _all_finite((*coefficients, *summaries)):
            if not _all_finite(coefficients):
                raise ValueError("temporal plan coefficients must be finite")
            raise ValueError("temporal plan summaries must be finite floating point")

    def detached(self) -> TemporalAggregationPlan:
        """Return this plan with tensors detached for teacher/student replay."""

        return TemporalAggregationPlan(
            local_coefficients=self.local_coefficients.detach(),
            global_coefficients=self.global_coefficients.detach(),
            local_selected=self.local_selected.detach(),
            global_selected=self.global_selected.detach(),
            local_mode=self.local_mode.detach(),
            local_available=self.local_available.detach(),
            global_available=self.global_available.detach(),
            local_gain=self.local_gain.detach(),
            global_gain=self.global_gain.detach(),
            local_uncertainty=self.local_uncertainty.detach(),
            global_uncertainty=self.global_uncertainty.detach(),
            available=self.available.detach(),
        )


@dataclass(frozen=True)
class RelationBranchOutput:
    message: Tensor
    predicted_gain: Tensor
    uncertainty: Tensor
    available: Tensor
    candidate_messages: Tensor
    candidate_predicted_gain: Tensor
    candidate_uncertainty: Tensor
    candidate_eligible: Tensor
    candidate_selected: Tensor
    diagnostics: dict[str, Tensor]

    def __post_init__(self) -> None:
        tensor_fields = (
            self.message, self.predicted_gain, self.uncertainty, self.available,
            self.candidate_messages, self.candidate_predicted_gain,
            self.candidate_uncertainty, self.candidate_eligible,
            self.candidate_selected,
        )
        if any(not isinstance(value, Tensor) for value in tensor_fields):
            raise TypeError("branch fields must be tensors")
        if self.message.ndim != 4:
            raise ValueError("message must have shape [B,C,N,D]")
        main = (self.predicted_gain, self.uncertainty, self.available)
        main_shape = self.message.shape[:-1]
        if any(value.shape != main_shape for value in main):
            raise ValueError("branch main tensors must share shape [B,C,N]")
        candidates = (
            self.candidate_predicted_gain,
            self.candidate_uncertainty,
            self.candidate_eligible,
            self.candidate_selected,
        )
        if self.candidate_messages.ndim != 5:
            raise ValueError("candidate_messages must have shape [B,C,N,K,D]")
        if self.candidate_messages.shape[:-1] != (*main_shape, self.candidate_messages.shape[-2]):
            raise ValueError("candidate_messages has invalid leading dimensions")
        candidate_shape = self.candidate_messages.shape[:-1]
        if any(value.shape != candidate_shape for value in candidates):
            raise ValueError("candidate tensors must share shape [B,C,N,K]")
        finite_fields = tuple(
            value for value in (self.message,) + main + (self.candidate_messages,) + candidates
            if value.is_floating_point()
        )
        if not _all_finite(finite_fields):
            raise ValueError("branch floating tensors must be finite")
        floating = (self.message, self.predicted_gain, self.uncertainty,
                    self.candidate_messages, self.candidate_predicted_gain,
                    self.candidate_uncertainty)
        if any(not value.is_floating_point() for value in floating):
            raise TypeError("branch values must be floating-point tensors")
        if self.available.dtype != torch.bool or self.candidate_eligible.dtype != torch.bool or self.candidate_selected.dtype != torch.bool:
            raise ValueError("availability and selection masks must be bool")
        if not isinstance(self.diagnostics, dict) or any(not isinstance(k, str) or not isinstance(v, Tensor) for k, v in self.diagnostics.items()):
            raise TypeError("diagnostics must be a dict[str, Tensor]")


def stack_branch_outputs(
    historical: RelationBranchOutput,
    temporal: RelationBranchOutput,
    variable: RelationBranchOutput,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Stack branch summaries in the protocol's fixed H/T/V order."""

    return (
        torch.stack((historical.message, temporal.message, variable.message), dim=-2),
        torch.stack((historical.predicted_gain, temporal.predicted_gain, variable.predicted_gain), dim=-1),
        torch.stack((historical.uncertainty, temporal.uncertainty, variable.uncertainty), dim=-1),
        torch.stack((historical.available, temporal.available, variable.available), dim=-1),
    )


__all__ = [
    "RelationBranchOutput",
    "TemporalAggregationPlan",
    "WeightedAggregationPlan",
    "stack_branch_outputs",
]
