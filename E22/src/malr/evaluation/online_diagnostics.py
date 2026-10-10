"""Read-only aggregates for relation-V2 online evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite

import torch
from torch import Tensor

from malr.models.routing_v2 import V2_ROUTE_OPTIONS


_RELATIONS = ("historical", "temporal", "variable")
_REGRET_TOLERANCE = 1.0e-7


class OnlineDiagnosticAccumulator:
    """Accumulate target-masked relation and route outcomes across batches."""

    def __init__(self) -> None:
        self._totals: dict[str, float] = {}
        self._counts: dict[str, float] = {}
        self._damaged = 0.0

    @staticmethod
    def _target_mask(values: Tensor, damaged: Tensor) -> Tensor:
        if values.shape[: damaged.ndim] != damaged.shape:
            raise ValueError("diagnostic tensors must begin with damaged target dimensions")
        return damaged.to(device=values.device, dtype=torch.bool).reshape(
            *damaged.shape, *((1,) * (values.ndim - damaged.ndim))
        ).expand_as(values)

    def _mean(self, key: str, values: Tensor, mask: Tensor) -> None:
        selected = values[mask]
        if selected.numel() == 0:
            return
        self._totals[key] = self._totals.get(key, 0.0) + float(selected.sum().detach().cpu())
        self._counts[key] = self._counts.get(key, 0.0) + float(selected.numel())

    def _rate(self, key: str, selected: Tensor, denominator: Tensor) -> None:
        selected_mask = selected.to(device=denominator.device, dtype=torch.bool)
        if selected_mask.shape != denominator.shape:
            raise ValueError("diagnostic rate tensors must have matching shapes")
        self._totals[key] = self._totals.get(key, 0.0) + float(
            (selected_mask & denominator).sum().detach().cpu()
        )
        self._counts[key] = self._counts.get(key, 0.0) + float(denominator.sum().detach().cpu())

    def update(
        self,
        *,
        relation_candidate: Mapping[str, Tensor],
        relation_selected: Mapping[str, Tensor],
        relation_predicted_gain: Mapping[str, Tensor],
        relation_actual_gain: Mapping[str, Tensor],
        temporal_local_selected: Tensor,
        temporal_global_selected: Tensor,
        route_index: Tensor,
        route_actual_gain: Tensor,
        oracle_route_index: Tensor,
        route_repair_regret: Tensor,
        damaged: Tensor,
    ) -> None:
        if damaged.dtype != torch.bool or route_index.shape != damaged.shape:
            raise ValueError("damaged and route_index must share shape [B,C,N]")
        if route_actual_gain.shape != damaged.shape or oracle_route_index.shape != damaged.shape:
            raise ValueError("route gains and oracle indices must match route_index")
        if route_repair_regret.shape != damaged.shape:
            raise ValueError("route repair regret must match route_index")
        if route_index.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64}:
            raise ValueError("route_index must be integral")
        if not route_repair_regret.is_floating_point():
            raise TypeError("route repair regret must be floating-point")
        if not torch.isfinite(route_repair_regret).all():
            raise ValueError("route repair regret must contain only finite values")
        if torch.any(route_repair_regret < -_REGRET_TOLERANCE):
            raise ValueError("route repair regret must be non-negative")
        self._damaged += float(damaged.sum().detach().cpu())
        for relation in _RELATIONS:
            try:
                candidate = relation_candidate[relation]
                selected = relation_selected[relation]
                predicted = relation_predicted_gain[relation]
                actual = relation_actual_gain[relation]
            except KeyError as error:
                raise ValueError(f"missing {relation} diagnostic tensors") from error
            if not (candidate.shape == selected.shape == predicted.shape == actual.shape):
                raise ValueError("relation diagnostic tensors must share shape")
            target = self._target_mask(candidate, damaged)
            candidate_mask = candidate.to(dtype=torch.bool) & target
            selected_mask = selected.to(dtype=torch.bool) & target
            self._rate(f"{relation}_candidate_rate", candidate.to(dtype=torch.bool), target)
            self._rate(f"{relation}_selected_rate", selected.to(dtype=torch.bool), target)
            self._rate(f"{relation}_zero_rate", ~selected.to(dtype=torch.bool), candidate_mask)
            self._mean(f"{relation}_predicted_gain", predicted, candidate_mask)
            self._mean(f"{relation}_actual_gain", actual, candidate_mask)

        temporal_target = self._target_mask(temporal_local_selected, damaged)
        self._rate("temporal_local_use_rate", temporal_local_selected, temporal_target)
        self._rate("temporal_global_use_rate", temporal_global_selected, temporal_target)
        target = damaged.to(device=route_index.device, dtype=torch.bool)
        for route_index_value, route_name in enumerate(V2_ROUTE_OPTIONS):
            chosen = route_index.eq(route_index_value)
            self._rate(f"route_{route_name}_rate", chosen, target)
            self._mean(f"route_{route_name}_actual_gain", route_actual_gain, chosen & target)
        self._rate("route_selection_error_rate", route_index.ne(oracle_route_index), target)
        regret = route_repair_regret.clamp_min(0.0)
        regret_target = damaged.to(device=regret.device, dtype=torch.bool)
        self._mean("route_repair_regret", regret, regret_target)
        self._rate(
            "route_repair_regret_positive_rate",
            regret.gt(_REGRET_TOLERANCE),
            regret_target,
        )

    def compute(self) -> dict[str, float]:
        keys = set(self._totals) | set(self._counts)
        result = {
            key: (self._totals.get(key, 0.0) / self._counts[key] if self._counts.get(key, 0.0) else 0.0)
            for key in keys
        }
        # Emit all contractual keys even when a relation has no candidates.
        for relation in _RELATIONS:
            for suffix in ("candidate_rate", "selected_rate", "zero_rate", "predicted_gain", "actual_gain"):
                result.setdefault(f"{relation}_{suffix}", 0.0)
        for route in V2_ROUTE_OPTIONS:
            result.setdefault(f"route_{route}_rate", 0.0)
            result.setdefault(f"route_{route}_actual_gain", 0.0)
        result.setdefault("route_repair_regret", 0.0)
        result.setdefault("route_repair_regret_positive_rate", 0.0)
        for key, value in result.items():
            if not key or not isfinite(value):
                raise ValueError("online diagnostics must be finite and named")
        return dict(sorted(result.items()))


__all__ = ["OnlineDiagnosticAccumulator"]
