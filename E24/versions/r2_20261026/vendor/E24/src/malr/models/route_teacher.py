"""Immutable Stage-1 relation-scoring snapshot for forecast fine-tuning."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from .candidates_v2 import TemporalCandidateBatch, VariableCandidateBatch
from .historical_relation import HistoricalRelationOperator
from .online_history import HistoricalCandidateBatch
from .relation_outputs import (
    RelationBranchOutput,
    TemporalAggregationPlan,
    WeightedAggregationPlan,
)
from .temporal_relation import DynamicTemporalRelation
from .variable_relation import DirectedVariableRelation, VariableRelationStatistics

if TYPE_CHECKING:
    from .online_stage1_v2 import RelationV2OnlineStage1Model


class FrozenRouteTeacher(nn.Module):
    """A self-contained, serialized copy of the Stage-1 route-producing modules."""

    def __init__(
        self,
        patch_projection: nn.Linear,
        historical_operator: HistoricalRelationOperator,
        temporal_operator: DynamicTemporalRelation,
        variable_operator: DirectedVariableRelation,
    ) -> None:
        super().__init__()
        self.patch_projection = patch_projection
        self.historical_operator = historical_operator
        self.temporal_operator = temporal_operator
        self.variable_operator = variable_operator
        self.requires_grad_(False)
        self.eval()

    @classmethod
    def from_stage1(cls, model: RelationV2OnlineStage1Model) -> FrozenRouteTeacher:
        """Deep-copy every learned tensor that can influence Stage-1 routing."""

        teacher = cls(
            deepcopy(model.core.W_P),
            deepcopy(model.historical_operator),
            deepcopy(model.temporal_operator),
            deepcopy(model.variable_operator),
        )
        if any(parameter.requires_grad for parameter in teacher.parameters()):
            raise RuntimeError("frozen route teacher contains a trainable parameter")
        return teacher

    @property
    def W_P(self) -> nn.Linear:
        """Expose the projection protocol consumed by historical candidate building."""

        return self.patch_projection

    def project_clean_patches(self, patches: Tensor) -> Tensor:
        return self.patch_projection(patches)

    def relation_plans(
        self,
        base_latents: Tensor,
        historical_candidates: HistoricalCandidateBatch,
        temporal_candidates: TemporalCandidateBatch,
        variable_candidates: VariableCandidateBatch,
        rho: Tensor,
        variable_statistics: VariableRelationStatistics,
    ) -> tuple[
        RelationBranchOutput,
        WeightedAggregationPlan,
        RelationBranchOutput,
        TemporalAggregationPlan,
        RelationBranchOutput,
        WeightedAggregationPlan,
    ]:
        """Score all three relation branches and return replayable plans."""

        historical, historical_plan = self.historical_operator.forward_with_plan(
            base_latents, historical_candidates
        )
        temporal, temporal_plan = self.temporal_operator.forward_with_plan(
            base_latents, temporal_candidates, rho
        )
        variable, variable_plan = self.variable_operator.forward_with_plan(
            base_latents, variable_candidates, variable_statistics
        )
        return (
            historical,
            historical_plan,
            temporal,
            temporal_plan,
            variable,
            variable_plan,
        )

    def state_sha256(self) -> str:
        """Hash names and exact tensor representation in deterministic key order."""

        digest = sha256()
        for name, value in sorted(self.state_dict().items()):
            tensor = value.detach().cpu()
            if tensor.layout != torch.strided:
                raise ValueError("teacher state tensors must be dense")
            if (tensor.is_floating_point() or tensor.is_complex()) and not torch.isfinite(
                tensor
            ).all():
                raise ValueError("teacher state tensors must be finite")
            contiguous = tensor.contiguous()
            metadata = f"{name}\0{contiguous.dtype}\0{tuple(contiguous.shape)}\0"
            digest.update(metadata.encode("utf-8"))
            digest.update(contiguous.view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def train(self, mode: bool = True) -> FrozenRouteTeacher:
        """Remain in deterministic evaluation mode under every parent mode change."""

        super().train(False)
        return self


__all__ = ["FrozenRouteTeacher"]
