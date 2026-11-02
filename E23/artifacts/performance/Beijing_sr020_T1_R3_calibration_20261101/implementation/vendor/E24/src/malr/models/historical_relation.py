"""Target-space historical relation operator."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .online_history import HistoricalCandidateBatch
from .relation_outputs import RelationBranchOutput, WeightedAggregationPlan


class HistoricalRelationOperator(nn.Module):
    def __init__(self, d_model: int, feature_dim: int = 11, max_items: int = 4) -> None:
        super().__init__()
        if d_model <= 0 or feature_dim <= 0 or max_items <= 0:
            raise ValueError("d_model, feature_dim, and max_items must be positive")
        self.d_model = d_model
        self.feature_dim = feature_dim
        self.max_items = max_items
        hidden = max(d_model, feature_dim)
        self.message_head = nn.Sequential(
            nn.Linear(d_model + feature_dim, hidden), nn.GELU(), nn.Linear(hidden, d_model)
        )
        self.gain_head = nn.Sequential(
            nn.Linear(2 * d_model + feature_dim, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        self.uncertainty_head = nn.Sequential(
            nn.Linear(2 * d_model + feature_dim, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )

    def _validate(self, target_nodes: Tensor, candidates: HistoricalCandidateBatch) -> None:
        if target_nodes.ndim != 4 or target_nodes.shape[-1] != self.d_model:
            raise ValueError("target_nodes must have shape [B,C,N,D]")
        source = candidates.source_nodes
        features = candidates.features
        if source.ndim != 5 or source.shape[:3] != target_nodes.shape[:3] or source.shape[-1] != self.d_model:
            raise ValueError("source_nodes must have shape [B,C,N,K,D]")
        if features.shape != (*source.shape[:-1], self.feature_dim):
            raise ValueError("features must have shape [B,C,N,K,feature_dim]")
        if candidates.eligible.shape != source.shape[:-1]:
            raise ValueError("eligible must have shape [B,C,N,K]")
        if not torch.isfinite(target_nodes).all() or not torch.isfinite(source).all() or not torch.isfinite(features).all():
            raise ValueError("operator inputs must be finite")

    def project_candidate_messages(
        self, target_nodes: Tensor, candidates: HistoricalCandidateBatch
    ) -> Tensor:
        """Project each historical candidate independently into target space."""

        self._validate(target_nodes, candidates)
        source = candidates.source_nodes
        features = candidates.features
        candidate_input = torch.cat((source, features), dim=-1)
        return self.message_head(candidate_input)

    def replay_message(
        self, candidate_messages: Tensor, plan: WeightedAggregationPlan
    ) -> Tensor:
        """Aggregate projected candidate messages with a previously computed plan."""

        message = (plan.coefficients.unsqueeze(-1) * candidate_messages).sum(dim=-2)
        return torch.where(
            plan.available.unsqueeze(-1), message, torch.zeros_like(message)
        )

    def forward_with_plan(
        self, target_nodes: Tensor, candidates: HistoricalCandidateBatch
    ) -> tuple[RelationBranchOutput, WeightedAggregationPlan]:
        self._validate(target_nodes, candidates)
        source = candidates.source_nodes
        features = candidates.features
        candidate_messages = self.project_candidate_messages(target_nodes, candidates)
        edge_input = torch.cat((target_nodes.unsqueeze(-2).expand_as(candidate_messages), candidate_messages, features), dim=-1)
        predicted_gain = self.gain_head(edge_input).squeeze(-1)
        candidate_uncertainty = F.softplus(self.uncertainty_head(edge_input).squeeze(-1)) + 1e-6
        eligible = candidates.eligible.to(device=target_nodes.device, dtype=torch.bool)
        admissible = eligible & predicted_gain.gt(0.0)
        score = predicted_gain - candidate_uncertainty
        width = score.shape[-1]
        take = min(self.max_items, width)
        selected = torch.zeros_like(admissible)
        if take:
            order = torch.argsort(torch.where(admissible, score, torch.full_like(score, -torch.inf)), dim=-1, descending=True, stable=True)
            chosen = order[..., :take]
            chosen_valid = admissible.gather(-1, chosen)
            selected.scatter_(-1, chosen, chosen_valid)
        weights = torch.where(selected, score, torch.full_like(score, -torch.inf))
        weights = torch.softmax(weights, dim=-1).masked_fill(~selected, 0.0)
        gain = (weights * predicted_gain).sum(dim=-1)
        uncertainty = (weights * candidate_uncertainty).sum(dim=-1)
        available = selected.any(dim=-1)
        plan = WeightedAggregationPlan(weights, selected, available)
        message = self.replay_message(candidate_messages, plan)
        gain = torch.where(available, gain, torch.zeros_like(gain))
        uncertainty = torch.where(available, uncertainty, torch.ones_like(uncertainty))
        output = RelationBranchOutput(
            message=message,
            predicted_gain=gain,
            uncertainty=uncertainty,
            available=available,
            candidate_messages=candidate_messages,
            candidate_predicted_gain=predicted_gain,
            candidate_uncertainty=candidate_uncertainty,
            candidate_eligible=eligible,
            candidate_selected=selected,
            diagnostics={},
        )
        return output, plan

    def forward(self, target_nodes: Tensor, candidates: HistoricalCandidateBatch) -> RelationBranchOutput:
        output, _ = self.forward_with_plan(target_nodes, candidates)
        return output


__all__ = ["HistoricalRelationOperator"]
