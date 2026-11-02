from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class BilateralContextOutput:
    query: Tensor
    left: Tensor
    right: Tensor
    left_available: Tensor
    right_available: Tensor


class BilateralContextEncoder(nn.Module):
    def __init__(self, d_model: int, hidden_multiplier: int = 2) -> None:
        super().__init__()
        if type(d_model) is not int or d_model <= 0:
            raise ValueError("d_model must be a positive integer")
        if type(hidden_multiplier) is not int or hidden_multiplier <= 0:
            raise ValueError("hidden_multiplier must be a positive integer")
        hidden = hidden_multiplier * d_model
        self.network = nn.Sequential(
            nn.LayerNorm(5 * d_model + 3),
            nn.Linear(5 * d_model + 3, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def forward(
        self,
        visible: Tensor,
        mask_embedding: Tensor,
        prior: Tensor,
        rho: Tensor,
        target_eligible: Tensor,
    ) -> BilateralContextOutput:
        if visible.ndim != 4:
            raise ValueError("visible must have shape [B,C,N,D]")
        if mask_embedding.shape != visible.shape:
            raise ValueError("mask_embedding must match visible")
        if prior.shape != visible.shape:
            raise ValueError("prior must match visible")
        if rho.shape != visible.shape[:-1]:
            raise ValueError("rho must have shape [B,C,N]")
        if target_eligible.shape != rho.shape or target_eligible.dtype != torch.bool:
            raise ValueError(
                "target_eligible must be Boolean with shape [B,C,N]"
            )
        if not torch.isfinite(rho).all():
            raise ValueError("rho must be finite")
        if not ((rho >= 0.0) & (rho <= 1.0)).all():
            raise ValueError("rho must lie in [0, 1]")

        patch_count = visible.shape[-2]
        index = torch.arange(patch_count, device=visible.device)
        signed_distance = index.view(1, -1) - index.view(-1, 1)
        distance = signed_distance.abs().to(dtype=visible.dtype)
        source_weight = rho.unsqueeze(-2) / (
            1.0 + distance.view(1, 1, patch_count, patch_count)
        )
        source_visible = (rho > 0).unsqueeze(-2)
        left_mask = source_visible & (signed_distance < 0).view(
            1, 1, patch_count, patch_count
        )
        right_mask = source_visible & (signed_distance > 0).view(
            1, 1, patch_count, patch_count
        )

        def summarize(side_mask: Tensor) -> tuple[Tensor, Tensor]:
            weights = torch.where(
                side_mask, source_weight, torch.zeros_like(source_weight)
            )
            denominator = weights.sum(dim=-1, keepdim=True)
            total = torch.einsum("bcij,bcjd->bcid", weights, visible)
            available = denominator.squeeze(-1) > 0
            normalized = total / denominator.clamp_min(
                torch.finfo(visible.dtype).eps
            )
            summary = torch.where(
                available.unsqueeze(-1), normalized, torch.zeros_like(total)
            )
            return summary, available

        left, left_available = summarize(left_mask)
        right, right_available = summarize(right_mask)
        features = torch.cat(
            (
                visible,
                mask_embedding,
                prior,
                left,
                right,
                rho.unsqueeze(-1),
                left_available.unsqueeze(-1).to(visible.dtype),
                right_available.unsqueeze(-1).to(visible.dtype),
            ),
            dim=-1,
        )
        query = self.network(features)
        query = torch.where(
            target_eligible.unsqueeze(-1), query, torch.zeros_like(query)
        )
        return BilateralContextOutput(
            query=query,
            left=left,
            right=right,
            left_available=left_available,
            right_available=right_available,
        )


@dataclass(frozen=True)
class FilteredEdgeSelection:
    weights: Tensor
    probabilities: Tensor
    hard_gate: Tensor
    selected: Tensor
    candidate_count: Tensor
    retained_count: Tensor
    keep_loss: Tensor


def hard_threshold_topk(
    probabilities: Tensor,
    base_scores: Tensor,
    eligible: Tensor,
    *,
    threshold: float,
    max_edges: int,
) -> tuple[Tensor, Tensor]:
    """Threshold first, then rank by confidence, base score, and index."""

    if probabilities.ndim < 1 or probabilities.shape != base_scores.shape:
        raise ValueError("probabilities and base_scores must have matching shapes")
    if eligible.shape != probabilities.shape:
        raise ValueError("eligible must match probabilities")
    if not 0.0 < float(threshold) < 1.0:
        raise ValueError("threshold must lie in (0, 1)")
    if type(max_edges) is not int or max_edges <= 0:
        raise ValueError("max_edges must be a positive integer")

    candidate_width = probabilities.shape[-1]
    output_shape = (*probabilities.shape[:-1], max_edges)
    if candidate_width == 0:
        return (
            torch.zeros_like(eligible, dtype=torch.bool),
            torch.full(
                output_shape,
                -1,
                dtype=torch.long,
                device=probabilities.device,
            ),
        )

    eligible = eligible.to(device=probabilities.device, dtype=torch.bool)
    accepted = eligible & (probabilities >= float(threshold))
    negative_infinity = torch.full_like(probabilities, -torch.inf)
    ranked_probability = torch.where(
        accepted, probabilities, negative_infinity
    )
    ranked_base = torch.where(accepted, base_scores, negative_infinity)

    base_order = torch.argsort(
        ranked_base, dim=-1, descending=True, stable=True
    )
    probability_in_base_order = ranked_probability.gather(-1, base_order)
    probability_order = torch.argsort(
        probability_in_base_order, dim=-1, descending=True, stable=True
    )
    order = base_order.gather(-1, probability_order)

    width = min(max_edges, candidate_width)
    chosen_raw = order[..., :width]
    chosen_valid = accepted.gather(-1, chosen_raw)
    chosen = torch.where(
        chosen_valid, chosen_raw, torch.full_like(chosen_raw, -1)
    )
    selected = torch.full(
        output_shape,
        -1,
        dtype=torch.long,
        device=probabilities.device,
    )
    selected[..., :width] = chosen

    gate_counts = torch.zeros_like(probabilities, dtype=torch.long)
    gate_counts.scatter_add_(
        -1,
        chosen_raw,
        chosen_valid.to(dtype=torch.long),
    )
    return gate_counts > 0, selected


class EdgeConfidenceFilter(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        auxiliary_dim: int,
        max_edges: int,
        threshold: float,
        hidden_multiplier: int,
        dropout: float,
        temperature: float,
        initial_keep_probability: float,
    ) -> None:
        super().__init__()
        if type(d_model) is not int or d_model <= 0:
            raise ValueError("d_model must be a positive integer")
        if type(auxiliary_dim) is not int or auxiliary_dim < 0:
            raise ValueError("auxiliary_dim must be a non-negative integer")
        if type(max_edges) is not int or max_edges <= 0:
            raise ValueError("max_edges must be a positive integer")
        if not 0.0 < float(threshold) < 1.0:
            raise ValueError("threshold must lie in (0, 1)")
        if type(hidden_multiplier) is not int or hidden_multiplier <= 0:
            raise ValueError("hidden_multiplier must be a positive integer")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if float(temperature) <= 0.0:
            raise ValueError("temperature must be positive")
        if not 0.0 < float(initial_keep_probability) < 1.0:
            raise ValueError("initial_keep_probability must lie in (0, 1)")

        self.d_model = d_model
        self.auxiliary_dim = auxiliary_dim
        self.max_edges = max_edges
        self.threshold = float(threshold)
        self.temperature = float(temperature)
        input_dim = 4 * d_model + auxiliary_dim
        hidden_dim = hidden_multiplier * d_model
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        initial_logit = torch.logit(
            torch.tensor(float(initial_keep_probability))
        ).item()
        nn.init.constant_(self.network[-1].bias, initial_logit)

    def forward(
        self,
        query: Tensor,
        keys: Tensor,
        auxiliary: Tensor,
        base_scores: Tensor,
        eligible: Tensor,
    ) -> FilteredEdgeSelection:
        if query.ndim < 1 or query.shape[-1] != self.d_model:
            raise ValueError("query must end in d_model")
        if keys.ndim != query.ndim + 1:
            raise ValueError("keys must add one candidate axis to query")
        if keys.shape[:-2] != query.shape[:-1] or keys.shape[-1] != self.d_model:
            raise ValueError("keys must match query node axes and d_model")
        candidate_shape = keys.shape[:-1]
        if auxiliary.shape != (*candidate_shape, self.auxiliary_dim):
            raise ValueError("auxiliary features have an invalid shape")
        if base_scores.shape != candidate_shape or eligible.shape != candidate_shape:
            raise ValueError("base_scores and eligible must match candidate axes")

        expanded_query = query.unsqueeze(-2).expand_as(keys)
        pair_features = torch.cat(
            (
                expanded_query,
                keys,
                (expanded_query - keys).abs(),
                expanded_query * keys,
                auxiliary,
            ),
            dim=-1,
        )
        logits = self.network(pair_features).squeeze(-1)
        eligible = eligible.to(device=logits.device, dtype=torch.bool)
        raw_probabilities = torch.sigmoid(logits)
        probabilities = torch.where(
            eligible, raw_probabilities, torch.zeros_like(raw_probabilities)
        )
        hard_gate, selected = hard_threshold_topk(
            probabilities,
            base_scores,
            eligible,
            threshold=self.threshold,
            max_edges=self.max_edges,
        )

        threshold_logit = torch.logit(
            logits.new_tensor(self.threshold)
        )
        soft_gate = torch.sigmoid(
            (logits - threshold_logit) / self.temperature
        )
        soft_gate = torch.where(eligible, soft_gate, torch.zeros_like(soft_gate))
        if self.training:
            gate = hard_gate.to(logits.dtype).detach() - soft_gate.detach() + soft_gate
        else:
            gate = hard_gate.to(logits.dtype)

        eligible_scores = torch.where(
            eligible, base_scores, torch.full_like(base_scores, -torch.inf)
        )
        maximum = eligible_scores.max(dim=-1, keepdim=True).values
        maximum = torch.where(
            torch.isfinite(maximum), maximum, torch.zeros_like(maximum)
        )
        exponentiated = torch.where(
            eligible,
            torch.exp(base_scores - maximum),
            torch.zeros_like(base_scores),
        )
        weighted_gate = gate * exponentiated
        denominator = weighted_gate.sum(dim=-1, keepdim=True)
        no_retained_edge = ~hard_gate.any(dim=-1, keepdim=True)
        safe_denominator = torch.where(
            no_retained_edge, torch.ones_like(denominator), denominator
        )
        weights = weighted_gate / safe_denominator

        candidate_count = eligible.sum(dim=-1)
        retained_count = hard_gate.sum(dim=-1)
        eligible_total = candidate_count.sum()
        if bool(eligible_total > 0):
            keep_loss = probabilities.sum() / eligible_total.to(probabilities.dtype)
        else:
            keep_loss = probabilities.sum() * 0.0
        return FilteredEdgeSelection(
            weights=weights,
            probabilities=probabilities,
            hard_gate=hard_gate,
            selected=selected,
            candidate_count=candidate_count,
            retained_count=retained_count,
            keep_loss=keep_loss,
        )


class MissingAwareEdgeFilter(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        temporal_topk: int,
        variable_topk: int,
        historical_topk: int,
        temporal_threshold: float,
        variable_threshold: float,
        historical_threshold: float,
        hidden_multiplier: int,
        dropout: float,
        temperature: float,
        initial_keep_probability: float,
    ) -> None:
        super().__init__()
        common = {
            "d_model": d_model,
            "hidden_multiplier": hidden_multiplier,
            "dropout": dropout,
            "temperature": temperature,
            "initial_keep_probability": initial_keep_probability,
        }
        self.context = BilateralContextEncoder(d_model, hidden_multiplier)
        self.temporal = EdgeConfidenceFilter(
            auxiliary_dim=2,
            max_edges=temporal_topk,
            threshold=temporal_threshold,
            **common,
        )
        self.variable = EdgeConfidenceFilter(
            auxiliary_dim=2 * d_model + 1,
            max_edges=variable_topk,
            threshold=variable_threshold,
            **common,
        )
        self.historical = EdgeConfidenceFilter(
            auxiliary_dim=3,
            max_edges=historical_topk,
            threshold=historical_threshold,
            **common,
        )

    def context_query(
        self,
        visible: Tensor,
        mask_embedding: Tensor,
        prior: Tensor,
        rho: Tensor,
        target_eligible: Tensor,
    ) -> BilateralContextOutput:
        return self.context(
            visible,
            mask_embedding,
            prior,
            rho,
            target_eligible,
        )
