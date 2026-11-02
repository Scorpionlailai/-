"""One-layer relation-aware message passing for routed residual recovery."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .routing import ROUTE_OPTIONS, route_relation_mask


@dataclass(frozen=True)
class GraphRecoveryOutput:
    latents: Tensor
    message: Tensor
    residual: Tensor
    weights: Tensor
    active_edges: Tensor


class RelationAwareResidualRecovery(nn.Module):
    """Aggregate routed edges once, then predict a residual correction."""

    def __init__(
        self,
        d_model: int,
        edge_feature_dim: int,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if type(d_model) is not int or d_model <= 0:
            raise ValueError("d_model must be a positive integer")
        if type(edge_feature_dim) is not int or edge_feature_dim <= 0:
            raise ValueError("edge_feature_dim must be a positive integer")
        hidden = d_model if hidden_dim is None else int(hidden_dim)
        if hidden <= 0:
            raise ValueError("hidden_dim must be positive")
        self.d_model = d_model
        self.edge_feature_dim = edge_feature_dim
        self.message_functions = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model + edge_feature_dim, hidden),
                    nn.GELU(),
                    nn.Linear(hidden, d_model),
                )
                for _ in range(3)
            ]
        )
        self.residual_mlp = nn.Sequential(
            nn.Linear(2 * d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def _validate(
        self,
        nodes: Tensor,
        source_nodes: Tensor,
        edge_features: Tensor,
        edge_type: Tensor,
        predicted_gain: Tensor,
        admitted: Tensor,
        route_index: Tensor,
        target_damaged: Tensor,
        self_edge: Tensor | None,
    ) -> None:
        if nodes.ndim != 4 or nodes.shape[-1] != self.d_model:
            raise ValueError("nodes must have shape [batch, variables, patches, d_model]")
        if (
            source_nodes.ndim != 5
            or source_nodes.shape[:3] != nodes.shape[:3]
            or source_nodes.shape[-1] != self.d_model
        ):
            raise ValueError("source_nodes must have shape [B,C,N,K,d_model]")
        expected_edges = (*nodes.shape[:3], source_nodes.shape[-2])
        if edge_features.shape != (*expected_edges, self.edge_feature_dim):
            raise ValueError("edge_features must have shape [B,C,N,K,edge_feature_dim]")
        if edge_type.shape != expected_edges or predicted_gain.shape != expected_edges or admitted.shape != expected_edges:
            raise ValueError("edge metadata must have shape [B,C,N,K]")
        if route_index.shape != nodes.shape[:-1] or target_damaged.shape != route_index.shape:
            raise ValueError("route_index and target_damaged must match node dimensions")
        integer_dtypes = {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }
        if edge_type.dtype not in integer_dtypes or route_index.dtype not in integer_dtypes:
            raise ValueError("edge_type and route_index must be integer tensors")
        if torch.any((edge_type < 0) | (edge_type >= 3)):
            raise ValueError("edge_type contains an unknown relation")
        if torch.any((route_index < 0) | (route_index >= len(ROUTE_OPTIONS))):
            raise ValueError("route_index contains an unknown route")
        for name, tensor in (
            ("nodes", nodes),
            ("source_nodes", source_nodes),
            ("edge_features", edge_features),
            ("predicted_gain", predicted_gain),
        ):
            if not tensor.is_floating_point() or not torch.isfinite(tensor).all():
                raise ValueError(f"{name} must be finite floating-point values")
        if self_edge is not None:
            if self_edge.shape != expected_edges:
                raise ValueError("self_edge must have shape [B,C,N,K]")
            if self_edge.to(dtype=torch.bool).any():
                raise ValueError("self-loop edges are not allowed")

    def forward(
        self,
        nodes: Tensor,
        source_nodes: Tensor,
        edge_features: Tensor,
        edge_type: Tensor,
        predicted_gain: Tensor,
        admitted: Tensor,
        route_index: Tensor,
        target_damaged: Tensor,
        self_edge: Tensor | None = None,
    ) -> GraphRecoveryOutput:
        self._validate(
            nodes,
            source_nodes,
            edge_features,
            edge_type,
            predicted_gain,
            admitted,
            route_index,
            target_damaged,
            self_edge,
        )
        edge_type = edge_type.to(device=nodes.device, dtype=torch.long)
        route_index = route_index.to(device=nodes.device, dtype=torch.long)
        admitted = admitted.to(device=nodes.device, dtype=torch.bool)
        target_damaged = target_damaged.to(device=nodes.device, dtype=torch.bool)
        selected_relations = route_relation_mask(device=nodes.device).index_select(
            0, route_index.reshape(-1)
        ).reshape(*route_index.shape, 3)
        relation_table = selected_relations.unsqueeze(-2).expand(
            *edge_type.shape, 3
        )
        type_allowed = relation_table.gather(
            -1, edge_type.clamp_min(0).unsqueeze(-1)
        ).squeeze(-1)
        active_edges = admitted & type_allowed & target_damaged.unsqueeze(-1)

        batch_size, variable_count, patch_count, d_model = nodes.shape
        edge_width = source_nodes.shape[-2]
        if edge_width == 0:
            weights = predicted_gain.new_zeros((*predicted_gain.shape,))
        else:
            masked_gain = predicted_gain.masked_fill(~active_edges, -torch.inf)
            has_active = active_edges.any(dim=-1, keepdim=True)
            maximum = masked_gain.max(dim=-1, keepdim=True).values
            maximum = torch.where(has_active, maximum, torch.zeros_like(maximum))
            numerator = torch.exp(masked_gain - maximum).masked_fill(
                ~active_edges, 0.0
            )
            weights = numerator / numerator.sum(dim=-1, keepdim=True).clamp_min(
                torch.finfo(predicted_gain.dtype).tiny
            )
            weights = torch.where(has_active, weights, torch.zeros_like(weights))

        edge_input = torch.cat((source_nodes, edge_features), dim=-1)
        messages = source_nodes.new_zeros(source_nodes.shape)
        for relation_index, message_function in enumerate(self.message_functions):
            transformed = message_function(edge_input.reshape(-1, d_model + self.edge_feature_dim))
            transformed = transformed.reshape(*source_nodes.shape[:-1], d_model)
            relation_mask = edge_type.eq(relation_index).unsqueeze(-1)
            messages = torch.where(relation_mask, transformed, messages)
        message = (weights.unsqueeze(-1) * messages).sum(dim=-2)
        residual_candidate = self.residual_mlp(torch.cat((nodes, message), dim=-1))
        has_active_target = active_edges.any(dim=-1).unsqueeze(-1)
        residual = torch.where(has_active_target, residual_candidate, torch.zeros_like(residual_candidate))
        latents = nodes + residual
        return GraphRecoveryOutput(
            latents=latents,
            message=message,
            residual=residual,
            weights=weights,
            active_edges=active_edges,
        )


__all__ = ["GraphRecoveryOutput", "RelationAwareResidualRecovery"]
