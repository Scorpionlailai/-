"""Eight-route fusion of precomputed historical, temporal, and variable messages."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .routing_v2 import V2_ROUTE_OPTIONS, route_relation_mask as v2_route_relation_mask


@dataclass(frozen=True)
class BranchRecoveryOutput:
    """Selected recovery and every route-level counterfactual it was selected from."""

    latents: Tensor
    message: Tensor
    residual: Tensor
    weights: Tensor
    active_relations: Tensor
    route_latents: Tensor
    route_messages: Tensor
    route_weights: Tensor


class BranchRoutedResidualRecovery(nn.Module):
    """Fuse target-space branch messages, then recover each admissible route once."""

    def __init__(self, d_model: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        if type(d_model) is not int or d_model <= 0:
            raise ValueError("d_model must be a positive integer")
        hidden = d_model if hidden_dim is None else int(hidden_dim)
        if hidden <= 0:
            raise ValueError("hidden_dim must be positive")
        self.d_model = d_model
        self.residual_mlp = nn.Sequential(
            nn.Linear(2 * d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def _validate_forward_inputs(
        self,
        nodes: Tensor,
        branch_messages: Tensor,
        branch_scores: Tensor,
        branch_available: Tensor,
        route_index: Tensor,
        target_damaged: Tensor,
    ) -> None:
        if nodes.ndim != 4 or nodes.shape[-1] != self.d_model:
            raise ValueError("nodes must have shape [B,C,N,d_model]")
        expected_relations = (*nodes.shape[:-1], 3)
        if branch_messages.shape != (*expected_relations, self.d_model):
            raise ValueError("branch_messages must have shape [B,C,N,3,d_model]")
        if branch_scores.shape != expected_relations:
            raise ValueError("branch_scores must have shape [B,C,N,3]")
        if branch_available.shape != expected_relations:
            raise ValueError("branch_available must have shape [B,C,N,3]")
        if route_index.shape != nodes.shape[:-1]:
            raise ValueError("route_index must have shape [B,C,N]")
        if target_damaged.shape != nodes.shape[:-1]:
            raise ValueError("target_damaged must have shape [B,C,N]")
        integer_dtypes = {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }
        if route_index.dtype not in integer_dtypes:
            raise ValueError("route_index must be an integer tensor")
        if torch.any((route_index < 0) | (route_index >= len(V2_ROUTE_OPTIONS))):
            raise ValueError("route_index contains an unknown route")
        if branch_available.dtype != torch.bool:
            raise ValueError("branch_available must be bool")
        for name, tensor in (
            ("nodes", nodes),
            ("branch_messages", branch_messages),
            ("branch_scores", branch_scores),
        ):
            if not tensor.is_floating_point() or not torch.isfinite(tensor).all():
                raise ValueError(f"{name} must be finite floating-point values")

    def forward(
        self,
        nodes: Tensor,
        branch_messages: Tensor,
        branch_scores: Tensor,
        branch_available: Tensor,
        route_index: Tensor,
        target_damaged: Tensor,
    ) -> BranchRecoveryOutput:
        """Materialize every route counterfactual, then gather the selected route."""

        self._validate_forward_inputs(
            nodes,
            branch_messages,
            branch_scores,
            branch_available,
            route_index,
            target_damaged,
        )
        route_index = route_index.to(device=nodes.device, dtype=torch.long)
        target_damaged = target_damaged.to(device=nodes.device, dtype=torch.bool)
        branch_available = branch_available.to(device=nodes.device)
        route_mask = v2_route_relation_mask(device=nodes.device)
        active = route_mask.view(1, 1, 1, len(V2_ROUTE_OPTIONS), 3) & branch_available.unsqueeze(-2)
        masked_score = branch_scores.unsqueeze(-2).masked_fill(~active, -torch.inf)
        has_active = active.any(dim=-1, keepdim=True)
        safe_score = torch.where(has_active, masked_score, torch.zeros_like(masked_score))
        route_weights = torch.softmax(safe_score, dim=-1)
        route_weights = torch.where(has_active, route_weights, torch.zeros_like(route_weights))
        route_messages = (
            route_weights.unsqueeze(-1) * branch_messages.unsqueeze(-3)
        ).sum(dim=-2)
        route_residuals = self.residual_mlp(
            torch.cat(
                (nodes.unsqueeze(-2).expand_as(route_messages), route_messages),
                dim=-1,
            )
        )
        route_residuals = route_residuals.masked_fill(~has_active, 0.0)
        route_residuals = route_residuals * target_damaged.unsqueeze(-1).unsqueeze(-1)
        route_latents = nodes.unsqueeze(-2) + route_residuals

        latent_index = route_index.unsqueeze(-1).unsqueeze(-1).expand(*route_index.shape, 1, self.d_model)
        relation_index = route_index.unsqueeze(-1).unsqueeze(-1).expand(*route_index.shape, 1, 3)
        return BranchRecoveryOutput(
            latents=route_latents.gather(-2, latent_index).squeeze(-2),
            message=route_messages.gather(-2, latent_index).squeeze(-2),
            residual=route_residuals.gather(-2, latent_index).squeeze(-2),
            weights=route_weights.gather(-2, relation_index).squeeze(-2),
            active_relations=active.gather(-2, relation_index).squeeze(-2),
            route_latents=route_latents,
            route_messages=route_messages,
            route_weights=route_weights,
        )

    def recover_candidates(
        self,
        nodes: Tensor,
        candidate_messages: Tensor,
        target_damaged: Tensor,
    ) -> Tensor:
        """Recover each candidate with the same residual MLP used by route fusion."""

        if nodes.ndim != 4 or nodes.shape[-1] != self.d_model:
            raise ValueError("nodes must have shape [B,C,N,d_model]")
        if (
            candidate_messages.ndim != 5
            or candidate_messages.shape[:3] != nodes.shape[:3]
            or candidate_messages.shape[-1] != self.d_model
        ):
            raise ValueError("candidate_messages must have shape [B,C,N,K,d_model]")
        if target_damaged.shape != nodes.shape[:-1]:
            raise ValueError("target_damaged must have shape [B,C,N]")
        for name, tensor in (("nodes", nodes), ("candidate_messages", candidate_messages)):
            if not tensor.is_floating_point() or not torch.isfinite(tensor).all():
                raise ValueError(f"{name} must be finite floating-point values")

        candidate_residual = self.residual_mlp(
            torch.cat(
                (nodes.unsqueeze(-2).expand_as(candidate_messages), candidate_messages),
                dim=-1,
            )
        )
        candidate_residual = candidate_residual * target_damaged.to(
            device=nodes.device, dtype=torch.bool
        ).unsqueeze(-1).unsqueeze(-1)
        return nodes.unsqueeze(-2) + candidate_residual


__all__ = ["BranchRecoveryOutput", "BranchRoutedResidualRecovery"]
