from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from malr.data.patching import PATCH_MISSING, PATCH_OBSERVED, PATCH_PARTIAL

from .relations import HistoricalRelation, TemporalRelation, VariableRelation


@dataclass(frozen=True)
class RelationMessages:
    temporal: Tensor
    variable: Tensor
    historical: Tensor


@dataclass(frozen=True)
class RecoveryOutput:
    latents: Tensor
    visible: Tensor
    prior: Tensor
    recovered: Tensor
    messages: RelationMessages
    diagnostics: dict[str, Tensor]


class MissingAwareRecovery(nn.Module):
    def __init__(
        self,
        c_in: int,
        patch_num: int,
        patch_len: int,
        d_model: int,
        *,
        temporal_topk: int = 6,
        variable_topk: int = 4,
        historical_relation: HistoricalRelation | None = None,
        use_temporal: bool = True,
        use_variable: bool = True,
        use_historical: bool = True,
    ) -> None:
        super().__init__()
        if min(c_in, patch_num, patch_len, d_model) <= 0:
            raise ValueError("recovery dimensions must be positive")
        self.c_in = c_in
        self.patch_num = patch_num
        self.patch_len = patch_len
        self.d_model = d_model
        self.use_temporal = use_temporal
        self.use_variable = use_variable
        self.use_historical = use_historical and historical_relation is not None

        self.W_m = nn.Linear(patch_len, d_model)
        self.variable_embedding = nn.Parameter(torch.empty(c_in, d_model))
        self.position_embedding = nn.Parameter(torch.empty(patch_num, d_model))
        self.state_embedding = nn.Parameter(torch.empty(3, d_model))
        self.W_q = nn.Linear(d_model, d_model)
        self.temporal = TemporalRelation(d_model, temporal_topk)
        self.variable = VariableRelation(d_model, variable_topk)
        self.historical = historical_relation
        self.fusion = nn.Linear(3 * d_model, d_model)
        nn.init.normal_(self.variable_embedding, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.02)
        nn.init.normal_(self.state_embedding, std=0.02)

    def fuse(self, visible: Tensor, prior: Tensor, recovered: Tensor, state: Tensor) -> Tensor:
        if visible.shape != prior.shape or visible.shape != recovered.shape:
            raise ValueError("visible, prior, and recovered tensors must match")
        if state.shape != visible.shape[:-1]:
            raise ValueError("state must match the node dimensions")
        observed = state == PATCH_OBSERVED
        partial = state == PATCH_PARTIAL
        missing = state == PATCH_MISSING
        if not torch.all(observed | partial | missing):
            raise ValueError("state contains an unknown patch state")
        damaged_output = prior + recovered + partial.unsqueeze(-1) * visible
        return torch.where(observed.unsqueeze(-1), visible, damaged_output)

    def _validate(
        self,
        values: Tensor,
        mask: Tensor,
        rho: Tensor,
        state: Tensor,
        base_latents: Tensor,
    ) -> None:
        expected = (self.c_in, self.patch_num, self.patch_len)
        if values.ndim != 4 or tuple(values.shape[1:]) != expected:
            raise ValueError(f"values must have shape [batch, {expected}]")
        if mask.shape != values.shape:
            raise ValueError("mask must match patch values")
        if rho.shape != values.shape[:-1] or state.shape != rho.shape:
            raise ValueError("rho and state must match patch nodes")
        latent_shape = (*values.shape[:-1], self.d_model)
        if base_latents.shape != latent_shape:
            raise ValueError("base_latents must match patch nodes and d_model")

    def forward(
        self,
        values: Tensor,
        mask: Tensor,
        rho: Tensor,
        state: Tensor,
        *,
        raw_context: Tensor | None = None,
        context_mask: Tensor | None = None,
        query_mean: Tensor | None = None,
        query_std: Tensor | None = None,
        train_std: Tensor | None = None,
        base_latents: Tensor,
        prototype_projection: nn.Module | None = None,
    ) -> RecoveryOutput:
        self._validate(values, mask, rho, state, base_latents)
        visible = base_latents
        query_visible = visible + self.W_m(mask.to(dtype=values.dtype))
        prior = (
            self.variable_embedding.view(1, self.c_in, 1, self.d_model)
            + self.position_embedding.view(1, 1, self.patch_num, self.d_model)
            + self.state_embedding[state]
        )
        query = self.W_q(query_visible + prior)
        damaged = state != PATCH_OBSERVED
        zeros = torch.zeros_like(visible)
        diagnostics: dict[str, Tensor] = {}

        if self.use_temporal:
            temporal = self.temporal.select(query, visible, rho, damaged)
            temporal_message = temporal.message
            diagnostics["temporal_selected"] = temporal.selected
            diagnostics["temporal_weights"] = temporal.weights
            diagnostics["temporal_candidate_count"] = temporal.candidate_count
        else:
            temporal_message = zeros
            diagnostics["temporal_candidate_count"] = torch.zeros_like(state)

        if self.use_variable:
            variable = self.variable.select(query, visible, rho, damaged)
            variable_message = variable.message
            diagnostics["variable_selected"] = variable.selected
            diagnostics["variable_weights"] = variable.weights
            diagnostics["variable_candidate_count"] = variable.candidate_count
        else:
            variable_message = zeros
            diagnostics["variable_candidate_count"] = torch.zeros_like(state)

        if self.use_historical:
            if any(
                item is None
                for item in (
                    raw_context,
                    context_mask,
                    query_mean,
                    query_std,
                    train_std,
                    prototype_projection,
                )
            ):
                raise ValueError("historical recovery requires context and normalization inputs")
            assert self.historical is not None
            historical = self.historical.select(
                raw_context,
                context_mask,
                state,
                query_mean,
                query_std,
                train_std,
                prototype_projection=prototype_projection,
            )
            historical_message = historical.message
            diagnostics["historical_selected"] = historical.selected
            diagnostics["historical_weights"] = historical.weights
            diagnostics["historical_candidate_count"] = historical.candidate_count
        else:
            historical_message = zeros
            diagnostics["historical_candidate_count"] = torch.zeros_like(state)

        messages = RelationMessages(
            temporal=temporal_message,
            variable=variable_message,
            historical=historical_message,
        )
        recovered = self.fusion(
            torch.cat(
                (messages.temporal, messages.variable, messages.historical), dim=-1
            )
        )
        latents = self.fuse(visible, prior, recovered, state)
        return RecoveryOutput(latents, visible, prior, recovered, messages, diagnostics)
