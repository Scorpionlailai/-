from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from malr.data.patching import PATCH_MISSING, PATCH_OBSERVED, PATCH_PARTIAL

from .filtering import EdgeConfidenceFilter, MissingAwareEdgeFilter
from .refinement import GlobalLatentRefiner
from .reconstruction import LatentReconstructionHead, aggregate_patch_predictions
from .relations import (
    HistoricalRelation,
    RelationSelection,
    TemporalRelation,
    VariableRelation,
)


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
    filter_keep_loss: Tensor


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
        fusion: str = "additive",
        residual_hidden_multiplier: int = 2,
        residual_dropout: float = 0.0,
        residual_output_init: str = "zero",
        damage_scaling: str = "missing_fraction",
        global_refiner_enabled: bool = False,
        global_refiner_heads: int = 4,
        global_refiner_ff_multiplier: int = 2,
        global_refiner_dropout: float = 0.0,
        global_refiner_output_init: str = "zero",
        edge_filter: MissingAwareEdgeFilter | None = None,
        relation_selection: str = "hard_topk",
        reconstruction_enabled: bool = False,
        reconstruction_window_len: int | None = None,
        reconstruction_patch_stride: int | None = None,
    ) -> None:
        super().__init__()
        if min(c_in, patch_num, patch_len, d_model) <= 0:
            raise ValueError("recovery dimensions must be positive")
        if fusion not in {"additive", "residual_mlp", "replacement_mlp"}:
            raise ValueError(
                "recovery.fusion must be additive, residual_mlp, or replacement_mlp"
            )
        if (
            type(residual_hidden_multiplier) is not int
            or residual_hidden_multiplier <= 0
        ):
            raise ValueError(
                "recovery.hidden_multiplier must be a positive non-boolean integer"
            )
        if not 0.0 <= residual_dropout < 1.0:
            raise ValueError("recovery.dropout must be in [0, 1)")
        if residual_output_init != "zero":
            raise ValueError("recovery.output_init must be zero")
        if damage_scaling != "missing_fraction":
            raise ValueError("recovery.damage_scaling must be missing_fraction")
        if global_refiner_enabled and fusion != "replacement_mlp":
            raise ValueError("global refiner requires replacement_mlp fusion")
        if relation_selection not in {"hard_topk", "missing_aware_filter"}:
            raise ValueError("unknown relation selection")
        if (relation_selection == "missing_aware_filter") != (
            edge_filter is not None
        ):
            raise ValueError(
                "missing_aware_filter selection requires exactly one edge filter"
            )
        self.c_in = c_in
        self.patch_num = patch_num
        self.patch_len = patch_len
        self.d_model = d_model
        self.use_temporal = use_temporal
        self.use_variable = use_variable
        self.use_historical = use_historical and historical_relation is not None
        self.fusion_mode = fusion
        self.damage_scaling = damage_scaling
        self.relation_selection = relation_selection
        self.edge_filter = edge_filter
        self.reconstruction_patch_stride = (
            patch_len // 2
            if reconstruction_patch_stride is None
            else reconstruction_patch_stride
        )
        self.reconstruction_window_len = (
            patch_num * self.reconstruction_patch_stride
            if reconstruction_window_len is None
            else reconstruction_window_len
        )
        if self.reconstruction_patch_stride <= 0:
            raise ValueError("reconstruction patch stride must be positive")
        if self.reconstruction_window_len <= 0:
            raise ValueError("reconstruction window length must be positive")
        self.reconstruction_head = (
            LatentReconstructionHead(d_model=d_model, patch_len=patch_len)
            if reconstruction_enabled
            else None
        )

        self.W_m = nn.Linear(patch_len, d_model)
        self.variable_embedding = nn.Parameter(torch.empty(c_in, d_model))
        self.position_embedding = nn.Parameter(torch.empty(patch_num, d_model))
        self.state_embedding = nn.Parameter(torch.empty(3, d_model))
        self.W_q = nn.Linear(d_model, d_model)
        self.temporal = TemporalRelation(d_model, temporal_topk)
        self.variable = VariableRelation(d_model, variable_topk)
        self.historical = historical_relation
        self.fusion = nn.Linear(3 * d_model, d_model)
        self.residual_mlp: nn.Sequential | None = None
        self.replacement_mlp: nn.Sequential | None = None
        self.global_refiner: GlobalLatentRefiner | None = None
        if fusion == "residual_mlp":
            hidden_dim = residual_hidden_multiplier * d_model
            self.residual_mlp = nn.Sequential(
                nn.LayerNorm(4 * d_model),
                nn.Linear(4 * d_model, hidden_dim),
                nn.GELU(),
                nn.Dropout(residual_dropout),
                nn.Linear(hidden_dim, d_model),
            )
            nn.init.zeros_(self.residual_mlp[-1].weight)
            nn.init.zeros_(self.residual_mlp[-1].bias)
        elif fusion == "replacement_mlp":
            hidden_dim = residual_hidden_multiplier * d_model
            self.replacement_mlp = nn.Sequential(
                nn.LayerNorm(5 * d_model),
                nn.Linear(5 * d_model, hidden_dim),
                nn.GELU(),
                nn.Dropout(residual_dropout),
                nn.Linear(hidden_dim, d_model),
            )
            nn.init.zeros_(self.replacement_mlp[-1].weight)
            nn.init.zeros_(self.replacement_mlp[-1].bias)
        if global_refiner_enabled:
            self.global_refiner = GlobalLatentRefiner(
                d_model=d_model,
                n_heads=global_refiner_heads,
                ff_multiplier=global_refiner_ff_multiplier,
                dropout=global_refiner_dropout,
                output_init=global_refiner_output_init,
            )
        nn.init.normal_(self.variable_embedding, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.02)
        nn.init.normal_(self.state_embedding, std=0.02)

    def reconstruct(self, latents: Tensor) -> tuple[Tensor, Tensor]:
        if self.reconstruction_head is None:
            raise ValueError("reconstruction head is not enabled")
        patches = self.reconstruction_head(latents)
        return aggregate_patch_predictions(
            patches,
            window_len=self.reconstruction_window_len,
            patch_len=self.patch_len,
            stride=self.reconstruction_patch_stride,
            validity=None,
        )

    def fuse(
        self,
        visible: Tensor,
        prior: Tensor,
        recovered: Tensor,
        state: Tensor,
        rho: Tensor,
    ) -> Tensor:
        if visible.shape != prior.shape or visible.shape != recovered.shape:
            raise ValueError("visible, prior, and recovered tensors must match")
        if state.shape != visible.shape[:-1]:
            raise ValueError("state must match the node dimensions")
        if rho.shape != state.shape:
            raise ValueError("rho must match the node dimensions")
        if not torch.isfinite(rho).all():
            raise ValueError("rho must contain only finite values")
        if not torch.all((rho >= 0.0) & (rho <= 1.0)):
            raise ValueError("rho must be in [0, 1]")
        observed = state == PATCH_OBSERVED
        partial = state == PATCH_PARTIAL
        missing = state == PATCH_MISSING
        if not torch.all(observed | partial | missing):
            raise ValueError("state contains an unknown patch state")
        if self.fusion_mode == "additive":
            partial = partial.unsqueeze(-1).to(visible.dtype)
            damaged_output = prior + recovered + partial * visible
            return torch.where(observed.unsqueeze(-1), visible, damaged_output)

        damage_strength = (1.0 - rho).unsqueeze(-1).to(visible.dtype)
        if self.fusion_mode == "residual_mlp":
            residual_output = visible + damage_strength * recovered
            return torch.where(observed.unsqueeze(-1), visible, residual_output)

        replacement_output = rho.unsqueeze(-1).to(visible.dtype) * visible
        replacement_output = replacement_output + damage_strength * recovered
        return torch.where(observed.unsqueeze(-1), visible, replacement_output)

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
        mask_embedding = self.W_m(mask.to(dtype=values.dtype))
        query_visible = visible + mask_embedding
        prior = (
            self.variable_embedding.view(1, self.c_in, 1, self.d_model)
            + self.position_embedding.view(1, 1, self.patch_num, self.d_model)
            + self.state_embedding[state]
        )
        query = self.W_q(query_visible + prior)
        damaged = state != PATCH_OBSERVED
        zeros = torch.zeros_like(visible)
        diagnostics: dict[str, Tensor] = {}
        filter_keep_terms: list[Tensor] = []
        filter_query = None
        if self.edge_filter is not None:
            filter_query = self.edge_filter.context_query(
                visible,
                mask_embedding,
                prior,
                rho,
                damaged,
            ).query

        def record_filter_diagnostics(
            name: str,
            selection: RelationSelection,
            head: EdgeConfidenceFilter,
        ) -> None:
            if selection.filter_probabilities is None:
                raise RuntimeError(f"{name} relation omitted filter probabilities")
            if selection.hard_gate is None or selection.retained_count is None:
                raise RuntimeError(f"{name} relation omitted hard filter outputs")
            if selection.pool_count is None or selection.keep_loss is None:
                raise RuntimeError(f"{name} relation omitted filter statistics")
            probabilities = selection.filter_probabilities
            hard_gate = selection.hard_gate
            diagnostics[f"{name}_filter_candidate_count"] = (
                selection.candidate_count.detach()
            )
            diagnostics[f"{name}_filter_pool_count"] = selection.pool_count.detach()
            diagnostics[f"{name}_filter_retained_count"] = (
                selection.retained_count.detach()
            )
            diagnostics[f"{name}_filter_candidate_confidence_sum"] = (
                probabilities.sum(dim=-1).detach()
            )
            diagnostics[f"{name}_filter_retained_confidence_sum"] = (
                (probabilities * hard_gate.to(probabilities.dtype))
                .sum(dim=-1)
                .detach()
            )
            diagnostics[f"{name}_filter_threshold"] = visible.new_tensor(
                head.threshold
            )
            diagnostics[f"{name}_filter_max_edges"] = visible.new_tensor(
                head.max_edges
            )
            if bool(selection.candidate_count.sum() > 0):
                filter_keep_terms.append(selection.keep_loss)

        if self.use_temporal:
            if self.edge_filter is None:
                temporal = self.temporal.select(query, visible, rho, damaged)
            else:
                assert filter_query is not None
                temporal = self.temporal.select(
                    query,
                    visible,
                    rho,
                    damaged,
                    filter_query=filter_query,
                    edge_filter=self.edge_filter.temporal,
                )
                record_filter_diagnostics(
                    "temporal", temporal, self.edge_filter.temporal
                )
            temporal_message = temporal.message
            diagnostics["temporal_selected"] = temporal.selected
            diagnostics["temporal_weights"] = temporal.weights
            diagnostics["temporal_candidate_count"] = temporal.candidate_count
        else:
            temporal_message = zeros
            diagnostics["temporal_candidate_count"] = torch.zeros_like(state)

        if self.use_variable:
            if self.edge_filter is None:
                variable = self.variable.select(query, visible, rho, damaged)
            else:
                assert filter_query is not None
                variable = self.variable.select(
                    query,
                    visible,
                    rho,
                    damaged,
                    filter_query=filter_query,
                    edge_filter=self.edge_filter.variable,
                    variable_embeddings=self.variable_embedding,
                )
                record_filter_diagnostics(
                    "variable", variable, self.edge_filter.variable
                )
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
            if self.edge_filter is None:
                historical = self.historical.select(
                    raw_context,
                    context_mask,
                    state,
                    query_mean,
                    query_std,
                    train_std,
                    prototype_projection=prototype_projection,
                )
            else:
                assert filter_query is not None
                historical = self.historical.select(
                    raw_context,
                    context_mask,
                    state,
                    query_mean,
                    query_std,
                    train_std,
                    prototype_projection=prototype_projection,
                    filter_query=filter_query,
                    edge_filter=self.edge_filter.historical,
                )
                record_filter_diagnostics(
                    "historical", historical, self.edge_filter.historical
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
        relation_messages = torch.cat(
            (messages.temporal, messages.variable, messages.historical), dim=-1
        )
        if self.fusion_mode == "residual_mlp":
            assert self.residual_mlp is not None
            recovered = self.residual_mlp(torch.cat((relation_messages, prior), dim=-1))
        elif self.fusion_mode == "replacement_mlp":
            assert self.replacement_mlp is not None
            recovered = self.replacement_mlp(
                torch.cat((relation_messages, prior, visible), dim=-1)
            )
        else:
            recovered = self.fusion(relation_messages)
        latents = self.fuse(visible, prior, recovered, state, rho)
        if self.fusion_mode == "residual_mlp":
            delta_norm = recovered.norm(dim=-1)
            applied_norm = ((1.0 - rho).unsqueeze(-1) * recovered).norm(dim=-1)
            partial_mask = state == PATCH_PARTIAL
            missing_mask = state == PATCH_MISSING
            zero = applied_norm.new_zeros(())
            diagnostics["residual_delta_norm"] = delta_norm.detach()
            diagnostics["residual_applied_norm"] = applied_norm.detach()
            diagnostics["residual_partial_mean_norm"] = (
                applied_norm[partial_mask].mean() if partial_mask.any() else zero
            ).detach()
            diagnostics["residual_missing_mean_norm"] = (
                applied_norm[missing_mask].mean() if missing_mask.any() else zero
            ).detach()
        elif self.fusion_mode == "replacement_mlp":
            replacement_delta = (1.0 - rho).unsqueeze(-1) * (recovered - visible)
            applied_norm = replacement_delta.norm(dim=-1)
            partial_mask = state == PATCH_PARTIAL
            missing_mask = state == PATCH_MISSING
            zero = applied_norm.new_zeros(())
            diagnostics["replacement_candidate_norm"] = recovered.norm(dim=-1).detach()
            diagnostics["replacement_applied_norm"] = applied_norm.detach()
            diagnostics["replacement_partial_mean_norm"] = (
                applied_norm[partial_mask].mean() if partial_mask.any() else zero
            ).detach()
            diagnostics["replacement_missing_mean_norm"] = (
                applied_norm[missing_mask].mean() if missing_mask.any() else zero
            ).detach()
        if self.global_refiner is not None:
            refinement = self.global_refiner(latents, prior, rho)
            latents = refinement.latents
            diagnostics["global_refiner_delta_norm"] = (
                refinement.delta.norm(dim=-1).detach()
            )
            diagnostics["global_refiner_applied_norm"] = (
                refinement.applied_delta.norm(dim=-1).detach()
            )
        filter_keep_loss = (
            torch.stack(filter_keep_terms).mean()
            if filter_keep_terms
            else latents.sum() * 0.0
        )
        return RecoveryOutput(
            latents,
            visible,
            prior,
            recovered,
            messages,
            diagnostics,
            filter_keep_loss,
        )
