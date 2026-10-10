"""Composition root for the first online relation-aware Stage-1 prototype.

This module is deliberately separate from the legacy ``Stage1Model``.  It makes
the new protocol executable without changing the old experiment path: online
memory writes are explicit, inference is read-only, and all relation branches
are converted to one common edge table before routing and message passing.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from malr.data.patching import PATCH_MISSING, PATCH_OBSERVED
from malr.prototypes.online_bank import OnlinePrototypeMemory, OnlinePrototypeMember
from malr.prototypes.retrieval import (
    RetrievalEncoderEMA,
    masked_contrastive_loss,
    masked_pair_loss,
)
from malr.training.unified_objective import UnifiedStage1Loss, unified_stage1_loss

from .candidates import (
    TemporalCandidateBatch,
    VARIABLE_LAGS,
    VariableCandidateBatch,
    build_temporal_candidates,
    build_variable_candidates,
)
from .edge_gain import (
    balanced_edge_gain_loss,
    edge_gain_admission,
    relative_edge_gain,
)
from .graph_recovery import GraphRecoveryOutput, RelationAwareResidualRecovery
from .online_history import HistoricalCandidateBatch, build_online_historical_candidates
from .patchtst import PatchTSTCore
from .routing import (
    ROUTE_OPTIONS,
    hard_route_select,
    oracle_route_labels,
    route_availability_mask,
    route_cross_entropy,
)


@dataclass(frozen=True)
class OnlineStage1Forward:
    forecast: Tensor
    normalized_forecast: Tensor
    base_latents: Tensor
    latents: Tensor
    graph: GraphRecoveryOutput
    temporal_candidates: TemporalCandidateBatch
    variable_candidates: VariableCandidateBatch
    source_nodes: Tensor
    edge_features: Tensor
    edge_type: Tensor
    eligible: Tensor
    predicted_gain: Tensor
    admitted: Tensor
    route_logits: Tensor
    route_index: Tensor
    route_available: Tensor
    diagnostics: dict[str, Tensor]


@dataclass(frozen=True)
class OnlineStage1TrainingResult:
    output: OnlineStage1Forward
    loss: UnifiedStage1Loss
    edge_gain_target: Tensor
    route_target: Tensor


class RelationEdgeGainScorer(nn.Module):
    """Three relation-specific heads predicting one shared gain scale."""

    def __init__(self, d_model: int, edge_feature_dim: int = 11, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden = d_model if hidden_dim is None else int(hidden_dim)
        if d_model <= 0 or edge_feature_dim <= 0 or hidden <= 0:
            raise ValueError("edge scorer dimensions must be positive")
        self.d_model = d_model
        self.edge_feature_dim = edge_feature_dim
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * d_model + edge_feature_dim, hidden),
                    nn.GELU(),
                    nn.Linear(hidden, 1),
                )
                for _ in range(3)
            ]
        )

    def forward(self, target: Tensor, source: Tensor, features: Tensor, edge_type: Tensor) -> Tensor:
        if target.shape != source.shape or target.shape[:-1] != features.shape[:-1]:
            raise ValueError("edge scorer target, source and feature shapes do not match")
        if features.shape[-1] != self.edge_feature_dim:
            raise ValueError("edge scorer feature dimension does not match")
        if edge_type.shape != target.shape[:-1]:
            raise ValueError("edge_type must match edge nodes")
        edge_type = edge_type.to(device=target.device, dtype=torch.long)
        if torch.any((edge_type < 0) | (edge_type >= 3)):
            raise ValueError("edge_type contains an unknown relation")
        inputs = torch.cat((target, source, features), dim=-1)
        outputs = []
        for head in self.heads:
            outputs.append(head(inputs.reshape(-1, inputs.shape[-1])).reshape(*inputs.shape[:-1]))
        stacked = torch.stack(outputs, dim=-1)
        return stacked.gather(-1, edge_type.unsqueeze(-1)).squeeze(-1)


def _huber(values: Tensor, target: Tensor, *, delta: float = 1.0) -> Tensor:
    absolute = (values - target).abs()
    return torch.where(absolute <= delta, 0.5 * absolute.square(), delta * (absolute - 0.5 * delta))


class OnlineStage1Model(nn.Module):
    """Online memory + routed residual recovery composition root.

    ``OnlinePrototypeMemory`` and ``RetrievalEncoderEMA`` are stateful services,
    not neural modules.  The trainable retrieval encoder is registered here;
    the service's EMA/frozen copies are exported through ``online_state``.
    """

    def __init__(
        self,
        core: PatchTSTCore,
        *,
        memory: OnlinePrototypeMemory,
        retrieval: RetrievalEncoderEMA,
        variable_count: int,
        historical_max_items: int = 4,
        edge_threshold: float = 0.05,
        historical_similarity_threshold: float = 0.7,
        residual_warmup_epochs: int = 10,
        relation_prior_decay: float = 0.9,
    ) -> None:
        super().__init__()
        if core.c_in != variable_count or memory.variable_count != variable_count:
            raise ValueError("core, memory and variable_count must agree")
        if retrieval.encoder.patch_len != core.patch_len:
            raise ValueError("retrieval patch length must match PatchTST")
        if historical_max_items <= 0 or residual_warmup_epochs <= 0:
            raise ValueError("historical_max_items and warm-up must be positive")
        if not 0.0 <= edge_threshold <= 1.0:
            raise ValueError("edge threshold must lie in [0, 1]")
        if not 0.0 <= historical_similarity_threshold <= 1.0:
            raise ValueError("historical similarity threshold must lie in [0, 1]")
        if not 0.0 <= relation_prior_decay < 1.0:
            raise ValueError("relation prior decay must lie in [0, 1)")
        self.core = core
        self.memory = memory
        self.retrieval = retrieval
        self.add_module("retrieval_encoder", retrieval.encoder)
        self.variable_count = variable_count
        self.historical_max_items = int(historical_max_items)
        self.edge_threshold = float(edge_threshold)
        self.historical_similarity_threshold = float(historical_similarity_threshold)
        self.residual_warmup_epochs = int(residual_warmup_epochs)
        self.relation_prior_decay = float(relation_prior_decay)
        self.edge_scorer = RelationEdgeGainScorer(core.d_model, edge_feature_dim=11)
        self.graph = RelationAwareResidualRecovery(core.d_model, edge_feature_dim=11)
        self.route_router = nn.Sequential(
            nn.Linear(core.d_model + 5, core.d_model),
            nn.GELU(),
            nn.Linear(core.d_model, len(ROUTE_OPTIONS)),
        )
        self.patch_decoder = nn.Linear(core.d_model, core.patch_len)
        self.register_buffer("relation_prior", torch.zeros(variable_count, variable_count, 7))
        self._current_epoch: int | None = None

    def to(self, *args: object, **kwargs: object) -> "OnlineStage1Model":
        """Move the service-owned EMA/frozen encoders with the trainable model."""

        result = super().to(*args, **kwargs)
        device = next(result.parameters()).device
        result.retrieval.ema_encoder.to(device)
        result.retrieval.frozen_encoder.to(device)
        return result

    @staticmethod
    def _starts(patch_starts: Tensor | None, batch: int, patch_count: int, stride: int, device: torch.device) -> Tensor:
        if patch_starts is None:
            base = torch.arange(patch_count, device=device, dtype=torch.long) * stride
            return base.unsqueeze(0).expand(batch, -1)
        starts = patch_starts.to(device=device, dtype=torch.long)
        if starts.ndim == 1:
            if starts.shape[0] != patch_count:
                raise ValueError("patch_starts must match patch count")
            starts = starts.unsqueeze(0).expand(batch, -1)
        if starts.shape != (batch, patch_count) or torch.any(starts < 0):
            raise ValueError("patch_starts must have shape [batch, patches]")
        return starts

    @staticmethod
    def _gather_time(nodes: Tensor, indices: Tensor) -> Tensor:
        safe = indices.clamp_min(0)
        batch = torch.arange(nodes.shape[0], device=nodes.device).view(-1, 1, 1, 1)
        variable = torch.arange(nodes.shape[1], device=nodes.device).view(1, -1, 1, 1)
        batch = batch.expand_as(safe)
        variable = variable.expand_as(safe)
        return nodes[batch, variable, safe]

    @staticmethod
    def _gather_variable(nodes: Tensor, variables: Tensor, patches: Tensor) -> Tensor:
        safe_variables = variables.clamp_min(0)
        safe_patches = patches.clamp_min(0)
        batch = torch.arange(nodes.shape[0], device=nodes.device).view(-1, 1, 1, 1)
        batch = batch.expand_as(safe_variables)
        return nodes[batch, safe_variables, safe_patches]

    def _historical_candidate_batch(
        self,
        values: Tensor,
        mask: Tensor,
        rho: Tensor,
        state: Tensor,
        starts: Tensor,
        base_latents: Tensor,
    ) -> HistoricalCandidateBatch:
        return build_online_historical_candidates(
            self.core,
            self.memory,
            self.retrieval,
            values,
            mask,
            rho,
            state,
            starts,
            base_latents,
            max_items=self.historical_max_items,
            similarity_threshold=self.historical_similarity_threshold,
        )

    def _historical_candidates(
        self,
        values: Tensor,
        mask: Tensor,
        rho: Tensor,
        state: Tensor,
        starts: Tensor,
        base_latents: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        batch = self._historical_candidate_batch(
            values, mask, rho, state, starts, base_latents
        )
        return batch.source_nodes, batch.features, batch.eligible, batch.candidate_count

    def _assemble_edges(
        self,
        base_latents: Tensor,
        temporal: TemporalCandidateBatch,
        variable: VariableCandidateBatch,
        historical_nodes: Tensor,
        historical_features: Tensor,
        historical_eligible: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        temporal_nodes = self._gather_time(base_latents, temporal.source_patch)
        variable_nodes = self._gather_variable(base_latents, variable.source_variable, variable.source_patch)
        feature_pad = lambda x: F.pad(x, (0, 11 - x.shape[-1]))
        source_nodes = torch.cat((historical_nodes, temporal_nodes, variable_nodes), dim=-2)
        edge_features = torch.cat((feature_pad(historical_features), feature_pad(temporal.features), variable.features), dim=-2)
        eligible = torch.cat((historical_eligible, temporal.eligible, variable.eligible), dim=-1)
        edge_type = torch.cat(
            (
                torch.zeros_like(historical_eligible, dtype=torch.long),
                torch.ones_like(temporal.eligible, dtype=torch.long),
                torch.full_like(variable.eligible, 2, dtype=torch.long),
            ),
            dim=-1,
        )
        return source_nodes, edge_features, edge_type, eligible

    @staticmethod
    def _apply_relation_capacity(
        admitted: Tensor,
        predicted_gain: Tensor,
        start: int,
        width: int,
        capacity: int,
    ) -> Tensor:
        """Keep only the highest predicted-gain edges in a relation slice."""

        if width == 0 or capacity <= 0:
            return admitted[..., start : start + width] & False
        relation = admitted[..., start : start + width]
        if width <= capacity:
            return relation
        gains = predicted_gain[..., start : start + width].masked_fill(~relation, -torch.inf)
        top_indices = gains.topk(min(capacity, width), dim=-1).indices
        retained = torch.zeros_like(relation)
        retained.scatter_(-1, top_indices, True)
        return relation & retained

    def forward(
        self,
        patch_values: Tensor,
        patch_mask: Tensor,
        rho: Tensor,
        state: Tensor,
        query_mean: Tensor,
        query_std: Tensor,
        *,
        patch_starts: Tensor | None = None,
        epoch_index: int = 0,
        patch_valid: Tensor | None = None,
    ) -> OnlineStage1Forward:
        if patch_values.ndim != 4 or patch_values.shape[1] != self.variable_count:
            raise ValueError("patch_values must have shape [batch, variables, patches, patch_len]")
        if patch_mask.shape != patch_values.shape or rho.shape != patch_values.shape[:-1] or state.shape != rho.shape:
            raise ValueError("patch mask, rho and state do not match patch values")
        if query_mean.shape != (patch_values.shape[0], self.variable_count) or query_std.shape != query_mean.shape:
            raise ValueError("query normalization statistics must have shape [batch, variables]")
        if type(epoch_index) is not int or epoch_index < 0:
            raise ValueError("epoch_index must be a non-negative integer")
        valid = torch.ones_like(state, dtype=torch.bool) if patch_valid is None else patch_valid.to(device=state.device, dtype=torch.bool)
        if valid.shape != state.shape:
            raise ValueError("patch_valid must match state")
        damaged = (state != PATCH_OBSERVED) & valid
        starts = self._starts(patch_starts, patch_values.shape[0], patch_values.shape[2], self.core.stride, patch_values.device)
        base_latents = self.core.project_clean_patches(patch_values)
        temporal = build_temporal_candidates(patch_values, patch_mask, rho, target_eligible=damaged)
        variable = build_variable_candidates(patch_values, patch_mask, rho, self.relation_prior.detach(), target_eligible=damaged)
        historical_nodes, historical_features, historical_eligible, historical_count = self._historical_candidates(
            patch_values, patch_mask, rho, state, starts, base_latents
        )
        source_nodes, edge_features, edge_type, eligible = self._assemble_edges(
            base_latents, temporal, variable, historical_nodes, historical_features, historical_eligible
        )
        target_nodes = base_latents.unsqueeze(-2).expand_as(source_nodes)
        predicted_gain = self.edge_scorer(target_nodes, source_nodes, edge_features, edge_type)
        admitted = edge_gain_admission(predicted_gain, eligible, threshold=self.edge_threshold)
        historical_width = historical_nodes.shape[-2]
        temporal_width = temporal.source_patch.shape[-1]
        admitted = torch.cat(
            (
                self._apply_relation_capacity(admitted, predicted_gain, 0, historical_width, self.historical_max_items),
                self._apply_relation_capacity(admitted, predicted_gain, historical_width, temporal_width, temporal.capacity),
                self._apply_relation_capacity(admitted, predicted_gain, historical_width + temporal_width, variable.source_variable.shape[-1], variable.capacity),
            ),
            dim=-1,
        )
        relation_available = torch.stack(
            (
                admitted[..., : historical_width].any(dim=-1),
                admitted[..., historical_width : historical_width + temporal_width].any(dim=-1),
                admitted[..., historical_width + temporal_width :].any(dim=-1),
            ),
            dim=-1,
        )
        route_available = route_availability_mask(relation_available)
        router_features = torch.cat(
            (
                base_latents,
                rho.unsqueeze(-1),
                (1.0 - rho).unsqueeze(-1),
                relation_available.to(base_latents.dtype),
            ),
            dim=-1,
        )
        route_logits = self.route_router(router_features)
        route_index = hard_route_select(route_logits, route_available)
        raw_graph = self.graph(
            base_latents,
            source_nodes,
            edge_features,
            edge_type,
            predicted_gain,
            admitted,
            route_index,
            damaged,
        )
        warmup = min(1.0, float(epoch_index) / float(self.residual_warmup_epochs))
        scaled_residual = raw_graph.residual * warmup
        scaled_graph = GraphRecoveryOutput(
            latents=base_latents + scaled_residual,
            message=raw_graph.message,
            residual=scaled_residual,
            weights=raw_graph.weights,
            active_edges=raw_graph.active_edges,
        )
        latents = scaled_graph.latents
        normalized_forecast = self.core.forecast_from_latents(latents)
        if query_mean.shape != query_std.shape:
            raise ValueError("query_mean and query_std must match")
        forecast = normalized_forecast * query_std.unsqueeze(-1) + query_mean.unsqueeze(-1)
        diagnostics = {
            "historical_candidate_count": historical_count,
            "historical_admitted_count": admitted[..., : historical_width].sum(dim=-1),
            "temporal_candidate_count": temporal.candidate_count,
            "temporal_admitted_count": admitted[..., historical_width : historical_width + temporal_width].sum(dim=-1),
            "variable_candidate_count": variable.candidate_count,
            "variable_admitted_count": admitted[..., historical_width + temporal_width :].sum(dim=-1),
            "route_index": route_index.detach(),
            "route_available": route_available.detach(),
            "admitted_edge_count": admitted.sum(dim=-1),
            "active_edge_count": scaled_graph.active_edges.sum(dim=-1),
            "message_norm": scaled_graph.message.norm(dim=-1),
            "residual_norm": scaled_residual.norm(dim=-1),
        }
        return OnlineStage1Forward(
            forecast=forecast,
            normalized_forecast=normalized_forecast,
            base_latents=base_latents,
            latents=latents,
            graph=scaled_graph,
            temporal_candidates=temporal,
            variable_candidates=variable,
            source_nodes=source_nodes,
            edge_features=edge_features,
            edge_type=edge_type,
            eligible=eligible,
            predicted_gain=predicted_gain,
            admitted=admitted,
            route_logits=route_logits,
            route_index=route_index,
            route_available=route_available,
            diagnostics=diagnostics,
        )

    def _edge_targets(self, output: OnlineStage1Forward, clean_raw_patches: Tensor, patch_mask: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        clean = clean_raw_patches
        missing = ~patch_mask
        target_nodes = output.base_latents
        baseline_values = self.patch_decoder(target_nodes)
        baseline_error = _huber(baseline_values, clean).masked_fill(~missing, 0.0).sum(dim=-1) / missing.sum(dim=-1).clamp_min(1).to(clean.dtype)
        source_values = self.patch_decoder(output.source_nodes)
        edge_missing = missing.unsqueeze(-2)
        edge_error = _huber(source_values, clean.unsqueeze(-2)).masked_fill(~edge_missing, 0.0).sum(dim=-1) / edge_missing.sum(dim=-1).clamp_min(1).to(clean.dtype)
        baseline_error = baseline_error.unsqueeze(-1).expand_as(edge_error)
        target = relative_edge_gain(baseline_error, edge_error)
        eligible = output.eligible & (state != PATCH_OBSERVED).unsqueeze(-1)
        return target, eligible

    @staticmethod
    def _route_errors(baseline: Tensor, gains: Tensor, edge_type: Tensor, eligible: Tensor) -> Tensor:
        relation_gain = []
        for relation in range(3):
            values = gains.masked_fill(~(eligible & edge_type.eq(relation)), 0.0)
            relation_gain.append(values.max(dim=-1).values.clamp_min(0.0).clamp_max(1.0))
        historical, temporal, variable = relation_gain
        combined = {
            0: torch.zeros_like(baseline),
            1: historical,
            2: temporal,
            3: variable,
            4: 1.0 - (1.0 - historical) * (1.0 - temporal),
            5: 1.0 - (1.0 - historical) * (1.0 - variable),
            6: 1.0 - (1.0 - temporal) * (1.0 - variable),
        }
        return torch.stack([baseline * (1.0 - combined[index]) for index in range(7)], dim=-1)

    def training_objective(
        self,
        patch_values: Tensor,
        patch_mask: Tensor,
        rho: Tensor,
        state: Tensor,
        query_mean: Tensor,
        query_std: Tensor,
        forecast_target: Tensor,
        clean_raw_patches: Tensor,
        *,
        patch_starts: Tensor | None = None,
        epoch_index: int = 0,
        patch_valid: Tensor | None = None,
    ) -> OnlineStage1TrainingResult:
        output = self(
            patch_values,
            patch_mask,
            rho,
            state,
            query_mean,
            query_std,
            patch_starts=patch_starts,
            epoch_index=epoch_index,
            patch_valid=patch_valid,
        )
        if clean_raw_patches.shape != patch_values.shape:
            raise ValueError("clean_raw_patches must match patch values")
        target_gain, eligible = self._edge_targets(output, clean_raw_patches, patch_mask, state)
        edge_loss = balanced_edge_gain_loss(output.predicted_gain, target_gain, eligible, output.edge_type)
        damaged = (state != PATCH_OBSERVED) & (torch.ones_like(state, dtype=torch.bool) if patch_valid is None else patch_valid.to(torch.bool))
        decoded = self.patch_decoder(output.latents)
        missing = (~patch_mask) & damaged.unsqueeze(-1)
        imputation = _huber(decoded, clean_raw_patches).masked_fill(~missing, 0.0).sum() / missing.sum().clamp_min(1).to(decoded.dtype)
        baseline = self.patch_decoder(output.base_latents)
        baseline_error = _huber(baseline, clean_raw_patches).masked_fill(~missing, 0.0).sum(dim=-1) / missing.sum(dim=-1).clamp_min(1).to(decoded.dtype)
        route_errors = self._route_errors(baseline_error, target_gain, output.edge_type, eligible)
        relation_available = torch.stack(
            [
                (output.admitted & output.edge_type.eq(index) & output.eligible).any(dim=-1)
                for index in range(3)
            ],
            dim=-1,
        )
        route_target_obj = oracle_route_labels(route_errors, baseline_error, relation_available)
        route_loss = route_cross_entropy(output.route_logits, route_target_obj.index, output.route_available)
        flat_query = patch_values.reshape(-1, patch_values.shape[-1])
        flat_mask = patch_mask.reshape(-1, patch_mask.shape[-1])
        flat_clean = clean_raw_patches.reshape(-1, clean_raw_patches.shape[-1])
        query_keys = self.retrieval.encoder(flat_query, flat_mask)
        clean_keys = self.retrieval.encoder(flat_clean, torch.ones_like(flat_mask))
        memory_loss = masked_pair_loss(query_keys, clean_keys) + 0.5 * masked_contrastive_loss(query_keys, clean_keys)
        forecast_loss = F.mse_loss(output.forecast, forecast_target)
        loss = unified_stage1_loss(forecast_loss, imputation, edge_loss, route_loss, memory_loss)
        return OnlineStage1TrainingResult(output, loss, target_gain, route_target_obj.index)

    def begin_epoch(self, epoch: int) -> None:
        self.retrieval.begin_epoch(epoch)
        self._current_epoch = epoch

    def stage_observed_batch(self, clean_raw_patches: Tensor, patch_mask: Tensor, rho: Tensor, state: Tensor, patch_starts: Tensor | None = None) -> int:
        if self._current_epoch is None:
            raise RuntimeError("begin_epoch must be called before staging prototypes")
        if clean_raw_patches.ndim != 4 or clean_raw_patches.shape[1] != self.variable_count:
            raise ValueError("clean_raw_patches must have shape [batch, variables, patches, patch_len]")
        starts = self._starts(patch_starts, clean_raw_patches.shape[0], clean_raw_patches.shape[2], self.core.stride, clean_raw_patches.device)
        flat = clean_raw_patches.reshape(-1, clean_raw_patches.shape[-1])
        keys = self.retrieval.encode_frozen(flat, torch.ones_like(flat, dtype=torch.bool)).reshape(*clean_raw_patches.shape[:3], -1)
        observed = state.eq(PATCH_OBSERVED) & rho.eq(1.0) & patch_mask.all(dim=-1)
        starts_cpu = starts.detach().cpu().numpy()
        raw_cpu = clean_raw_patches.detach().cpu().numpy()
        keys_cpu = keys.detach().cpu().numpy()
        observed_cpu = observed.detach().cpu().numpy()
        staged = 0
        for b in range(clean_raw_patches.shape[0]):
            for variable in range(self.variable_count):
                for patch in range(clean_raw_patches.shape[2]):
                    staged += int(
                        self.memory.stage_patch(
                            variable,
                            int(starts_cpu[b, patch]),
                            raw_cpu[b, variable, patch],
                            keys_cpu[b, variable, patch],
                            observed=bool(observed_cpu[b, variable, patch]),
                        )
                    )
        return staged

    def end_epoch(self, epoch: int) -> None:
        if self._current_epoch != epoch:
            raise ValueError("end_epoch must close the active epoch")
        self.retrieval.update_ema()
        ema_encoder = self.retrieval.ema_encoder
        ema_device = next(ema_encoder.parameters()).device

        def encode_members(
            members: tuple[OnlinePrototypeMember, ...],
        ) -> np.ndarray:
            raw_patches = torch.as_tensor(
                np.stack([np.asarray(member.raw_patch) for member in members]),
                dtype=torch.float32,
                device=ema_device,
            )
            with torch.no_grad():
                clean_keys = ema_encoder(
                    raw_patches,
                    torch.ones_like(raw_patches, dtype=torch.bool),
                )
            return clean_keys.cpu().numpy()

        self.memory.rekey_batch(encode_members)
        self.memory.publish_epoch(epoch)
        self._current_epoch = None

    def update_relation_prior(self, output: OnlineStage1Forward, edge_gain_target: Tensor) -> None:
        if edge_gain_target.shape != output.predicted_gain.shape:
            raise ValueError("edge_gain_target must match model edges")
        variable_width = output.variable_candidates.source_variable.shape[-1]
        variable_offset = output.source_nodes.shape[-2] - variable_width
        variable_gain = edge_gain_target[..., variable_offset:]
        variable_eligible = output.eligible[..., variable_offset:] & output.edge_type[..., variable_offset:].eq(2)
        candidates = output.variable_candidates
        source = candidates.source_variable.clamp_min(0)
        target = torch.arange(
            self.variable_count, device=source.device, dtype=source.dtype
        ).view(1, self.variable_count, 1, 1).expand_as(source)
        lag_index = candidates.lag + 3
        flat_index = (
            (source * self.variable_count + target) * len(VARIABLE_LAGS)
            + lag_index
        ).reshape(-1)
        weights = variable_eligible.to(self.relation_prior.dtype).reshape(-1)
        flat_size = self.variable_count * self.variable_count * len(VARIABLE_LAGS)
        sums = torch.zeros(flat_size, dtype=self.relation_prior.dtype, device=source.device)
        counts = torch.zeros_like(sums)
        sums.scatter_add_(0, flat_index, variable_gain.detach().reshape(-1) * weights)
        counts.scatter_add_(0, flat_index, weights)
        sums = sums.reshape_as(self.relation_prior)
        counts = counts.reshape_as(self.relation_prior)
        valid = counts > 0
        means = sums / counts.clamp_min(1.0)
        self.relation_prior.copy_(torch.where(valid, self.relation_prior * self.relation_prior_decay + (1.0 - self.relation_prior_decay) * means, self.relation_prior))

    def update_training_statistics(self, result: OnlineStage1TrainingResult) -> None:
        """Apply legacy relation-prior observations after an optimizer step."""

        self.update_relation_prior(result.output, result.edge_gain_target)

    def online_state(self) -> dict[str, object]:
        relation_state = {"format_version": 1, "decay": self.relation_prior_decay, "values": self.relation_prior.detach().cpu().clone()}
        ema_state = self.retrieval.state_dict()
        ema_hash = sha256()
        for name, value in ema_state["ema_state"]:  # type: ignore[index]
            ema_hash.update(name.encode())
            ema_hash.update(np.asarray(value, dtype=np.float32).tobytes())
        relation_hash = sha256(self.relation_prior.detach().cpu().numpy().tobytes()).hexdigest()
        return {
            "epoch": self.memory.snapshot().epoch,
            "memory_state": self.memory.state_dict(),
            "ema_state": ema_state,
            "relation_prior_state": relation_state,
            "memory_hash": self.memory.snapshot_hash(),
            "ema_hash": ema_hash.hexdigest(),
            "relation_prior_hash": relation_hash,
        }

    def load_online_state(self, online_state: Mapping[str, object]) -> None:
        self.memory.load_state_dict(online_state["memory_state"])
        self.retrieval.load_state_dict(online_state["ema_state"])
        relation_state = online_state["relation_prior_state"]
        if not isinstance(relation_state, Mapping) or relation_state.get("format_version") != 1:
            raise ValueError("legacy online checkpoint requires relation state format 1")
        values = relation_state.get("values")
        if not isinstance(values, Tensor) or values.shape != self.relation_prior.shape:
            raise ValueError("online checkpoint relation prior shape does not match model")
        self.relation_prior.copy_(values.to(self.relation_prior))


__all__ = [
    "OnlineStage1Forward",
    "OnlineStage1Model",
    "OnlineStage1TrainingResult",
    "RelationEdgeGainScorer",
]
