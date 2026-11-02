"""E22 composition root: protected E21 H/V plus an independent temporal repair."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Iterator

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from malr.data.patching import PATCH_OBSERVED
from malr.models.candidates_v2 import (
    TEMPORAL_FEATURE_NAMES,
    VariableCandidateBatch,
    build_temporal_candidates,
    build_variable_candidates,
)
from malr.models.dynamic_temporal_context import (
    DynamicTemporalContextOutput,
    DynamicTemporalContextRepair,
)
from malr.models.branch_recovery import BranchRecoveryOutput
from malr.models.explicit_repair import ExplicitRepairBatch, build_explicit_repair
from malr.models.forecast_fusion import ForecastHVFusion
from malr.models.online_stage1_v2 import (
    RelationV2Forward,
    RelationV2OnlineStage1Model,
    VariableStatisticsUpdate,
)
from malr.models.relation_outputs import RelationBranchOutput, stack_branch_outputs
from malr.models.routing_v2 import (
    direct_single_route_select,
    route_availability_mask,
    route_relation_mask,
)
from malr.models.reconstruction import aggregate_patch_predictions
from malr.prototypes.retrieval import masked_contrastive_loss, masked_pair_loss


@dataclass(frozen=True)
class DynamicContextForward:
    forecast: Tensor | None
    normalized_forecast: Tensor | None
    base_latents: Tensor
    latents: Tensor
    forecast_latents: Tensor | None
    reconstructed_patches: Tensor
    hv_latents: Tensor
    hv_route_weights: Tensor
    historical: RelationBranchOutput
    variable: RelationBranchOutput
    variable_candidates: VariableCandidateBatch | None
    graph: BranchRecoveryOutput
    route_logits: Tensor
    route_index: Tensor
    route_available: Tensor
    temporal: DynamicTemporalContextOutput
    explicit: ExplicitRepairBatch
    diagnostics: dict[str, Tensor]


@dataclass(frozen=True)
class DynamicContextLoss:
    total: Tensor
    repair: Tensor
    gain: Tensor
    weighted_gain: Tensor
    memory: Tensor
    weighted_memory: Tensor
    forecast: Tensor


@dataclass(frozen=True)
class DynamicContextTrainingResult:
    output: DynamicContextForward
    loss: DynamicContextLoss
    candidate_gain_targets: Tensor
    variable_statistics_update: VariableStatisticsUpdate
    objective: str


class DynamicContextOnlineModel(RelationV2OnlineStage1Model):
    """Use E21 H/V operators unchanged and add T after their fused latent."""

    def __init__(
        self,
        *args: object,
        temporal_residual_limit: float = 2.0,
        stage2_repair_lr_scale: float = 0.25,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.temporal_residual_limit = float(temporal_residual_limit)
        self.stage2_repair_lr_scale = float(stage2_repair_lr_scale)
        if not np.isfinite(self.temporal_residual_limit) or self.temporal_residual_limit <= 0:
            raise ValueError("temporal_residual_limit must be finite and positive")
        if not np.isfinite(self.stage2_repair_lr_scale) or not 0.0 < self.stage2_repair_lr_scale <= 1.0:
            raise ValueError("stage2_repair_lr_scale must lie in (0, 1]")
        self.dynamic_temporal = DynamicTemporalContextRepair(
            d_model=self.core.d_model,
            patch_len=self.core.patch_len,
            feature_dim=len(TEMPORAL_FEATURE_NAMES),
            residual_limit=self.temporal_residual_limit,
        )
        self.install_forecast_hv_fusion()
        self.forecast_core: nn.Module | None = None
        self.forecast_mask_projection: nn.Linear | None = None
        self._training_stage = "repair"
        # The old E21 T operator remains only for checkpoint compatibility.  It
        # is never called by this class and never enters either E22 optimizer.
        self.temporal_operator.requires_grad_(False)
        self.configure_training_stage("repair")

    def _empty_old_temporal(self, nodes: Tensor) -> RelationBranchOutput:
        return self._empty_branch(nodes, nodes.shape[2] - 1)

    def _route_logits_hv(self, branch_gain: Tensor) -> Tensor:
        logits = branch_gain.new_full((*branch_gain.shape[:-1], 8), -1.0e9)
        logits[..., 0] = 0.0
        logits[..., 1] = branch_gain[..., 0]
        logits[..., 2] = -1.0e9
        logits[..., 3] = branch_gain[..., 2]
        return logits

    def _forward_hv_branches(
        self,
        patch_values: Tensor,
        patch_mask: Tensor,
        rho: Tensor,
        state: Tensor,
        query_mean: Tensor,
        query_std: Tensor,
        starts: Tensor,
        base_latents: Tensor,
        damaged: Tensor,
    ) -> tuple[RelationBranchOutput, RelationBranchOutput, VariableCandidateBatch | None]:
        has_damaged = bool(damaged.any())
        if not has_damaged:
            return (
                self._empty_branch(base_latents, self.historical_topk),
                self._empty_branch(
                    base_latents, (self.variable_count - 1) * patch_values.shape[2]
                ),
                None,
            )
        stats_shape = (*query_mean.shape, 1, 1)
        retrieval_values = patch_values * query_std.reshape(stats_shape) + query_mean.reshape(stats_shape)
        variable_candidates = build_variable_candidates(
            patch_values,
            patch_mask,
            rho,
            target_eligible=damaged,
            distance_boundaries=self.temporal_distance_boundaries,
        )
        if self.route_teacher is None:
            historical_candidates = self._historical_candidates(
                retrieval_values,
                patch_mask,
                rho,
                state,
                starts,
                base_latents,
                query_mean,
                query_std,
            )
            historical = self.historical_operator(base_latents, historical_candidates)
            variable = self.variable_operator(
                base_latents, variable_candidates, self.variable_statistics
            )
            return historical, variable, variable_candidates

        teacher = self.route_teacher
        assert teacher is not None
        with torch.no_grad():
            teacher_base = teacher.project_clean_patches(patch_values)
            teacher_historical_candidates = self._historical_candidates(
                retrieval_values,
                patch_mask,
                rho,
                state,
                starts,
                teacher_base,
                query_mean,
                query_std,
                projection_core=teacher,
            )
            teacher_historical, historical_plan = teacher.historical_operator.forward_with_plan(
                teacher_base, teacher_historical_candidates
            )
            teacher_variable, variable_plan = teacher.variable_operator.forward_with_plan(
                teacher_base, variable_candidates, self.variable_statistics
            )
            historical_plan = historical_plan.detached()
            variable_plan = variable_plan.detached()
        historical_candidates = self._historical_candidates(
            retrieval_values,
            patch_mask,
            rho,
            state,
            starts,
            base_latents,
            query_mean,
            query_std,
        )
        historical_messages = self.historical_operator.project_candidate_messages(
            base_latents, historical_candidates
        )
        historical = self._with_student_messages(
            teacher_historical,
            historical_messages,
            self.historical_operator.replay_message(historical_messages, historical_plan),
        )
        variable_messages = self.variable_operator.project_candidate_messages(
            base_latents, variable_candidates
        )
        variable = self._with_student_messages(
            teacher_variable,
            variable_messages,
            self.variable_operator.replay_message(variable_messages, variable_plan),
        )
        return historical, variable, variable_candidates

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
        patch_validity: Tensor | None = None,
        disabled_relations: frozenset[str] = frozenset(),
    ) -> DynamicContextForward:
        if patch_values.ndim != 4 or patch_values.shape[1] != self.variable_count:
            raise ValueError("patch_values must have shape [B,C,N,P]")
        if patch_mask.shape != patch_values.shape or rho.shape != patch_values.shape[:-1] or state.shape != rho.shape:
            raise ValueError("patch mask, rho, and state do not match patch values")
        if query_mean.shape != (patch_values.shape[0], self.variable_count) or query_std.shape != query_mean.shape:
            raise ValueError("query normalization statistics have invalid shape")
        if not disabled_relations <= frozenset(("historical", "temporal", "variable")):
            raise ValueError("disabled_relations contains an unknown relation")
        valid = torch.ones_like(state, dtype=torch.bool) if patch_valid is None else patch_valid.to(device=state.device, dtype=torch.bool)
        if valid.shape != state.shape:
            raise ValueError("patch_valid must match state")
        coordinate_validity = (
            valid.unsqueeze(-1).expand_as(patch_mask)
            if patch_validity is None
            else patch_validity.to(device=patch_values.device, dtype=torch.bool)
        )
        if coordinate_validity.shape != patch_mask.shape:
            raise ValueError("patch_validity must match patch_mask")
        damaged = state.ne(PATCH_OBSERVED) & valid
        starts = self._starts(
            patch_starts,
            patch_values.shape[0],
            patch_values.shape[2],
            self.core.stride,
            patch_values.device,
        )
        base_latents = self.core.project_clean_patches(patch_values)
        historical, variable, variable_candidates = self._forward_hv_branches(
            patch_values,
            patch_mask,
            rho,
            state,
            query_mean,
            query_std,
            starts,
            base_latents,
            damaged,
        )
        legacy_temporal = self._empty_old_temporal(base_latents)
        branch_messages, branch_gain, branch_uncertainty, _ = stack_branch_outputs(
            historical, legacy_temporal, variable
        )
        branch_available = torch.stack(
            (historical.available, legacy_temporal.available, variable.available), dim=-1
        )
        disabled = torch.tensor(
            [name in disabled_relations for name in ("historical", "temporal", "variable")],
            device=patch_values.device,
            dtype=torch.bool,
        )
        branch_available = branch_available & ~disabled.view(1, 1, 1, 3)
        branch_available[..., 1] = False
        historical = self._with_availability(historical, branch_available[..., 0])
        variable = self._with_availability(variable, branch_available[..., 2])
        branch_messages, branch_gain, branch_uncertainty, _ = stack_branch_outputs(
            historical, legacy_temporal, variable
        )
        route_available = route_availability_mask(branch_available).clone()
        route_available[..., 2] = False
        route_available[..., 4:] = False
        if epoch_index <= self.bootstrap_epochs:
            uses_historical = route_relation_mask(device=patch_values.device)[:, 0]
            route_available = route_available & ~uses_historical.view(1, 1, 1, -1)
        if not bool(damaged.any()):
            route_available = torch.zeros_like(route_available)
            route_available[..., 0] = True
        route_logits = self._route_logits_hv(branch_gain)
        route_index = direct_single_route_select(branch_gain, branch_available)
        graph = self.graph(
            base_latents,
            branch_messages,
            branch_gain - branch_uncertainty,
            branch_available,
            route_index,
            damaged,
        )
        missing_indicator = (~patch_mask.to(torch.bool)) & coordinate_validity
        assert self.forecast_fusion is not None
        hv_latents, hv_route_weights = self.forecast_fusion(
            graph.route_latents,
            route_available,
            route_logits.detach(),
            missing_indicator,
        )
        hv_patches = self.patch_decoder(hv_latents)
        temporal_candidates = build_temporal_candidates(
            patch_values,
            patch_mask,
            rho,
            starts,
            self.temporal_distance_boundaries,
            target_eligible=damaged,
        )
        temporal = self.dynamic_temporal(
            hv_latents,
            hv_patches,
            patch_values,
            patch_mask,
            coordinate_validity,
            rho,
            temporal_candidates,
            damaged,
        )
        if "temporal" in disabled_relations:
            temporal = replace(
                temporal,
                residual_patches=torch.zeros_like(temporal.residual_patches),
            )
        candidate_patches = torch.where(
            patch_mask.to(torch.bool),
            patch_values,
            hv_patches + temporal.residual_patches,
        )
        explicit = build_explicit_repair(
            candidate_patches,
            patch_values,
            patch_mask,
            coordinate_validity,
            query_mean,
            query_std,
            window_len=self.core.seq_len,
            patch_len=self.core.patch_len,
            stride=self.core.stride,
            strict_provenance=False,
        )
        normalized_forecast: Tensor | None = None
        forecast: Tensor | None = None
        forecast_latents: Tensor | None = None
        if self.forecast_core is not None:
            assert self.forecast_mask_projection is not None
            forecast_latents = self.forecast_core.project_clean_patches(explicit.repaired_patches)
            forecast_missing = (~explicit.observation_patches) & explicit.validity_patches
            forecast_latents = forecast_latents + self.forecast_mask_projection(
                forecast_missing.to(forecast_latents.dtype)
            )
            normalized_forecast = self.forecast_core.forecast_from_latents(forecast_latents)
            forecast = normalized_forecast * query_std.unsqueeze(-1) + query_mean.unsqueeze(-1)
        diagnostics = {
            "route_index": route_index.detach(),
            "route_available": route_available.detach(),
            "hv_route_weights": hv_route_weights.detach(),
            "damaged": damaged.detach(),
            "missing_indicator": missing_indicator.detach(),
            "temporal_local_selected": temporal.local_selected.detach(),
            "temporal_global_selected": temporal.global_selected.detach(),
        }
        return DynamicContextForward(
            forecast=forecast,
            normalized_forecast=normalized_forecast,
            base_latents=base_latents,
            latents=hv_latents,
            forecast_latents=forecast_latents,
            reconstructed_patches=explicit.repaired_patches,
            hv_latents=hv_latents,
            hv_route_weights=hv_route_weights,
            historical=historical,
            variable=variable,
            variable_candidates=variable_candidates,
            graph=graph,
            route_logits=route_logits,
            route_index=route_index,
            route_available=route_available,
            temporal=temporal,
            explicit=explicit,
            diagnostics=diagnostics,
        )

    def install_forecast_predictor(self) -> None:
        if self.forecast_core is not None or self.forecast_mask_projection is not None:
            raise RuntimeError("forecast predictor is already installed")
        self.forecast_core = deepcopy(self.core)
        self.forecast_core.head.linear.reset_parameters()  # type: ignore[attr-defined]
        self.forecast_mask_projection = nn.Linear(
            self.core.patch_len, self.core.d_model, bias=False
        ).to(device=self.core.W_P.weight.device, dtype=self.core.W_P.weight.dtype)
        nn.init.zeros_(self.forecast_mask_projection.weight)

    def install_forecast_hv_fusion(self) -> None:
        """Keep the E21 H/V fusion singleton across CLI handoff paths."""

        if self.forecast_fusion is None:
            super().install_forecast_hv_fusion()

    def configure_training_stage(self, stage: str) -> None:
        if stage not in {"repair", "forecast"}:
            raise ValueError("stage must be 'repair' or 'forecast'")
        if stage == "forecast" and self.route_teacher is None:
            raise RuntimeError("forecast stage requires a frozen route teacher")
        if stage == "forecast" and self.forecast_core is None:
            raise RuntimeError("forecast stage requires an installed predictor")
        self.requires_grad_(False)
        self._training_stage = stage
        self.core.W_P.requires_grad_(True)
        self.patch_decoder.requires_grad_(True)
        self.graph.requires_grad_(True)
        self.forecast_fusion.requires_grad_(True)  # type: ignore[union-attr]
        self.dynamic_temporal.requires_grad_(True)
        self.historical_operator.requires_grad_(True)
        self.variable_operator.requires_grad_(True)
        if stage == "repair":
            self.retrieval_encoder.requires_grad_(True)
        else:
            for module in (
                self.historical_operator.message_head,
                self.variable_operator.source_embedding,
                self.variable_operator.target_embedding,
                self.variable_operator.lag_embedding,
                self.variable_operator.translator,
            ):
                module.requires_grad_(True)
            assert self.forecast_core is not None and self.forecast_mask_projection is not None
            self.forecast_core.requires_grad_(True)
            self.forecast_mask_projection.requires_grad_(True)
        if self.route_teacher is not None:
            self.route_teacher.requires_grad_(False)
            self.route_teacher.eval()
        self.temporal_operator.requires_grad_(False)
        self.mask_projection.requires_grad_(False)
        self.core.W_pos.requires_grad_(False)
        self.core.encoder.requires_grad_(False)
        self.core.head.requires_grad_(False)

    def configure_forecast_finetuning(self, **_: object) -> None:
        self.configure_training_stage("forecast")

    def backbone_parameters(self) -> Iterator[nn.Parameter]:
        modules: list[nn.Module] = [self.core]
        if self.forecast_core is not None:
            modules.append(self.forecast_core)
        if self.forecast_mask_projection is not None:
            modules.append(self.forecast_mask_projection)
        seen: set[int] = set()
        for module in modules:
            for parameter in module.parameters():
                if parameter.requires_grad and id(parameter) not in seen:
                    seen.add(id(parameter))
                    yield parameter

    def relation_parameters(self) -> Iterator[nn.Parameter]:
        modules: tuple[nn.Module, ...] = (
            self.retrieval_encoder,
            self.historical_operator,
            self.variable_operator,
            self.graph,
            self.patch_decoder,
            self.dynamic_temporal,
            self.forecast_fusion,  # type: ignore[arg-type]
        )
        backbone_ids = {id(parameter) for parameter in self.backbone_parameters()}
        seen = set(backbone_ids)
        for module in modules:
            for parameter in module.parameters():
                if parameter.requires_grad and id(parameter) not in seen:
                    seen.add(id(parameter))
                    yield parameter

    def repair_parameters(self) -> Iterator[nn.Parameter]:
        self.configure_training_stage("repair")
        yield from self.backbone_parameters()
        yield from self.relation_parameters()

    def forecast_parameters(self) -> Iterator[nn.Parameter]:
        self.configure_training_stage("forecast")
        yield from self.backbone_parameters()
        yield from self.relation_parameters()

    @staticmethod
    def _empty_statistics_update(output: DynamicContextForward, device: torch.device) -> VariableStatisticsUpdate:
        shape = output.variable.candidate_eligible.shape
        empty = torch.zeros(shape, dtype=torch.long, device=device)
        return VariableStatisticsUpdate(
            source_variable=empty,
            target_variable=empty,
            signed_bucket=empty,
            gain=torch.zeros(shape, device=device),
            eligible=torch.zeros(shape, dtype=torch.bool, device=device),
            route_errors=torch.zeros((*shape[:-1], 8), device=device),
            route_available=torch.zeros((*shape[:-1], 8), dtype=torch.bool, device=device),
        )

    def training_objective(
        self,
        patch_values: Tensor,
        patch_mask: Tensor,
        rho: Tensor,
        state: Tensor,
        query_mean: Tensor,
        query_std: Tensor,
        forecast_target: Tensor,
        clean_raw_patches: Tensor | None,
        *,
        patch_starts: Tensor | None = None,
        epoch_index: int = 0,
        patch_valid: Tensor | None = None,
        patch_validity: Tensor | None = None,
        objective: str = "repair",
    ) -> DynamicContextTrainingResult:
        if objective not in {"repair", "forecast"}:
            raise ValueError("dynamic context objective must be 'repair' or 'forecast'")
        if objective == "repair" and clean_raw_patches is None:
            raise ValueError("repair objective requires clean_raw_patches")
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
            patch_validity=patch_validity,
        )
        if objective == "forecast":
            if output.forecast is None:
                raise RuntimeError("forecast objective requires an installed predictor")
            forecast_loss = F.mse_loss(output.forecast, forecast_target)
            zero = forecast_loss * 0.0
            loss = DynamicContextLoss(forecast_loss, zero, zero, zero, zero, zero, forecast_loss)
            return DynamicContextTrainingResult(
                output,
                loss,
                torch.zeros_like(output.variable.candidate_predicted_gain),
                self._empty_statistics_update(output, patch_values.device),
                objective,
            )

        assert clean_raw_patches is not None
        coordinate_validity = (
            patch_valid.unsqueeze(-1).expand_as(patch_mask)
            if patch_validity is None
            else patch_validity.to(device=patch_mask.device, dtype=torch.bool)
        )
        clean_sequence, _ = aggregate_patch_predictions(
            clean_raw_patches,
            window_len=self.core.seq_len,
            patch_len=self.core.patch_len,
            stride=self.core.stride,
            validity=coordinate_validity,
        )
        missing = output.explicit.sequence_missing
        if missing.any():
            repair = F.smooth_l1_loss(
                output.explicit.standardized_sequence[missing], clean_sequence[missing]
            )
        else:
            repair = output.explicit.standardized_sequence.sum() * 0.0
        damaged = state.ne(PATCH_OBSERVED) & (
            torch.ones_like(state, dtype=torch.bool)
            if patch_valid is None
            else patch_valid.to(device=state.device, dtype=torch.bool)
        )
        missing_mask = (~patch_mask.to(torch.bool)) & coordinate_validity & damaged.unsqueeze(-1)
        target_eligible = damaged & missing_mask.any(dim=-1)
        gains: list[Tensor] = []
        eligible: list[Tensor] = []
        for branch in (output.historical, output.variable):
            branch_gain, branch_eligible = self._candidate_counterfactual_gain(
                RelationV2Forward(
                    forecast=output.forecast if output.forecast is not None else forecast_target,
                    normalized_forecast=output.normalized_forecast if output.normalized_forecast is not None else forecast_target,
                    base_latents=output.base_latents,
                    latents=output.latents,
                    forecast_latents=output.forecast_latents if output.forecast_latents is not None else output.latents,
                    reconstructed_patches=output.reconstructed_patches,
                    historical=output.historical,
                    temporal=self._empty_old_temporal(output.base_latents),
                    variable=output.variable,
                    variable_candidates=output.variable_candidates,
                    graph=output.graph,
                    route_logits=output.route_logits,
                    route_index=output.route_index,
                    route_available=output.route_available,
                    diagnostics=output.diagnostics,
                ),
                branch,
                query_mean,
                query_std,
                clean_raw_patches,
                missing_mask,
                target_eligible,
            )
            gains.append(branch_gain)
            eligible.append(branch_eligible)
        predicted = torch.cat(
            (output.historical.candidate_predicted_gain, output.variable.candidate_predicted_gain),
            dim=-1,
        )
        targets = torch.cat(gains, dim=-1)
        all_eligible = torch.cat(eligible, dim=-1)
        relation_type = torch.cat(
            (
                torch.zeros_like(output.historical.candidate_eligible, dtype=torch.long),
                torch.ones_like(output.variable.candidate_eligible, dtype=torch.long),
            ),
            dim=-1,
        )
        from malr.models.edge_gain import balanced_edge_gain_loss

        gain = balanced_edge_gain_loss(
            predicted,
            targets,
            all_eligible,
            relation_type,
            relation_type_count=2,
        )
        stats_shape = (*query_mean.shape, 1, 1)
        retrieval_query = patch_values * query_std.reshape(stats_shape) + query_mean.reshape(stats_shape)
        flat_query = retrieval_query.reshape(-1, patch_values.shape[-1])
        flat_mask = patch_mask.reshape(-1, patch_mask.shape[-1])
        flat_clean = clean_raw_patches.reshape(-1, clean_raw_patches.shape[-1])
        flat_damaged = target_eligible.reshape(-1)
        if flat_damaged.any():
            query_keys = self.retrieval.encoder(flat_query[flat_damaged], flat_mask[flat_damaged])
            clean_keys = self.retrieval.encoder(flat_clean[flat_damaged], torch.ones_like(flat_mask[flat_damaged]))
            contrastive = masked_contrastive_loss(query_keys, clean_keys)
            normalizer = max(float(np.log(max(query_keys.shape[0], 2))), 1.0)
            memory = masked_pair_loss(query_keys, clean_keys) + 0.5 * contrastive / normalizer
        else:
            memory = gain * 0.0
        weighted_gain = self.gain_loss_weight * gain
        weighted_memory = self.memory_loss_weight * memory
        total = repair + weighted_gain + weighted_memory
        loss = DynamicContextLoss(total, repair, gain, weighted_gain, memory, weighted_memory, repair * 0.0)
        variable_shape = output.variable.candidate_eligible.shape
        variable_target = torch.arange(self.variable_count, device=patch_values.device, dtype=torch.long).view(1, self.variable_count, 1, 1).expand(variable_shape)
        variable_gain = gains[-1]
        variable_eligible = eligible[-1]
        source_variable = torch.zeros_like(variable_target)
        signed_bucket = torch.zeros_like(variable_target)
        if output.variable_candidates is not None:
            source_variable = output.variable_candidates.source_variable
            signed_bucket = output.variable.diagnostics["signed_bucket"]
        update = VariableStatisticsUpdate(
            source_variable=source_variable,
            target_variable=variable_target,
            signed_bucket=signed_bucket,
            gain=variable_gain,
            eligible=variable_eligible,
            route_errors=torch.zeros((*variable_shape[:-1], 8), device=patch_values.device),
            route_available=torch.zeros((*variable_shape[:-1], 8), dtype=torch.bool, device=patch_values.device),
        )
        return DynamicContextTrainingResult(output, loss, targets, update, objective)

    def update_training_statistics(self, result: DynamicContextTrainingResult) -> None:
        if not self.training or result.objective == "forecast":
            return
        update = result.variable_statistics_update
        self.variable_statistics.update(
            update.source_variable,
            update.target_variable,
            update.signed_bucket,
            update.gain,
            update.eligible,
        )


__all__ = [
    "DynamicContextForward",
    "DynamicContextLoss",
    "DynamicContextOnlineModel",
    "DynamicContextTrainingResult",
]
