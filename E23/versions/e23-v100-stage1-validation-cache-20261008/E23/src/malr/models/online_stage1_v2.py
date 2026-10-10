"""Composition root for target-space relation V2 online recovery."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from hashlib import sha256

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from malr.data.patching import PATCH_OBSERVED
from malr.prototypes.online_bank import OnlinePrototypeMemory, OnlinePrototypeMember
from malr.prototypes.retrieval import (
    RetrievalEncoderEMA,
    masked_contrastive_loss,
    masked_pair_loss,
)
from malr.training.unified_objective import UnifiedStage1Loss, unified_stage1_loss

from .branch_recovery import BranchRecoveryOutput, BranchRoutedResidualRecovery
from .candidates_v2 import (
    TEMPORAL_FEATURE_NAMES,
    VARIABLE_FEATURE_NAMES,
    TemporalCandidateBatch,
    VariableCandidateBatch,
    build_temporal_candidates,
    build_variable_candidates,
)
from .historical_relation import HistoricalRelationOperator
from .forecast_fusion import ForecastHVFusion
from .edge_gain import (
    balanced_edge_gain_loss,
    counterfactual_relative_gain,
    masked_patch_huber_error,
)
from .online_history import HistoricalCandidateBatch, build_online_historical_candidates
from .patchtst import PatchTSTCore
from .relation_outputs import RelationBranchOutput, stack_branch_outputs
from .route_teacher import FrozenRouteTeacher
from .routing_v2 import (
    RouteStabilityTracker,
    V2_ROUTE_OPTIONS,
    direct_single_route_select,
    oracle_route_labels,
    route_availability_mask,
    route_relation_mask,
)
from .temporal_relation import DynamicTemporalRelation
from .variable_relation import DirectedVariableRelation, VariableRelationStatistics


_RELATION_NAMES = frozenset(("historical", "temporal", "variable"))


@dataclass(frozen=True)
class RelationV2Forward:
    forecast: Tensor
    normalized_forecast: Tensor
    base_latents: Tensor
    latents: Tensor
    forecast_latents: Tensor
    reconstructed_patches: Tensor
    historical: RelationBranchOutput
    temporal: RelationBranchOutput
    variable: RelationBranchOutput
    variable_candidates: VariableCandidateBatch | None
    graph: BranchRecoveryOutput
    route_logits: Tensor
    route_index: Tensor
    route_available: Tensor
    diagnostics: dict[str, Tensor]


@dataclass(frozen=True)
class VariableStatisticsUpdate:
    """Detached V2 statistics observations, applied only by the trainer."""

    source_variable: Tensor
    target_variable: Tensor
    signed_bucket: Tensor
    gain: Tensor
    eligible: Tensor
    route_errors: Tensor
    route_available: Tensor


@dataclass(frozen=True)
class RelationV2TrainingResult:
    output: RelationV2Forward
    loss: UnifiedStage1Loss
    candidate_gain_targets: Tensor
    route_target: Tensor
    route_errors: Tensor
    branch_repair_losses: Tensor
    variable_statistics_update: VariableStatisticsUpdate
    objective: str


def pairwise_message_cosine(branch_messages: Tensor) -> Tensor:
    """Return the H/T, H/V, and T/V cosine similarities for each target."""

    if branch_messages.ndim != 5 or branch_messages.shape[-2] != 3:
        raise ValueError("branch_messages must have shape [B,C,N,3,D]")
    normalized = torch.nn.functional.normalize(branch_messages, dim=-1, eps=1e-8)
    return torch.stack(
        (
            (normalized[..., 0, :] * normalized[..., 1, :]).sum(dim=-1),
            (normalized[..., 0, :] * normalized[..., 2, :]).sum(dim=-1),
            (normalized[..., 1, :] * normalized[..., 2, :]).sum(dim=-1),
        ),
        dim=-1,
    )


class RelationV2OnlineStage1Model(nn.Module):
    """Compose H/T/V target-space branches without changing legacy V1 behavior."""

    def __init__(
        self,
        core: PatchTSTCore,
        *,
        memory: OnlinePrototypeMemory,
        retrieval: RetrievalEncoderEMA,
        variable_count: int,
        temporal_distance_boundaries: tuple[int, ...],
        historical_topk: int,
        temporal_topk: int,
        variable_topk: int,
        candidate_gain_threshold: float,
        route_improvement_threshold: float,
        route_stability_min_targets: int,
        bootstrap_epochs: int,
        joint_end_epoch: int,
        historical_similarity_threshold: float = 0.7,
        repair_loss_weight: float = 1.0,
        gain_loss_weight: float = 0.20,
        memory_loss_weight: float = 0.05,
    ) -> None:
        super().__init__()
        if core.c_in != variable_count or memory.variable_count != variable_count:
            raise ValueError("core, memory and variable_count must agree")
        if retrieval.encoder.patch_len != core.patch_len:
            raise ValueError("retrieval patch length must match PatchTST")
        if not temporal_distance_boundaries:
            raise ValueError("temporal_distance_boundaries must not be empty")
        if tuple(sorted(set(temporal_distance_boundaries))) != temporal_distance_boundaries:
            raise ValueError("temporal_distance_boundaries must be strictly increasing")
        if any(boundary <= 0 for boundary in temporal_distance_boundaries):
            raise ValueError("temporal_distance_boundaries must be positive")
        if any(value <= 0 for value in (historical_topk, temporal_topk, variable_topk)):
            raise ValueError("relation top-k values must be positive")
        if not 0.0 <= historical_similarity_threshold <= 1.0:
            raise ValueError("historical similarity threshold must lie in [0, 1]")
        if bootstrap_epochs <= 0 or joint_end_epoch < bootstrap_epochs:
            raise ValueError("stage boundaries must satisfy 0 < bootstrap <= joint end")
        for name, value in (
            ("repair_loss_weight", repair_loss_weight),
            ("gain_loss_weight", gain_loss_weight),
            ("memory_loss_weight", memory_loss_weight),
        ):
            scalar = torch.as_tensor(value)
            if not torch.isfinite(scalar) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if repair_loss_weight <= 0.0:
            raise ValueError("repair_loss_weight must be positive")

        self.core = core
        self.memory = memory
        self.retrieval = retrieval
        self.add_module("retrieval_encoder", retrieval.encoder)
        self.variable_count = int(variable_count)
        self.temporal_distance_boundaries = temporal_distance_boundaries
        self.historical_topk = int(historical_topk)
        self.temporal_topk = int(temporal_topk)
        self.variable_topk = int(variable_topk)
        self.candidate_gain_threshold = float(candidate_gain_threshold)
        self.historical_similarity_threshold = float(historical_similarity_threshold)
        self.bootstrap_epochs = int(bootstrap_epochs)
        self.joint_end_epoch = int(joint_end_epoch)
        self.repair_loss_weight = float(repair_loss_weight)
        self.gain_loss_weight = float(gain_loss_weight)
        self.memory_loss_weight = float(memory_loss_weight)

        self.historical_operator = HistoricalRelationOperator(
            core.d_model, feature_dim=11, max_items=historical_topk
        )
        self.temporal_operator = DynamicTemporalRelation(
            core.d_model,
            feature_dim=len(TEMPORAL_FEATURE_NAMES),
            distance_band_count=len(temporal_distance_boundaries) + 1,
            candidate_gain_threshold=candidate_gain_threshold,
        )
        lag_band_count = len(temporal_distance_boundaries) + 1
        signed_bucket_count = 2 * lag_band_count + 1
        self.variable_statistics = VariableRelationStatistics(
            variable_count, signed_bucket_count, min_count=route_stability_min_targets
        )
        self.variable_operator = DirectedVariableRelation(
            core.d_model,
            feature_dim=len(VARIABLE_FEATURE_NAMES),
            variable_count=variable_count,
            signed_bucket_count=signed_bucket_count,
            topk=variable_topk,
            candidate_gain_threshold=candidate_gain_threshold,
        )
        self.graph = BranchRoutedResidualRecovery(core.d_model)
        self.patch_decoder = nn.Linear(core.d_model, core.patch_len)
        self.mask_projection = nn.Linear(core.patch_len, core.d_model, bias=False)
        nn.init.zeros_(self.mask_projection.weight)
        self.route_statistics = RouteStabilityTracker(
            base_margin=route_improvement_threshold,
            min_targets=route_stability_min_targets,
        )
        self.route_teacher: FrozenRouteTeacher | None = None
        self.forecast_fusion: ForecastHVFusion | None = None
        self._forecast_fusion_mode = "hard"
        self._current_epoch: int | None = None

    @property
    def has_frozen_route_teacher(self) -> bool:
        return self.route_teacher is not None

    def install_frozen_route_teacher(self) -> str:
        """Snapshot the current finalized Stage-1 route-producing function once."""

        if self.route_teacher is not None:
            raise RuntimeError("frozen route teacher is already installed")
        self.route_teacher = FrozenRouteTeacher.from_stage1(self)
        if any(parameter.requires_grad for parameter in self.route_teacher.parameters()):
            raise RuntimeError("installed route teacher must be frozen")
        self.route_teacher.eval()
        return self.route_teacher.state_sha256()

    @property
    def has_forecast_hv_fusion(self) -> bool:
        return self.forecast_fusion is not None

    def install_forecast_hv_fusion(self) -> None:
        """Install the E21 forecast gate without changing legacy checkpoints."""

        if self.forecast_fusion is not None:
            raise RuntimeError("forecast H/V fusion is already installed")
        reference = self.core.W_P.weight
        self.forecast_fusion = ForecastHVFusion(
            d_model=self.core.d_model,
            patch_len=self.core.patch_len,
        ).to(device=reference.device, dtype=reference.dtype)

    def train(self, mode: bool = True) -> "RelationV2OnlineStage1Model":
        result = super().train(mode)
        if result.route_teacher is not None:
            result.route_teacher.eval()
        return result

    def to(self, *args: object, **kwargs: object) -> "RelationV2OnlineStage1Model":
        result = super().to(*args, **kwargs)
        device = next(result.parameters()).device
        result.retrieval.ema_encoder.to(device)
        result.retrieval.frozen_encoder.to(device)
        return result

    @staticmethod
    def _starts(
        patch_starts: Tensor | None,
        batch: int,
        patch_count: int,
        stride: int,
        device: torch.device,
    ) -> Tensor:
        if patch_starts is None:
            return (torch.arange(patch_count, device=device, dtype=torch.long) * stride).unsqueeze(0).expand(batch, -1)
        starts = patch_starts.to(device=device, dtype=torch.long)
        if starts.ndim == 1:
            if starts.shape[0] != patch_count:
                raise ValueError("patch_starts must match patch count")
            starts = starts.unsqueeze(0).expand(batch, -1)
        if starts.shape != (batch, patch_count) or torch.any(starts < 0):
            raise ValueError("patch_starts must have shape [batch, patches]")
        return starts

    @staticmethod
    def _empty_branch(nodes: Tensor, width: int) -> RelationBranchOutput:
        shape = nodes.shape[:-1]
        messages = nodes.new_zeros(*shape, width, nodes.shape[-1])
        scalar = nodes.new_zeros(*shape, width)
        available = torch.zeros(*shape, dtype=torch.bool, device=nodes.device)
        selected = torch.zeros(*shape, width, dtype=torch.bool, device=nodes.device)
        return RelationBranchOutput(
            message=nodes.new_zeros(*shape, nodes.shape[-1]),
            predicted_gain=nodes.new_zeros(*shape),
            uncertainty=nodes.new_ones(*shape),
            available=available,
            candidate_messages=messages,
            candidate_predicted_gain=scalar,
            candidate_uncertainty=scalar,
            candidate_eligible=selected,
            candidate_selected=selected,
            diagnostics={},
        )

    def _historical_candidates(
        self,
        values: Tensor,
        mask: Tensor,
        rho: Tensor,
        state: Tensor,
        starts: Tensor,
        base_latents: Tensor,
        query_mean: Tensor,
        query_std: Tensor,
        *,
        projection_core: PatchTSTCore | FrozenRouteTeacher | None = None,
    ) -> HistoricalCandidateBatch:
        return build_online_historical_candidates(
            self.core if projection_core is None else projection_core,  # type: ignore[arg-type]
            self.memory,
            self.retrieval,
            values,
            mask,
            rho,
            state,
            starts,
            base_latents,
            max_items=self.historical_topk,
            similarity_threshold=self.historical_similarity_threshold,
            query_mean=query_mean,
            query_std=query_std,
        )

    @staticmethod
    def _with_availability(branch: RelationBranchOutput, available: Tensor) -> RelationBranchOutput:
        return RelationBranchOutput(
            message=branch.message,
            predicted_gain=branch.predicted_gain,
            uncertainty=branch.uncertainty,
            available=available,
            candidate_messages=branch.candidate_messages,
            candidate_predicted_gain=branch.candidate_predicted_gain,
            candidate_uncertainty=branch.candidate_uncertainty,
            candidate_eligible=branch.candidate_eligible,
            candidate_selected=branch.candidate_selected,
            diagnostics=branch.diagnostics,
        )

    @staticmethod
    def _with_student_messages(
        teacher_branch: RelationBranchOutput,
        candidate_messages: Tensor,
        message: Tensor,
    ) -> RelationBranchOutput:
        """Carry trainable message values under immutable teacher decisions."""

        return RelationBranchOutput(
            message=message,
            predicted_gain=teacher_branch.predicted_gain.detach(),
            uncertainty=teacher_branch.uncertainty.detach(),
            available=teacher_branch.available.detach(),
            candidate_messages=candidate_messages,
            candidate_predicted_gain=teacher_branch.candidate_predicted_gain.detach(),
            candidate_uncertainty=teacher_branch.candidate_uncertainty.detach(),
            candidate_eligible=teacher_branch.candidate_eligible.detach(),
            candidate_selected=teacher_branch.candidate_selected.detach(),
            diagnostics={
                name: value.detach()
                for name, value in teacher_branch.diagnostics.items()
            },
        )

    def backbone_parameters(self) -> Iterator[nn.Parameter]:
        yield from (parameter for parameter in self.core.parameters() if parameter.requires_grad)
        yield from (
            parameter
            for parameter in self.mask_projection.parameters()
            if parameter.requires_grad
        )

    def relation_parameters(self) -> Iterator[nn.Parameter]:
        relation_modules: tuple[nn.Module, ...] = (
            self.retrieval_encoder,
            self.historical_operator,
            self.temporal_operator,
            self.variable_operator,
            self.graph,
            self.patch_decoder,
        )
        if self._forecast_fusion_mode == "teacher_hv":
            assert self.forecast_fusion is not None
            relation_modules += (self.forecast_fusion,)
        for module in relation_modules:
            yield from (parameter for parameter in module.parameters() if parameter.requires_grad)

    def routing_parameters(self) -> Iterator[nn.Parameter]:
        yield from ()

    def configure_forecast_finetuning(
        self,
        *,
        frozen_route_teacher: bool = False,
        forecast_fusion: str = "hard",
    ) -> None:
        """Freeze Stage-1 scoring while keeping forecast feature paths trainable.

        Gain and uncertainty heads were calibrated against decoded repair
        utility during Stage 1.  Letting forecast-only gradients update those
        heads turns their outputs into unconstrained routing logits and erases
        that calibration.  Stage 2 therefore trains only PatchTST, the mask
        projection, relation message builders, and graph recovery.
        """

        if type(frozen_route_teacher) is not bool:
            raise TypeError("frozen_route_teacher must be bool")
        if forecast_fusion not in {"hard", "teacher_hv"}:
            raise ValueError("forecast_fusion must be 'hard' or 'teacher_hv'")
        if frozen_route_teacher and self.route_teacher is None:
            raise RuntimeError(
                "install a frozen route teacher before configuring teacher fine-tuning"
            )
        if forecast_fusion == "teacher_hv":
            if self.route_teacher is None:
                raise RuntimeError(
                    "install a frozen route teacher before configuring teacher H/V fusion"
                )
            if self.forecast_fusion is None:
                raise RuntimeError(
                    "install forecast H/V fusion before configuring teacher H/V fusion"
                )
        self.requires_grad_(False)
        trainable_modules: tuple[nn.Module, ...] = (
            self.core,
            self.mask_projection,
            self.historical_operator.message_head,
            self.variable_operator.source_embedding,
            self.variable_operator.target_embedding,
            self.variable_operator.lag_embedding,
            self.variable_operator.translator,
            self.graph,
        )
        if forecast_fusion == "hard":
            trainable_modules += (
                self.temporal_operator.candidate_projector,
                self.temporal_operator.fusion_gate,
            )
        for module in trainable_modules:
            module.requires_grad_(True)

        if forecast_fusion == "teacher_hv":
            assert self.forecast_fusion is not None
            self.forecast_fusion.requires_grad_(True)
        self._forecast_fusion_mode = forecast_fusion

        if self.route_teacher is not None:
            self.route_teacher.requires_grad_(False)
            self.route_teacher.eval()

    def training_phase(self, epoch_index: int) -> str:
        """Return the explicit T/V bootstrap, joint, or calibration phase."""

        if type(epoch_index) is not int or epoch_index < 0:
            raise ValueError("epoch_index must be a non-negative integer")
        if epoch_index <= self.bootstrap_epochs:
            return "bootstrap"
        if epoch_index <= self.joint_end_epoch:
            return "joint"
        return "calibration"

    def update_training_statistics(self, result: RelationV2TrainingResult) -> None:
        """Apply detached relation observations only after an optimizer step."""

        if not self.training:
            return
        if result.objective == "forecast":
            return
        update = result.variable_statistics_update
        self.variable_statistics.update(
            update.source_variable,
            update.target_variable,
            update.signed_bucket,
            update.gain,
            update.eligible,
        )
        self.route_statistics.update(update.route_errors, update.route_available)

    @staticmethod
    def _mean_supervised_patch_error(error: Tensor, eligible: Tensor) -> Tensor:
        """Average only target patches with an artificial missing coordinate."""

        if error.shape != eligible.shape:
            raise ValueError("error and eligible must match [B,C,N]")
        selected = error[eligible]
        if selected.numel() == 0:
            return error.sum() * 0.0
        return selected.mean()

    def _candidate_counterfactual_gain(
        self,
        output: RelationV2Forward,
        branch: RelationBranchOutput,
        query_mean: Tensor,
        query_std: Tensor,
        clean_raw_patches: Tensor,
        missing_mask: Tensor,
        target_eligible: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Decode residual-transformed branch candidates before assigning gain."""

        candidate_latents = self.graph.recover_candidates(
            output.base_latents,
            branch.candidate_messages,
            target_eligible,
        )
        baseline_prediction = self._decode_raw_patches(
            output.base_latents, query_mean, query_std
        )
        candidate_prediction = self._decode_raw_patches(
            candidate_latents, query_mean, query_std
        )
        gain = counterfactual_relative_gain(
            baseline_prediction,
            candidate_prediction,
            clean_raw_patches,
            missing_mask,
        )
        eligible = branch.candidate_eligible & target_eligible.unsqueeze(-1)
        return gain, eligible

    def _decode_raw_patches(
        self, latents: Tensor, query_mean: Tensor, query_std: Tensor
    ) -> Tensor:
        """Decode normalized patch coordinates and restore raw query units."""

        decoded = self.patch_decoder(latents)
        stats_shape = (*query_mean.shape, *((1,) * (decoded.ndim - query_mean.ndim)))
        return decoded * query_std.reshape(stats_shape) + query_mean.reshape(stats_shape)

    @staticmethod
    def _straight_through_route_latents(
        route_latents: Tensor,
        route_logits: Tensor,
        route_available: Tensor,
        route_index: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Use the legacy hard-latent/ST-weight route semantics."""

        single_latents = route_latents[..., :4, :]
        single_logits = route_logits[..., :4]
        single_available = route_available[..., :4]
        masked_logits = single_logits.masked_fill(~single_available, -torch.inf)
        soft = torch.softmax(masked_logits, dim=-1)
        hard = F.one_hot(route_index.to(torch.long), num_classes=4).to(soft.dtype)
        straight_through = hard + soft - soft.detach()
        return (
            (straight_through.unsqueeze(-1) * single_latents).sum(dim=-2),
            straight_through,
        )

    @staticmethod
    def _fixed_soft_route_latents(
        route_latents: Tensor,
        route_logits: Tensor,
        route_available: Tensor,
        route_index: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Keep a hard forward route while distributing fixed-teacher gradients."""

        single_latents = route_latents[..., :4, :]
        single_logits = route_logits[..., :4]
        single_available = route_available[..., :4]
        masked_logits = single_logits.masked_fill(~single_available, -torch.inf)
        soft = torch.softmax(masked_logits, dim=-1)
        hard = F.one_hot(route_index.to(torch.long), num_classes=4).to(soft.dtype)
        straight_through = hard + soft - soft.detach()
        hard_latents = (hard.unsqueeze(-1) * single_latents).sum(dim=-2)
        soft_latents = (soft.unsqueeze(-1) * single_latents).sum(dim=-2)
        return (
            soft_latents + (hard_latents - soft_latents).detach(),
            straight_through,
        )

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
        differentiable_route: bool = False,
        detach_route_scores: bool = False,
        disabled_relations: frozenset[str] = frozenset(),
    ) -> RelationV2Forward:
        if patch_values.ndim != 4 or patch_values.shape[1] != self.variable_count:
            raise ValueError("patch_values must have shape [batch, variables, patches, patch_len]")
        if patch_mask.shape != patch_values.shape or rho.shape != patch_values.shape[:-1] or state.shape != rho.shape:
            raise ValueError("patch mask, rho and state do not match patch values")
        if query_mean.shape != (patch_values.shape[0], self.variable_count) or query_std.shape != query_mean.shape:
            raise ValueError("query normalization statistics must have shape [batch, variables]")
        if type(epoch_index) is not int or epoch_index < 0:
            raise ValueError("epoch_index must be a non-negative integer")
        if not disabled_relations <= _RELATION_NAMES:
            raise ValueError("disabled_relations contains an unknown relation")
        valid = torch.ones_like(state, dtype=torch.bool) if patch_valid is None else patch_valid.to(device=state.device, dtype=torch.bool)
        if valid.shape != state.shape:
            raise ValueError("patch_valid must match state")
        coordinate_validity = (
            valid.unsqueeze(-1).expand_as(patch_mask)
            if patch_validity is None
            else patch_validity.to(device=patch_mask.device, dtype=torch.bool)
        )
        if coordinate_validity.shape != patch_mask.shape:
            raise ValueError("patch_validity must match patch_mask")
        if type(differentiable_route) is not bool:
            raise TypeError("differentiable_route must be bool")
        if type(detach_route_scores) is not bool:
            raise TypeError("detach_route_scores must be bool")

        damaged = state.ne(PATCH_OBSERVED) & valid
        starts = self._starts(patch_starts, patch_values.shape[0], patch_values.shape[2], self.core.stride, patch_values.device)
        base_latents = self.core.project_clean_patches(patch_values)
        has_damaged = bool(damaged.any())
        if has_damaged:
            stats_shape = (*query_mean.shape, 1, 1)
            retrieval_values = (
                patch_values * query_std.reshape(stats_shape)
                + query_mean.reshape(stats_shape)
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
                temporal_candidates = build_temporal_candidates(
                    patch_values, patch_mask, rho, starts, self.temporal_distance_boundaries,
                    target_eligible=damaged,
                )
                temporal = self.temporal_operator(base_latents, temporal_candidates, rho)
                variable_candidates = build_variable_candidates(
                    patch_values, patch_mask, rho, target_eligible=damaged,
                    distance_boundaries=self.temporal_distance_boundaries,
                )
                variable = self.variable_operator(
                    base_latents, variable_candidates, self.variable_statistics
                )
            else:
                teacher = self.route_teacher
                temporal_candidates = build_temporal_candidates(
                    patch_values, patch_mask, rho, starts, self.temporal_distance_boundaries,
                    target_eligible=damaged,
                )
                variable_candidates = build_variable_candidates(
                    patch_values, patch_mask, rho, target_eligible=damaged,
                    distance_boundaries=self.temporal_distance_boundaries,
                )
                with torch.no_grad():
                    teacher_base_latents = teacher.project_clean_patches(patch_values)
                    teacher_historical_candidates = self._historical_candidates(
                        retrieval_values,
                        patch_mask,
                        rho,
                        state,
                        starts,
                        teacher_base_latents,
                        query_mean,
                        query_std,
                        projection_core=teacher,
                    )
                    (
                        teacher_historical,
                        historical_plan,
                        teacher_temporal,
                        temporal_plan,
                        teacher_variable,
                        variable_plan,
                    ) = teacher.relation_plans(
                        teacher_base_latents,
                        teacher_historical_candidates,
                        temporal_candidates,
                        variable_candidates,
                        rho,
                        self.variable_statistics,
                    )
                    historical_plan = historical_plan.detached()
                    temporal_plan = temporal_plan.detached()
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
                historical_candidate_messages = (
                    self.historical_operator.project_candidate_messages(
                        base_latents, historical_candidates
                    )
                )
                historical = self._with_student_messages(
                    teacher_historical,
                    historical_candidate_messages,
                    self.historical_operator.replay_message(
                        historical_candidate_messages, historical_plan
                    ),
                )
                temporal_candidate_messages = (
                    self.temporal_operator.project_candidate_messages(
                        base_latents, temporal_candidates
                    )
                )
                temporal = self._with_student_messages(
                    teacher_temporal,
                    temporal_candidate_messages,
                    self.temporal_operator.replay_message(
                        base_latents, temporal_candidate_messages, temporal_plan
                    ),
                )
                variable_candidate_messages = (
                    self.variable_operator.project_candidate_messages(
                        base_latents, variable_candidates
                    )
                )
                variable = self._with_student_messages(
                    teacher_variable,
                    variable_candidate_messages,
                    self.variable_operator.replay_message(
                        variable_candidate_messages, variable_plan
                    ),
                )
        else:
            historical = self._empty_branch(base_latents, self.historical_topk)
            temporal = self._empty_branch(base_latents, patch_values.shape[2] - 1)
            variable = self._empty_branch(base_latents, (self.variable_count - 1) * patch_values.shape[2])

        branches = (historical, temporal, variable)
        branch_available = torch.stack([branch.available for branch in branches], dim=-1)
        disabled = torch.tensor(
            [name in disabled_relations for name in ("historical", "temporal", "variable")],
            device=patch_values.device,
            dtype=torch.bool,
        )
        branch_available = branch_available & ~disabled.view(1, 1, 1, 3)
        historical, temporal, variable = (
            self._with_availability(branch, branch_available[..., index])
            for index, branch in enumerate(branches)
        )
        branch_messages, branch_gain, branch_uncertainty, _ = stack_branch_outputs(
            historical, temporal, variable
        )
        policy_branch_gain = branch_gain.detach() if detach_route_scores else branch_gain
        policy_branch_uncertainty = (
            branch_uncertainty.detach()
            if detach_route_scores
            else branch_uncertainty
        )
        route_available = route_availability_mask(branch_available)
        # Keep the internal eight-route tensor shape for graph/checkpoint
        # compatibility, but expose only none/H/T/V to the online policy.
        route_available = route_available.clone()
        route_available[..., 4:] = False
        if epoch_index <= self.bootstrap_epochs:
            uses_historical = route_relation_mask(device=patch_values.device)[:, 0]
            route_available = route_available & ~uses_historical.view(1, 1, 1, -1)
        if not has_damaged:
            route_available = torch.zeros_like(route_available)
            route_available[..., 0] = True
        route_logits = policy_branch_gain.new_full(
            (*policy_branch_gain.shape[:-1], len(V2_ROUTE_OPTIONS)), -1.0e9
        )
        route_logits[..., 0] = 0.0
        route_logits[..., 1:4] = policy_branch_gain
        route_index = direct_single_route_select(
            policy_branch_gain,
            route_available[..., 1:4],
        )
        graph = self.graph(
            base_latents,
            branch_messages,
            policy_branch_gain - policy_branch_uncertainty,
            branch_available,
            route_index,
            damaged,
        )
        decoded = self.patch_decoder(graph.latents)
        reconstructed = torch.where(patch_mask, patch_values, decoded)
        missing_indicator = (~patch_mask.to(torch.bool)) & coordinate_validity
        if self._forecast_fusion_mode == "teacher_hv":
            if self.route_teacher is None or self.forecast_fusion is None:
                raise RuntimeError(
                    "teacher H/V forecast fusion requires installed teacher and fusion"
                )
            routed_forecast_latents, forecast_route_weights = self.forecast_fusion(
                graph.route_latents,
                route_available,
                route_logits.detach(),
                missing_indicator,
            )
        elif differentiable_route:
            route_latent_function = (
                self._straight_through_route_latents
                if self.route_teacher is None
                else self._fixed_soft_route_latents
            )
            routed_forecast_latents, forecast_route_weights = (
                route_latent_function(
                    graph.route_latents,
                    route_logits,
                    route_available,
                    route_index,
                )
            )
        else:
            routed_forecast_latents = graph.latents
            forecast_route_weights = F.one_hot(
                route_index.to(torch.long), num_classes=4
            ).to(graph.latents.dtype)
        forecast_latents = routed_forecast_latents + self.mask_projection(
            missing_indicator.to(patch_values.dtype)
        )
        normalized_forecast = self.core.forecast_from_latents(forecast_latents)
        forecast = normalized_forecast * query_std.unsqueeze(-1) + query_mean.unsqueeze(-1)
        diagnostics = {
            "route_index": route_index.detach(),
            "route_available": route_available.detach(),
            "damaged": damaged.detach(),
            "historical_candidate_count": historical.candidate_eligible.sum(dim=-1),
            "temporal_candidate_count": temporal.candidate_eligible.sum(dim=-1),
            "variable_candidate_count": variable.candidate_eligible.sum(dim=-1),
            "forecast_route_weights": forecast_route_weights.detach(),
            "missing_indicator": missing_indicator.detach(),
        }
        return RelationV2Forward(
            forecast=forecast,
            normalized_forecast=normalized_forecast,
            base_latents=base_latents,
            latents=graph.latents,
            forecast_latents=forecast_latents,
            reconstructed_patches=reconstructed,
            historical=historical,
            temporal=temporal,
            variable=variable,
            variable_candidates=variable_candidates if has_damaged else None,
            graph=graph,
            route_logits=route_logits,
            route_index=route_index,
            route_available=route_available,
            diagnostics=diagnostics,
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
        objective: str = "joint",
    ) -> RelationV2TrainingResult:
        """Train on decoded H/T/V candidates and single-route outcomes."""

        if objective not in {"joint", "repair", "forecast"}:
            raise ValueError("objective must be 'joint', 'repair', or 'forecast'")
        if objective != "forecast" and (
            clean_raw_patches is None or clean_raw_patches.shape != patch_values.shape
        ):
            raise ValueError("clean_raw_patches must match patch values")
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
            differentiable_route=objective == "forecast",
            detach_route_scores=objective == "forecast",
        )
        forecast_loss = F.mse_loss(output.forecast, forecast_target)
        if objective == "forecast":
            zero = forecast_loss * 0.0
            loss = unified_stage1_loss(
                forecast_loss,
                zero,
                zero,
                zero,
                zero,
                lambda_imputation_huber=self.repair_loss_weight,
                lambda_edge=self.gain_loss_weight,
                lambda_route=0.0,
                lambda_memory=self.memory_loss_weight,
            )
            variable_shape = output.variable.candidate_eligible.shape
            empty_long = torch.zeros(
                variable_shape, dtype=torch.long, device=patch_values.device
            )
            return RelationV2TrainingResult(
                output=output,
                loss=loss,
                candidate_gain_targets=torch.zeros_like(
                    torch.cat(
                        (
                            output.historical.candidate_predicted_gain,
                            output.temporal.candidate_predicted_gain,
                            output.variable.candidate_predicted_gain,
                        ),
                        dim=-1,
                    )
                ),
                route_target=output.route_index.detach(),
                route_errors=torch.zeros_like(output.route_logits),
                branch_repair_losses=torch.zeros(
                    3, dtype=patch_values.dtype, device=patch_values.device
                ),
                variable_statistics_update=VariableStatisticsUpdate(
                    source_variable=empty_long,
                    target_variable=empty_long,
                    signed_bucket=empty_long,
                    gain=torch.zeros(variable_shape, device=patch_values.device),
                    eligible=torch.zeros(
                        variable_shape, dtype=torch.bool, device=patch_values.device
                    ),
                    route_errors=torch.zeros_like(output.route_logits),
                    route_available=torch.zeros_like(
                        output.route_available, dtype=torch.bool
                    ),
                ),
                objective=objective,
            )
        assert clean_raw_patches is not None
        valid = (
            torch.ones_like(state, dtype=torch.bool)
            if patch_valid is None
            else patch_valid.to(device=state.device, dtype=torch.bool)
        )
        damaged = state.ne(PATCH_OBSERVED) & valid
        missing_mask = (~patch_mask.to(device=patch_values.device, dtype=torch.bool)) & damaged.unsqueeze(-1)
        target_eligible = damaged & missing_mask.any(dim=-1)

        candidate_gain: list[Tensor] = []
        candidate_eligible: list[Tensor] = []
        candidate_predicted_gain: list[Tensor] = []
        candidate_type: list[Tensor] = []
        for relation_index, branch in enumerate(
            (output.historical, output.temporal, output.variable)
        ):
            gain, eligible = self._candidate_counterfactual_gain(
                output,
                branch,
                query_mean,
                query_std,
                clean_raw_patches,
                missing_mask,
                target_eligible,
            )
            candidate_gain.append(gain)
            candidate_eligible.append(eligible)
            candidate_predicted_gain.append(branch.candidate_predicted_gain)
            candidate_type.append(
                torch.full_like(branch.candidate_eligible, relation_index, dtype=torch.long)
            )
        candidate_gain_targets = torch.cat(candidate_gain, dim=-1)
        all_candidate_eligible = torch.cat(candidate_eligible, dim=-1)
        edge_loss = balanced_edge_gain_loss(
            torch.cat(candidate_predicted_gain, dim=-1),
            candidate_gain_targets,
            all_candidate_eligible,
            torch.cat(candidate_type, dim=-1),
        )

        baseline_prediction = self._decode_raw_patches(
            output.base_latents, query_mean, query_std
        )
        baseline_error = masked_patch_huber_error(
            baseline_prediction, clean_raw_patches, missing_mask
        )
        route_prediction = self._decode_raw_patches(
            output.graph.route_latents, query_mean, query_std
        )
        route_errors = masked_patch_huber_error(
            route_prediction, clean_raw_patches, missing_mask
        )
        # Route 0 is the exact unrecovered baseline, independent of the
        # branch-fusion implementation details.
        route_errors = route_errors.clone()
        route_errors[..., 0] = baseline_error

        relation_available = torch.stack(
            (output.historical.available, output.temporal.available, output.variable.available),
            dim=-1,
        ) & target_eligible.unsqueeze(-1)
        # Every available H/T/V branch is directly responsible for decoding
        # the hidden clean values.  Relation types are balanced equally so a
        # high candidate count cannot dominate this primary objective.
        branch_errors = route_errors[..., 1:4]
        reduce_dims = tuple(range(relation_available.ndim - 1))
        relation_counts = relation_available.sum(dim=reduce_dims)
        branch_repair_losses = (
            (branch_errors * relation_available.to(branch_errors.dtype)).sum(
                dim=reduce_dims
            )
            / relation_counts.clamp_min(1).to(branch_errors.dtype)
        )
        active_relations = relation_counts.gt(0)
        imputation_loss = (
            branch_repair_losses * active_relations.to(branch_errors.dtype)
        ).sum() / active_relations.sum().clamp_min(1).to(branch_errors.dtype)

        policy_relation_available = (
            output.route_available[..., 1:4] & target_eligible.unsqueeze(-1)
        )
        route_target_obj = oracle_route_labels(
            route_errors,
            baseline_error,
            policy_relation_available,
            complexity_margin=torch.zeros(
                3, device=route_errors.device, dtype=route_errors.dtype
            ),
            allow_combinations=False,
        )
        # The route oracle is diagnostic only.  Training and inference both
        # select argmax positive branch_gain, so a separate route classifier
        # and its overlapping cross-entropy objective are unnecessary.
        route_loss = route_errors.sum() * 0.0

        stats_shape = (*query_mean.shape, 1, 1)
        retrieval_query = (
            patch_values * query_std.reshape(stats_shape)
            + query_mean.reshape(stats_shape)
        )
        flat_query = retrieval_query.reshape(-1, patch_values.shape[-1])
        flat_mask = patch_mask.reshape(-1, patch_mask.shape[-1])
        flat_clean = clean_raw_patches.reshape(-1, clean_raw_patches.shape[-1])
        flat_damaged = target_eligible.reshape(-1)
        query_keys = self.retrieval.encoder(
            flat_query[flat_damaged], flat_mask[flat_damaged]
        )
        clean_keys = self.retrieval.encoder(
            flat_clean[flat_damaged], torch.ones_like(flat_mask[flat_damaged])
        )
        contrastive_loss = masked_contrastive_loss(query_keys, clean_keys)
        # A raw InfoNCE cross-entropy grows approximately as log(batch keys),
        # which made L_memory depend on batch size and overpower reconstruction
        # despite a smaller coefficient.  Normalize by the uniform-prediction
        # baseline so the configured weight has a stable interpretation.
        contrastive_normalizer = max(float(np.log(max(query_keys.shape[0], 2))), 1.0)
        memory_loss = masked_pair_loss(query_keys, clean_keys) + (
            0.5 * contrastive_loss / contrastive_normalizer
        )
        loss = unified_stage1_loss(
            forecast_loss,
            imputation_loss,
            edge_loss,
            route_loss,
            memory_loss,
            lambda_imputation_huber=self.repair_loss_weight,
            lambda_edge=self.gain_loss_weight,
            lambda_route=0.0,
            lambda_memory=self.memory_loss_weight,
        )
        if objective == "repair":
            loss = replace(
                loss,
                total=(
                    loss.weighted_imputation_huber
                    + loss.weighted_edge_gain
                    + loss.weighted_route
                    + loss.weighted_memory
                ),
            )
        variable_target = torch.arange(
            self.variable_count, device=patch_values.device, dtype=torch.long
        ).view(1, self.variable_count, 1, 1).expand_as(
            output.variable.candidate_eligible
        )
        variable_gain = candidate_gain[-1]
        variable_eligible = candidate_eligible[-1]
        source_variable = torch.zeros_like(variable_target)
        signed_bucket = torch.zeros_like(variable_target)
        if output.variable_candidates is not None:
            source_variable = output.variable_candidates.source_variable
            signed_bucket = output.variable.diagnostics["signed_bucket"]
        variable_statistics_update = VariableStatisticsUpdate(
            source_variable=source_variable,
            target_variable=variable_target,
            signed_bucket=signed_bucket,
            gain=variable_gain,
            eligible=variable_eligible,
            route_errors=route_errors.detach(),
            route_available=(route_target_obj.available & target_eligible.unsqueeze(-1)).detach(),
        )
        return RelationV2TrainingResult(
            output=output,
            loss=loss,
            candidate_gain_targets=candidate_gain_targets,
            route_target=route_target_obj.index,
            route_errors=route_errors,
            branch_repair_losses=branch_repair_losses,
            variable_statistics_update=variable_statistics_update,
            objective=objective,
        )

    def begin_epoch(self, epoch: int) -> None:
        self.retrieval.begin_epoch(epoch)
        self._current_epoch = epoch

    def stage_observed_batch(
        self,
        clean_raw_patches: Tensor,
        patch_mask: Tensor,
        rho: Tensor,
        state: Tensor,
        patch_starts: Tensor | None = None,
    ) -> int:
        if self._current_epoch is None:
            raise RuntimeError("begin_epoch must be called before staging prototypes")
        if clean_raw_patches.ndim != 4 or clean_raw_patches.shape[1] != self.variable_count:
            raise ValueError("clean_raw_patches must have shape [batch, variables, patches, patch_len]")
        starts = self._starts(
            patch_starts,
            clean_raw_patches.shape[0],
            clean_raw_patches.shape[2],
            self.core.stride,
            clean_raw_patches.device,
        )
        flat = clean_raw_patches.reshape(-1, clean_raw_patches.shape[-1])
        keys = self.retrieval.encode_frozen(
            flat, torch.ones_like(flat, dtype=torch.bool)
        ).reshape(*clean_raw_patches.shape[:3], -1)
        observed = state.eq(PATCH_OBSERVED) & rho.eq(1.0) & patch_mask.all(dim=-1)
        starts_cpu = starts.detach().cpu().numpy()
        raw_cpu = clean_raw_patches.detach().cpu().numpy()
        keys_cpu = keys.detach().cpu().numpy()
        observed_cpu = observed.detach().cpu().numpy()
        staged = 0
        for batch_index in range(clean_raw_patches.shape[0]):
            for variable_index in range(self.variable_count):
                for patch_index in range(clean_raw_patches.shape[2]):
                    staged += int(
                        self.memory.stage_patch(
                            variable_index,
                            int(starts_cpu[batch_index, patch_index]),
                            raw_cpu[batch_index, variable_index, patch_index],
                            keys_cpu[batch_index, variable_index, patch_index],
                            observed=bool(observed_cpu[batch_index, variable_index, patch_index]),
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
                    raw_patches, torch.ones_like(raw_patches, dtype=torch.bool)
                )
            return clean_keys.cpu().numpy()

        self.memory.rekey_batch(encode_members)
        self.memory.publish_epoch(epoch)
        self._current_epoch = None

    @staticmethod
    def _statistics_state(module: nn.Module) -> dict[str, Tensor]:
        return {
            name: value.detach().cpu().clone()
            for name, value in module.state_dict().items()
            if isinstance(value, Tensor)
        }

    @staticmethod
    def _statistics_hash(*states: Mapping[str, Tensor]) -> str:
        digest = sha256()
        for state in states:
            for name in sorted(state):
                digest.update(name.encode())
                digest.update(state[name].detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def online_state(self) -> dict[str, object]:
        """Return the V2 online state used by trainer read-only validation."""

        variable_state = self._statistics_state(self.variable_statistics)
        route_state = self._statistics_state(self.route_statistics)
        ema_state = self.retrieval.state_dict()
        ema_hash = sha256()
        for name, value in ema_state["ema_state"]:  # type: ignore[index]
            ema_hash.update(name.encode())
            ema_hash.update(np.asarray(value, dtype=np.float32).tobytes())
        return {
            "epoch": self.memory.snapshot().epoch,
            "memory_state": self.memory.state_dict(),
            "ema_state": ema_state,
            "relation_prior_state": {
                "format_version": 2,
                "variable_statistics": variable_state,
                "route_statistics": route_state,
            },
            "memory_hash": self.memory.snapshot_hash(),
            "ema_hash": ema_hash.hexdigest(),
            "relation_prior_hash": self._statistics_hash(variable_state, route_state),
        }

    def load_online_state(self, online_state: Mapping[str, object]) -> None:
        self.memory.load_state_dict(online_state["memory_state"])
        self.retrieval.load_state_dict(online_state["ema_state"])
        relation_state = online_state["relation_prior_state"]
        if not isinstance(relation_state, Mapping) or relation_state.get("format_version") != 2:
            raise ValueError("relation-v2 checkpoint requires relation state format 2")
        variable_state = relation_state.get("variable_statistics")
        route_state = relation_state.get("route_statistics")
        if not isinstance(variable_state, Mapping) or not isinstance(route_state, Mapping):
            raise ValueError("relation-v2 checkpoint statistics are incomplete")
        self.variable_statistics.load_state_dict(variable_state, strict=True)
        self.route_statistics.load_state_dict(route_state, strict=True)


__all__ = [
    "RelationV2Forward",
    "RelationV2OnlineStage1Model",
    "RelationV2TrainingResult",
    "VariableStatisticsUpdate",
    "pairwise_message_cosine",
]
