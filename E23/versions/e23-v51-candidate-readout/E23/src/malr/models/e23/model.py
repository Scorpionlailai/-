"""Parallel H/V/T evidence, one explicit repair and a mask-aware forecast head."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

import torch
from torch import Tensor, nn

from malr.data.patching import patchify_with_state, patchify_preserving_values
from malr.models.explicit_repair import ExplicitRepairBatch, build_explicit_repair
from malr.models.repaired_forecast import MaskAwareForecastModel

from .contracts import ExpertEvidence, ExpertInputs, RouterOutput
from .exposure import sample_specialist_exposure
from .historical import HistoricalExpert
from .historical_variable import HistoricalVariableAdapter
from .iterative import IterationDiagnostics, IterativeTVHController
from .feature_tv import FeatureTVHController
from .historical_candidates import HistoricalCandidateProvider
from .stacked import StackLayerDiagnostics, StackedTVHController
from .multihead_router import CandidateMultiHeadRouter
from .post_fusion import EvidenceAwareFusionRefiner
from .point_refiner import PointResidualRefiner
from .hybrid_point import HybridPointRefiner
from .fine_history import FineHistoricalExpert
from .window_context import WindowContextEncoder
from .point_evidence import PointEvidenceRouter, OverlapConfidence
from .point_candidates import PointCandidateReadout
from .router import ExpertRouter
from .temporal import MultiScaleTemporalExpert
from .temporal_attention import VisibleTemporalAttentionExpert
from .variable import SparseVariableExpert


@dataclass(frozen=True)
class E23RepairOutput:
    explicit: ExplicitRepairBatch
    router: RouterOutput
    evidence: Mapping[str, ExpertEvidence]
    base_latents: Tensor
    specialist_selected: Tensor | None = None
    iterations: tuple[IterationDiagnostics, ...] | None = None
    stack_layers: tuple[StackLayerDiagnostics, ...] | None = None
    point_candidate_attention: Mapping[str, object] | None = None


class MultiExpertRepairEncoder(nn.Module):
    def __init__(
        self, variable_count: int, window_length: int, patch_length: int, stride: int,
        d_model: int = 16, retrieval_dim: int = 16, *, historical_topk: int = 4,
        historical_threshold: float = 0.7, variable_topk: int = 4,
        historical_top_p: float | None = None, variable_top_p: float | None = None,
        historical_candidate_cap: int = 8,
        historical_retrieval: str = "one_group",
        variable_selection: str = "global_topk",
        variable_aggregation: str = "scalar_v1",
        variable_offsets: tuple[int, ...] = (-1, 0, 1), temporal_local_radius: int = 2,
        temporal_boundaries: tuple[int, ...] = (2, 4, 8, 16),
        temporal_selection: str = "banded_v1",
        post_fusion_refiner: str = "none",
        multihead_router: str = "none",
        historical_archive: Tensor | None = None,
        bounded_history: bool = False,
        recovery_schedule: str = "parallel_v1",
        recovery_rounds: int = 1,
        recovery_layers: int = 1, expert_groups: int = 1, activation_checkpointing: bool = False,
        point_refiner_span: int = 0,
        point_feature_width: int = 0,
        point_refiner_arch: str = "legacy",
        historical_query_length: int = 0,
        front_context: str = "none",
        point_router: str = "none", overlap_fusion: str = "none",
        point_candidate_readout: str = "none",
    ) -> None:
        super().__init__()
        if min(variable_count, window_length, patch_length, stride, d_model, retrieval_dim) < 1:
            raise ValueError("E23 model geometry must be positive")
        if stride > patch_length or window_length < patch_length:
            raise ValueError("invalid E23 patch geometry")
        if recovery_schedule not in ("parallel_v1", "iterative_tv_h_v1", "stacked_parallel_tv_h_v1", "feature_tv_h_v1"):
            raise ValueError("unsupported recovery schedule")
        if type(recovery_rounds) is not int or recovery_rounds not in (1, 3):
            raise ValueError("unsupported recovery rounds")
        feature_tv = recovery_schedule == "feature_tv_h_v1"
        contextual = recovery_schedule in ("iterative_tv_h_v1", "feature_tv_h_v1")
        if not contextual and recovery_rounds != 1:
            raise ValueError("parallel recovery requires one round")
        stacked = recovery_schedule == "stacked_parallel_tv_h_v1"
        if type(recovery_layers) is not int or type(expert_groups) is not int or min(recovery_layers, expert_groups) < 1 or type(activation_checkpointing) is not bool:
            raise ValueError("invalid stacked recovery geometry")
        if not stacked and not feature_tv and (recovery_layers != 1 or expert_groups != 1 or activation_checkpointing):
            raise ValueError("stack parameters require stacked recovery")
        if feature_tv and (recovery_layers not in (1,2) or recovery_rounds != 1 or expert_groups != 1 or activation_checkpointing):
            raise ValueError("unsupported feature TV/H geometry")
        if (contextual or stacked) and (
            variable_selection != "window_context_topk_v1" or variable_aggregation != "scalar_v1"
            or historical_retrieval != "one_group" or historical_top_p is not None
            or variable_top_p is not None or temporal_selection != "banded_v1"
            or multihead_router != "none" or post_fusion_refiner != "none"
            or historical_archive is not None
        ):
            raise ValueError("iterative recovery requires the v28 expert policies")
        self.recovery_schedule = recovery_schedule
        self.recovery_rounds = recovery_rounds
        self.recovery_layers = recovery_layers
        self.expert_groups = expert_groups
        self.activation_checkpointing = activation_checkpointing
        self.variable_count = variable_count
        self.window_length = window_length
        self.patch_length = patch_length
        self.stride = stride
        self.d_model = d_model
        self.patch_count = ((window_length + patch_length - 1) // patch_length
                            if stride == patch_length else
                            1 + (window_length + stride - patch_length) // stride)
        self.base_encoder = nn.Linear(2 * patch_length, d_model)
        self.stacked = None
        if stacked:
            self.historical = HistoricalCandidateProvider(patch_length,d_model,retrieval_dim,variable_count,
                topk=historical_topk,similarity_threshold=historical_threshold,
                retrieval_strategy=historical_retrieval)
            self.stacked = StackedTVHController(variable_count,patch_length,d_model,layers=recovery_layers,
                groups=expert_groups,activation_checkpointing=activation_checkpointing,
                variable_topk=variable_topk,variable_offsets=variable_offsets,
                temporal_local_radius=temporal_local_radius,temporal_boundaries=temporal_boundaries)
        else:
            self.historical = HistoricalExpert(
                patch_length, d_model, retrieval_dim, variable_count,
                topk=historical_topk, similarity_threshold=historical_threshold,
                top_p=historical_top_p, candidate_cap=historical_candidate_cap,
                retrieval_strategy=historical_retrieval,
                contextual=contextual,
            )
            self.variable = SparseVariableExpert(
                variable_count, patch_length, d_model,
                topk=variable_topk, offsets=variable_offsets,
                top_p=variable_top_p,
                selection=variable_selection,
                aggregation=variable_aggregation,
                contextual=contextual,
            )
            if temporal_selection == "banded_v1":
                self.temporal = MultiScaleTemporalExpert(
                    patch_length, d_model, local_radius=temporal_local_radius,
                    boundaries=temporal_boundaries,
                    contextual=contextual,
                )
            elif temporal_selection == "visible_attention_v1":
                self.temporal = VisibleTemporalAttentionExpert(
                    patch_length, d_model, boundaries=temporal_boundaries,
                )
            else:
                raise ValueError("unsupported E23 temporal selection")
            if multihead_router == "none":
                self.router = ExpertRouter(d_model)
            elif multihead_router == "eight_candidate_heads_v1" and d_model in (64, 128) and temporal_selection == "banded_v1":
                self.router = CandidateMultiHeadRouter(d_model, head_count=8)
            elif multihead_router == "eight_full_width_gated_v1" and d_model == 64 and temporal_selection == "banded_v1":
                self.router = CandidateMultiHeadRouter(d_model, head_count=8, fusion_mode="gated_sum")
            else:
                raise ValueError("unsupported E23 multihead router")
        if historical_query_length:
            if historical_query_length != 16 or stride != patch_length or point_refiner_arch not in ("hybrid_v1", "cnn_v1", "none_v1"):
                raise ValueError("fine history requires nonoverlap geometry")
            with torch.random.fork_rng(devices=[]):
                self.historical = FineHistoricalExpert(patch_length, d_model, variable_count,
                    topk=historical_topk, threshold=historical_threshold)
        self.history_adapter = (
            HistoricalVariableAdapter(
                variable_count, patch_length, d_model, topk=variable_topk,
                archive=historical_archive, bounded_values=bounded_history,
            ) if historical_archive is not None else None
        )
        self.decoder = nn.Linear(d_model, patch_length)
        if post_fusion_refiner == "none":
            self.post_fusion_refiner = None
        elif post_fusion_refiner == "evidence_factorized_v1":
            self.post_fusion_refiner = EvidenceAwareFusionRefiner(d_model, self.patch_count)
        else:
            raise ValueError("unsupported E23 post-fusion refiner")
        self.iterative = (FeatureTVHController(variable_count, d_model, patch_length, stride,
                          window_length, recovery_layers) if feature_tv else
                          IterativeTVHController(d_model, recovery_rounds) if contextual else None)
        self.point_refiner = None
        if type(point_refiner_span) is not int or point_refiner_span not in (0, 32, 96):
            raise ValueError("unsupported point refiner span")
        if point_feature_width and not point_refiner_span:
            raise ValueError("point features require a point refiner")
        if point_refiner_span:
            if recovery_schedule != "iterative_tv_h_v1" or recovery_rounds != 1:
                raise ValueError("point refiner requires original v31-B recovery")
            # Additional repair capacity must not consume the original head RNG.
            with torch.random.fork_rng(devices=[]):
                if point_refiner_arch in ("hybrid_v1", "cnn_v1"):
                    self.point_refiner = HybridPointRefiner(d_model, window_length,
                        patch_length, stride, point_refiner_span,
                        attention=point_refiner_arch == "hybrid_v1")
                elif point_refiner_arch == "legacy":
                    self.point_refiner = PointResidualRefiner(d_model, window_length,
                        patch_length, stride, point_refiner_span,
                        point_feature_width=point_feature_width)
                else:
                    raise ValueError("unknown point refiner architecture")

        if front_context not in ("none", "window_cnn_v1"):
            raise ValueError("unsupported front context")
        self.front_context = None
        if front_context != "none":
            if (d_model,window_length,patch_length,stride,recovery_schedule,recovery_rounds,point_refiner_span) != (64,96,16,8,"iterative_tv_h_v1",1,32):
                raise ValueError("front context requires original v41 geometry")
            with torch.random.fork_rng(devices=[]):
                self.front_context = WindowContextEncoder(d_model,window_length,patch_length,stride)
        self.point_router = None
        self.overlap_confidence = None
        self.point_candidate_readout = None
        if point_candidate_readout not in ("none", "eight_head_v1"):
            raise ValueError("unsupported point candidate readout")
        if point_router not in ("none", "pointwise_v1") or overlap_fusion not in ("none", "confidence_v1"):
            raise ValueError("unsupported point evidence policy")
        if (point_router != "none" or overlap_fusion != "none" or point_candidate_readout != "none") and self.front_context is None:
            raise ValueError("point evidence requires window context")
        with torch.random.fork_rng(devices=[]):
            if point_router != "none":self.point_router = PointEvidenceRouter(d_model)
            if overlap_fusion != "none":self.overlap_confidence = OverlapConfidence(d_model)
            if point_candidate_readout != "none":self.point_candidate_readout = PointCandidateReadout(d_model)

    def forward(
        self, values: Tensor, observed_mask: Tensor, query_mean: Tensor,
        query_std: Tensor, patch_starts: Tensor, *, valid_mask: Tensor | None = None,
        epoch: int = 0, explore_variable: bool = False,
        specialist_exposure: float = 0.0, specialist_seed: int = 0,
        disabled_experts: frozenset[str] = frozenset(),
        disabled_groups: frozenset[tuple[int,int]] = frozenset(),
    ) -> E23RepairOutput:
        if values.ndim != 3 or values.shape[1:] != (self.window_length, self.variable_count):
            raise ValueError("E23 values must have shape [B,T,C]")
        if observed_mask.shape != values.shape:
            raise ValueError("E23 observed mask must match values")
        patches = patchify_with_state(
            values, observed_mask, self.patch_length, self.stride,
            validity=valid_mask,
        )
        batch = values.shape[0]
        if patches.values.shape[2] != self.patch_count:
            raise ValueError("E23 patch count differs from predictor geometry")
        if patch_starts.shape not in ((batch, self.patch_count),
                                      (batch, self.variable_count, self.patch_count)):
            raise ValueError("E23 patch starts shape mismatch")
        patch_values = patches.values.masked_fill(~patches.validity, 0)
        observed = patches.effective_mask
        valid = patches.validity
        if self.iterative is not None or self.stacked is not None:
            patch_values = patch_values.masked_fill(~observed, 0)
        base = self.base_encoder(torch.cat((
            patch_values, observed.to(values.dtype),
        ), dim=-1))
        points = visible_features = None
        if self.front_context is not None:
            if self.point_router is not None or self.overlap_confidence is not None or self.point_candidate_readout is not None:
                context, points, visible_features = self.front_context(values, observed_mask, query_mean, query_std,
                    patches.validity, valid_mask=valid_mask, return_points=True)
            else:
                context = self.front_context(values, observed_mask, query_mean, query_std,
                                              patches.validity, valid_mask=valid_mask)
            base = base + context
        damaged = patches.rho.lt(1.0)
        inputs = ExpertInputs(
            base_latents=base, patch_values=patch_values,
            patch_mask=observed, patch_validity=valid,
            rho=patches.rho, patch_starts=patch_starts,
            query_mean=query_mean, query_std=query_std,
            damaged=damaged,
        )
        iterations = None; stack_layers = None
        if self.stacked is not None:
            result = self.stacked(inputs,historical=self.historical,epoch=epoch,explore_variable=explore_variable,
                specialist_exposure=specialist_exposure,specialist_seed=specialist_seed,
                disabled_experts=disabled_experts,disabled_groups=disabled_groups)
            routed,evidence = result.router,dict(result.evidence)
            selected,stack_layers = result.specialist_selected,result.layers
        elif self.iterative is not None:
            iteration = self.iterative(
                inputs, historical=self.historical, variable=self.variable,
                temporal=self.temporal, router=self.router, epoch=epoch,
                explore_variable=explore_variable, specialist_exposure=specialist_exposure,
                specialist_seed=specialist_seed, disabled_experts=disabled_experts,
            )
            routed, evidence = iteration.router, dict(iteration.evidence)
            selected, iterations = iteration.specialist_selected, iteration.iterations
        else:
            if disabled_experts:
                raise ValueError("disabled experts require iterative recovery")
            evidence = {
                "historical": self.historical(inputs),
                "variable": self.variable(inputs, epoch=epoch, explore=explore_variable),
                "temporal": self.temporal(inputs),
            }
            selected = None
            routed_evidence = evidence
            if specialist_exposure:
                availability = torch.stack([evidence[name].available for name in
                                            ("historical", "variable", "temporal")], dim=-1)
                allowed, selected = sample_specialist_exposure(
                    damaged, availability, fraction=specialist_exposure, seed=specialist_seed,
                )
                routed_evidence = {
                    name: replace(evidence[name], available=allowed[..., index])
                    for index, name in enumerate(("historical", "variable", "temporal"))
                }
            routed = self.router(inputs, routed_evidence)
        if self.history_adapter is not None:
            correction, historical_variable = self.history_adapter(
                routed.latents, inputs, evidence["historical"],
                epoch=epoch, explore=explore_variable,
            )
            evidence["historical_variable"] = historical_variable
            routed = replace(routed, latents=routed.latents + correction)
        if self.post_fusion_refiner is not None:
            routed = replace(
                routed, latents=self.post_fusion_refiner(routed, patches.rho),
            )
        candidates = self.decoder(routed.latents)
        if self.point_router is not None:
            correction, point_weights = self.point_router(inputs, routed, evidence, self.decoder, points, visible_features)
            candidates = candidates + correction
            routed = replace(routed, point_weights=point_weights)
        candidate_weights = None
        candidate_attention = None
        if self.point_candidate_readout is not None:
            correction, candidate_attention = self.point_candidate_readout(
                inputs, routed, evidence, self.decoder, points, visible_features, candidates)
            # Direct numerical repair residual is learned on the global training scale.
            candidates = candidates+correction/query_std[:, :, None, None]
        if self.overlap_confidence is not None:
            candidate_weights = self.overlap_confidence(inputs, routed, self.decoder, points, visible_features, candidates)
        explicit = build_explicit_repair(
            candidates, patch_values, observed, valid,
            query_mean, query_std,
            window_len=self.window_length, patch_len=self.patch_length,
            stride=self.stride,
            candidate_weights=candidate_weights,
        )
        # Patchify replicates the final time point to fill the final patch.
        # Invalid replicated entries must not enter the forecast projector.
        explicit = replace(
            explicit, repaired_patches=explicit.repaired_patches.masked_fill(
                ~explicit.validity_patches, 0,
            ),
        )
        if self.point_refiner is not None:
            if isinstance(self.point_refiner, HybridPointRefiner):
                explicit = self.point_refiner(explicit, routed, query_mean, query_std,
                                               specialist_selected=selected)
            else:
                explicit = self.point_refiner(explicit, routed, query_mean, query_std)
        return E23RepairOutput(explicit, routed, evidence, base, selected, iterations, stack_layers,
                               candidate_attention)


class E23ForecastModel(nn.Module):
    def __init__(
        self, repair: MultiExpertRepairEncoder, predictor: MaskAwareForecastModel,
        *, forecast_input: str = "explicit_v1",
        forecast_padding: str = "native",
    ) -> None:
        super().__init__()
        core = predictor.core
        if (core.c_in, core.seq_len) != (repair.variable_count, repair.window_length):
            raise ValueError("repair and forecast sequence geometry differs")
        self.repatch_for_predictor = (core.patch_len, core.stride) != (repair.patch_length, repair.stride)
        if self.repatch_for_predictor and (forecast_input != "explicit_v1" or
                repair.stride != repair.patch_length or (core.patch_len, core.stride) != (16, 8)):
            raise ValueError("different patch geometry requires nonoverlap repair and original explicit predictor")
        self.repair = repair
        self.predictor = predictor
        if forecast_input not in ("explicit_v1", "implicit_latent_v1"):
            raise ValueError("unsupported E23 forecast input")
        if forecast_input == "implicit_latent_v1" and predictor.variable_mixer is not None:
            raise ValueError("implicit E23 input requires the original PatchTST predictor")
        if repair.stacked is not None and (forecast_input != "explicit_v1" or predictor.variable_mixer is not None):
            raise ValueError("stacked recovery requires the original explicit predictor")
        self.forecast_input = forecast_input
        if forecast_padding not in ("native", "replicate_last_value_v1"):
            raise ValueError("unsupported forecast padding")
        if forecast_padding != "native" and forecast_input != "explicit_v1":
            raise ValueError("common forecast padding requires explicit input")
        self.forecast_padding = forecast_padding
        if forecast_input == "implicit_latent_v1":
            self.implicit_adapter = nn.Linear(repair.d_model, core.d_model, bias=False)
            nn.init.zeros_(self.implicit_adapter.weight)

    def forward(
        self, values: Tensor, observed_mask: Tensor, query_mean: Tensor,
        query_std: Tensor, patch_starts: Tensor, *, valid_mask: Tensor | None = None,
        epoch: int = 0, explore_variable: bool = False,
        specialist_exposure: float = 0.0, specialist_seed: int = 0,
    ) -> tuple[Tensor, E23RepairOutput]:
        output = self.repair(
            values, observed_mask, query_mean, query_std, patch_starts,
            valid_mask=valid_mask, epoch=epoch, explore_variable=explore_variable,
            specialist_exposure=specialist_exposure, specialist_seed=specialist_seed,
        )
        if self.forecast_input == "explicit_v1":
            explicit = output.explicit
            if self.repatch_for_predictor:
                patches = patchify_preserving_values(explicit.normalized_sequence,
                    explicit.sequence_observed, self.predictor.core.patch_len,
                    self.predictor.core.stride, validity=explicit.sequence_validity)
                predictor_values = patches.values.masked_fill(~patches.validity, 0)
                observation, validity = patches.effective_mask, patches.validity
            else:
                predictor_values = explicit.repaired_patches
                observation, validity = explicit.observation_patches, explicit.validity_patches
            if self.forecast_padding == "replicate_last_value_v1":
                last = explicit.normalized_sequence[:, -1, :][:, :, None, None]
                predictor_values = torch.where(validity, predictor_values, last)
            forecast = self.predictor(
                predictor_values, observation,
                query_mean, query_std, patch_validity=validity,
            )
        else:
            patches = patchify_with_state(
                values, observed_mask, self.repair.patch_length, self.repair.stride,
                validity=valid_mask,
            )
            original = patches.values.masked_fill(~patches.effective_mask, 0)
            missing = (~patches.effective_mask) & patches.validity
            latents = self.predictor.core.project_clean_patches(original)
            latents = latents + self.predictor.mask_projection(missing.to(original.dtype))
            correction = self.implicit_adapter(output.router.latents)
            latents = latents + correction * patches.rho.lt(1.0)[..., None]
            normalized = self.predictor.core.forecast_from_latents(latents)
            forecast = normalized * query_std.unsqueeze(-1) + query_mean.unsqueeze(-1)
        return forecast, output


__all__ = ["E23RepairOutput", "MultiExpertRepairEncoder", "E23ForecastModel",
           "sample_specialist_exposure"]
