"""Parallel H/V/T evidence, one explicit repair and a mask-aware forecast head."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

import torch
from torch import Tensor, nn

from malr.data.patching import patchify_with_state
from malr.models.explicit_repair import ExplicitRepairBatch, build_explicit_repair
from malr.models.repaired_forecast import MaskAwareForecastModel

from .contracts import ExpertEvidence, ExpertInputs, RouterOutput
from .historical import HistoricalExpert
from .historical_variable import HistoricalVariableAdapter
from .multihead_router import CandidateMultiHeadRouter
from .post_fusion import EvidenceAwareFusionRefiner
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


def sample_specialist_exposure(
    damaged: Tensor, availability: Tensor, *, fraction: float, seed: int,
) -> tuple[Tensor, Tensor]:
    """Expose exactly one usable expert for a fixed-seed subset of damaged targets."""
    if availability.shape != (*damaged.shape, 3):
        raise ValueError("specialist availability must have shape [B,C,N,3]")
    if not 0 <= fraction <= 1 or seed < 0:
        raise ValueError("specialist fraction/seed is invalid")
    generator = torch.Generator(device=damaged.device).manual_seed(seed)
    eligible = damaged.to(torch.bool).flatten().nonzero().flatten()
    selected = torch.zeros(damaged.numel(), dtype=torch.bool, device=damaged.device)
    count = round(fraction * len(eligible))
    if count:
        choice = torch.randperm(len(eligible), generator=generator, device=damaged.device)[:count]
        selected[eligible[choice]] = True
    selected = selected.reshape_as(damaged)
    random = torch.rand(availability.shape, generator=generator, device=damaged.device)
    winner = random.masked_fill(~availability.to(torch.bool), -1).argmax(dim=-1)
    singleton = torch.nn.functional.one_hot(winner, num_classes=3).to(torch.bool)
    allowed = torch.where(selected[..., None], singleton & availability, availability)
    return allowed, selected


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
    ) -> None:
        super().__init__()
        if min(variable_count, window_length, patch_length, stride, d_model, retrieval_dim) < 1:
            raise ValueError("E23 model geometry must be positive")
        if stride > patch_length or window_length < patch_length:
            raise ValueError("invalid E23 patch geometry")
        self.variable_count = variable_count
        self.window_length = window_length
        self.patch_length = patch_length
        self.stride = stride
        self.d_model = d_model
        self.patch_count = 1 + (window_length + stride - patch_length) // stride
        self.base_encoder = nn.Linear(2 * patch_length, d_model)
        self.historical = HistoricalExpert(
            patch_length, d_model, retrieval_dim, variable_count,
            topk=historical_topk, similarity_threshold=historical_threshold,
            top_p=historical_top_p, candidate_cap=historical_candidate_cap,
            retrieval_strategy=historical_retrieval,
        )
        self.variable = SparseVariableExpert(
            variable_count, patch_length, d_model,
            topk=variable_topk, offsets=variable_offsets,
            top_p=variable_top_p,
            selection=variable_selection,
            aggregation=variable_aggregation,
        )
        if temporal_selection == "banded_v1":
            self.temporal = MultiScaleTemporalExpert(
                patch_length, d_model, local_radius=temporal_local_radius,
                boundaries=temporal_boundaries,
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

    def forward(
        self, values: Tensor, observed_mask: Tensor, query_mean: Tensor,
        query_std: Tensor, patch_starts: Tensor, *, valid_mask: Tensor | None = None,
        epoch: int = 0, explore_variable: bool = False,
        specialist_exposure: float = 0.0, specialist_seed: int = 0,
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
        base = self.base_encoder(torch.cat((
            patch_values, observed.to(values.dtype),
        ), dim=-1))
        damaged = patches.rho.lt(1.0)
        inputs = ExpertInputs(
            base_latents=base, patch_values=patch_values,
            patch_mask=observed, patch_validity=valid,
            rho=patches.rho, patch_starts=patch_starts,
            query_mean=query_mean, query_std=query_std,
            damaged=damaged,
        )
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
        explicit = build_explicit_repair(
            candidates, patch_values, observed, valid,
            query_mean, query_std,
            window_len=self.window_length, patch_len=self.patch_length,
            stride=self.stride,
        )
        # Patchify replicates the final time point to fill the final patch.
        # Invalid replicated entries must not enter the forecast projector.
        explicit = replace(
            explicit, repaired_patches=explicit.repaired_patches.masked_fill(
                ~explicit.validity_patches, 0,
            ),
        )
        return E23RepairOutput(explicit, routed, evidence, base, selected)


class E23ForecastModel(nn.Module):
    def __init__(
        self, repair: MultiExpertRepairEncoder, predictor: MaskAwareForecastModel,
        *, forecast_input: str = "explicit_v1",
    ) -> None:
        super().__init__()
        core = predictor.core
        if (core.c_in, core.seq_len, core.patch_len, core.stride) != (
            repair.variable_count, repair.window_length, repair.patch_length, repair.stride,
        ):
            raise ValueError("repair and forecast patch geometry differs")
        self.repair = repair
        self.predictor = predictor
        if forecast_input not in ("explicit_v1", "implicit_latent_v1"):
            raise ValueError("unsupported E23 forecast input")
        if forecast_input == "implicit_latent_v1" and predictor.variable_mixer is not None:
            raise ValueError("implicit E23 input requires the original PatchTST predictor")
        self.forecast_input = forecast_input
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
            forecast = self.predictor(
                explicit.repaired_patches, explicit.observation_patches,
                query_mean, query_std, patch_validity=explicit.validity_patches,
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
