"""Sparse directed cross-variable evidence from visible local source patches."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from .contracts import CandidateEvidence, ExpertEvidence, ExpertInputs
from .nucleus import nucleus_prefix


@dataclass(frozen=True)
class VariableSourceCache:
    visible_values: Tensor
    source_observed: Tensor
    lag: Tensor
    source_fraction: Tensor
    consistency: Tensor
    raw_quality: Tensor
    logits: Tensor
    valid: Tensor
    available: Tensor
    source_ix: Tensor
    patch_ix: Tensor
    provenance: Mapping[str, Any]
    empty_evidence: ExpertEvidence | None = None


DYNAMIC_POLICY = "window_context_nucleus32_p90_cap16_v1"


class SparseVariableExpert(nn.Module):
    """Select source variables first, then gather at most three patches per edge."""

    def __init__(
        self, variable_count: int, patch_length: int, d_model: int,
        *, topk: int = 4, offsets: tuple[int, ...] = (-1, 0, 1),
        top_p: float | None = None, selection: str = "global_topk",
        aggregation: str = "scalar_v1",
        contextual: bool = False,
        calibration: str = "none",
    ) -> None:
        super().__init__()
        if min(variable_count, patch_length, d_model, topk) < 1:
            raise ValueError("variable count, patch length, width and topk must be positive")
        if not offsets or len(set(offsets)) != len(offsets):
            raise ValueError("offsets must be nonempty and distinct")
        if top_p is not None and not 0 < top_p <= 1:
            raise ValueError("variable top-p must lie in (0, 1]")
        self.dynamic_selection = selection == DYNAMIC_POLICY
        if self.dynamic_selection:
            if topk != 16 or top_p is not None or not contextual or aggregation != "scalar_v1":
                raise ValueError("dynamic V requires contextual scalar V with cap 16")
            selection = "window_context_topk_v1"
        if selection not in ("global_topk", "window_context_topk_v1",
                             "window_context_topk_v2", "window_affine_v1"):
            raise ValueError("unsupported variable selection")
        if selection != "global_topk" and (top_p is not None or 0 not in offsets):
            raise ValueError("contextual variable selection requires top-k and zero offset")
        if aggregation not in ("scalar_v1", "channelwise_v1", "channelwise_bounded_v1"):
            raise ValueError("unsupported variable aggregation")
        self.variable_count = variable_count
        self.patch_length = patch_length
        self.d_model = d_model
        self.topk = min(topk, variable_count - 1) if top_p is None else variable_count - 1
        self.top_p = top_p
        self.selection = selection
        self.aggregation = aggregation
        self.offsets = offsets
        self.source_embedding = nn.Embedding(variable_count, d_model)
        self.target_embedding = nn.Embedding(variable_count, d_model)
        if calibration not in ("none", "identity_scale_v1"):
            raise ValueError("unsupported variable calibration")
        if calibration != "none" and (not contextual or selection != "window_context_topk_v1"):
            raise ValueError("variable calibration requires original contextual V")
        self.calibration = None
        if calibration != "none":
            with torch.random.fork_rng(devices=[]):
                self.calibration = nn.Sequential(nn.Linear(2*d_model+2*patch_length+4,d_model),
                    nn.GELU(),nn.Linear(d_model,d_model))
                nn.init.zeros_(self.calibration[-1].weight)
                nn.init.zeros_(self.calibration[-1].bias)
        if selection == "window_affine_v1":
            self.register_buffer("pair_slope", torch.ones(variable_count, variable_count))
            self.register_buffer("pair_intercept", torch.zeros(variable_count, variable_count))
        self.message_projector = nn.Sequential(
            nn.Linear(2 * patch_length + d_model + 3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.quality_projector = nn.Sequential(nn.Linear(4, d_model), nn.Tanh())
        self.contextual = contextual
        if contextual:
            if selection != "window_context_topk_v1" or aggregation != "scalar_v1":
                raise ValueError("contextual V requires window_context_topk_v1 and scalar_v1")
            self.context_norm = nn.LayerNorm(d_model)
            self.source_context_projection = nn.Linear(d_model, d_model, bias=False)
            self.context_query = nn.Linear(d_model, d_model, bias=False)
            self.context_key = nn.Linear(d_model, d_model, bias=False)
            nn.init.zeros_(self.source_context_projection.weight)
            nn.init.zeros_(self.context_query.weight)
        if aggregation in ("channelwise_v1", "channelwise_bounded_v1"):
            self.channel_projector = nn.Linear(d_model, d_model, bias=False)
            nn.init.zeros_(self.channel_projector.weight)

    @torch.no_grad()
    def set_affine_map(self, slope: Tensor, intercept: Tensor) -> None:
        if self.selection != "window_affine_v1":
            raise ValueError("affine map is only valid for window_affine_v1")
        shape = (self.variable_count, self.variable_count)
        if slope.shape != shape or intercept.shape != shape:
            raise ValueError("V affine map shape mismatch")
        if not torch.isfinite(slope).all() or not torch.isfinite(intercept).all():
            raise ValueError("V affine map must be finite")
        self.pair_slope.copy_(slope)
        self.pair_intercept.copy_(intercept)

    def _selected_sources(self, *, epoch: int, explore: bool) -> tuple[Tensor, Tensor, Tensor]:
        # Only the tiny C-by-C training prior is dense. Sample values are gathered
        # after this selection, so the candidate MLP never sees rejected edges.
        relation = self.source_embedding.weight @ self.target_embedding.weight.T
        relation = relation / sqrt(self.d_model)
        targets = torch.arange(self.variable_count, device=relation.device)
        ranked = torch.argsort(
            relation.T.masked_fill(torch.eye(self.variable_count, device=relation.device, dtype=torch.bool), -torch.inf),
            dim=-1, descending=True, stable=True,
        )
        if self.top_p is not None:
            selected = ranked[:, :self.topk]
            prior = relation.T.gather(1, selected)
            keep = nucleus_prefix(prior, torch.ones_like(prior, dtype=torch.bool), p=self.top_p)
            if explore:
                others = ~keep
                count = others.sum(-1)
                rank = others.long().cumsum(-1) - 1
                choice = (epoch + targets) % count.clamp_min(1)
                keep = keep | (others & (rank == choice[:, None]) & (count[:, None] > 0))
            return selected, prior, keep
        selected = ranked[:, :self.topk].clone()
        if explore and self.variable_count - 1 > self.topk:
            choices = ranked[:, self.topk:self.variable_count - 1]
            rotation = (epoch + targets) % choices.shape[1]
            selected[:, -1] = choices.gather(1, rotation[:, None]).squeeze(1)
        prior = relation.T.gather(1, selected)
        return selected, prior, torch.ones_like(selected, dtype=torch.bool)

    def _contextual_sources(
        self, inputs: ExpertInputs, *, epoch: int, explore: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Rank directed sources using only observed local coordinates."""
        if self.dynamic_selection:
            selected, prior, keep, _ = self._dynamic_sources(inputs, epoch=epoch, explore=explore)
            return selected, prior, keep
        values, mask, validity = inputs.patch_values, inputs.patch_mask, inputs.patch_validity
        batch, variables, patches, _ = values.shape
        o = len(self.offsets)
        relation = self.source_embedding.weight @ self.target_embedding.weight.T
        prior_all = (relation / sqrt(self.d_model)).T
        target_ix = torch.arange(variables, device=values.device)
        patch_ix = torch.arange(patches, device=values.device)
        offsets = torch.tensor(self.offsets, device=values.device)
        raw = patch_ix[:, None] + offsets[None, :]
        in_range = (raw >= 0) & (raw < patches)
        local = raw.clamp(0, patches - 1)
        # [B,N,S,O,P] is gathered once; the candidate MLP only sees selected edges.
        local_values = values[:, :, local].permute(0, 2, 1, 3, 4)
        local_valid = validity[:, :, local].permute(0, 2, 1, 3, 4)
        local_observed = (mask & validity)[:, :, local].permute(0, 2, 1, 3, 4)
        local_valid = local_valid & in_range[None, :, None, :, None]
        local_observed = local_observed & local_valid
        visible_fraction = (local_observed.sum(dim=(-1, -2)).float()
                            / local_valid.sum(dim=(-1, -2)).clamp_min(1))
        aligned = self.offsets.index(0)
        aligned_observed = local_observed[..., aligned, :]
        target_observed = (mask & validity).permute(0, 2, 1, 3)
        common = target_observed[:, :, :, None, :] & aligned_observed[:, :, None, :, :]
        target_values = values.permute(0, 2, 1, 3)[:, :, :, None, :]
        source_values = local_values[..., aligned, :][:, :, None, :, :]
        # Mask operands before subtraction so hidden/padded values never enter scores.
        if self.selection == "window_affine_v1":
            source_values = (self.pair_slope[None, None, :, :, None] * source_values
                             + self.pair_intercept[None, None, :, :, None])
        difference = (source_values.masked_fill(~common, 0)
                      - target_values.masked_fill(~common, 0)).abs()
        error = difference.sum(-1) / common.sum(-1).clamp_min(1)
        consistency = torch.exp(-error) * common.any(-1)
        context_scale = 0.5 if self.selection == "window_context_topk_v1" else 1.0
        scores = (prior_all[None, None, :, :] + context_scale * visible_fraction[:, :, None, :]
                  + context_scale * consistency).permute(0, 2, 1, 3)
        has_source = local_observed.any(dim=(-1, -2))[:, None, :, :]
        scores = scores.masked_fill(~has_source, -1e9)
        scores = scores.masked_fill(
            torch.eye(variables, device=values.device, dtype=torch.bool)[None, :, None, :],
            -torch.inf,
        )
        ranked = torch.argsort(scores, dim=-1, descending=True, stable=True)
        selected = ranked[..., :self.topk].clone()
        if explore and variables - 1 > self.topk:
            choices = ranked[..., self.topk:variables - 1]
            rotation = ((epoch + target_ix[None, :, None]) % choices.shape[-1])
            rotation = rotation.expand(batch, variables, patches)
            selected[..., -1] = choices.gather(-1, rotation[..., None]).squeeze(-1)
        prior = prior_all[None, :, None, :].expand(batch, variables, patches, variables)
        selected_prior = prior.gather(-1, selected)
        return selected, selected_prior, torch.ones_like(selected, dtype=torch.bool)

    def _dynamic_sources(self, inputs: ExpertInputs, *, epoch: int, explore: bool):
        """Score at most 32 prior-shortlisted sources per target and patch.

        The nucleus uses the entire eligible shortlist before the cap is applied.
        Fixed padded output slots support batched downstream operations; keep marks
        actual sources. No unobserved value participates in local scoring.
        """
        values = inputs.patch_values
        observed = inputs.patch_mask & inputs.patch_validity
        batch, variables, patches, _ = values.shape
        prior_all = (self.source_embedding.weight @ self.target_embedding.weight.T).T / sqrt(self.d_model)
        diagonal = torch.eye(variables, device=values.device, dtype=torch.bool)
        ranked = prior_all.masked_fill(diagonal, -torch.inf).argsort(dim=-1, descending=True, stable=True)
        pool_size = min(32, variables - 1)
        pool = ranked[:, :pool_size].clone()
        if explore and variables - 1 > pool_size:
            omitted = ranked[:, pool_size:variables-1]
            rotation = (epoch + torch.arange(variables, device=values.device)) % omitted.shape[-1]
            pool[:, -1] = omitted.gather(-1, rotation[:, None]).squeeze(-1)
        offsets = torch.tensor(self.offsets, device=values.device)
        raw_patch = torch.arange(patches, device=values.device)[:, None] + offsets[None, :]
        in_range = (raw_patch >= 0) & (raw_patch < patches)
        local = raw_patch.clamp(0, patches-1)
        # [B,C,N,M,O,P]; local comparisons grow with C*M, rather than C*C.
        source_values = values[:, pool[:, None, :, None], local[None, :, None, :]]
        source_observed = observed[:, pool[:, None, :, None], local[None, :, None, :]]
        source_validity = inputs.patch_validity[:, pool[:, None, :, None], local[None, :, None, :]]
        source_validity = source_validity & in_range[None, None, :, None, :, None]
        source_observed = source_observed & source_validity
        fraction = source_observed.sum((-1,-2)).float() / source_validity.sum((-1,-2)).clamp_min(1)
        aligned = self.offsets.index(0)
        common = observed[..., None, :] & source_observed[..., aligned, :]
        difference = (source_values[..., aligned, :].masked_fill(~common, 0)
                      - values[..., None, :].masked_fill(~common, 0)).abs()
        consistency = torch.exp(-difference.sum(-1) / common.sum(-1).clamp_min(1)) * common.any(-1)
        prior = prior_all.gather(-1, pool)[None, :, None, :].expand(batch, variables, patches, pool_size)
        scores = prior + 0.5*fraction + 0.5*consistency
        eligible = source_observed.any((-1,-2))
        keep_all = nucleus_prefix(scores, eligible, p=0.9)
        order = scores.masked_fill(~eligible, -torch.inf).argsort(dim=-1, descending=True, stable=True)
        slots = order[..., :self.topk]
        pool_expanded = pool[None, :, None, :].expand_as(scores)
        selected = pool_expanded.gather(-1, slots)
        keep = keep_all.gather(-1, slots)
        safe_scores = scores.masked_fill(~eligible, -torch.inf)
        safe_scores = torch.where(eligible.any(-1, keepdim=True), safe_scores, torch.zeros_like(scores))
        probabilities = safe_scores.softmax(-1) * eligible
        statistics = {
            "source_pool_size": pool_size,
            "selected_source_count": keep.sum(-1).detach(),
            "uncapped_source_count": keep_all.sum(-1).detach(),
            "cap_hit": (keep_all.sum(-1) > self.topk).detach(),
            "retained_probability_mass": (probabilities.gather(-1, slots)*keep).sum(-1).detach(),
            "selection_policy": DYNAMIC_POLICY,
            "nucleus_p": 0.9, "source_cap": 16,
        }
        return selected, prior.gather(-1, slots), keep, statistics

    def prepare_sources(self, inputs: ExpertInputs, *, epoch: int, explore: bool) -> VariableSourceCache:
        values, mask, validity = inputs.patch_values, inputs.patch_mask, inputs.patch_validity
        base = inputs.base_latents
        if values.ndim != 4:
            raise ValueError("patch values must have shape [B,C,N,P]")
        batch, variables, patches, length = values.shape
        if variables != self.variable_count or length != self.patch_length:
            raise ValueError("variable count or patch length does not match expert")
        if mask.shape != values.shape or validity.shape != values.shape:
            raise ValueError("patch mask/validity shape mismatch")
        if base.shape != (batch, variables, patches, self.d_model):
            raise ValueError("base latent shape mismatch")
        if inputs.damaged.shape != (batch, variables, patches):
            raise ValueError("damaged shape mismatch")
        if epoch < 0:
            raise ValueError("epoch must be nonnegative")
        empty_message = base.new_zeros(base.shape)
        empty_available = torch.zeros_like(inputs.damaged, dtype=torch.bool)
        if self.topk == 0:
            empty = ExpertEvidence(
                empty_message, empty_message, empty_available,
                {"source_variables": torch.empty((1, 0), device=values.device, dtype=torch.long),
                 "candidate_count": torch.zeros_like(inputs.damaged, dtype=torch.long)},
                CandidateEvidence(
                    base.new_zeros(batch, variables, patches, 1, self.d_model),
                    base.new_zeros(batch, variables, patches, 1, 4),
                    base.new_zeros(batch, variables, patches, 1),
                    torch.zeros((batch, variables, patches, 1), dtype=torch.bool, device=base.device),
                ),
            )
            blank = base.new_empty(0)
            return VariableSourceCache(
                blank, blank.to(torch.bool), blank, blank, blank, blank,
                base.new_zeros(batch, variables, patches, 1), empty.candidates.valid,
                empty.available, blank.to(torch.long), blank.to(torch.long), empty.provenance, empty,
            )

        selection_statistics = {}
        if self.dynamic_selection:
            source, prior, source_selected, selection_statistics = self._dynamic_sources(
                inputs, epoch=epoch, explore=explore,
            )
        elif self.selection != "global_topk":
            source, prior, source_selected = self._contextual_sources(
                inputs, epoch=epoch, explore=explore,
            )
        else:
            source, prior, source_selected = self._selected_sources(epoch=epoch, explore=explore)
        k, o = self.topk, len(self.offsets)
        target_patch = torch.arange(patches, device=values.device)
        offsets = torch.tensor(self.offsets, device=values.device)
        raw_patch = target_patch[:, None] + offsets[None, :]
        in_range = (raw_patch >= 0) & (raw_patch < patches)
        if self.selection != "global_topk":
            source_ix = source[..., None].expand(batch, variables, patches, k, o)
            patch_ix = raw_patch.clamp(0, patches - 1)[None, None, :, None, :].expand_as(source_ix)
            batch_ix = torch.arange(batch, device=values.device)[:, None, None, None, None]
            source_values = values[batch_ix, source_ix, patch_ix]
            source_observed = (mask & validity)[batch_ix, source_ix, patch_ix]
            selected_mask = source_selected[..., None]
            selected_prior = prior[..., None].expand(batch, variables, patches, k, o)
        else:
            source_ix = source[:, None, :, None].expand(variables, patches, k, o)
            patch_ix = raw_patch.clamp(0, patches - 1)[None, :, None, :].expand_as(source_ix)
            source_values = values[:, source_ix, patch_ix]
            source_observed = (mask & validity)[:, source_ix, patch_ix]
            selected_mask = source_selected[None, :, None, :, None]
            selected_prior = prior[None, :, None, :, None].expand(batch, variables, patches, k, o)
        target_observed = (mask & validity)[:, :, :, None, None, :]
        target_values = values[:, :, :, None, None, :]
        if self.selection == "window_affine_v1":
            target_ix = torch.arange(variables, device=values.device)[None, :, None, None, None]
            source_values = (self.pair_slope[target_ix, source_ix][..., None] * source_values
                             + self.pair_intercept[target_ix, source_ix][..., None])
        visible_values = (source_values.masked_fill(~source_observed, 0) if self.dynamic_selection
                          else source_values * source_observed)

        candidate_valid = (
            source_observed.any(dim=-1)
            & in_range[None, None, :, None, :]
            & inputs.damaged[..., None, None]
            & selected_mask
        )
        available = candidate_valid.any(dim=(-1, -2))
        source_fraction = source_observed.float().mean(dim=-1)
        common = source_observed & target_observed
        overlap = common.float().mean(dim=-1)
        if self.dynamic_selection:
            error = (source_values.masked_fill(~common, 0)
                     - target_values.masked_fill(~common, 0)).abs().sum(-1)
        else:
            error = ((source_values - target_values).abs() * common).sum(-1)
        error = error / common.sum(-1).clamp_min(1)
        consistency = torch.exp(-error) * common.any(dim=-1)
        signed_lag = offsets.to(values.dtype).view(1, 1, 1, 1, o)
        lag = signed_lag.expand(batch, variables, patches, k, o)
        candidate_score = (
            selected_prior + 0.5 * source_fraction + 0.5 * consistency
            - 0.1 * lag.abs()
        )
        candidate_score = candidate_score.masked_fill(~candidate_valid, -1e9)
        raw_quality = torch.stack((selected_prior, source_fraction, overlap * consistency, lag), dim=-1)
        provenance = {"source_variables": source.detach(), "source_patch_indices": patch_ix.detach(),
                      "source_selected": source_selected.detach(),
                      "candidate_valid": candidate_valid.detach(),
                      "candidate_count": candidate_valid.sum(dim=(-1, -2)).detach(),
                      **selection_statistics}
        return VariableSourceCache(
            visible_values, source_observed, lag, source_fraction, consistency,
            raw_quality, candidate_score, candidate_valid, available, source_ix, patch_ix, provenance,
        )

    def from_cache(
        self, inputs: ExpertInputs, cache: VariableSourceCache,
        *, round_embedding: Tensor | None = None,
        source_context_allowed: Tensor | None = None,
    ) -> ExpertEvidence:
        if cache.empty_evidence is not None:
            return cache.empty_evidence
        base, values = inputs.base_latents, inputs.patch_values
        batch, variables, patches, _ = base.shape
        k, o = self.topk, len(self.offsets)
        if cache.visible_values.shape != (batch, variables, patches, k, o, self.patch_length):
            raise ValueError("variable cache shape mismatch")
        visible_values, source_observed = cache.visible_values, cache.source_observed
        lag, source_fraction, consistency = cache.lag, cache.source_fraction, cache.consistency
        candidate_valid, available = cache.valid, cache.available
        candidate_score = cache.logits
        source_state = None
        if self.contextual:
            batch_ix = torch.arange(batch, device=base.device)[:, None, None, None, None]
            source_state = self.context_norm(base[batch_ix, cache.source_ix, cache.patch_ix])
            target_state = self.context_norm(base)
            if round_embedding is not None:
                target_state = target_state + round_embedding
            query = self.context_query(target_state)[:, :, :, None, None, :]
            key = self.context_key(source_state)
            match = (query * key).sum(-1) / sqrt(self.d_model)
            adjustment = 0.5 * torch.tanh(match)
            if source_context_allowed is not None:
                if source_context_allowed.shape != base.shape[:-1]:
                    raise ValueError("variable source context mask shape mismatch")
                adjustment = adjustment * source_context_allowed[..., None, None]
            candidate_score = candidate_score + adjustment
            candidate_score = candidate_score.masked_fill(~candidate_valid, -1e9)
        weights = torch.softmax(candidate_score.flatten(-2), dim=-1).reshape_as(candidate_score)
        weights = weights * candidate_valid
        weights = weights / weights.sum(dim=(-1, -2), keepdim=True).clamp_min(1e-8)

        expanded_base = base[:, :, :, None, None, :].expand(
            batch, variables, patches, k, o, self.d_model
        )
        features = torch.cat((
            visible_values, source_observed.to(values.dtype), expanded_base,
            lag[..., None], source_fraction[..., None], consistency[..., None],
        ), dim=-1)
        if self.dynamic_selection:
            # Only active edges enter the MLP. Padded slots remain for downstream
            # batching, so this does not imply the entire repairer scales by count.
            candidate_messages = base.new_zeros(*features.shape[:-1], self.d_model)
            candidate_messages[candidate_valid] = self.message_projector(features[candidate_valid])
        else:
            candidate_messages = self.message_projector(features)
        if self.calibration is not None:
            batch_ix = torch.arange(batch,device=base.device)[:,None,None,None,None]
            source_mean = inputs.query_mean[batch_ix,cache.source_ix]
            source_std = inputs.query_std[batch_ix,cache.source_ix]
            raw_source = torch.where(source_observed,
                torch.where(source_observed,visible_values,0)*source_std[...,None]+source_mean[...,None],0)
            source_identity = self.source_embedding(cache.source_ix)
            target_identity = self.target_embedding.weight[None,:,None,None,None,:].expand_as(source_identity)
            target_mean = inputs.query_mean[:,:,None,None,None].expand_as(source_mean)
            target_std = inputs.query_std[:,:,None,None,None].expand_as(source_std)
            statistics = torch.stack((source_mean,source_std.clamp_min(1e-6).log(),
                                      target_mean,target_std.clamp_min(1e-6).log()),-1)
            conditioning = torch.cat((source_identity,target_identity,raw_source,
                                      source_observed.to(base.dtype),statistics),-1)
            correction = self.calibration(conditioning).masked_fill(~candidate_valid[...,None],0)
            candidate_messages = candidate_messages+correction
        if source_state is not None:
            source_message = self.source_context_projection(source_state)
            if source_context_allowed is not None:
                source_message = source_message * source_context_allowed[..., None, None, None]
            candidate_messages = candidate_messages + source_message
        provenance = {**cache.provenance, "weights": weights.detach()}
        if self.aggregation in ("channelwise_v1", "channelwise_bounded_v1"):
            # Every output channel scores the same sparse visible candidates.
            # Zero initialization exactly reproduces scalar v12 weights at startup.
            channel_delta = self.channel_projector(candidate_messages)
            if self.aggregation == "channelwise_bounded_v1":
                channel_delta = 0.5 * torch.tanh(channel_delta)
            channel_logits = candidate_score[..., None] + channel_delta
            channel_weights = torch.softmax(channel_logits.flatten(-3, -2), dim=-2).reshape_as(channel_logits)
            channel_weights = channel_weights * candidate_valid[..., None]
            channel_weights = channel_weights / channel_weights.sum(dim=(-3, -2), keepdim=True).clamp_min(1e-8)
            message = (channel_weights * candidate_messages).sum(dim=(-3, -2))
            provenance["channel_weights"] = channel_weights.detach()
        else:
            message = (weights[..., None] * candidate_messages).sum(dim=(-3, -2))
        raw_quality = cache.raw_quality
        quality = self.quality_projector(
            (weights[..., None] * raw_quality).sum(dim=(-3, -2))
        )
        message = torch.where(available[..., None], message, torch.zeros_like(base))
        quality = torch.where(available[..., None], quality, torch.zeros_like(base))
        return ExpertEvidence(
            message, quality, available,
            provenance,
            CandidateEvidence(
                candidate_messages.flatten(-3, -2),
                raw_quality.flatten(-3, -2),
                weights.flatten(-2), candidate_valid.flatten(-2),
            ),
        )

    def forward(self, inputs: ExpertInputs, *, epoch: int, explore: bool) -> ExpertEvidence:
        return self.from_cache(inputs, self.prepare_sources(inputs, epoch=epoch, explore=explore))


__all__ = ["SparseVariableExpert", "VariableSourceCache"]
