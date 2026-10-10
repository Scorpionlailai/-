"""Shared point temporal features and feature-dependent variable transfer.

Original observations, validity, and repaired latent context remain distinct.
The extra information enters T/V evidence before the existing H/V/T router.
"""
from __future__ import annotations

from dataclasses import replace
from math import log

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .contracts import ExpertEvidence, ExpertInputs
from .exposure import sample_specialist_exposure
from .iterative import IterationDiagnostics, IterativeRepairState, IterativeTVHController
from .router import EXPERTS


class SharedTemporalFeatureBank(nn.Module):
    def __init__(self, variables: int, width: int, patch_length: int,
                 stride: int, window_length: int) -> None:
        super().__init__()
        self.variables, self.width = variables, width
        self.patch_length, self.stride, self.window_length = patch_length, stride, window_length
        self.input_projection = nn.Conv1d(2, width, 5, padding=2)
        self.long_filter = nn.Conv1d(width, width, 71, padding=35, groups=width)
        self.short_filter = nn.Conv1d(width, width, 5, padding=2, groups=width)
        self.feature_mix = nn.Conv1d(width, width, 1)
        self.feature_norm = nn.GroupNorm(1, width)
        self.state_projection = nn.Linear(width, width, bias=False)
        nn.init.zeros_(self.state_projection.weight)
        self.query = nn.Conv1d(width, width, 7, padding=3, groups=width)
        self.key = nn.Conv1d(width, width, 7, padding=3, groups=width)
        self.value = nn.Conv1d(width, width, 7, padding=3, groups=width)
        self.pair_prior = nn.Parameter(torch.zeros(variables, variables, width))
        self.temporal_output = nn.Linear(width, width)
        self.variable_output = nn.Linear(width, width)
        self.temporal_quality = nn.Sequential(nn.Linear(3, width), nn.Tanh())
        self.variable_quality = nn.Sequential(nn.Linear(3, width), nn.Tanh())
        self.temporal_scale = nn.Parameter(torch.tensor(0.01))
        self.variable_scale = nn.Parameter(torch.tensor(0.01))

    def _fold(self, values: Tensor) -> Tensor:
        b, c, n, p = values.shape
        length = (n - 1) * self.stride + p
        return F.fold(values.reshape(b*c, n, p).transpose(1, 2),
                      output_size=(1, length), kernel_size=(1, p),
                      stride=(1, self.stride)).reshape(b, c, length)

    def _pool(self, points: Tensor, inputs: ExpertInputs) -> Tensor:
        b, c, d, length = points.shape
        p, n = self.patch_length, inputs.patch_values.shape[2]
        padded = (n-1)*self.stride+p
        points = F.pad(points, (0, padded-length)).reshape(b*c, d, 1, padded)
        patches = F.unfold(points, kernel_size=(1, p), stride=(1, self.stride))
        patches = patches.reshape(b, c, d, p, n).permute(0, 1, 4, 2, 3)
        valid = inputs.patch_validity[..., None, :].to(points.dtype)
        return (patches*valid).sum(-1)/valid.sum(-1).clamp_min(1)

    def _encode(self, inputs: ExpertInputs, reference: Tensor,
                context_allowed: Tensor | None) -> tuple[Tensor, Tensor, Tensor]:
        observed = inputs.patch_mask.bool() & inputs.patch_validity.bool()
        valid = inputs.patch_validity.to(inputs.base_latents.dtype)
        count = self._fold(valid).clamp_min(1)
        visible = self._fold(inputs.patch_values.masked_fill(~observed, 0))/count
        mask = self._fold(observed.to(visible.dtype))/count
        visible, mask = visible[..., :self.window_length], mask[..., :self.window_length]
        b, c, length = visible.shape
        delta = inputs.base_latents-reference
        if context_allowed is not None:
            delta = delta*context_allowed[..., None]
        delta = self.state_projection(delta)
        n, d = delta.shape[-2:]
        p = self.patch_length
        context_patches = delta[..., None]*valid[..., None, :]
        context_patches = context_patches.permute(0, 1, 3, 4, 2).reshape(b*c, d*p, n)
        context = F.fold(context_patches, output_size=(1, count.shape[-1]),
                         kernel_size=(1, p), stride=(1, self.stride))
        context = context.reshape(b, c, d, -1)/count[..., None, :]
        context = context[..., :length].reshape(b*c, d, length)
        source = torch.stack((visible, mask), 2).reshape(b*c, 2, length)
        features = self.input_projection(source)+context
        features = self.feature_mix(F.gelu(self.long_filter(features)+self.short_filter(features)))
        features = self.feature_norm(features).reshape(b, c, d, length)
        # Reliability is computed from original observations, never latent estimates.
        flat_mask = mask.reshape(b*c, 1, length)
        confidence = .75*F.avg_pool1d(flat_mask, 71, stride=1, padding=35,
                                     count_include_pad=False)+.25*flat_mask.mean(-1, keepdim=True)
        confidence = confidence.reshape(b, c, 1, length)
        return features, confidence, mask

    def temporal_evidence(self, inputs: ExpertInputs, old: ExpertEvidence,
                          reference: Tensor, context_allowed: Tensor | None) -> ExpertEvidence:
        features, confidence, mask = self._encode(inputs, reference, context_allowed)
        pooled = self._pool(features, inputs)
        # A full-window summary supplements the dense local feature field.
        global_feature = (features*confidence).sum(-1)/confidence.sum(-1).clamp_min(1e-6)
        message = self.temporal_output(pooled+global_feature[:, :, None, :])
        visible_fraction = mask.mean(-1)[:, :, None].expand_as(inputs.rho)
        local_fraction = self._pool(confidence, inputs).squeeze(-1)
        available = visible_fraction.gt(0) & inputs.damaged.bool()
        quality = self.temporal_quality(torch.stack((inputs.rho, local_fraction, visible_fraction), -1))
        return replace(old, message=old.message+self.temporal_scale*message*available[..., None],
            quality=old.quality+self.temporal_scale*quality*available[..., None],
            available=old.available | available,
            provenance={**old.provenance,'candidate_count':old.provenance['candidate_count']+available.long(),
                        'shared_point_features':True})

    def variable_transfer(self, inputs: ExpertInputs, reference: Tensor,
                          context_allowed: Tensor | None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        features, confidence, mask = self._encode(inputs, reference, context_allowed)
        b, c, d, length = features.shape
        flat = features.reshape(b*c, d, length)
        q = F.normalize(self.query(flat).reshape(b, c, d, length), dim=-1)
        k = F.normalize(self.key(flat).reshape(b, c, d, length), dim=-1)
        scores = 3*torch.einsum('bcdt,bsdt->bcsd', q, k)+self.pair_prior
        source_confidence = self._pool(confidence, inputs).squeeze(-1).permute(0, 2, 1)
        scores = scores[:, :, None, :, :]+source_confidence[:, None, :, :, None].clamp_min(.01).log()
        n = inputs.rho.shape[-1]
        diagonal = torch.eye(c, dtype=torch.bool, device=features.device)
        visible_source = mask.any(-1)
        valid = (visible_source[:, None, None, :].expand(b, c, n, c)
                 & ~diagonal[None, :, None, :] & inputs.damaged[..., None].bool())
        weights = scores.masked_fill(~valid[..., None], -1e9).softmax(-2)
        weights = weights*valid[..., None]
        weights = weights/weights.sum(-2, keepdim=True).clamp_min(1e-12)
        source = self.value(flat).reshape(b, c, d, length)
        source = self._pool(source, inputs).permute(0, 2, 1, 3)
        message = (weights*source[:, None, :, :, :]).sum(-2)
        transferred_confidence = (weights.mean(-1)*source_confidence[:, None, :, :]).sum(-1)
        quality = self.variable_quality(torch.stack((inputs.rho, transferred_confidence,
                                                    weights.mean(-1).amax(-1)), -1))
        return self.variable_output(message), quality, valid, weights

    def variable_evidence(self, inputs: ExpertInputs, old: ExpertEvidence,
                          reference: Tensor, context_allowed: Tensor | None) -> ExpertEvidence:
        message, quality, valid, weights = self.variable_transfer(inputs, reference, context_allowed)
        available = valid.any(-1)
        return replace(old, message=old.message+self.variable_scale*message*available[..., None],
            quality=old.quality+self.variable_scale*quality*available[..., None],
            available=old.available | available,
            provenance={**old.provenance,'candidate_count':old.provenance['candidate_count']+valid.sum(-1),
                        'feature_variable_weights':weights,'shared_point_features':True})


class FeatureTVHController(IterativeTVHController):
    def __init__(self, variables: int, width: int, patch_length: int,
                 stride: int, window_length: int, layers: int) -> None:
        super().__init__(width, 1)
        if layers not in (1, 2):
            raise ValueError('Feature TV/H trial supports one or two layers')
        self.layers = layers
        # Extra repair parameters must not change the original forecast-head RNG.
        with torch.random.fork_rng(devices=[]):
            self.feature_banks = nn.ModuleList([
                SharedTemporalFeatureBank(variables, width, patch_length, stride, window_length)
                for _ in range(layers)])
            self.later_step_logits = nn.Parameter(torch.full((layers-1,), log(.1/.9)))

    def forward(self, inputs: ExpertInputs, *, historical, variable, temporal, router,
                epoch: int, explore_variable: bool, specialist_exposure: float,
                specialist_seed: int, disabled_experts: frozenset[str] = frozenset()) -> IterativeRepairState:
        if not set(disabled_experts).issubset(EXPERTS):
            raise ValueError('Unknown disabled expert')
        h_cache = historical.prepare_candidates(inputs)
        v_cache = variable.prepare_sources(inputs, epoch=epoch, explore=explore_variable)
        t_cache = temporal.prepare_sources(inputs)
        visible_variables = t_cache.observed.any((-1, -2))
        t_available = visible_variables[..., None].expand_as(inputs.damaged) & inputs.damaged.bool()
        v_available = (visible_variables.sum(-1, keepdim=True)-visible_variables.long()).gt(0)
        v_available = v_available[..., None].expand_as(inputs.damaged) & inputs.damaged.bool()
        availability = torch.stack((h_cache.available, v_cache.available | v_available, t_available), -1)
        for index, name in enumerate(EXPERTS):
            if name in disabled_experts:availability[..., index]=False
        selected, allowed = None, availability
        if specialist_exposure:
            allowed, selected = sample_specialist_exposure(inputs.damaged, availability,
                fraction=specialist_exposure, seed=specialist_seed)
        context_allowed = None if selected is None else ~selected
        state, iterations = inputs.base_latents, []
        for layer, bank in enumerate(self.feature_banks):
            embedding = self.round_embedding.weight[layer]
            current = replace(inputs, base_latents=state)
            # Re-encode actual observations and the updated latent context each layer.
            t = temporal.from_cache(current, t_cache, round_embedding=embedding,
                                    source_context_allowed=context_allowed)
            t = bank.temporal_evidence(current, t, inputs.base_latents, context_allowed)
            t = replace(t, available=t.available & allowed[..., 2])
            dt = router.expert_delta(current, 'temporal', t)
            gt = router.preview_gate(current, 'temporal', t, dt, allowed=allowed[..., 2], round_embedding=embedding)
            after_t = replace(inputs, base_latents=state+gt[..., None]*dt)
            v = variable.from_cache(after_t, v_cache, round_embedding=embedding,
                                    source_context_allowed=context_allowed)
            v = bank.variable_evidence(after_t, v, inputs.base_latents, context_allowed)
            v = replace(v, available=v.available & allowed[..., 1])
            dv = router.expert_delta(after_t, 'variable', v)
            gv = router.preview_gate(after_t, 'variable', v, dv, allowed=allowed[..., 1], round_embedding=embedding)
            after_v = replace(inputs, base_latents=after_t.base_latents+gv[..., None]*dv)
            h = historical.from_cache(after_v, h_cache, round_embedding=embedding)
            h = replace(h, available=h.available & allowed[..., 0])
            dh = router.expert_delta(after_v, 'historical', h)
            evidence = {'historical':h, 'variable':v, 'temporal':t}
            scale = 1 if layer==0 else self.later_step_logits[layer-1].sigmoid()
            deltas = {'historical':dh*scale,'variable':dv*scale,'temporal':dt*scale}
            routed = router.route_precomputed(current, evidence, deltas, round_embedding=embedding)
            iterations.append(IterationDiagnostics(layer+1, routed.latents, routed.weights,
                torch.stack((gt, gv), -1), allowed, (routed.latents-state).norm(dim=-1)))
            state = routed.latents
        return IterativeRepairState(routed, evidence, tuple(iterations), selected)
