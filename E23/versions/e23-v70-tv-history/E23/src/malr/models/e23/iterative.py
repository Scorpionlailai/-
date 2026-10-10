"""Shared T/V state updates followed by one historical-aware arbitration."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

import torch
from torch import Tensor, nn

from .contracts import ExpertEvidence, ExpertInputs, RouterOutput
from .exposure import sample_specialist_exposure
from .historical import HistoricalExpert
from .router import EXPERTS, ExpertRouter
from .temporal import MultiScaleTemporalExpert
from .variable import SparseVariableExpert


@dataclass(frozen=True)
class IterationDiagnostics:
    round_index: int
    latents: Tensor
    weights: Tensor
    preview_gates: Tensor
    allowed: Tensor
    update_norm: Tensor


@dataclass(frozen=True)
class IterativeRepairState:
    router: RouterOutput
    evidence: Mapping[str, ExpertEvidence]
    iterations: tuple[IterationDiagnostics, ...]
    specialist_selected: Tensor | None


class IterativeTVHController(nn.Module):
    def __init__(self, d_model: int, rounds: int) -> None:
        super().__init__()
        if type(rounds) is not int or rounds not in (1, 3) or d_model < 1:
            raise ValueError("iterative recovery requires one or three rounds")
        self.rounds = rounds
        self.round_embedding = nn.Embedding(3, d_model)
        nn.init.zeros_(self.round_embedding.weight)

    def forward(
        self, inputs: ExpertInputs, *, historical: HistoricalExpert,
        variable: SparseVariableExpert, temporal: MultiScaleTemporalExpert,
        router: ExpertRouter, epoch: int, explore_variable: bool,
        specialist_exposure: float, specialist_seed: int,
        disabled_experts: frozenset[str] = frozenset(),
    ) -> IterativeRepairState:
        if not set(disabled_experts).issubset(EXPERTS):
            raise ValueError("unknown disabled expert")
        h_cache = historical.prepare_candidates(inputs)
        v_cache = variable.prepare_sources(inputs, epoch=epoch, explore=explore_variable)
        t_cache = temporal.prepare_sources(inputs)
        visible_sources = t_cache.observed.any(-1)
        temporal_available = (visible_sources.sum(-1, keepdim=True) - visible_sources.long()).gt(0)
        temporal_available = temporal_available & inputs.damaged.to(torch.bool)
        availability = torch.stack((h_cache.available, v_cache.available, temporal_available), -1)
        if historical.tv_query is not None:
            from .tv_history import fully_hidden_with_context
            # Candidate identities are unknown until TV supplies its query. This
            # potential mask permits H-only exposure; actual matches still gate H.
            availability[..., 0] |= fully_hidden_with_context(inputs)
        for i, name in enumerate(EXPERTS):
            if name in disabled_experts:
                availability[..., i] = False
        selected = None
        allowed = availability
        if specialist_exposure:
            allowed, selected = sample_specialist_exposure(
                inputs.damaged, availability, fraction=specialist_exposure, seed=specialist_seed,
            )
        # A singleton target must not consume other experts through source states.
        source_context_allowed = None if selected is None else ~selected
        state = inputs.base_latents
        iterations = []
        for round_index in range(self.rounds):
            embedding = self.round_embedding.weight[round_index]
            current = replace(inputs, base_latents=state)
            t = temporal.from_cache(current, t_cache, round_embedding=embedding,
                                    source_context_allowed=source_context_allowed)
            t = replace(t, available=t.available & allowed[..., 2])
            dt = router.expert_delta(current, "temporal", t)
            gt = router.preview_gate(current, "temporal", t, dt, allowed=allowed[..., 2],
                                     round_embedding=embedding)
            after_t = replace(inputs, base_latents=state + gt[..., None] * dt)
            v = variable.from_cache(after_t, v_cache, round_embedding=embedding,
                                    source_context_allowed=source_context_allowed)
            v = replace(v, available=v.available & allowed[..., 1])
            dv = router.expert_delta(after_t, "variable", v)
            gv = router.preview_gate(after_t, "variable", v, dv, allowed=allowed[..., 1],
                                     round_embedding=embedding)
            after_v = replace(inputs, base_latents=after_t.base_latents + gv[..., None] * dv)
            current_h_cache = historical.prepare_tv_candidates(after_v, h_cache)
            h = historical.from_cache(after_v, current_h_cache, round_embedding=embedding)
            h = replace(h, available=h.available & allowed[..., 0])
            dh = router.expert_delta(after_v, "historical", h)
            evidence = {"historical": h, "variable": v, "temporal": t}
            routed = router.route_precomputed(current, evidence,
                {"historical": dh, "variable": dv, "temporal": dt}, round_embedding=embedding)
            iterations.append(IterationDiagnostics(round_index + 1, routed.latents, routed.weights,
                torch.stack((gt, gv), -1), allowed, (routed.latents - state).norm(dim=-1)))
            state = routed.latents
        return IterativeRepairState(routed, evidence, tuple(iterations), selected)
