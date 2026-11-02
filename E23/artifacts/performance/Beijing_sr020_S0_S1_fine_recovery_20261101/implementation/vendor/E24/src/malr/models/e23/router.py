"""One residual arbitration step over parallel H/V/T evidence."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from .contracts import ExpertEvidence, ExpertInputs, RouterOutput


EXPERTS = ("historical", "variable", "temporal")


class ExpertRouter(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        if type(d_model) is not int or d_model <= 0:
            raise ValueError("d_model must be a positive integer")
        self.d_model = d_model
        self.base_norm = nn.LayerNorm(d_model)
        self.residual = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model),
            ) for name in EXPERTS
        })
        self.scales = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(0.1)) for name in EXPERTS
        })
        self.type_embedding = nn.Embedding(len(EXPERTS), d_model)
        self.shared_score = nn.Sequential(
            nn.Linear(4 * d_model + 3, d_model), nn.GELU(), nn.Linear(d_model, 1),
        )
        self.none_score = nn.Linear(d_model + 3, 1)
        self.none_bias = nn.Parameter(torch.tensor(0.0))

    def _validate(self, inputs: ExpertInputs, evidence: Mapping[str, ExpertEvidence]) -> None:
        if set(evidence) != set(EXPERTS):
            raise ValueError("router evidence must contain exactly historical, variable and temporal")
        base = inputs.base_latents
        if base.ndim != 4 or base.shape[-1] != self.d_model:
            raise ValueError("base latent shape must be [B,C,N,D]")
        b, c, n, _ = base.shape
        patch_shape = inputs.patch_values.shape
        if len(patch_shape) != 4 or patch_shape[:3] != (b, c, n):
            raise ValueError("patch values have wrong shape")
        if inputs.patch_mask.shape != patch_shape or inputs.patch_validity.shape != patch_shape:
            raise ValueError("patch mask/validity shape mismatch")
        if inputs.rho.shape != (b, c, n) or inputs.damaged.shape != (b, c, n):
            raise ValueError("rho/damaged shape mismatch")
        if inputs.patch_starts.shape not in {(b, n), (b, c, n)}:
            raise ValueError("patch starts shape mismatch")
        if inputs.query_mean.shape != (b, c) or inputs.query_std.shape != (b, c):
            raise ValueError("query statistics shape mismatch")
        if not torch.isfinite(base).all() or not torch.isfinite(inputs.rho).all():
            raise ValueError("router input contains nonfinite values")
        for name in EXPERTS:
            item = evidence[name]
            if item.message.shape != base.shape or item.quality.shape != base.shape:
                raise ValueError(f"{name} evidence shape mismatch")
            if item.available.shape != (b, c, n):
                raise ValueError(f"{name} availability shape mismatch")
            if not torch.isfinite(item.message).all() or not torch.isfinite(item.quality).all():
                raise ValueError(f"{name} evidence contains nonfinite values")

    def _geometry(self, inputs: ExpertInputs) -> Tensor:
        valid = inputs.patch_validity.to(dtype=torch.bool)
        observed = inputs.patch_mask.to(dtype=torch.bool) & valid
        denom = valid.sum(dim=-1).clamp_min(1)
        observed_ratio = observed.sum(dim=-1).to(inputs.base_latents.dtype) / denom
        missing_ratio = ((~observed) & valid).sum(dim=-1).to(inputs.base_latents.dtype) / denom
        starts = inputs.patch_starts.to(inputs.base_latents.dtype)
        if starts.ndim == 2:
            starts = starts[:, None, :].expand_as(observed_ratio)
        relative = starts / starts.amax(dim=-1, keepdim=True).clamp_min(1)
        return torch.stack((observed_ratio, missing_ratio, relative), dim=-1)

    def _condition(self, inputs: ExpertInputs, embedding: Tensor | None) -> Tensor:
        normalized = self.base_norm(inputs.base_latents)
        if embedding is not None:
            if embedding.shape != (self.d_model,) or not torch.isfinite(embedding).all():
                raise ValueError("round embedding shape or values are invalid")
            normalized = normalized + embedding
        return normalized

    def expert_delta(self, inputs: ExpertInputs, name: str, item: ExpertEvidence) -> Tensor:
        if name not in EXPERTS:
            raise ValueError("unknown expert")
        return self.scales[name] * self.residual[name](
            torch.cat((self.base_norm(inputs.base_latents), item.message), -1)
        )

    def none_logit(self, inputs: ExpertInputs, *, round_embedding: Tensor | None = None) -> Tensor:
        return self.none_score(torch.cat((self._condition(inputs, round_embedding),
                                         self._geometry(inputs)), -1)).squeeze(-1) + self.none_bias

    def expert_logit(
        self, inputs: ExpertInputs, name: str, item: ExpertEvidence, delta: Tensor,
        *, round_embedding: Tensor | None = None,
    ) -> Tensor:
        if name not in EXPERTS:
            raise ValueError("unknown expert")
        normalized = self._condition(inputs, round_embedding)
        kind = self.type_embedding.weight[EXPERTS.index(name)].expand_as(normalized)
        return self.shared_score(torch.cat((normalized, delta, item.quality, kind,
                                             self._geometry(inputs)), -1)).squeeze(-1)

    def preview_gate(
        self, inputs: ExpertInputs, name: str, item: ExpertEvidence, delta: Tensor,
        *, allowed: Tensor, round_embedding: Tensor | None = None,
    ) -> Tensor:
        if allowed.shape != inputs.damaged.shape:
            raise ValueError("preview allowed shape mismatch")
        score = self.expert_logit(inputs, name, item, delta, round_embedding=round_embedding)
        none = self.none_logit(inputs, round_embedding=round_embedding)
        eligible = allowed.to(torch.bool) & inputs.damaged.to(torch.bool) & item.available.to(torch.bool)
        return torch.sigmoid(score - none) * eligible

    def route_precomputed(
        self, inputs: ExpertInputs, evidence: Mapping[str, ExpertEvidence],
        deltas: Mapping[str, Tensor], *, round_embedding: Tensor | None = None,
    ) -> RouterOutput:
        self._validate(inputs, evidence)
        if set(deltas) != set(EXPERTS):
            raise ValueError("precomputed deltas must contain exactly H/V/T")
        for name, delta in deltas.items():
            if delta.shape != inputs.base_latents.shape or not torch.isfinite(delta).all():
                raise ValueError(f"{name} precomputed delta shape or values are invalid")
        base = inputs.base_latents
        normalized = self._condition(inputs, round_embedding)
        geometry = self._geometry(inputs)
        none = self.none_score(torch.cat((normalized, geometry), -1)).squeeze(-1) + self.none_bias
        stacked = torch.stack([deltas[name] for name in EXPERTS], -2)
        kinds = self.type_embedding.weight.view(1, 1, 1, len(EXPERTS), self.d_model)
        features = torch.cat((
            normalized[..., None, :].expand_as(stacked), stacked,
            torch.stack([evidence[name].quality for name in EXPERTS], -2),
            kinds.expand_as(stacked),
            geometry[..., None, :].expand(*geometry.shape[:-1], len(EXPERTS), 3),
        ), -1)
        scores = self.shared_score(features).squeeze(-1)
        if not torch.isfinite(scores).all() or not torch.isfinite(none).all():
            raise ValueError("router score is nonfinite")
        available = torch.stack((torch.ones_like(inputs.damaged, dtype=torch.bool),
                                 *(evidence[name].available.to(torch.bool) & inputs.damaged.to(torch.bool)
                                   for name in EXPERTS)), -1)
        logits = torch.cat((none[..., None], scores.masked_fill(~available[..., 1:], -torch.inf)), -1)
        weights = torch.softmax(logits, -1)
        latents = base
        for index, name in enumerate(EXPERTS, 1):
            latents = latents + weights[..., index, None] * deltas[name]
        return RouterOutput(latents, weights, deltas, available)

    def forward(
        self, inputs: ExpertInputs, evidence: Mapping[str, ExpertEvidence],
    ) -> RouterOutput:
        self._validate(inputs, evidence)
        base = inputs.base_latents
        normalized = self.base_norm(base)
        geometry = self._geometry(inputs)
        none_logit = self.none_score(torch.cat((normalized, geometry), dim=-1)).squeeze(-1)
        none_logit = none_logit + self.none_bias
        deltas: dict[str, Tensor] = {}
        delta_values: list[Tensor] = []
        available = [torch.ones_like(inputs.damaged, dtype=torch.bool)]
        for name in EXPERTS:
            item = evidence[name]
            delta = self.scales[name] * self.residual[name](
                torch.cat((normalized, item.message), dim=-1)
            )
            deltas[name] = delta
            delta_values.append(delta)
            eligible = item.available.to(torch.bool) & inputs.damaged.to(torch.bool)
            available.append(eligible)
        stacked = torch.stack(delta_values, dim=-2)
        kinds = self.type_embedding.weight.view(1, 1, 1, len(EXPERTS), self.d_model)
        score_inputs = torch.cat((
            normalized[..., None, :].expand_as(stacked), stacked,
            torch.stack([evidence[name].quality for name in EXPERTS], dim=-2),
            kinds.expand_as(stacked),
            geometry[..., None, :].expand(*geometry.shape[:-1], len(EXPERTS), 3),
        ), dim=-1)
        scores = self.shared_score(score_inputs).squeeze(-1)
        if not torch.isfinite(scores).all() or not torch.isfinite(none_logit).all():
            raise ValueError("router score is nonfinite")
        availability = torch.stack(available, dim=-1)
        logits = torch.cat((
            none_logit[..., None], scores.masked_fill(~availability[..., 1:], -torch.inf),
        ), dim=-1)
        weights = torch.softmax(logits, dim=-1)
        latents = base
        for index, name in enumerate(EXPERTS, 1):
            latents = latents + weights[..., index, None] * deltas[name]
        return RouterOutput(latents, weights, deltas, availability)


__all__ = ["ExpertRouter", "EXPERTS"]
