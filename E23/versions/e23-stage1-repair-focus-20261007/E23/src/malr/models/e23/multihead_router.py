"""Parallel H/V/T heads that attend to expert candidates before aggregation."""

from __future__ import annotations

from collections.abc import Mapping
from math import sqrt

import torch
from torch import Tensor, nn

from .contracts import ExpertEvidence, ExpertInputs, RouterOutput
from .router import EXPERTS, ExpertRouter


class CandidateMultiHeadRouter(nn.Module):
    def __init__(
        self, d_model: int = 128, *, head_count: int = 8,
        fusion_mode: str = "concat",
    ) -> None:
        super().__init__()
        if (type(d_model) is not int or type(head_count) is not int or head_count < 1
                or d_model < 1 or (fusion_mode == "concat" and d_model % head_count)):
            raise ValueError("router width must be positive and divisible by head count")
        if fusion_mode not in ("concat", "gated_sum"):
            raise ValueError("unsupported candidate head fusion mode")
        self.d_model = d_model
        self.head_count = head_count
        self.fusion_mode = fusion_mode
        self.head_width = d_model // head_count if fusion_mode == "concat" else d_model
        projected_width = head_count * self.head_width
        if fusion_mode == "gated_sum":
            self.base_projection = nn.Linear(d_model, projected_width)
            self.head_gate = nn.Sequential(
                nn.Linear(4 * self.head_width + 3, self.head_width),
                nn.GELU(), nn.Linear(self.head_width, 1),
            )
            self.none_head_score = nn.Linear(d_model + 3, 1)
        self.base_norm = nn.LayerNorm(self.head_width)
        self.query = nn.ModuleDict({name: nn.Linear(d_model, projected_width) for name in EXPERTS})
        self.key = nn.ModuleDict({name: nn.Linear(d_model, projected_width) for name in EXPERTS})
        self.value = nn.ModuleDict({name: nn.Linear(d_model, projected_width) for name in EXPERTS})
        self.quality = nn.ModuleDict({
            "historical": nn.Linear(5, projected_width),
            "variable": nn.Linear(4, projected_width),
            "temporal": nn.Linear(5, projected_width),
        })
        self.residual = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(2 * self.head_width, self.head_width),
                                nn.GELU(), nn.Linear(self.head_width, self.head_width))
            for name in EXPERTS
        })
        self.scales = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(0.1)) for name in EXPERTS
        })
        self.type_embedding = nn.Embedding(len(EXPERTS), self.head_width)
        self.head_embedding = nn.Embedding(head_count, self.head_width)
        self.shared_score = nn.Sequential(
            nn.Linear(5 * self.head_width + 3, self.head_width),
            nn.GELU(), nn.Linear(self.head_width, 1),
        )
        self.none_score = nn.Linear(2 * self.head_width + 3, 1)
        self.none_bias = nn.Parameter(torch.tensor(0.0))

    def _aggregate(self, name: str, base: Tensor, item: ExpertEvidence) -> tuple[Tensor, Tensor, Tensor]:
        candidates = item.candidates
        if candidates is None:
            raise ValueError(f"{name} candidate evidence is missing")
        b, c, n, d = base.shape
        messages, quality, weights, valid = (
            candidates.messages, candidates.quality, candidates.weights, candidates.valid,
        )
        k = messages.shape[-2] if messages.ndim == 5 else -1
        qdim = self.quality[name].in_features
        if (messages.shape != (b, c, n, k, d) or k < 1
                or quality.shape != (b, c, n, k, qdim)
                or weights.shape != (b, c, n, k) or valid.shape != weights.shape):
            raise ValueError(f"{name} candidate shape mismatch")
        if not torch.isfinite(messages).all() or not torch.isfinite(quality).all() or not torch.isfinite(weights).all():
            raise ValueError(f"{name} candidate contains nonfinite values")
        heads, width = self.head_count, self.head_width
        query = self.query[name](base).reshape(b, c, n, heads, width)
        key = self.key[name](messages).reshape(b, c, n, k, heads, width).transpose(-3, -2)
        value = self.value[name](messages).reshape(b, c, n, k, heads, width).transpose(-3, -2)
        projected_quality = self.quality[name](quality).reshape(
            b, c, n, k, heads, width,
        ).transpose(-3, -2)
        scores = (query[..., None, :] * key).sum(-1) / sqrt(width)
        scores = scores + torch.log(weights.clamp_min(1e-8))[..., None, :]
        allowed = valid[..., None, :] & item.available[..., None, None].to(torch.bool)
        scores = scores.masked_fill(~allowed, -1e9)
        attention = torch.softmax(scores, dim=-1) * allowed
        attention = attention / attention.sum(-1, keepdim=True).clamp_min(1e-8)
        message = (attention[..., None] * value).sum(-2)
        summary_quality = (attention[..., None] * projected_quality).sum(-2)
        return message, summary_quality, attention

    def forward(self, inputs: ExpertInputs, evidence: Mapping[str, ExpertEvidence]) -> RouterOutput:
        ExpertRouter._validate(self, inputs, evidence)
        base = inputs.base_latents
        b, c, n, _ = base.shape
        heads, width = self.head_count, self.head_width
        base_heads = (base.reshape(b, c, n, heads, width)
                      if self.fusion_mode == "concat" else
                      self.base_projection(base).reshape(b, c, n, heads, width))
        normalized = self.base_norm(base_heads)
        geometry = ExpertRouter._geometry(self, inputs)
        head_embedding = self.head_embedding.weight.view(1, 1, 1, heads, width)
        head_embedding = head_embedding.expand(b, c, n, heads, width)
        none = self.none_score(torch.cat((normalized, head_embedding,
                                          geometry[..., None, :].expand(b, c, n, heads, 3)), -1)).squeeze(-1)
        none = none + self.none_bias
        expert_head_deltas = {}
        attention = {}
        delta_list = []
        quality_list = []
        availability = [torch.ones_like(inputs.damaged, dtype=torch.bool)]
        for name in EXPERTS:
            message, quality, attention[name] = self._aggregate(name, base, evidence[name])
            delta = self.scales[name] * self.residual[name](torch.cat((normalized, message), -1))
            expert_head_deltas[name] = delta
            delta_list.append(delta)
            quality_list.append(quality)
            availability.append(evidence[name].available.to(torch.bool) & inputs.damaged.to(torch.bool)
                              & evidence[name].candidates.valid.any(-1))
        stacked = torch.stack(delta_list, -2)
        kinds = self.type_embedding.weight.view(1, 1, 1, 1, len(EXPERTS), width)
        score_features = torch.cat((
            normalized[..., None, :].expand_as(stacked), stacked,
            torch.stack(quality_list, -2),
            kinds.expand_as(stacked),
            head_embedding[..., None, :].expand_as(stacked),
            geometry[..., None, None, :].expand(b, c, n, heads, len(EXPERTS), 3),
        ), -1)
        scores = self.shared_score(score_features).squeeze(-1)
        if not torch.isfinite(scores).all() or not torch.isfinite(none).all():
            raise ValueError("multihead router score is nonfinite")
        available = torch.stack(availability, -1)
        logits = torch.cat((none[..., None],
                            scores.masked_fill(~available[..., None, 1:], -torch.inf)), -1)
        head_weights = torch.softmax(logits, -1)
        head_deltas = (head_weights[..., 1:, None] * stacked).sum(-2)
        if self.fusion_mode == "concat":
            latents = (base_heads + head_deltas).reshape(b, c, n, self.d_model)
            deltas = {name: delta.reshape(b, c, n, self.d_model)
                      for name, delta in expert_head_deltas.items()}
            return RouterOutput(latents, head_weights.mean(-2), deltas, available,
                                head_weights, attention)
        head_quality = (head_weights[..., 1:, None]
                        * torch.stack(quality_list, -2)).sum(-2)
        gate_features = torch.cat((
            normalized, head_deltas, head_quality, head_embedding,
            geometry[..., None, :].expand(b, c, n, heads, 3),
        ), -1)
        gate_scores = self.head_gate(gate_features).squeeze(-1)
        none_gate_score = self.none_head_score(torch.cat((base, geometry), -1)).squeeze(-1)
        if not torch.isfinite(gate_scores).all() or not torch.isfinite(none_gate_score).all():
            raise ValueError("head gate score is nonfinite")
        has_evidence = available[..., 1:].any(-1)
        gate_logits = torch.cat((none_gate_score[..., None],
                                 gate_scores.masked_fill(~has_evidence[..., None], -torch.inf)), -1)
        head_gate_weights = torch.softmax(gate_logits, -1)
        latents = base + (head_gate_weights[..., 1:, None] * head_deltas).sum(-2)
        expert_mass = (head_gate_weights[..., 1:, None] * head_weights[..., 1:]).sum(-2)
        none_mass = head_gate_weights[..., 0] + (
            head_gate_weights[..., 1:] * head_weights[..., 0]
        ).sum(-1)
        effective_weights = torch.cat((none_mass[..., None], expert_mass), -1)
        deltas = {
            name: (head_gate_weights[..., 1:, None]
                   * head_weights[..., index, None] * expert_head_deltas[name]).sum(-2)
            for index, name in enumerate(EXPERTS, 1)
        }
        return RouterOutput(latents, effective_weights, deltas, available,
                            head_weights, attention, head_gate_weights, head_deltas)


__all__ = ["CandidateMultiHeadRouter"]
