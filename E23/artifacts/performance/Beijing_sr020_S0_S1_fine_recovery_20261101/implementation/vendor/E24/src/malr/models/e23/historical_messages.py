"""Independent historical messages over shared immutable candidate values."""
from __future__ import annotations
from math import sqrt
import torch
from torch import Tensor, nn
from .contracts import CandidateEvidence, ExpertEvidence, ExpertInputs
from .historical_candidates import HistoricalCandidateCache

def initialize_historical_messages(self, patch_length: int, d_model: int, contextual: bool):
    self.message_projector = nn.Sequential(
        nn.Linear(patch_length + d_model + 5, d_model), nn.GELU(), nn.Linear(d_model, d_model),
    )
    self.quality_projector = nn.Sequential(nn.Linear(5, d_model), nn.Tanh())
    self.contextual = contextual
    if contextual:
        self.context_norm = nn.LayerNorm(d_model)
        self.context_query = nn.Linear(d_model, d_model, bias=False)
        self.context_key = nn.Linear(patch_length, d_model, bias=False)
        nn.init.zeros_(self.context_query.weight)

def empty_historical(base: Tensor, patch_length: int, d_model: int, topk: int, *, candidate_count: Tensor | None = None) -> ExpertEvidence:
    zeros = torch.zeros_like(base)
    b, c, n, _ = base.shape
    if candidate_count is None:
        candidate_count = torch.zeros((b, c, n), dtype=torch.long, device=base.device)
    return ExpertEvidence(
        zeros, zeros, torch.zeros((b, c, n), dtype=torch.bool, device=base.device),
        {"candidate_count": candidate_count,
         "candidate_valid": torch.zeros((b, c, n, topk), dtype=torch.bool, device=base.device),
         "candidate_starts": torch.zeros((b, c, n, topk), dtype=torch.long, device=base.device),
         "candidate_raw": torch.zeros((b, c, n, topk, patch_length), dtype=base.dtype, device=base.device),
         "weights": torch.zeros((b, c, n, topk), dtype=base.dtype, device=base.device),
         "visible_error": torch.zeros((b, c, n, topk), dtype=base.dtype, device=base.device)},
        CandidateEvidence(
            base.new_zeros(b, c, n, topk, d_model),
            base.new_zeros(b, c, n, topk, 5),
            base.new_zeros(b, c, n, topk),
            torch.zeros((b, c, n, topk), dtype=torch.bool, device=base.device),
        ),
    )


def historical_from_cache(
    self, inputs: ExpertInputs, cache: HistoricalCandidateCache,
    *, round_embedding: Tensor | None = None,
) -> ExpertEvidence:
    base = inputs.base_latents
    batch, variables, patches, _ = base.shape
    count = getattr(self, "topk", cache.normalized_raw.shape[-2])
    if cache.normalized_raw.shape != (batch, variables, patches, count, self.patch_length):
        raise ValueError("historical cache shape mismatch")
    if cache.empty:
        return empty_historical(base, self.patch_length, self.d_model, count)
    valid, available = cache.valid, cache.available
    normalized_raw, raw_quality = cache.normalized_raw, cache.raw_quality
    logits = cache.base_logits
    if self.contextual:
        state = self.context_norm(base)
        if round_embedding is not None:
            state = state + round_embedding
        query = self.context_query(state)
        key = self.context_key(normalized_raw)
        compatibility = (query[..., None, :] * key).sum(-1) / sqrt(self.d_model)
        logits = logits + 0.5 * torch.tanh(compatibility)
    logits = logits.masked_fill(~valid, -1e9)
    weights = torch.softmax(logits, -1) * valid
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
    message_input = torch.cat((
        normalized_raw,
        base[..., None, :].expand(batch, variables, patches, count, self.d_model),
        raw_quality,
    ), dim=-1)
    candidate_messages = self.message_projector(message_input)
    message = (candidate_messages * weights[..., None]).sum(dim=-2)
    quality = self.quality_projector((raw_quality * weights[..., None]).sum(dim=-2))
    zeros = torch.zeros_like(base)
    message = torch.where(available[..., None], message, zeros)
    quality = torch.where(available[..., None], quality, zeros)
    return ExpertEvidence(
        message, quality, available,
        {**cache.provenance, "weights": weights.detach()},
        CandidateEvidence(candidate_messages, raw_quality, weights, valid),
    )


class HistoricalMessageEncoder(nn.Module):
    def __init__(self, patch_length: int, d_model: int, *, contextual: bool):
        super().__init__()
        if min(patch_length, d_model) < 1:
            raise ValueError("historical message geometry must be positive")
        self.patch_length = patch_length
        self.d_model = d_model
        initialize_historical_messages(self, patch_length, d_model, contextual)

    from_cache = historical_from_cache
