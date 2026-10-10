"""Historical evidence from paired, published training prototypes."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from malr.prototypes.online_bank import OnlinePrototypeSnapshot
from malr.prototypes.retrieval import MaskAwareRetrievalEncoder, RetrievalEncoderEMA
from malr.prototypes.tensor_index import TensorPrototypeIndex, TensorRetrieval

from .contracts import CandidateEvidence, ExpertEvidence, ExpertInputs
from .nucleus import nucleus_prefix


@dataclass(frozen=True)
class HistoricalCandidateCache:
    normalized_raw: Tensor
    raw_quality: Tensor
    base_logits: Tensor
    valid: Tensor
    available: Tensor
    provenance: Mapping[str, Any]
    empty: bool = False


class HistoricalExpert(nn.Module):
    def __init__(
        self, patch_length: int, d_model: int, retrieval_dim: int, variable_count: int,
        *, topk: int = 4, similarity_threshold: float = 0.7,
        top_p: float | None = None, candidate_cap: int = 8,
        retrieval_strategy: str = "one_group",
        contextual: bool = False,
    ) -> None:
        super().__init__()
        if min(patch_length, d_model, retrieval_dim, variable_count, topk) < 1:
            raise ValueError("historical geometry and topk must be positive")
        if not -1.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity threshold must lie in [-1, 1]")
        if candidate_cap < 1:
            raise ValueError("historical candidate cap must be positive")
        if top_p is not None and not 0 < top_p <= 1:
            raise ValueError("historical top-p must lie in (0, 1]")
        if retrieval_strategy not in ("one_group", "two_group_visible_rerank_v1"):
            raise ValueError("unsupported historical retrieval strategy")
        if retrieval_strategy != "one_group" and top_p is not None:
            raise ValueError("two-group historical retrieval requires top-k")
        self.patch_length = patch_length
        self.d_model = d_model
        self.variable_count = variable_count
        self.topk = topk if top_p is None else candidate_cap
        self.top_p = top_p
        self.retrieval_strategy = retrieval_strategy
        self.similarity_threshold = similarity_threshold
        self.key_encoder = MaskAwareRetrievalEncoder(patch_length, retrieval_dim)
        self.retrieval = RetrievalEncoderEMA(self.key_encoder)
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
        self.index: TensorPrototypeIndex | None = None

    def _apply(self, fn):
        super()._apply(fn)
        if hasattr(self, "retrieval"):
            self.retrieval.ema_encoder._apply(fn)
            self.retrieval.frozen_encoder._apply(fn)
        # A published index must be rebuilt on the new device by the trainer.
        if hasattr(self, "index") and self.index is not None:
            if self.index.device != next(self.parameters()).device:
                self.index = None
        return self

    def refresh_snapshot(self, snapshot: OnlinePrototypeSnapshot) -> None:
        device = next(self.parameters()).device
        index = TensorPrototypeIndex.from_snapshot(snapshot, device=device)
        if index.variable_count != self.variable_count:
            raise ValueError("snapshot variable count does not match H expert")
        if index.patch_length not in (0, self.patch_length):
            raise ValueError("snapshot patch length does not match H expert")
        if index.key_dim not in (0, self.key_encoder.retrieval_dim):
            raise ValueError("snapshot key width does not match H encoder")
        self.index = index

    def _empty(self, base: Tensor, *, candidate_count: Tensor | None = None) -> ExpertEvidence:
        zeros = torch.zeros_like(base)
        b, c, n, _ = base.shape
        if candidate_count is None:
            candidate_count = torch.zeros((b, c, n), dtype=torch.long, device=base.device)
        return ExpertEvidence(
            zeros, zeros, torch.zeros((b, c, n), dtype=torch.bool, device=base.device),
            {"candidate_count": candidate_count,
             "candidate_valid": torch.zeros((b, c, n, self.topk), dtype=torch.bool, device=base.device),
             "candidate_starts": torch.zeros((b, c, n, self.topk), dtype=torch.long, device=base.device),
             "candidate_raw": torch.zeros((b, c, n, self.topk, self.patch_length), dtype=base.dtype, device=base.device),
             "weights": torch.zeros((b, c, n, self.topk), dtype=base.dtype, device=base.device),
             "visible_error": torch.zeros((b, c, n, self.topk), dtype=base.dtype, device=base.device)},
            CandidateEvidence(
                base.new_zeros(b, c, n, self.topk, self.d_model),
                base.new_zeros(b, c, n, self.topk, 5),
                base.new_zeros(b, c, n, self.topk),
                torch.zeros((b, c, n, self.topk), dtype=torch.bool, device=base.device),
            ),
        )

    def _empty_cache(self, base: Tensor) -> HistoricalCandidateCache:
        b, c, n, _ = base.shape
        empty = self._empty(base)
        return HistoricalCandidateCache(
            base.new_zeros(b, c, n, self.topk, self.patch_length),
            base.new_zeros(b, c, n, self.topk, 5),
            base.new_zeros(b, c, n, self.topk),
            empty.candidates.valid, empty.available, empty.provenance, True,
        )

    def prepare_candidates(self, inputs: ExpertInputs) -> HistoricalCandidateCache:
        base = inputs.base_latents
        values = inputs.patch_values
        mask = inputs.patch_mask.to(torch.bool) & inputs.patch_validity.to(torch.bool)
        if values.ndim != 4:
            raise ValueError("historical patch values must have shape [B,C,N,P]")
        batch, variables, patches, length = values.shape
        if variables != self.variable_count or length != self.patch_length:
            raise ValueError("historical geometry does not match input")
        if base.shape != (batch, variables, patches, self.d_model):
            raise ValueError("historical base latent shape mismatch")
        if inputs.query_mean.shape != (batch, variables) or inputs.query_std.shape != (batch, variables):
            raise ValueError("historical query statistics shape mismatch")
        if not torch.isfinite(inputs.query_mean).all() or not torch.isfinite(inputs.query_std).all() or (inputs.query_std <= 0).any():
            raise ValueError("historical query statistics must be finite with positive std")
        if self.index is None or self.index.member_keys.numel() == 0:
            return self._empty_cache(base)
        damaged = inputs.damaged.to(torch.bool) & mask.any(dim=-1)
        if not damaged.any():
            return self._empty_cache(base)
        flat_values = values.reshape(-1, length)
        flat_mask = mask.reshape(-1, length)
        query_keys = self.retrieval.encode_frozen(flat_values, flat_mask).reshape(
            batch, variables, patches, -1,
        )
        if self.retrieval_strategy == "two_group_visible_rerank_v1":
            shortlist = self.index.query_two_groups(
                query_keys, inputs.patch_starts, damaged, member_topk=self.topk,
            )
            shortlisted_valid = shortlist.valid & (
                shortlist.key_similarity >= self.similarity_threshold
            )
            mean = inputs.query_mean[:, :, None, None, None]
            std = inputs.query_std[:, :, None, None, None]
            shortlist_normalized = (shortlist.raw_patches - mean) / std
            visible = mask[..., None, :]
            difference = (shortlist_normalized.masked_fill(~visible, 0)
                          - values[..., None, :].masked_fill(~visible, 0)).abs()
            shortlist_error = difference.sum(-1) / visible.sum(-1).clamp_min(1)
            rerank = (shortlist.key_similarity - shortlist_error).masked_fill(
                ~shortlisted_valid, -torch.inf,
            )
            order = torch.argsort(rerank, dim=-1, descending=True, stable=True)[..., :self.topk]
            def gather(value: Tensor) -> Tensor:
                if value.ndim == order.ndim + 1:
                    return value.gather(-2, order[..., None].expand(*order.shape, value.shape[-1]))
                return value.gather(-1, order)
            candidates = TensorRetrieval(
                gather(shortlist.raw_patches), gather(shortlist.key_similarity),
                gather(shortlist.starts), gather(shortlist.group_support),
                gather(shortlisted_valid), gather(shortlist.group_ids),
            )
        else:
            candidates = self.index.query(query_keys, inputs.patch_starts, damaged, topk=self.topk)
        valid = candidates.valid & (candidates.key_similarity >= self.similarity_threshold)
        available = valid.any(dim=-1)
        mean = inputs.query_mean[:, :, None, None, None]
        std = inputs.query_std[:, :, None, None, None]
        normalized_raw = (candidates.raw_patches - mean) / std
        visible = mask[..., None, :]
        if self.retrieval_strategy == "two_group_visible_rerank_v1":
            visible_error = (
                normalized_raw.masked_fill(~visible, 0)
                - values[..., None, :].masked_fill(~visible, 0)
            ).abs().sum(-1) / visible.sum(-1).clamp_min(1)
        else:
            visible_error = (
                (normalized_raw - values[..., None, :]).abs() * visible
            ).sum(-1) / visible.sum(-1).clamp_min(1)
        visible_error = visible_error.masked_fill(~valid, 0)
        if self.top_p is not None:
            valid = nucleus_prefix(candidates.key_similarity - visible_error,
                                   valid, p=self.top_p)
            available = valid.any(dim=-1)
        scores = candidates.key_similarity.masked_fill(~valid, -1e9)
        weights = torch.softmax(scores - visible_error, dim=-1) * valid
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        sorted_similarity = candidates.key_similarity.masked_fill(~valid, -1e9)
        top_two = torch.topk(sorted_similarity, k=min(2, self.topk), dim=-1).values
        margin = (
            (top_two[..., 0] - top_two[..., 1]).clamp_min(0)
            if self.topk >= 2 else torch.zeros_like(top_two[..., 0])
        )
        margin = torch.where(valid.sum(-1) >= 2, margin, torch.zeros_like(margin))
        fraction = mask.float().mean(-1)
        support = torch.log1p(candidates.group_support.to(values.dtype))
        raw_quality = torch.stack((
            candidates.key_similarity,
            -visible_error,
            margin[..., None].expand_as(weights),
            fraction[..., None].expand_as(weights),
            support,
        ), dim=-1)
        raw_quality = torch.where(valid[..., None], raw_quality, torch.zeros_like(raw_quality))
        return HistoricalCandidateCache(
            normalized_raw, raw_quality, scores - visible_error, valid, available,
            {"candidate_count": valid.sum(dim=-1).detach(),
             "candidate_valid": valid.detach(),
             "candidate_starts": candidates.starts.detach(),
             "candidate_raw": candidates.raw_patches.detach(),
             "candidate_group_ids": candidates.group_ids.detach(),
             "visible_error": visible_error.detach(),
             "key_similarity": candidates.key_similarity.detach()},
        )

    def from_cache(
        self, inputs: ExpertInputs, cache: HistoricalCandidateCache,
        *, round_embedding: Tensor | None = None,
    ) -> ExpertEvidence:
        base = inputs.base_latents
        batch, variables, patches, _ = base.shape
        if cache.normalized_raw.shape != (batch, variables, patches, self.topk, self.patch_length):
            raise ValueError("historical cache shape mismatch")
        if cache.empty:
            return self._empty(base)
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
            base[..., None, :].expand(batch, variables, patches, self.topk, self.d_model),
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

    def forward(self, inputs: ExpertInputs) -> ExpertEvidence:
        return self.from_cache(inputs, self.prepare_candidates(inputs))


__all__ = ["HistoricalExpert", "HistoricalCandidateCache"]
