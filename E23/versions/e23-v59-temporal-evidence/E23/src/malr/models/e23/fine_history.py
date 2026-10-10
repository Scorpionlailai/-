"""Retrieve 16-point parents, then project cropped candidates at output resolution."""
from dataclasses import replace

import torch

from .historical_candidates import HistoricalCandidateCache, HistoricalCandidateProvider
from .historical_messages import HistoricalMessageEncoder


class FineHistoricalExpert(HistoricalCandidateProvider):
    def __init__(self, output_length, width, variable_count, *, topk=4, threshold=.7):
        if output_length not in (4, 8, 16):
            raise ValueError('fine history requires 4/8/16 point output')
        super().__init__(16, width, width, variable_count, topk=topk,
                         similarity_threshold=threshold)
        self.output_length = output_length
        self.query_length = self.query_stride = 16
        self.messages = HistoricalMessageEncoder(output_length, width, contextual=True)

    def parent_inputs(self, inputs):
        b, c, n, p = inputs.patch_values.shape
        ratio = 16 // p
        if p != self.output_length or n * p != 96 or n % ratio:
            raise ValueError('fine historical parent geometry mismatch')
        parents = n // ratio
        values = inputs.patch_values.reshape(b, c, parents, 16)
        mask = inputs.patch_mask.reshape(b, c, parents, 16).bool()
        validity = inputs.patch_validity.reshape(b, c, parents, 16).bool()
        observed = mask & validity
        values = values.masked_fill(~observed, 0)
        fraction = observed.sum(-1) / validity.sum(-1).clamp_min(1)
        base = inputs.base_latents.reshape(b, c, parents, ratio, -1).mean(-2)
        starts = inputs.patch_starts[..., ::ratio]
        return replace(inputs, base_latents=base, patch_values=values,
                       patch_mask=observed, patch_validity=validity, rho=fraction,
                       damaged=fraction.lt(1), patch_starts=starts)

    def prepare_candidates(self, inputs):
        parent = self.parent_inputs(inputs)
        cache = super().prepare_candidates(parent)
        b, c, n, k, _ = cache.normalized_raw.shape
        p = self.output_length
        ratio = 16 // p

        def crop(raw):
            return raw.reshape(b, c, n, k, ratio, p).permute(0, 1, 2, 4, 3, 5).reshape(b, c, n*ratio, k, p)

        valid = cache.valid.repeat_interleave(ratio, 2) & inputs.damaged[..., None]
        quality = cache.raw_quality.repeat_interleave(ratio, 2)
        quality = quality.masked_fill(~valid[..., None], 0)
        provenance = {name: (value.repeat_interleave(ratio, 2) if torch.is_tensor(value) else value)
                      for name, value in cache.provenance.items()}
        offsets = (torch.arange(n*ratio, device=valid.device) % ratio * p)[None, None, :, None]
        provenance['parent_candidate_starts'] = provenance['candidate_starts']
        provenance['candidate_starts'] = provenance['candidate_starts'] + offsets
        provenance['parent_query_starts'] = parent.patch_starts.repeat_interleave(ratio, -1)
        provenance['candidate_raw'] = crop(cache.provenance['candidate_raw'])
        provenance['candidate_valid'] = valid.detach()
        provenance['candidate_count'] = valid.sum(-1).detach()
        provenance['historical_query_length'] = 16
        return HistoricalCandidateCache(crop(cache.normalized_raw), quality,
            cache.base_logits.repeat_interleave(ratio, 2), valid, valid.any(-1),
            provenance, cache.empty)

    def from_cache(self, inputs, cache, *, round_embedding=None):
        return self.messages.from_cache(inputs, cache, round_embedding=round_embedding)

    def forward(self, inputs):
        return self.from_cache(inputs, self.prepare_candidates(inputs))
