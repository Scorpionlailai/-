"""Stable nucleus selection over a fixed, masked candidate axis."""

from __future__ import annotations

from math import isfinite

import torch
from torch import Tensor


def nucleus_prefix(logits: Tensor, eligible: Tensor, *, p: float) -> Tensor:
    """Return the shortest stable ranked prefix with probability mass at least p."""
    if logits.shape != eligible.shape or logits.ndim < 1 or logits.shape[-1] == 0:
        raise ValueError("nucleus logits and eligibility must have matching nonempty shape")
    if not isfinite(p) or not 0 < p <= 1:
        raise ValueError("nucleus p must lie in (0, 1]")
    eligible = eligible.to(torch.bool)
    if not torch.isfinite(logits[eligible]).all():
        raise ValueError("eligible nucleus logits must be finite")
    ranked = torch.argsort(logits.masked_fill(~eligible, -torch.inf),
                           dim=-1, descending=True, stable=True)
    ranked_eligible = eligible.gather(-1, ranked)
    ranked_logits = logits.gather(-1, ranked).masked_fill(~ranked_eligible, -torch.inf)
    empty = ~ranked_eligible.any(dim=-1, keepdim=True)
    safe_logits = torch.where(empty, torch.zeros_like(ranked_logits), ranked_logits)
    probability = torch.softmax(safe_logits, dim=-1) * ranked_eligible
    cumulative = probability.cumsum(dim=-1)
    exclusive = torch.cat((torch.zeros_like(cumulative[..., :1]), cumulative[..., :-1]), dim=-1)
    selected_ranked = ranked_eligible & (exclusive < p)
    return torch.zeros_like(eligible).scatter(-1, ranked, selected_ranked)


__all__ = ["nucleus_prefix"]
