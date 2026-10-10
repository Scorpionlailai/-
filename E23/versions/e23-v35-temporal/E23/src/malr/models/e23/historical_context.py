"""Original-observation context for historical retrieval; no generated values."""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from malr.models.explicit_repair import _aggregate_allow_empty


def context_windows(values: Tensor, mask: Tensor, *, patch_len: int, stride: int,
                    patches: int) -> tuple[Tensor, Tensor]:
    """One patch of left/right context, zero padded outside the lookback.

    Outputs [B,C,N,3P]. Missing and padded coordinates are always zero.
    """
    if values.ndim != 3 or mask.shape != values.shape:
        raise ValueError('historical context expects matching [B,T,C] inputs')
    positions = (torch.arange(patches, device=values.device)[:, None] * stride
                 + torch.arange(-patch_len, 2 * patch_len, device=values.device)[None])
    valid = (positions >= 0) & (positions < values.shape[1])
    indices = positions.clamp(0, values.shape[1] - 1)
    visible = mask.to(torch.bool).permute(0, 2, 1)[:, :, indices] & valid
    result = values.permute(0, 2, 1)[:, :, indices].masked_fill(~visible, 0)
    return result, visible


def input_context(inputs, *, patch_len: int, stride: int) -> tuple[Tensor, Tensor]:
    """Reconstruct the original observed sequence from overlapping patches."""
    n = inputs.patch_values.shape[2]
    # The last patch contains stride replicated padding, as in E23's geometry.
    window_len = (n - 1) * stride + patch_len - stride
    observed = inputs.patch_mask.to(torch.bool) & inputs.patch_validity.to(torch.bool)
    values, coverage = _aggregate_allow_empty(
        inputs.patch_values, observed, window_len=window_len,
        patch_len=patch_len, stride=stride,
    )
    visible = coverage.gt(0)
    global_values = values * inputs.query_std[:, None] + inputs.query_mean[:, None]
    return context_windows(global_values, visible, patch_len=patch_len, stride=stride, patches=n)


def member_key_inputs(members, *, device, contextual: bool) -> tuple[Tensor, Tensor]:
    if contextual:
        if any(m.context_values is None for m in members):
            raise ValueError('contextual historical bank member lacks original context')
        values = np.stack([m.context_values for m in members])
        mask = np.stack([m.context_mask for m in members])
    else:
        values = np.stack([m.raw_patch for m in members])
        mask = np.ones_like(values, dtype=np.bool_)
    return (torch.as_tensor(values, dtype=torch.float32, device=device),
            torch.as_tensor(mask, dtype=torch.bool, device=device))


def retrieval_training_inputs(batch, model):
    """Global-scale masked and clean keys; clean values are loss targets only."""
    p, stride, n = model.patch_length, model.stride, model.patch_count
    observed = batch.observed_mask.to(torch.bool)
    query = (batch.normalized_observed * batch.query_std[:, None]
             + batch.query_mean[:, None]).masked_fill(~observed, 0)
    target = batch.normalized_target * batch.query_std[:, None] + batch.query_mean[:, None]
    if model.historical.query_mode == 'global_context_v1':
        values, mask = context_windows(query, observed, patch_len=p, stride=stride, patches=n)
        clean, validity = context_windows(target, torch.ones_like(observed),
                                         patch_len=p, stride=stride, patches=n)
    else:
        from malr.data.patching import patchify_with_state, patchify_preserving_values
        q = patchify_with_state(query, observed, p, stride)
        c = patchify_preserving_values(target, torch.ones_like(observed), p, stride)
        values, mask, clean, validity = q.values, q.effective_mask, c.values, c.validity
    from malr.data.patching import patchify_with_state
    damaged = patchify_with_state(query, observed, p, stride).rho.lt(1)
    eligible = damaged & mask.any(-1)
    return values[eligible], mask[eligible], clean[eligible], validity[eligible]
