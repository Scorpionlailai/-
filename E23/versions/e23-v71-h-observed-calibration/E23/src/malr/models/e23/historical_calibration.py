"""Observed-only ridge level alignment with exact leave-one-out reliability."""
import torch


def calibrate_historical_candidates(inputs, normalized_raw, valid):
    observed = inputs.patch_mask.bool() & inputs.patch_validity.bool()
    visible = observed[..., None, :] & valid[..., None]
    values = inputs.patch_values[..., None, :].masked_fill(~visible, 0)
    raw_visible = normalized_raw.masked_fill(~visible, 0)
    residual = values - raw_visible
    count = visible.sum(-1).to(normalized_raw.dtype)
    total = residual.sum(-1)
    # Four virtual observations with zero offset shrink sparse fits.
    limit = (2 * torch.sqrt(inputs.query_std.square() + .1**2)
             / inputs.query_std)[:, :, None, None]

    def bounded(value):
        return torch.maximum(torch.minimum(value, limit), -limit)

    offset = bounded(total / (count + 4))
    offset = torch.where(valid & count.gt(0), offset, 0)
    adjusted = normalized_raw + offset[..., None]
    # Fit without each evaluated visible point, including the same bound.
    loo_offset = (total[..., None] - residual) / (count[..., None] + 3)
    loo_offset = torch.maximum(torch.minimum(loo_offset, limit[..., None]), -limit[..., None])
    error = (residual - loo_offset).abs().masked_fill(~visible, 0).sum(-1) / count.clamp_min(1)
    error = torch.where(valid & count.gt(0), error, 0)
    return adjusted, error, offset
