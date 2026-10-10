"""Historical reconstruction supervision for recent level and trend amplitude."""
import torch


def recent_state_loss(candidate, target, hidden, query_std, *, level_weight=.2,trend_weight=.2):
    if candidate.shape[1] != 96 or candidate.shape != target.shape or hidden.shape != candidate.shape:
        raise ValueError("recent state loss requires aligned 96-point history")
    # Visible/invalid coordinates contribute zero and never carry a target gradient.
    error = (torch.where(hidden, candidate, 0)-torch.where(hidden, target, 0))*query_std.detach()[:, None]
    tail = error[:, -32:]
    x = torch.linspace(-1, 1, 32, device=tail.device, dtype=tail.dtype)
    level = tail.mean(1).square().mean()
    trend = ((tail*x[None, :, None]).sum(1)/x.square().sum()).square().mean()
    loss = level_weight*level+trend_weight*trend
    return loss, dict(recent_level=level, recent_trend_amplitude=trend, recent_structure_weighted=loss)
