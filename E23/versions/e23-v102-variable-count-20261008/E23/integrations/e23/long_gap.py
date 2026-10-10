"""Stage1 supervision of long hidden runs in train-standardized coordinates."""
import torch
from torch import Tensor

MIN_LENGTH = 16
LEVEL_WEIGHT = .2
TREND_WEIGHT = .2


def long_gap_moments(candidate: Tensor, target: Tensor, hidden: Tensor,
                     query_std: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return length-weighted level/trend residuals, coordinates, and runs.

    Runs are clipped to the lookback and split by any observed or invalid point.
    Trend is the fitted residual amplitude on each run's [-1,1] coordinate.
    It penalizes reconstruction error; it does not force the signal to be linear.
    """
    if (candidate.ndim != 3 or target.shape != candidate.shape
            or hidden.shape != candidate.shape or hidden.dtype != torch.bool
            or query_std.shape != (candidate.shape[0], candidate.shape[2])):
        raise ValueError('long-gap candidate/target/hidden/scale shape mismatch')
    if not torch.isfinite(query_std).all() or (query_std <= 0).any():
        raise ValueError('long-gap loss requires positive finite scales')
    B, T, C = candidate.shape
    if T < 1:
        raise ValueError('long-gap lookback must be nonempty')
    mask = hidden.transpose(1, 2).reshape(B*C, T)
    previous = torch.cat((torch.zeros_like(mask[:, :1]), mask[:, :-1]), -1)
    starts = mask & ~previous
    ids = (starts.long().cumsum(-1)-1).clamp_min(0)
    weights = mask.to(candidate.dtype)
    count = torch.zeros_like(weights).scatter_add(1, ids, weights)
    position = torch.arange(T, device=candidate.device)[None].expand_as(ids)
    run_start = torch.where(starts, position, 0).cummax(-1).values
    length = count.gather(1, ids)
    eligible_points = (mask & (length >= MIN_LENGTH)).reshape(B, C, T).transpose(1, 2)
    u = (2*(position-run_start).to(candidate.dtype)/(length-1).clamp_min(1)-1)
    u = torch.where(mask, u, 0)
    # Clear operands before subtraction: excluded NaN/Inf must not enter fitting.
    safe_candidate = torch.where(eligible_points, candidate, 0)
    safe_target = torch.where(eligible_points, target, 0)
    error = ((safe_candidate-safe_target)*query_std.detach()[:, None]).transpose(1, 2).reshape(B*C, T)
    sum_error = torch.zeros_like(error).scatter_add(1, ids, error)
    sum_trend = torch.zeros_like(error).scatter_add(1, ids, error*u)
    sum_u2 = torch.zeros_like(error).scatter_add(1, ids, u.square())
    level = sum_error/count.clamp_min(1)
    trend = sum_trend/sum_u2.clamp_min(1)
    eligible = count >= MIN_LENGTH
    eligible_count = torch.where(eligible, count, 0)
    denominator = eligible_count.sum().clamp_min(1)
    level_loss = (eligible_count*level.square()).sum()/denominator
    trend_loss = (eligible_count*trend.square()).sum()/denominator
    return level_loss, trend_loss, eligible_count.sum().detach(), eligible.sum().detach()
