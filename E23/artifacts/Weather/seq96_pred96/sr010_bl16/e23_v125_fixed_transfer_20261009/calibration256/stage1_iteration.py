"""Stage 1 global-scale objectives, separate selections, and repair accounting."""
import math

import numpy as np
import torch

REPAIR_KEYS = ('repair96_mse', 'repair96_mae', 'repair32_mse', 'repair32_mae')


def selection_updates(metrics, best):
    values = {criterion: float(metrics['global_' + criterion]) for criterion in ('mse', 'mae')}
    if not all(math.isfinite(value) and value >= 0 for value in values.values()):
        raise ValueError('Invalid validation repair metrics')
    updates = [criterion for criterion, value in values.items() if value < best[criterion]]
    for criterion in updates:
        best[criterion] = values[criterion]
    return updates


def repair_loss(repaired, target, observed, valid, query_std, weights):
    error = (repaired - target) * query_std[:, None]
    hidden = ~observed & valid
    if not hidden.any() or not hidden[:, -32:].any():
        raise ValueError('Repair objective requires full and tail hidden coordinates')
    values = {'full_mse': error[hidden].square().mean(),
              'full_mae': error[hidden].abs().mean(),
              'tail_mse': error[:, -32:][hidden[:, -32:]].square().mean(),
              'tail_mae': error[:, -32:][hidden[:, -32:]].abs().mean()}
    if set(weights) != set(values) or any(v < 0 or not math.isfinite(v) for v in weights.values()):
        raise ValueError('Expected four nonnegative finite objective weights')
    loss = sum(values[name] * weight for name, weight in weights.items())
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite repair loss')
    return loss


def repair_totals(repaired, target, observed, valid, query_std):
    error = ((repaired - target) * query_std[:, None]).double()
    hidden = ~observed & valid
    totals = {}
    for length in (96, 32):
        part = error[:, -length:][hidden[:, -length:]]
        totals[f'repair{length}_squared'] = part.square().sum().item()
        totals[f'repair{length}_absolute'] = part.abs().sum().item()
        totals[f'hidden{length}_coordinates'] = part.numel()
    return totals


def summarize_draws(rows):
    if len(rows) < 2:
        raise ValueError('Need multiple draws to report sample mask standard deviation')
    result = {key: {'mean': float(np.mean([r[key] for r in rows])),
                    'mask_std_ddof1': float(np.std([r[key] for r in rows], ddof=1))}
              for key in REPAIR_KEYS}
    result['coordinate_weighted'] = {}
    for key in REPAIR_KEYS:
        count = 'hidden96_coordinates' if key.startswith('repair96') else 'hidden32_coordinates'
        result['coordinate_weighted'][key] = sum(r[key]*r[count] for r in rows)/sum(r[count] for r in rows)
    return result
