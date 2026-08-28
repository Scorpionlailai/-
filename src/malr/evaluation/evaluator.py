from __future__ import annotations

from collections.abc import Iterable

import torch

from malr.models.stage1 import Stage1Model
from malr.training.trainer import PreparedBatch

from .metrics import ForecastMetrics


@torch.no_grad()
def evaluate_model(
    model: Stage1Model,
    batches: Iterable[PreparedBatch],
    *,
    device: str | torch.device = "cpu",
) -> ForecastMetrics:
    evaluation_device = torch.device(device)
    model.to(evaluation_device).eval()
    squared_error = 0.0
    absolute_error = 0.0
    element_count = 0
    for batch in batches:
        current = batch.to(evaluation_device)
        output = model(
            current.patch_values,
            current.patch_mask,
            current.rho,
            current.state,
            current.query_mean,
            current.query_std,
            raw_context=current.raw_context,
            context_mask=current.context_mask,
            train_std=current.train_std,
        )
        error = output.forecast - current.forecast_target
        squared_error += float(error.square().sum().cpu())
        absolute_error += float(error.abs().sum().cpu())
        element_count += error.numel()
    if element_count == 0:
        raise ValueError("test batch iterator produced no batches")
    return ForecastMetrics(
        mse=squared_error / element_count,
        mae=absolute_error / element_count,
    )
