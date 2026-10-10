from __future__ import annotations

import pytest
import torch

from malr.training.unified_objective import (
    unified_stage1_loss,
    residual_injection_weight,
)


def test_residual_injection_weight_is_zero_to_one_linear_warmup() -> None:
    assert residual_injection_weight(0) == pytest.approx(0.0)
    assert residual_injection_weight(5) == pytest.approx(0.5)
    assert residual_injection_weight(10) == pytest.approx(1.0)
    assert residual_injection_weight(20) == pytest.approx(1.0)


def test_unified_stage1_loss_uses_fixed_weights_and_backpropagates() -> None:
    forecast = torch.tensor(1.0, requires_grad=True)
    imputation_huber = torch.tensor(2.0, requires_grad=True)
    edge_gain = torch.tensor(3.0, requires_grad=True)
    route = torch.tensor(4.0, requires_grad=True)
    memory = torch.tensor(5.0, requires_grad=True)

    result = unified_stage1_loss(
        forecast,
        imputation_huber,
        edge_gain,
        route,
        memory,
    )

    assert result.total.item() == pytest.approx(1.6)
    assert result.weighted_imputation_huber.item() == pytest.approx(0.2)
    assert result.weighted_edge_gain.item() == pytest.approx(0.15)
    assert result.weighted_route.item() == pytest.approx(0.2)
    assert result.weighted_memory.item() == pytest.approx(0.05)
    result.total.backward()
    assert all(value.grad is not None for value in (forecast, imputation_huber, edge_gain, route, memory))


def test_unified_stage1_loss_rejects_negative_weights_and_invalid_epoch() -> None:
    with pytest.raises(ValueError, match="loss weights"):
        unified_stage1_loss(
            torch.tensor(1.0),
            torch.tensor(1.0),
            torch.tensor(1.0),
            torch.tensor(1.0),
            torch.tensor(1.0),
            lambda_route=-0.1,
        )
    with pytest.raises(ValueError, match="epoch_index"):
        residual_injection_weight(-1)
    with pytest.raises(ValueError, match="warmup_epochs"):
        residual_injection_weight(0, warmup_epochs=0)
