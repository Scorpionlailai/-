from __future__ import annotations

import pytest
import torch

from malr.models.routing import (
    ROUTE_OPTIONS,
    hard_route_select,
    oracle_route_labels,
    route_availability_mask,
    route_cross_entropy,
    route_relation_mask,
)


def test_route_relation_mask_and_availability_mask_follow_fixed_seven_way_order() -> None:
    assert ROUTE_OPTIONS == (
        "none",
        "historical",
        "temporal",
        "variable",
        "historical_temporal",
        "historical_variable",
        "temporal_variable",
    )
    relation_mask = route_relation_mask()
    available = route_availability_mask(
        torch.tensor([[True, True, True], [False, True, True]])
    )

    assert relation_mask.shape == (7, 3)
    assert relation_mask.tolist() == [
        [False, False, False],
        [True, False, False],
        [False, True, False],
        [False, False, True],
        [True, True, False],
        [True, False, True],
        [False, True, True],
    ]
    assert available.tolist() == [
        [True, True, True, True, True, True, True],
        [True, False, True, True, False, False, True],
    ]


def test_oracle_route_applies_double_improvement_gate_and_no_positive_fallback() -> None:
    baseline = torch.ones(3)
    combination_error = torch.tensor(
        [
            [1.0, 0.9, 0.5, 0.8, 0.48, 0.7, 0.7],
            [1.0, 0.9, 0.5, 0.8, 0.47, 0.7, 0.7],
            [1.0, 1.1, 1.2, 1.1, 1.3, 1.2, 1.4],
        ]
    )
    relation_available = torch.ones(3, 3, dtype=torch.bool)

    result = oracle_route_labels(combination_error, baseline, relation_available)

    assert result.index.tolist() == [2, 4, 0]
    assert result.one_hot.shape == (3, 7)
    assert result.one_hot.argmax(dim=-1).tolist() == [2, 4, 0]
    assert result.relative_gain.tolist() == pytest.approx([0.5, 0.53, 0.0])


def test_hard_route_select_masks_unavailable_logits_and_route_loss_backpropagates() -> None:
    logits = torch.tensor([[1.0, 9.0, 2.0, 1.0, 8.0, 7.0, 6.0]], requires_grad=True)
    available = torch.tensor([[True, False, True, True, False, False, True]])

    selected = hard_route_select(logits, available)
    loss = route_cross_entropy(logits, torch.tensor([2]), available)
    loss.backward()

    assert selected.tolist() == [6]
    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert logits.grad[0, 1].item() == 0.0


def test_route_validation_rejects_unavailable_oracle_and_invalid_shapes() -> None:
    with pytest.raises(ValueError, match="combination_error"):
        oracle_route_labels(torch.ones(2, 6), torch.ones(2), torch.ones(2, 3, dtype=torch.bool))
    with pytest.raises(ValueError, match="oracle_index"):
        route_cross_entropy(
            torch.zeros(1, 7),
            torch.tensor([1]),
            torch.tensor([[True, False, True, True, False, False, True]]),
        )
