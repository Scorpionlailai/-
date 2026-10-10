from __future__ import annotations

import pytest
import torch

from malr.models.edge_gain import (
    EDGE_RELATION_TYPES,
    balanced_edge_gain_loss,
    counterfactual_relative_gain,
    edge_gain_admission,
    relative_edge_gain,
)


def test_relative_edge_gain_preserves_positive_zero_negative_and_detaches() -> None:
    baseline = torch.tensor([2.0, 2.0, 2.0], requires_grad=True)
    edge_error = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

    gain = relative_edge_gain(baseline, edge_error)

    torch.testing.assert_close(gain, torch.tensor([0.5, 0.0, -0.5]))
    assert not gain.requires_grad


def test_relative_edge_gain_is_finite_when_baseline_error_is_zero() -> None:
    gain = relative_edge_gain(torch.tensor([0.0]), torch.tensor([1.0]), eps=1e-3)

    assert torch.isfinite(gain).all()
    assert gain.item() == pytest.approx(-1000.0)


def test_counterfactual_gain_uses_transformed_candidate_prediction() -> None:
    """Changing a recovered candidate must change its real decoded gain label."""

    baseline = torch.tensor([[[[0.0, 0.0]]]])
    candidate = torch.tensor([[[[[1.0, 2.0], [4.0, 4.0]]]]])
    target = torch.tensor([[[[1.0, 2.0]]]])
    missing = torch.ones_like(target, dtype=torch.bool)

    gain = counterfactual_relative_gain(baseline, candidate, target, missing)

    assert gain[0, 0, 0, 0].item() > 0.0
    assert gain[0, 0, 0, 1].item() < 0.0


def test_counterfactual_gain_bounds_near_zero_baseline_outliers() -> None:
    baseline = torch.zeros(1, 1, 1, 1)
    candidate = torch.ones(1, 1, 1, 1, 1)
    target = torch.zeros(1, 1, 1, 1)
    missing = torch.ones_like(target, dtype=torch.bool)

    gain = counterfactual_relative_gain(baseline, candidate, target, missing)

    assert gain.item() == pytest.approx(-1.0)


def test_edge_gain_admission_applies_eligible_mask_and_inclusive_threshold() -> None:
    predicted = torch.tensor([[0.05, 0.049, 0.30]])
    eligible = torch.tensor([[True, True, False]])

    admitted = edge_gain_admission(predicted, eligible)

    assert admitted.tolist() == [[True, False, False]]


def test_balanced_edge_gain_loss_weights_relation_types_equally() -> None:
    predicted = torch.tensor([0.0, 1.0, 1.0], requires_grad=True)
    target = torch.zeros(3, requires_grad=True)
    eligible = torch.ones(3, dtype=torch.bool)
    relation_type = torch.tensor([0, 0, 1])

    loss = balanced_edge_gain_loss(predicted, target, eligible, relation_type)

    # Historical mean is (0 + 0.5) / 2 = 0.25, temporal mean is 0.5;
    # the relation-type means, not the raw candidate counts, are averaged.
    assert loss.item() == pytest.approx(0.375)
    loss.backward()
    assert predicted.grad is not None
    assert target.grad is None
    assert EDGE_RELATION_TYPES == ("historical", "temporal", "variable")


def test_balanced_edge_gain_loss_returns_differentiable_zero_without_candidates() -> None:
    predicted = torch.empty(0, requires_grad=True)
    target = torch.empty(0)
    eligible = torch.empty(0, dtype=torch.bool)
    relation_type = torch.empty(0, dtype=torch.long)

    loss = balanced_edge_gain_loss(predicted, target, eligible, relation_type)

    assert loss.item() == 0.0
    loss.backward()
    assert predicted.grad is not None
