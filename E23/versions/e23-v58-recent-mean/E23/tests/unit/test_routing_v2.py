from __future__ import annotations

import pytest
import torch

from malr.models.routing_v2 import (
    RouteStabilityTracker,
    V2_ROUTE_OPTIONS,
    direct_single_route_select,
    hard_route_select,
    oracle_route_labels,
    route_availability_mask,
    route_relation_mask,
)


def test_route_mask_follows_fixed_eight_way_order() -> None:
    assert V2_ROUTE_OPTIONS[-1] == "historical_temporal_variable"
    assert route_relation_mask().tolist()[-1] == [True, True, True]
    assert route_relation_mask().shape == (8, 3)


def test_availability_requires_each_relation_in_a_route() -> None:
    available = route_availability_mask(
        torch.tensor([[True, False, True], [True, True, True]])
    )

    assert available.tolist() == [
        [True, True, False, True, False, True, False, False],
        [True, True, True, True, True, True, True, True],
    ]


def test_triple_must_beat_best_double_by_margin() -> None:
    baseline = torch.ones(2)
    errors = torch.tensor(
        [
            [1.0, 0.89, 0.95, 0.96, 0.78, 0.84, 0.86, 0.69],
            [1.0, 0.89, 0.95, 0.96, 0.78, 0.84, 0.86, 0.72],
        ]
    )

    result = oracle_route_labels(
        errors,
        baseline,
        torch.ones(2, 3, dtype=torch.bool),
        complexity_margin=torch.tensor([0.10, 0.10, 0.10]),
    )

    assert result.index.tolist() == [7, 4]


def test_double_compares_only_available_strict_subsets() -> None:
    errors = torch.tensor([[1.0, 0.89, 0.20, 0.95, 0.95, 0.80, 0.95, 0.95]])

    result = oracle_route_labels(
        errors,
        torch.ones(1),
        torch.tensor([[True, False, True]]),
        complexity_margin=torch.tensor([0.10, 0.10, 0.10]),
    )

    assert result.index.tolist() == [5]


def test_oracle_breaks_admitted_error_ties_in_favor_of_simpler_route() -> None:
    result = oracle_route_labels(
        torch.tensor([[1.0, 0.80, 0.90, 0.90, 0.80, 0.90, 0.90, 0.80]]),
        torch.ones(1),
        torch.ones(1, 3, dtype=torch.bool),
        complexity_margin=torch.zeros(3),
    )

    assert result.index.tolist() == [1]


def test_tracker_uses_base_margin_until_the_minimum_target_count() -> None:
    tracker = RouteStabilityTracker(min_targets=128)
    errors = torch.ones(127, 8)
    errors[:, 7] = 0.5
    available = torch.ones(127, 8, dtype=torch.bool)

    tracker.update(errors, available)

    assert tracker.count.tolist() == [381, 381, 127]
    assert tracker.margins().tolist() == pytest.approx([0.10, 0.10, 0.10])


def test_tracker_enables_the_larger_ema_standard_deviation_at_count_128() -> None:
    tracker = RouteStabilityTracker(min_targets=128)
    errors = torch.ones(127, 8)
    errors[:, 7] = 0.5
    available = torch.ones(127, 8, dtype=torch.bool)
    tracker.update(errors, available)

    next_errors = torch.ones(1, 8)
    next_errors[:, 7] = 0.9
    tracker.update(next_errors, torch.ones(1, 8, dtype=torch.bool))

    assert tracker.count[2].item() == 128
    assert tracker.margins()[2].item() == pytest.approx(0.12)


def test_tracker_is_read_only_in_evaluation_mode_and_round_trips_state() -> None:
    tracker = RouteStabilityTracker()
    tracker.update(torch.tensor([[1.0] * 7 + [0.5]]), torch.ones(1, 8, dtype=torch.bool))
    restored = RouteStabilityTracker()
    restored.load_state_dict(tracker.state_dict())
    before = {name: value.clone() for name, value in restored.state_dict().items()}

    restored.eval()
    restored.update(torch.tensor([[1.0] * 7 + [0.1]]), torch.ones(1, 8, dtype=torch.bool))

    for name, value in before.items():
        assert torch.equal(restored.state_dict()[name], value)


def test_hard_selection_uses_eight_way_availability() -> None:
    selected = hard_route_select(
        torch.tensor([[1.0, 9.0, 2.0, 1.0, 8.0, 7.0, 6.0, 10.0]]),
        torch.tensor([[True, False, True, True, False, False, True, False]]),
    )

    assert selected.tolist() == [6]


def test_direct_single_route_select_uses_largest_positive_branch_gain() -> None:
    gains = torch.tensor(
        [
            [[0.20, 0.60, 0.40], [0.20, 0.10, 0.00], [-0.10, -0.20, -0.30]],
        ]
    )
    available = torch.tensor(
        [
            [[True, True, True], [True, False, True], [False, False, False]],
        ]
    )

    selected = direct_single_route_select(gains, available)

    assert selected.tolist() == [[2, 1, 0]]


def test_oracle_can_be_restricted_to_none_and_single_routes() -> None:
    errors = torch.tensor(
        [
            [1.0, 0.95, 0.80, 0.90, 0.10, 0.10, 0.10, 0.10],
            [1.0, 1.10, 1.20, 1.30, 0.01, 0.01, 0.01, 0.01],
        ]
    )

    result = oracle_route_labels(
        errors,
        torch.ones(2),
        torch.ones(2, 3, dtype=torch.bool),
        complexity_margin=torch.zeros(3),
        allow_combinations=False,
    )

    assert result.index.tolist() == [2, 0]
    assert not result.available[:, 4:].any()


def test_tracker_clips_near_zero_parent_relative_gains() -> None:
    tracker = RouteStabilityTracker(min_targets=1)
    errors = torch.tensor([[1.0e-8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
    available = torch.ones(1, 8, dtype=torch.bool)

    tracker.update(errors, available)

    assert torch.isfinite(tracker.margins()).all()
    assert torch.all(tracker.margins() <= 1.0)
