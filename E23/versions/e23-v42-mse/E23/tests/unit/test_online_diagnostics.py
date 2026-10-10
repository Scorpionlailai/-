from __future__ import annotations

import torch
import pytest

from malr.models.routing_v2 import V2_ROUTE_OPTIONS


def test_online_diagnostics_reports_all_routes_and_relations() -> None:
    """Dropping a relation or route metric must be visible in evaluation output."""

    from malr.evaluation.online_diagnostics import OnlineDiagnosticAccumulator

    accumulator = OnlineDiagnosticAccumulator()
    damaged = torch.tensor([[[True, True]]])
    relation_candidate = {
        name: torch.tensor([[[[True, False], [True, True]]]])
        for name in ("historical", "temporal", "variable")
    }
    relation_selected = {
        name: torch.tensor([[[[True, False], [False, True]]]])
        for name in relation_candidate
    }
    relation_predicted_gain = {
        name: torch.tensor([[[[0.2, 0.0], [0.1, 0.3]]]])
        for name in relation_candidate
    }
    relation_actual_gain = {
        name: torch.tensor([[[[0.1, 0.0], [-0.1, 0.2]]]])
        for name in relation_candidate
    }
    accumulator.update(
        relation_candidate=relation_candidate,
        relation_selected=relation_selected,
        relation_predicted_gain=relation_predicted_gain,
        relation_actual_gain=relation_actual_gain,
        temporal_local_selected=torch.tensor([[[[True, False], [False, False]]]]),
        temporal_global_selected=torch.tensor([[[[False, False], [False, True]]]]),
        route_index=torch.tensor([[[1, 7]]]),
        route_actual_gain=torch.tensor([[[0.1, 0.2]]]),
        oracle_route_index=torch.tensor([[[1, 6]]]),
        route_repair_regret=torch.tensor([[[0.0, 0.25]]]),
        damaged=damaged,
    )
    result = accumulator.compute()

    for relation in ("historical", "temporal", "variable"):
        assert f"{relation}_candidate_rate" in result
        assert f"{relation}_selected_rate" in result
        assert f"{relation}_zero_rate" in result
        assert f"{relation}_predicted_gain" in result
        assert f"{relation}_actual_gain" in result
    for route in V2_ROUTE_OPTIONS:
        assert f"route_{route}_rate" in result
        assert f"route_{route}_actual_gain" in result
    assert "temporal_local_use_rate" in result
    assert "temporal_global_use_rate" in result
    assert result["route_selection_error_rate"] == 0.5
    assert result["route_repair_regret"] == pytest.approx(0.125)
    assert result["route_repair_regret_positive_rate"] == pytest.approx(0.5)


def _update_with_route_repair_regret(route_repair_regret: torch.Tensor) -> dict[str, float]:
    from malr.evaluation.online_diagnostics import OnlineDiagnosticAccumulator

    damaged = torch.tensor([[[True, True]]])
    relation_boolean = {
        name: torch.zeros(1, 1, 2, 1, dtype=torch.bool)
        for name in ("historical", "temporal", "variable")
    }
    relation_gain = {
        name: torch.zeros(1, 1, 2, 1)
        for name in relation_boolean
    }
    accumulator = OnlineDiagnosticAccumulator()
    accumulator.update(
        relation_candidate=relation_boolean,
        relation_selected=relation_boolean,
        relation_predicted_gain=relation_gain,
        relation_actual_gain=relation_gain,
        temporal_local_selected=torch.zeros(1, 1, 2, 1, dtype=torch.bool),
        temporal_global_selected=torch.zeros(1, 1, 2, 1, dtype=torch.bool),
        route_index=torch.zeros(1, 1, 2, dtype=torch.long),
        route_actual_gain=torch.zeros(1, 1, 2),
        oracle_route_index=torch.zeros(1, 1, 2, dtype=torch.long),
        route_repair_regret=route_repair_regret,
        damaged=damaged,
    )
    return accumulator.compute()


def test_online_diagnostics_clamps_tiny_negative_route_repair_regret() -> None:
    result = _update_with_route_repair_regret(torch.tensor([[[-1.0e-7, 0.0]]]))

    assert result["route_repair_regret"] == 0.0
    assert result["route_repair_regret_positive_rate"] == 0.0


@pytest.mark.parametrize(
    ("regret", "error"),
    (
        (torch.zeros(1, 2), ValueError),
        (torch.tensor([[[-2.0e-7, 0.0]]]), ValueError),
        (torch.tensor([[[float("nan"), 0.0]]]), ValueError),
        (torch.zeros(1, 1, 2, dtype=torch.long), TypeError),
    ),
)
def test_online_diagnostics_rejects_invalid_route_repair_regret(
    regret: torch.Tensor,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        _update_with_route_repair_regret(regret)


def test_online_diagnostics_emits_zero_regret_for_no_damaged_targets() -> None:
    from malr.evaluation.online_diagnostics import OnlineDiagnosticAccumulator

    accumulator = OnlineDiagnosticAccumulator()
    result = accumulator.compute()

    assert result["route_repair_regret"] == 0.0
    assert result["route_repair_regret_positive_rate"] == 0.0
