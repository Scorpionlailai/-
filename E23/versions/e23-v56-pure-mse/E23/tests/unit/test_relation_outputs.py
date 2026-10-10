import torch
import pytest
from torch.profiler import ProfilerActivity, profile

from malr.models.relation_outputs import (
    RelationBranchOutput,
    TemporalAggregationPlan,
    stack_branch_outputs,
)


def branch_with_constant(value: float) -> RelationBranchOutput:
    scalar = torch.full((1, 1, 1), value)
    candidate = torch.full((1, 1, 1, 1), value)
    return RelationBranchOutput(
        message=scalar.unsqueeze(-1),
        predicted_gain=scalar,
        uncertainty=torch.zeros_like(scalar),
        available=torch.ones_like(scalar, dtype=torch.bool),
        candidate_messages=candidate.unsqueeze(-1),
        candidate_predicted_gain=candidate,
        candidate_uncertainty=torch.zeros_like(candidate),
        candidate_eligible=torch.ones_like(candidate, dtype=torch.bool),
        candidate_selected=torch.ones_like(candidate, dtype=torch.bool),
        diagnostics={},
    )


def test_stack_branch_outputs_preserves_htv_order() -> None:
    outputs = [branch_with_constant(value) for value in (1.0, 2.0, 3.0)]
    messages, gain, uncertainty, available = stack_branch_outputs(*outputs)
    assert messages.shape[-2] == 3
    assert messages[0, 0, 0, :, 0].tolist() == [1.0, 2.0, 3.0]
    assert gain[0, 0, 0].tolist() == [1.0, 2.0, 3.0]
    assert available.dtype == torch.bool


def test_relation_branch_requires_exact_main_ranks() -> None:
    output = branch_with_constant(1.0)
    with pytest.raises(ValueError, match="shape \[B,C,N,D\]"):
        RelationBranchOutput(
            message=torch.ones(1, 1, 1),
            predicted_gain=torch.ones(1, 1),
            uncertainty=torch.ones(1, 1),
            available=torch.ones(1, 1, dtype=torch.bool),
            candidate_messages=torch.ones(1, 1, 1, 1),
            candidate_predicted_gain=torch.ones(1, 1, 1),
            candidate_uncertainty=torch.ones(1, 1, 1),
            candidate_eligible=torch.ones(1, 1, 1, dtype=torch.bool),
            candidate_selected=torch.ones(1, 1, 1, dtype=torch.bool),
            diagnostics={},
        )
    with pytest.raises(ValueError, match="shape \[B,C,N,D\]"):
        RelationBranchOutput(
            message=torch.ones(1, 1, 1, 1, 1),
            predicted_gain=torch.ones(1, 1, 1, 1),
            uncertainty=torch.ones(1, 1, 1, 1),
            available=torch.ones(1, 1, 1, 1, dtype=torch.bool),
            candidate_messages=torch.ones(1, 1, 1, 1, 1, 1),
            candidate_predicted_gain=torch.ones(1, 1, 1, 1, 1),
            candidate_uncertainty=torch.ones(1, 1, 1, 1, 1),
            candidate_eligible=torch.ones(1, 1, 1, 1, 1, dtype=torch.bool),
            candidate_selected=torch.ones(1, 1, 1, 1, 1, dtype=torch.bool),
            diagnostics={},
        )


def test_relation_branch_validates_predicted_gain_shape() -> None:
    output = branch_with_constant(1.0)
    with pytest.raises(ValueError, match="share shape \[B,C,N\]"):
        RelationBranchOutput(
            message=output.message,
            predicted_gain=torch.ones(1, 1, 2),
            uncertainty=output.uncertainty,
            available=output.available,
            candidate_messages=output.candidate_messages,
            candidate_predicted_gain=output.candidate_predicted_gain,
            candidate_uncertainty=output.candidate_uncertainty,
            candidate_eligible=output.candidate_eligible,
            candidate_selected=output.candidate_selected,
            diagnostics={},
        )


def test_relation_branch_reports_non_tensor_fields_cleanly() -> None:
    output = branch_with_constant(1.0)
    with pytest.raises(TypeError, match="tensors"):
        RelationBranchOutput(
            message=output.message,
            predicted_gain=output.predicted_gain,
            uncertainty=output.uncertainty,
            available=output.available,
            candidate_messages=None,
            candidate_predicted_gain=output.candidate_predicted_gain,
            candidate_uncertainty=output.candidate_uncertainty,
            candidate_eligible=output.candidate_eligible,
            candidate_selected=output.candidate_selected,
            diagnostics={},
        )


def _scalar_reads(profiler: profile) -> int:
    return sum(event.count for event in profiler.key_averages() if event.key == "aten::item")


def test_relation_branch_finite_validation_reads_one_scalar() -> None:
    with profile(activities=[ProfilerActivity.CPU]) as recorded:
        branch_with_constant(1.0)

    assert _scalar_reads(recorded) == 1


def test_temporal_plan_finite_validation_reads_one_scalar() -> None:
    coefficients = torch.ones(1, 1, 1, 2)
    selected = torch.ones_like(coefficients, dtype=torch.bool)
    summary = torch.zeros(1, 1, 1)
    available = torch.ones_like(summary, dtype=torch.bool)
    with profile(activities=[ProfilerActivity.CPU]) as recorded:
        TemporalAggregationPlan(
            local_coefficients=coefficients,
            global_coefficients=coefficients,
            local_selected=selected,
            global_selected=selected,
            local_mode=torch.ones_like(summary, dtype=torch.long),
            local_available=available,
            global_available=available,
            local_gain=summary,
            global_gain=summary,
            local_uncertainty=summary,
            global_uncertainty=summary,
            available=available,
        )

    assert _scalar_reads(recorded) == 1


def test_relation_branch_rejects_nonfinite_candidate_gain() -> None:
    output = branch_with_constant(1.0)
    output.candidate_predicted_gain[0, 0, 0, 0] = float("nan")

    with pytest.raises(ValueError, match="branch floating tensors must be finite"):
        RelationBranchOutput(**vars(output))


def test_temporal_plan_rejects_nonfinite_summary() -> None:
    coefficients = torch.ones(1, 1, 1, 2)
    selected = torch.ones_like(coefficients, dtype=torch.bool)
    summary = torch.zeros(1, 1, 1)
    available = torch.ones_like(summary, dtype=torch.bool)

    with pytest.raises(ValueError, match="temporal plan summaries must be finite"):
        TemporalAggregationPlan(
            local_coefficients=coefficients,
            global_coefficients=coefficients,
            local_selected=selected,
            global_selected=selected,
            local_mode=torch.ones_like(summary, dtype=torch.long),
            local_available=available,
            global_available=available,
            local_gain=summary,
            global_gain=summary,
            local_uncertainty=summary,
            global_uncertainty=torch.full_like(summary, float("inf")),
            available=available,
        )
