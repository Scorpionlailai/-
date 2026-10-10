from __future__ import annotations

import torch

from malr.models.dynamic_context_model import DynamicContextOnlineModel
from tests.integration.test_online_stage1 import _fixture


def _dynamic_fixture() -> tuple[DynamicContextOnlineModel, dict[str, torch.Tensor]]:
    legacy, data = _fixture()
    model = DynamicContextOnlineModel(
        legacy.core,
        memory=legacy.memory,
        retrieval=legacy.retrieval,
        variable_count=2,
        temporal_distance_boundaries=(2, 4, 8, 16),
        historical_topk=4,
        temporal_topk=6,
        variable_topk=4,
        candidate_gain_threshold=0.0,
        route_improvement_threshold=0.10,
        route_stability_min_targets=128,
        bootstrap_epochs=20,
        joint_end_epoch=80,
        temporal_residual_limit=2.0,
    )
    data = dict(data)
    data["patch_validity"] = torch.ones_like(data["values"], dtype=torch.bool)
    return model, data


def test_dynamic_context_has_independent_full_lookback_t_and_hv_only_base() -> None:
    model, data = _dynamic_fixture()
    output = model(
        data["values"],
        data["mask"],
        data["rho"],
        data["state"],
        data["query_mean"],
        data["query_std"],
        patch_starts=data["starts"],
        patch_validity=data["patch_validity"],
    )

    assert output.forecast is None
    assert output.explicit.normalized_sequence.shape == (1, 96, 2)
    assert output.hv_route_weights.shape[-1] == 4
    assert output.hv_route_weights[..., 2].eq(0).all()
    assert output.temporal.gate_weights.shape[-1] == 3
    assert not hasattr(model, "route_router")


def test_dynamic_context_stage1_has_only_repair_gain_and_memory_loss() -> None:
    model, data = _dynamic_fixture()
    result = model.training_objective(
        data["values"],
        data["mask"],
        data["rho"],
        data["state"],
        data["query_mean"],
        data["query_std"],
        data["forecast_target"],
        data["clean"],
        patch_starts=data["starts"],
        patch_validity=data["patch_validity"],
        objective="repair",
    )
    assert result.output.forecast is None
    assert result.loss.total.requires_grad
    assert result.loss.forecast.item() == 0.0
    model.zero_grad(set_to_none=True)
    result.loss.total.backward()
    assert model.core.W_P.weight.grad is not None
    assert model.dynamic_temporal.residual_head[-1].weight.grad is not None
    assert model.core.head.linear.weight.grad is None
    assert model.temporal_operator.candidate_projector[0].weight.grad is None


def test_dynamic_context_stage2_uses_only_forecast_loss_after_teacher_handoff() -> None:
    model, data = _dynamic_fixture()
    teacher_sha = model.install_frozen_route_teacher()
    model.install_forecast_predictor()
    model.configure_training_stage("forecast")
    result = model.training_objective(
        data["values"],
        data["mask"],
        data["rho"],
        data["state"],
        data["query_mean"],
        data["query_std"],
        data["forecast_target"],
        None,
        patch_starts=data["starts"],
        patch_validity=data["patch_validity"],
        objective="forecast",
    )
    assert result.output.forecast is not None
    torch.testing.assert_close(result.loss.total, result.loss.forecast)
    assert result.loss.repair.item() == 0.0
    assert result.loss.gain.item() == 0.0
    assert result.loss.memory.item() == 0.0
    model.zero_grad(set_to_none=True)
    result.loss.total.backward()
    assert model.forecast_core is not None
    assert model.forecast_core.head.linear.weight.grad is not None
    assert model.forecast_mask_projection is not None
    assert model.forecast_mask_projection.weight.grad is not None
    assert model.route_teacher is not None
    assert model.route_teacher.state_sha256() == teacher_sha


def test_dynamic_context_teacher_handoff_preserves_hv_route_plan() -> None:
    model, data = _dynamic_fixture()
    ordinary = model(
        data["values"], data["mask"], data["rho"], data["state"],
        data["query_mean"], data["query_std"], patch_starts=data["starts"],
        patch_validity=data["patch_validity"],
    )
    model.install_frozen_route_teacher()
    split = model(
        data["values"], data["mask"], data["rho"], data["state"],
        data["query_mean"], data["query_std"], patch_starts=data["starts"],
        patch_validity=data["patch_validity"],
    )
    for actual, expected in (
        (split.historical.message, ordinary.historical.message),
        (split.variable.message, ordinary.variable.message),
        (split.route_logits, ordinary.route_logits),
        (split.hv_route_weights, ordinary.hv_route_weights),
    ):
        torch.testing.assert_close(actual, expected)


def test_dynamic_context_never_calls_legacy_temporal_operator() -> None:
    model, data = _dynamic_fixture()
    with torch.no_grad():
        for parameter in model.temporal_operator.parameters():
            parameter.fill_(float("nan"))
    output = model(
        data["values"], data["mask"], data["rho"], data["state"],
        data["query_mean"], data["query_std"], patch_starts=data["starts"],
        patch_validity=data["patch_validity"],
    )
    assert torch.isfinite(output.historical.message).all()
    assert torch.isfinite(output.variable.message).all()
    assert torch.isfinite(output.explicit.normalized_sequence).all()
