from __future__ import annotations

from copy import deepcopy

import torch
import pytest

from malr.data.patching import patchify_with_state
from malr.evaluation.evaluator import evaluate_online_model
from malr.models.online_stage1_v2 import RelationV2OnlineStage1Model
from malr.models.routing_v2 import direct_single_route_select
from malr.training.trainer import PreparedBatch
from tests.integration.test_online_stage1 import _fixture


def _v2_fixture() -> tuple[RelationV2OnlineStage1Model, dict[str, torch.Tensor]]:
    legacy, data = _fixture()
    model = RelationV2OnlineStage1Model(
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
    )
    return model, data


def _v2_fully_observed_fixture() -> tuple[RelationV2OnlineStage1Model, dict[str, torch.Tensor]]:
    model, data = _v2_fixture()
    data = dict(data)
    data["values"] = data["clean"]
    data["mask"] = torch.ones_like(data["clean"], dtype=torch.bool)
    data["rho"] = torch.ones_like(data["rho"])
    data["state"] = torch.zeros_like(data["state"])
    return model, data


def model_forward(
    model: RelationV2OnlineStage1Model,
    data: dict[str, torch.Tensor],
    *,
    epoch_index: int,
):
    return model(
        data["values"],
        data["mask"],
        data["rho"],
        data["state"],
        data["query_mean"],
        data["query_std"],
        patch_starts=data["starts"],
        epoch_index=epoch_index,
    )


def _populate_historical_teacher_fixture(
    model: RelationV2OnlineStage1Model,
    data: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Publish causal prototypes and make every similarity admissible."""

    model.begin_epoch(1)
    model.stage_observed_batch(
        data["clean"], data["mask"], data["rho"], data["state"], data["starts"]
    )
    model.end_epoch(1)
    model.historical_similarity_threshold = 0.0
    shifted = dict(data)
    shifted["starts"] = data["starts"] + 128
    return shifted


def _assert_nonzero_finite_gradient(module: torch.nn.Module) -> None:
    gradients = [
        parameter.grad
        for parameter in module.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient) > 0 for gradient in gradients)


def test_relation_v2_runs_from_empty_memory_with_eight_routes() -> None:
    model, data = _v2_fixture()
    output = model_forward(model, data, epoch_index=0)
    assert output.forecast.shape == (1, 2, 96)
    assert output.route_logits.shape == (1, 2, 12, 8)
    assert output.graph.route_latents.shape[-2] == 8
    assert not output.historical.available.any()


def test_online_policy_exposes_only_none_and_single_relation_routes() -> None:
    model, data = _v2_fixture()
    output = model_forward(model, data, epoch_index=40)

    assert int(output.route_index.max()) <= 3
    assert not output.route_available[..., 4:].any()


def test_online_policy_uses_the_largest_positive_branch_gain() -> None:
    model, data = _v2_fixture()
    output = model_forward(model, data, epoch_index=40)
    gains = torch.stack(
        (
            output.historical.predicted_gain,
            output.temporal.predicted_gain,
            output.variable.predicted_gain,
        ),
        dim=-1,
    )

    torch.testing.assert_close(output.route_logits[..., 1:4], gains)
    torch.testing.assert_close(output.route_logits[..., 0], torch.zeros_like(gains[..., 0]))
    assert torch.equal(
        output.route_index,
        direct_single_route_select(gains, output.route_available[..., 1:4]),
    )
    assert not hasattr(model, "route_router")


def test_training_objective_has_explicit_repair_and_forecast_modes() -> None:
    model, data = _v2_fixture()
    repair = model.training_objective(
        data["values"], data["mask"], data["rho"], data["state"],
        data["query_mean"], data["query_std"], data["forecast_target"], data["clean"],
        patch_starts=data["starts"], epoch_index=40, objective="repair",
    )
    forecast = model.training_objective(
        data["values"], data["mask"], data["rho"], data["state"],
        data["query_mean"], data["query_std"], data["forecast_target"], data["clean"],
        patch_starts=data["starts"], epoch_index=40, objective="forecast",
    )

    torch.testing.assert_close(
        repair.loss.total,
        repair.loss.weighted_imputation_huber
        + repair.loss.weighted_edge_gain
        + repair.loss.weighted_memory,
    )
    assert repair.loss.weighted_imputation_huber.item() == pytest.approx(
        repair.loss.imputation_huber.item()
    )
    assert repair.loss.weighted_edge_gain.item() == pytest.approx(
        0.20 * repair.loss.edge_gain.item()
    )
    assert repair.loss.weighted_memory.item() == pytest.approx(
        0.05 * repair.loss.memory.item()
    )
    assert repair.loss.route.item() == 0.0
    assert repair.loss.weighted_route.item() == 0.0
    model.zero_grad(set_to_none=True)
    repair.loss.total.backward()
    assert model.core.head.linear.weight.grad is None
    torch.testing.assert_close(forecast.loss.total, forecast.loss.forecast)


def test_forecast_objective_is_independent_of_clean_reconstruction_targets() -> None:
    """Stage 2 consumes the masked input and forecast target only."""

    model, data = _v2_fixture()
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
        epoch_index=40,
        objective="forecast",
    )

    torch.testing.assert_close(result.loss.total, result.loss.forecast)
    for component in (
        result.loss.imputation_huber,
        result.loss.edge_gain,
        result.loss.route,
        result.loss.memory,
    ):
        assert component.item() == pytest.approx(0.0)
    assert result.objective == "forecast"


def test_forecast_finetuning_updates_messages_but_freezes_route_scores() -> None:
    """Forecast gradients reach TVH messages without erasing gain calibration."""

    model, data = _v2_fixture()
    model.configure_forecast_finetuning()
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
        epoch_index=40,
        objective="forecast",
    )
    model.zero_grad(set_to_none=True)
    result.loss.total.backward()

    assert model.mask_projection.weight.grad is not None
    assert float(model.mask_projection.weight.grad.abs().sum()) > 0.0
    assert model.graph.residual_mlp[0].weight.grad is not None
    assert float(model.graph.residual_mlp[0].weight.grad.abs().sum()) > 0.0
    assert model.temporal_operator.candidate_projector[2].weight.grad is not None
    assert float(
        model.temporal_operator.candidate_projector[2].weight.grad.abs().sum()
    ) > 0.0
    for branch in (
        model.historical_operator,
        model.temporal_operator,
        model.variable_operator,
    ):
        assert all(not parameter.requires_grad for parameter in branch.gain_head.parameters())
        assert all(
            not parameter.requires_grad
            for parameter in branch.uncertainty_head.parameters()
        )
        assert all(parameter.grad is None for parameter in branch.gain_head.parameters())
        assert all(
            parameter.grad is None for parameter in branch.uncertainty_head.parameters()
        )


def test_forecast_finetuning_has_an_explicit_parameter_allowlist() -> None:
    model, _ = _v2_fixture()
    model.configure_forecast_finetuning()

    trainable = {name for name, value in model.named_parameters() if value.requires_grad}
    assert trainable
    assert any(name.startswith("core.") for name in trainable)
    assert any(name.startswith("historical_operator.message_head.") for name in trainable)
    assert any(
        name.startswith("temporal_operator.candidate_projector.")
        for name in trainable
    )
    assert any(name.startswith("variable_operator.translator.") for name in trainable)
    assert any(name.startswith("graph.") for name in trainable)
    assert "mask_projection.weight" in trainable
    assert not any("gain_head" in name for name in trainable)
    assert not any("uncertainty_head" in name for name in trainable)
    assert not any(name.startswith("retrieval_encoder.") for name in trainable)
    assert not any(name.startswith("patch_decoder.") for name in trainable)


def test_frozen_teacher_split_reproduces_the_unsplit_stage1_forward() -> None:
    """Snapshotting identical weights must not alter the first forecast or route."""

    torch.manual_seed(101)
    model, data = _v2_fixture()
    data = _populate_historical_teacher_fixture(model, data)
    model.eval()
    ordinary = model_forward(model, data, epoch_index=100)

    model.install_frozen_route_teacher()
    split = model_forward(model, data, epoch_index=100)

    for actual, expected in (
        (split.forecast, ordinary.forecast),
        (split.historical.message, ordinary.historical.message),
        (split.temporal.message, ordinary.temporal.message),
        (split.variable.message, ordinary.variable.message),
        (split.route_logits, ordinary.route_logits),
    ):
        torch.testing.assert_close(actual, expected)
    assert torch.equal(split.route_available, ordinary.route_available)
    assert torch.equal(split.route_index, ordinary.route_index)


def test_fixed_soft_backward_is_enabled_only_for_the_frozen_teacher() -> None:
    """An unselected T path keeps legacy zero gradients but learns under E20."""

    torch.manual_seed(102)
    model, data = _v2_fixture()
    data = _populate_historical_teacher_fixture(model, data)
    with torch.no_grad():
        for operator, gain in (
            (model.historical_operator, 3.0),
            (model.temporal_operator, 2.0),
            (model.variable_operator, 1.0),
        ):
            operator.gain_head[-1].weight.zero_()
            operator.gain_head[-1].bias.fill_(gain)
    model.configure_forecast_finetuning()
    model.train()
    legacy = model.training_objective(
        data["values"], data["mask"], data["rho"], data["state"],
        data["query_mean"], data["query_std"], data["forecast_target"], None,
        patch_starts=data["starts"], epoch_index=100, objective="forecast",
    )
    damaged = data["state"].ne(0)
    assert torch.equal(
        legacy.output.route_index[damaged],
        torch.ones_like(legacy.output.route_index[damaged]),
    )
    model.zero_grad(set_to_none=True)
    legacy.loss.total.backward()
    legacy_temporal_gradient = model.temporal_operator.candidate_projector[-1].weight.grad
    assert legacy_temporal_gradient is not None
    assert torch.count_nonzero(legacy_temporal_gradient) == 0

    model.zero_grad(set_to_none=True)
    model.install_frozen_route_teacher()
    model.configure_forecast_finetuning(frozen_route_teacher=True)
    teacher = model.training_objective(
        data["values"], data["mask"], data["rho"], data["state"],
        data["query_mean"], data["query_std"], data["forecast_target"], None,
        patch_starts=data["starts"], epoch_index=100, objective="forecast",
    )
    torch.testing.assert_close(teacher.output.forecast, legacy.output.forecast)
    assert torch.equal(teacher.output.route_index, legacy.output.route_index)
    model.zero_grad(set_to_none=True)
    teacher.loss.total.backward()
    teacher_temporal_gradient = model.temporal_operator.candidate_projector[-1].weight.grad
    assert teacher_temporal_gradient is not None
    assert torch.isfinite(teacher_temporal_gradient).all()
    assert torch.count_nonzero(teacher_temporal_gradient) > 0


def test_frozen_teacher_routes_are_invariant_while_student_receives_forecast_gradients() -> None:
    """A forecast update may change message content but never the teacher policy."""

    torch.manual_seed(103)
    model, data = _v2_fixture()
    data = _populate_historical_teacher_fixture(model, data)
    with torch.no_grad():
        model.historical_operator.gain_head[-1].weight.zero_()
        model.historical_operator.gain_head[-1].bias.fill_(1.0)
    model.install_frozen_route_teacher()
    model.configure_forecast_finetuning(frozen_route_teacher=True)
    model.train()

    before_output = model_forward(model, data, epoch_index=100)
    before_route = {
        "gain": torch.stack(
            (
                before_output.historical.predicted_gain,
                before_output.temporal.predicted_gain,
                before_output.variable.predicted_gain,
            ),
            dim=-1,
        ).detach().clone(),
        "uncertainty": torch.stack(
            (
                before_output.historical.uncertainty,
                before_output.temporal.uncertainty,
                before_output.variable.uncertainty,
            ),
            dim=-1,
        ).detach().clone(),
        "available": before_output.route_available.detach().clone(),
        "route_index": before_output.route_index.detach().clone(),
    }
    before_parameters = {
        "core": model.core.W_P.weight.detach().clone(),
        "historical": model.historical_operator.message_head[-1].weight.detach().clone(),
        "temporal": model.temporal_operator.candidate_projector[-1].weight.detach().clone(),
        "variable": model.variable_operator.translator[-1].weight.detach().clone(),
        "graph": model.graph.residual_mlp[-1].weight.detach().clone(),
        "mask": model.mask_projection.weight.detach().clone(),
    }
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1.0e-3,
    )
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
        epoch_index=100,
        objective="forecast",
    )
    optimizer.zero_grad(set_to_none=True)
    result.loss.total.backward()

    for module in (
        model.core,
        model.mask_projection,
        model.historical_operator.message_head,
        model.temporal_operator.candidate_projector,
        model.temporal_operator.fusion_gate,
        model.variable_operator.source_embedding,
        model.variable_operator.target_embedding,
        model.variable_operator.lag_embedding,
        model.variable_operator.translator,
        model.graph,
    ):
        _assert_nonzero_finite_gradient(module)
    assert model.route_teacher is not None
    assert all(parameter.grad is None for parameter in model.route_teacher.parameters())

    optimizer.step()
    after_output = model_forward(model, data, epoch_index=100)
    after_route = {
        "gain": torch.stack(
            (
                after_output.historical.predicted_gain,
                after_output.temporal.predicted_gain,
                after_output.variable.predicted_gain,
            ),
            dim=-1,
        ),
        "uncertainty": torch.stack(
            (
                after_output.historical.uncertainty,
                after_output.temporal.uncertainty,
                after_output.variable.uncertainty,
            ),
            dim=-1,
        ),
        "available": after_output.route_available,
        "route_index": after_output.route_index,
    }
    for name in before_route:
        assert torch.equal(before_route[name], after_route[name])
    after_parameters = {
        "core": model.core.W_P.weight,
        "historical": model.historical_operator.message_head[-1].weight,
        "temporal": model.temporal_operator.candidate_projector[-1].weight,
        "variable": model.variable_operator.translator[-1].weight,
        "graph": model.graph.residual_mlp[-1].weight,
        "mask": model.mask_projection.weight,
    }
    assert all(
        not torch.equal(before_parameters[name], after_parameters[name])
        for name in before_parameters
    )


def test_frozen_teacher_checkpoint_round_trips_strictly() -> None:
    """An E20 model state is self-contained after a compatible container is installed."""

    torch.manual_seed(107)
    model, data = _v2_fixture()
    model.install_frozen_route_teacher()
    model.eval()
    expected = model_forward(model, data, epoch_index=100)
    expected_digest = model.route_teacher.state_sha256()
    checkpoint = deepcopy(model.state_dict())

    restored, _ = _v2_fixture()
    restored.install_frozen_route_teacher()
    incompatible = restored.load_state_dict(checkpoint, strict=True)
    restored.eval()
    actual = model_forward(restored, data, epoch_index=100)

    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    assert restored.route_teacher is not None
    assert restored.route_teacher.state_sha256() == expected_digest
    assert all(not parameter.requires_grad for parameter in restored.route_teacher.parameters())
    assert not restored.route_teacher.training
    torch.testing.assert_close(actual.forecast, expected.forecast)
    assert torch.equal(actual.route_index, expected.route_index)


def test_teacher_hv_fusion_trains_hv_and_gate_without_teacher_gradients() -> None:
    """Forecast MSE trains H/V fusion while T stays excluded and teacher stays frozen."""

    torch.manual_seed(109)
    model, data = _v2_fixture()
    data = _populate_historical_teacher_fixture(model, data)
    with torch.no_grad():
        model.historical_operator.gain_head[-1].weight.zero_()
        model.historical_operator.gain_head[-1].bias.fill_(1.0)
    model.install_frozen_route_teacher()
    model.install_forecast_hv_fusion()
    model.configure_forecast_finetuning(
        frozen_route_teacher=True,
        forecast_fusion="teacher_hv",
    )
    model.train()

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
        epoch_index=100,
        objective="forecast",
    )
    weights = result.output.diagnostics["forecast_route_weights"]
    assert torch.equal(weights[..., 2], torch.zeros_like(weights[..., 2]))
    torch.testing.assert_close(
        weights.sum(dim=-1),
        torch.ones_like(weights[..., 0]),
    )

    model.zero_grad(set_to_none=True)
    result.loss.total.backward()

    assert model.forecast_fusion is not None
    _assert_nonzero_finite_gradient(model.forecast_fusion)
    _assert_nonzero_finite_gradient(model.historical_operator.message_head)
    _assert_nonzero_finite_gradient(model.variable_operator.translator)
    _assert_nonzero_finite_gradient(model.graph)
    assert model.route_teacher is not None
    assert all(parameter.grad is None for parameter in model.route_teacher.parameters())
    assert all(
        parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
        for parameter in model.temporal_operator.candidate_projector.parameters()
    )
    assert all(
        not parameter.requires_grad
        for parameter in model.temporal_operator.candidate_projector.parameters()
    )
    assert all(
        not parameter.requires_grad
        for parameter in model.temporal_operator.fusion_gate.parameters()
    )
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert not any(
        name.startswith("temporal_operator.candidate_projector.")
        for name in trainable_names
    )
    assert not any(
        name.startswith("temporal_operator.fusion_gate.")
        for name in trainable_names
    )


def test_hard_forecast_mode_ignores_an_installed_fusion_module() -> None:
    """Installing the E21 container must not change explicit E20 hard behavior."""

    torch.manual_seed(113)
    model, data = _v2_fixture()
    data = _populate_historical_teacher_fixture(model, data)
    model.install_frozen_route_teacher()
    model.eval()
    expected = model_forward(model, data, epoch_index=100)

    model.install_forecast_hv_fusion()
    model.configure_forecast_finetuning(
        frozen_route_teacher=True,
        forecast_fusion="hard",
    )
    model.eval()
    actual = model_forward(model, data, epoch_index=100)

    torch.testing.assert_close(actual.forecast, expected.forecast)
    torch.testing.assert_close(actual.forecast_latents, expected.forecast_latents)
    assert torch.equal(
        actual.diagnostics["forecast_route_weights"],
        expected.diagnostics["forecast_route_weights"],
    )
    assert torch.equal(actual.route_index, expected.route_index)


def test_teacher_hv_checkpoint_round_trips_strictly() -> None:
    """An E21 checkpoint restores both frozen teacher and trainable fusion state."""

    torch.manual_seed(127)
    model, data = _v2_fixture()
    model.install_frozen_route_teacher()
    model.install_forecast_hv_fusion()
    model.configure_forecast_finetuning(
        frozen_route_teacher=True,
        forecast_fusion="teacher_hv",
    )
    model.eval()
    expected = model_forward(model, data, epoch_index=100)
    assert model.route_teacher is not None
    expected_digest = model.route_teacher.state_sha256()
    checkpoint = deepcopy(model.state_dict())
    assert any(name.startswith("route_teacher.") for name in checkpoint)
    assert any(name.startswith("forecast_fusion.") for name in checkpoint)

    restored, _ = _v2_fixture()
    restored.install_frozen_route_teacher()
    restored.install_forecast_hv_fusion()
    incompatible = restored.load_state_dict(checkpoint, strict=True)
    restored.configure_forecast_finetuning(
        frozen_route_teacher=True,
        forecast_fusion="teacher_hv",
    )
    restored.eval()
    actual = model_forward(restored, data, epoch_index=100)

    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    assert restored.route_teacher is not None
    assert restored.route_teacher.state_sha256() == expected_digest
    torch.testing.assert_close(actual.forecast, expected.forecast)
    torch.testing.assert_close(
        actual.diagnostics["forecast_route_weights"],
        expected.diagnostics["forecast_route_weights"],
    )
    assert torch.equal(actual.route_index, expected.route_index)


def test_fully_observed_input_is_an_exact_recovery_bypass() -> None:
    model, data = _v2_fully_observed_fixture()
    output = model_forward(model, data, epoch_index=40)
    expected = model.core.forecast_from_latents(output.base_latents)
    expected = expected * data["query_std"].unsqueeze(-1) + data["query_mean"].unsqueeze(-1)
    torch.testing.assert_close(output.forecast, expected)
    torch.testing.assert_close(output.latents, output.base_latents)
    torch.testing.assert_close(output.reconstructed_patches, data["values"])


def test_v2_parameter_groups_are_disjoint_and_complete() -> None:
    model, _ = _v2_fixture()
    groups = [
        list(model.backbone_parameters()),
        list(model.relation_parameters()),
    ]
    grouped = [id(parameter) for group in groups for parameter in group]
    expected = [id(parameter) for parameter in model.parameters() if parameter.requires_grad]
    assert len(grouped) == len(set(grouped))
    assert set(grouped) == set(expected)


def test_repair_loss_directly_balances_every_available_relation_branch() -> None:
    model, data = _v2_fixture()
    model.begin_epoch(1)
    model.stage_observed_batch(
        data["clean"], data["mask"], data["rho"], data["state"], data["starts"]
    )
    model.end_epoch(1)

    result = model.training_objective(
        data["values"], data["mask"], data["rho"], data["state"],
        data["query_mean"], data["query_std"], data["forecast_target"], data["clean"],
        patch_starts=data["starts"], epoch_index=40, objective="repair",
    )
    target = data["state"].ne(0) & (~data["mask"]).any(dim=-1)
    available = torch.stack(
        (
            result.output.historical.available,
            result.output.temporal.available,
            result.output.variable.available,
        ),
        dim=-1,
    ) & target.unsqueeze(-1)
    expected = []
    for relation_index in range(3):
        selected = available[..., relation_index]
        expected.append(
            result.route_errors[..., relation_index + 1][selected].mean()
            if selected.any()
            else result.route_errors.sum() * 0.0
        )
    expected_losses = torch.stack(expected)

    torch.testing.assert_close(result.branch_repair_losses, expected_losses)
    torch.testing.assert_close(
        result.loss.imputation_huber,
        expected_losses[available.any(dim=(0, 1, 2))].mean(),
    )


def test_route_errors_are_decoded_from_all_route_latents() -> None:
    """Each counterfactual route is scored after its own residual recovery."""

    model, data = _v2_fixture()
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
        epoch_index=40,
    )

    assert result.route_errors.shape == (1, 2, 12, 8)
    assert not torch.equal(result.route_errors[..., 4], result.route_errors[..., 1])
    assert result.route_target.shape == (1, 2, 12)


def test_counterfactual_errors_use_raw_decoder_coordinates() -> None:
    """A zero normalized decoder must reconstruct each raw query mean."""

    model, data = _v2_fixture()
    data = dict(data)
    data["query_mean"] = torch.tensor([[10.0, -4.0]])
    data["query_std"] = torch.tensor([[2.0, 3.0]])
    data["clean"] = data["query_mean"].unsqueeze(-1).unsqueeze(-1).expand_as(
        data["clean"]
    ).clone()
    data["values"] = torch.zeros_like(data["values"])
    data["forecast_target"] = torch.zeros_like(data["forecast_target"])
    with torch.no_grad():
        model.patch_decoder.weight.zero_()
        model.patch_decoder.bias.zero_()

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
        epoch_index=40,
    )

    target_eligible = (~data["mask"]).any(dim=-1) & data["state"].ne(0)
    torch.testing.assert_close(
        result.route_errors[target_eligible],
        torch.zeros_like(result.route_errors[target_eligible]),
    )
    assert result.loss.imputation_huber.item() == 0.0


def test_training_phase_has_inclusive_bootstrap_and_joint_boundaries() -> None:
    """Epoch boundaries must not accidentally expose H routes one epoch early."""

    model, _ = _v2_fixture()

    assert model.training_phase(20) == "bootstrap"
    assert model.training_phase(21) == "joint"
    assert model.training_phase(80) == "joint"
    assert model.training_phase(81) == "calibration"


def test_v2_evaluation_reports_ablations_without_mutating_online_state() -> None:
    """Full and relation-disabled inference must leave online H/V/route state intact."""

    model, data = _v2_fixture()
    batch = PreparedBatch(
        patch_values=data["values"], patch_mask=data["mask"], rho=data["rho"],
        state=data["state"], query_mean=data["query_mean"], query_std=data["query_std"],
        forecast_target=data["forecast_target"], clean_raw_patches=data["clean"],
        patch_valid=torch.ones_like(data["rho"], dtype=torch.bool),
        raw_context=torch.empty(0), context_mask=torch.empty(0, dtype=torch.bool),
        train_std=torch.ones(2), patch_starts=data["starts"],
    )
    before = model.online_state()
    result = evaluate_online_model(model, [batch])
    after = model.online_state()

    assert set(result.inference_ablations) == {
        "without_historical", "without_temporal", "without_variable", "historical_only"
    }
    assert result.imputation_mse >= 0.0
    assert result.diagnostics["route_repair_regret"] >= 0.0
    assert 0.0 <= result.diagnostics["route_repair_regret_positive_rate"] <= 1.0
    assert before["memory_hash"] == after["memory_hash"]
    assert before["ema_hash"] == after["ema_hash"]
    assert before["relation_prior_hash"] == after["relation_prior_hash"]


def test_v2_evaluation_oracle_excludes_policy_disabled_historical_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repair regret compares only routes the active policy may select."""

    import malr.evaluation.evaluator as evaluator_module

    model, data = _v2_fixture()
    data = _populate_historical_teacher_fixture(model, data)
    with torch.no_grad():
        model.historical_operator.gain_head[-1].weight.zero_()
        model.historical_operator.gain_head[-1].bias.fill_(1.0)
    batch = PreparedBatch(
        patch_values=data["values"], patch_mask=data["mask"], rho=data["rho"],
        state=data["state"], query_mean=data["query_mean"], query_std=data["query_std"],
        forecast_target=data["forecast_target"], clean_raw_patches=data["clean"],
        patch_valid=torch.ones_like(data["rho"], dtype=torch.bool),
        raw_context=torch.empty(0), context_mask=torch.empty(0, dtype=torch.bool),
        train_std=torch.ones(2), patch_starts=data["starts"],
    )
    captured: dict[str, object] = {}
    original_forward = evaluator_module._forward_v2
    original_decode = model._decode_raw_patches

    def capture_full_forward(model_arg, current, disabled):
        output = original_forward(model_arg, current, disabled)
        if not disabled:
            captured["full"] = output
        return output

    def controlled_decode(latents, query_mean, query_std):
        output = captured.get("full")
        if output is not None and latents is output.base_latents:
            return data["clean"].new_zeros(data["clean"].shape)
        if output is not None and latents is output.latents:
            return data["clean"].new_zeros(data["clean"].shape)
        if output is not None and latents is output.graph.route_latents:
            prediction = data["clean"].new_zeros(
                *data["clean"].shape[:-1], latents.shape[-2], data["clean"].shape[-1]
            )
            prediction[..., 1, :] = data["clean"]
            return prediction
        return original_decode(latents, query_mean, query_std)

    monkeypatch.setattr(evaluator_module, "_forward_v2", capture_full_forward)
    monkeypatch.setattr(model, "_decode_raw_patches", controlled_decode)
    result = evaluate_online_model(model, [batch])
    output = captured["full"]

    damaged = batch.state.ne(0) & batch.patch_valid
    missing = (~batch.patch_mask) & damaged.unsqueeze(-1)
    target = damaged & missing.any(dim=-1)
    assert target.any()
    assert (output.historical.available & target).any()
    assert not output.route_available[..., 1][target].any()

    baseline_prediction = torch.zeros_like(batch.clean_raw_patches)
    baseline_points = torch.nn.functional.huber_loss(
        baseline_prediction, batch.clean_raw_patches, reduction="none"
    )
    baseline_error = baseline_points.masked_fill(~missing, 0.0).sum(dim=-1)
    baseline_error = baseline_error / missing.sum(dim=-1).clamp_min(1)

    route_prediction = torch.zeros(
        *batch.clean_raw_patches.shape[:-1], 8, batch.clean_raw_patches.shape[-1]
    )
    route_prediction[..., 1, :] = batch.clean_raw_patches
    route_target = batch.clean_raw_patches.unsqueeze(-2).expand_as(route_prediction)
    route_missing = missing.unsqueeze(-2).expand_as(route_prediction)
    route_points = torch.nn.functional.huber_loss(
        route_prediction, route_target, reduction="none"
    )
    route_errors = route_points.masked_fill(~route_missing, 0.0).sum(dim=-1)
    route_errors = route_errors / route_missing.sum(dim=-1).clamp_min(1)
    route_errors[..., 0] = baseline_error

    allowed = output.route_available[..., :4] & target.unsqueeze(-1)
    eligible_errors = route_errors[..., :4].masked_fill(~allowed, torch.inf)
    oracle_error, oracle_index = eligible_errors.min(dim=-1)
    selected_error = route_errors.gather(
        -1, output.route_index.unsqueeze(-1)
    ).squeeze(-1)
    expected_regret = (
        (selected_error - oracle_error) / (baseline_error + 1.0e-8)
    ).clamp_min(0.0)
    expected_mean = float(expected_regret[target].mean())
    expected_positive_rate = float(expected_regret[target].gt(1.0e-7).float().mean())
    expected_selection_error = float(output.route_index[target].ne(oracle_index[target]).float().mean())

    assert route_errors[..., 1][output.historical.available & target].eq(0.0).all()
    assert baseline_error[output.historical.available & target].gt(0.0).all()
    assert oracle_index[target].ne(1).all()
    assert result.diagnostics["route_repair_regret"] == pytest.approx(expected_mean)
    assert result.diagnostics["route_repair_regret_positive_rate"] == pytest.approx(
        expected_positive_rate
    )
    assert result.diagnostics["route_selection_error_rate"] == pytest.approx(
        expected_selection_error
    )


def test_v2_evaluation_excludes_final_patch_padding_from_imputation_metrics() -> None:
    """The padded tail must not become missing evidence for online evaluation."""

    model, _ = _v2_fixture()
    raw = torch.zeros(1, 96, 2)
    raw[:, 88, 0] = 2.0
    raw[:, 95, 0] = 10.0
    observed = torch.ones_like(raw, dtype=torch.bool)
    observed[:, 88, 0] = False
    patches = patchify_with_state(raw, observed, length=16, stride=8)
    clean_patches = patchify_with_state(
        raw, torch.ones_like(raw, dtype=torch.bool), length=16, stride=8
    )
    batch = PreparedBatch(
        patch_values=patches.values,
        patch_mask=patches.effective_mask,
        rho=patches.rho,
        state=patches.state,
        query_mean=torch.zeros(1, 2),
        query_std=torch.ones(1, 2),
        forecast_target=torch.zeros(1, 2, 96),
        clean_raw_patches=clean_patches.values,
        patch_valid=patches.validity.any(dim=-1),
        raw_context=torch.empty(0),
        context_mask=torch.empty(0, dtype=torch.bool),
        train_std=torch.ones(2),
        patch_validity=patches.validity,
    )
    with torch.no_grad():
        model.patch_decoder.weight.zero_()
        model.patch_decoder.bias.zero_()

    result = evaluate_online_model(model, [batch])

    assert not batch.patch_validity[0, 0, -1, 8:].any()
    assert result.imputation_mse == pytest.approx(4.0)
    assert result.imputation_mae == pytest.approx(2.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_relation_v2_cuda_forward_backward_is_finite() -> None:
    """The V2 relation objective remains finite through a real CUDA backward pass."""

    model, data = _v2_fixture()
    model = model.to("cuda:0")
    cuda = {key: value.to("cuda:0") for key, value in data.items()}
    result = model.training_objective(
        cuda["values"], cuda["mask"], cuda["rho"], cuda["state"],
        cuda["query_mean"], cuda["query_std"], cuda["forecast_target"], cuda["clean"],
        patch_starts=cuda["starts"], epoch_index=40,
    )
    result.loss.total.backward()
    assert torch.isfinite(result.loss.total)
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)
