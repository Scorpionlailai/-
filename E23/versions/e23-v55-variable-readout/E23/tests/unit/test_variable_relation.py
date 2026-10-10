from __future__ import annotations

import pytest
import torch

from malr.models.candidates_v2 import VariableCandidateBatch
from malr.models.variable_relation import (
    DirectedVariableRelation,
    VariableRelationStatistics,
    signed_lag_bucket,
)


def _legacy_variable(
    module: DirectedVariableRelation, nodes: torch.Tensor, candidates: VariableCandidateBatch,
    statistics: VariableRelationStatistics,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pre-plan V selection/softmax formula, without plan or replay helpers."""

    source_variable = candidates.source_variable.to(device=nodes.device, dtype=torch.long)
    source_patch = candidates.source_patch.to(device=nodes.device, dtype=torch.long)
    source = nodes[torch.arange(nodes.shape[0], device=nodes.device).view(-1, 1, 1, 1), source_variable, source_patch]
    bucket = signed_lag_bucket(candidates.lag.to(nodes.device), candidates.lag_band.to(nodes.device, torch.long), module.lag_band_count).long()
    target_variable = torch.arange(nodes.shape[1], device=nodes.device).view(1, -1, 1, 1).expand_as(source_variable)
    features = candidates.features.to(device=nodes.device, dtype=nodes.dtype)
    messages = module.translator(torch.cat((source, module.source_embedding(source_variable), module.target_embedding(target_variable), module.lag_embedding(bucket), features), dim=-1))
    mean, std, stable = statistics.features(source_variable, target_variable, bucket)
    edge = torch.cat((nodes.unsqueeze(-2).expand_as(messages), messages, features, mean.to(nodes).unsqueeze(-1), std.to(nodes).unsqueeze(-1), stable.to(nodes).unsqueeze(-1)), dim=-1)
    gain = module.gain_head(edge).squeeze(-1)
    uncertainty = torch.nn.functional.softplus(module.uncertainty_head(edge).squeeze(-1)) + torch.finfo(nodes.dtype).eps
    score = gain - uncertainty
    admissible = candidates.eligible & gain.gt(module.candidate_gain_threshold)
    one_per_source = torch.zeros_like(admissible)
    for source_id in range(module.variable_count):
        group = admissible & source_variable.eq(source_id)
        best = score.masked_fill(~group, -torch.inf).argmax(dim=-1, keepdim=True)
        one_per_source |= torch.zeros_like(group).scatter(-1, best, True) & group.any(dim=-1, keepdim=True)
    order = torch.argsort(score.masked_fill(~one_per_source, -torch.inf), dim=-1, descending=True, stable=True)
    ranked = one_per_source.gather(-1, order)
    rank = torch.arange(order.shape[-1], device=nodes.device).view(*((1,) * (order.ndim - 1)), -1)
    selected = torch.zeros_like(one_per_source).scatter(-1, order, ranked & rank.lt(module.topk))
    masked_score = score.masked_fill(~selected, -torch.inf)
    maximum = masked_score.max(dim=-1, keepdim=True).values
    maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    numerator = torch.where(selected, torch.exp(masked_score - maximum), torch.zeros_like(score))
    coefficients = numerator / numerator.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(score.dtype).tiny)
    available = selected.any(dim=-1)
    message = (coefficients.unsqueeze(-1) * messages).sum(dim=-2)
    return torch.where(available.unsqueeze(-1), message, torch.zeros_like(message)), coefficients, selected, available


def variable_fixture(
    source: int, target: int, lag: int, *, width: int = 1
) -> tuple[torch.Tensor, VariableCandidateBatch]:
    """One real cross-variable candidate at target variable one, patch zero."""

    nodes = torch.zeros(1, 3, 1, 4)
    shape = (1, 3, 1, width)
    source_variable = torch.zeros(shape, dtype=torch.long)
    source_patch = torch.zeros(shape, dtype=torch.long)
    lags = torch.zeros(shape, dtype=torch.long)
    lag_band = torch.zeros(shape, dtype=torch.long)
    eligible = torch.zeros(shape, dtype=torch.bool)
    source_variable[0, target, 0, 0] = source
    lags[0, target, 0, 0] = lag
    lag_band[0, target, 0, 0] = min(abs(lag), 4)
    eligible[0, target, 0, 0] = True
    return nodes, VariableCandidateBatch(
        source_variable=source_variable,
        source_patch=source_patch,
        lag=lags,
        lag_band=lag_band,
        eligible=eligible,
        is_synchronous=lags.eq(0),
        features=torch.zeros(*shape, 11),
        candidate_count=eligible.sum(dim=-1),
    )


def variable_many_lag_fixture() -> tuple[
    DirectedVariableRelation, torch.Tensor, VariableCandidateBatch, VariableRelationStatistics
]:
    nodes, single = variable_fixture(source=0, target=1, lag=-2, width=3)
    lag = torch.tensor([[[[0, 0, 0]], [[-2, -1, 0]], [[0, 0, 0]]]])
    eligible = torch.zeros_like(single.eligible)
    eligible[0, 1, 0] = True
    candidates = VariableCandidateBatch(
        source_variable=torch.zeros_like(single.source_variable),
        source_patch=torch.zeros_like(single.source_patch),
        lag=lag,
        lag_band=lag.abs().clamp_max(4),
        eligible=eligible,
        is_synchronous=lag.eq(0),
        features=torch.zeros(1, 3, 1, 3, 11),
        candidate_count=eligible.sum(dim=-1),
    )
    return (
        DirectedVariableRelation(4, 11, variable_count=3, signed_bucket_count=11, topk=4),
        nodes,
        candidates,
        VariableRelationStatistics(3, 11),
    )


def variable_two_source_fixture() -> tuple[
    DirectedVariableRelation, torch.Tensor, VariableCandidateBatch, VariableRelationStatistics
]:
    """Two admitted ordered sources so weighted softmax gradients are nonzero."""

    nodes = torch.randn(1, 3, 1, 4)
    source_variable = torch.zeros(1, 3, 1, 2, dtype=torch.long)
    source_variable[0, 1, 0] = torch.tensor([0, 2])
    lag = torch.zeros(1, 3, 1, 2, dtype=torch.long)
    lag[0, 1, 0] = torch.tensor([-1, 1])
    eligible = torch.zeros(1, 3, 1, 2, dtype=torch.bool)
    eligible[0, 1, 0] = True
    candidates = VariableCandidateBatch(
        source_variable=source_variable,
        source_patch=torch.zeros_like(source_variable),
        lag=lag,
        lag_band=lag.abs(),
        eligible=eligible,
        is_synchronous=lag.eq(0),
        features=torch.zeros(1, 3, 1, 2, 11),
        candidate_count=eligible.sum(dim=-1),
    )
    return (
        DirectedVariableRelation(4, 11, variable_count=3, signed_bucket_count=11, topk=4),
        nodes,
        candidates,
        VariableRelationStatistics(3, 11),
    )


def test_signed_lag_bucket_distinguishes_direction_and_synchrony() -> None:
    bucket = signed_lag_bucket(
        torch.tensor([-7, -1, 0, 1, 9]), torch.tensor([4, 1, 0, 1, 4]), 5
    )

    assert bucket.tolist() == [4, 1, 5, 7, 10]


def test_ordered_variable_translation_is_not_symmetric() -> None:
    torch.manual_seed(3)
    module = DirectedVariableRelation(4, 11, variable_count=3, signed_bucket_count=11, topk=4)
    nodes, forward_candidates = variable_fixture(source=0, target=1, lag=-1)
    _, reverse_candidates = variable_fixture(source=1, target=0, lag=1)
    stats = VariableRelationStatistics(3, 11)

    forward = module(nodes, forward_candidates, stats).candidate_messages
    reverse = module(nodes, reverse_candidates, stats).candidate_messages

    assert not torch.equal(forward, reverse)


def test_variable_selection_keeps_at_most_one_lag_per_source() -> None:
    module, nodes, candidates, stats = variable_many_lag_fixture()

    output = module(nodes, candidates, stats)
    selected_source = candidates.source_variable[output.candidate_selected]

    assert selected_source.unique().numel() == selected_source.numel()
    assert selected_source.numel() <= 4


def test_relation_statistics_updates_only_explicit_eligible_entries() -> None:
    stats = VariableRelationStatistics(2, 11, decay=0.9, min_count=2)

    stats.update(
        torch.tensor([0, 0]),
        torch.tensor([1, 1]),
        torch.tensor([4, 4]),
        torch.tensor([0.4, -0.2]),
        torch.tensor([True, False]),
    )

    assert stats.count[0, 1, 4].item() == 1
    assert stats.mean[0, 1, 4].item() == pytest.approx(0.4)


def test_no_positive_gain_leaves_relation_unavailable() -> None:
    module, nodes, candidates, stats = variable_many_lag_fixture()
    with torch.no_grad():
        module.gain_head[-1].weight.zero_()
        module.gain_head[-1].bias.fill_(-1.0)

    output = module(nodes, candidates, stats)

    assert not output.available.any()
    assert torch.equal(output.message, torch.zeros_like(nodes))
    assert torch.equal(output.candidate_selected, torch.zeros_like(candidates.eligible))


def test_variable_relation_rejects_nan_nodes_and_statistics_gain() -> None:
    module, nodes, candidates, stats = variable_many_lag_fixture()
    nodes[0, 0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        module(nodes, candidates, stats)
    with pytest.raises(ValueError, match="finite"):
        stats.update(
            torch.tensor([0]), torch.tensor([1]), torch.tensor([4]),
            torch.tensor([torch.nan]), torch.tensor([True]),
        )


def test_relation_statistics_state_dict_round_trip_preserves_features() -> None:
    stats = VariableRelationStatistics(2, 11, decay=0.9, min_count=2)
    stats.update(
        torch.tensor([0, 0]), torch.tensor([1, 1]), torch.tensor([4, 4]),
        torch.tensor([0.4, 0.2]), torch.tensor([True, True]),
    )
    restored = VariableRelationStatistics(2, 11, decay=0.9, min_count=2)
    restored.load_state_dict(stats.state_dict())
    source = torch.tensor([0])
    target = torch.tensor([1])
    bucket = torch.tensor([4])

    expected = stats.features(source, target, bucket)
    actual = restored.features(source, target, bucket)

    for received, wanted in zip(actual, expected):
        assert torch.equal(received, wanted)


def test_variable_relation_has_finite_source_gradients() -> None:
    torch.manual_seed(5)
    module, _, candidates, stats = variable_many_lag_fixture()
    nodes = torch.randn(1, 3, 1, 4, requires_grad=True)

    output = module(nodes, candidates, stats)
    output.message.square().mean().backward()

    assert nodes.grad is not None
    assert torch.isfinite(nodes.grad).all()


def test_variable_relation_replays_its_aggregation_plan() -> None:
    """A plan must own the selected weighted aggregation, not recompute it."""

    module, nodes, candidates, statistics = variable_many_lag_fixture()

    output, plan = module.forward_with_plan(nodes, candidates, statistics)
    projected = module.project_candidate_messages(nodes, candidates)
    replayed = module.replay_message(projected, plan)
    expected_message, expected_coefficients, expected_selected, expected_available = _legacy_variable(module, nodes, candidates, statistics)

    torch.testing.assert_close(replayed, output.message)
    torch.testing.assert_close(output.message, expected_message)
    torch.testing.assert_close(plan.coefficients, expected_coefficients)
    assert torch.equal(plan.selected, expected_selected)
    assert torch.equal(plan.available, expected_available)
    torch.testing.assert_close(
        plan.coefficients.sum(dim=-1)[plan.available],
        torch.ones_like(plan.available, dtype=nodes.dtype)[plan.available],
    )


def test_variable_forward_preserves_legacy_gain_and_uncertainty_gradients() -> None:
    module, nodes, candidates, statistics = variable_two_source_fixture()
    with torch.no_grad():
        module.gain_head[-1].weight.zero_()
        module.gain_head[-1].bias.fill_(1.0)
        module.uncertainty_head[-1].weight.zero_()
        module.uncertainty_head[-1].bias.zero_()
    reference = DirectedVariableRelation(4, 11, variable_count=3, signed_bucket_count=11, topk=4)
    reference.load_state_dict(module.state_dict())
    reference_statistics = VariableRelationStatistics(3, 11)

    module(nodes, candidates, statistics).message.square().mean().backward()
    expected, _, _, _ = _legacy_variable(reference, nodes, candidates, reference_statistics)
    expected.square().mean().backward()

    for actual_head, expected_head in ((module.gain_head, reference.gain_head), (module.uncertainty_head, reference.uncertainty_head)):
        for actual, expected_parameter in zip(actual_head.parameters(), expected_head.parameters()):
            assert actual.grad is not None
            torch.testing.assert_close(actual.grad, expected_parameter.grad, rtol=0.0, atol=0.0)
    assert any(parameter.grad is not None and parameter.grad.abs().sum().gt(0) for parameter in reference.gain_head.parameters())
    assert any(parameter.grad is not None and parameter.grad.abs().sum().gt(0) for parameter in reference.uncertainty_head.parameters())
