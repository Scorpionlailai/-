from __future__ import annotations

import torch

from malr.models.candidates_v2 import TemporalCandidateBatch, build_temporal_candidates
from malr.models.temporal_relation import (
    DynamicTemporalRelation,
    select_temporal_candidates,
)


def temporal_fixture(
    lags: tuple[int, ...], bands: tuple[int, ...]
) -> TemporalCandidateBatch:
    """Create a hand-checked one-target candidate pool for selector tests."""

    width = len(lags)
    shape = (1, 1, 1, width)
    lag = torch.tensor(lags).reshape(shape)
    band = torch.tensor(bands).reshape(shape)
    return TemporalCandidateBatch(
        source_patch=10 + lag,
        signed_patch_lag=lag,
        signed_time_offset=8 * lag,
        direction=lag.sign(),
        distance_band=band,
        eligible=torch.ones(shape, dtype=torch.bool),
        is_local=band.eq(0),
        features=torch.zeros(*shape, 12),
        candidate_count=torch.full(shape[:-1], width, dtype=torch.long),
    )


def test_bilateral_local_selection_uses_nearest_left_and_right() -> None:
    candidates = temporal_fixture(lags=(-4, -1, 1, 3), bands=(1, 0, 0, 1))

    selection = select_temporal_candidates(candidates, torch.ones(1, 1, 1, 4))

    assert selection.local_mode.item() == 1
    assert selection.local_selected.flatten().tolist() == [False, True, True, False]


def test_one_sided_local_selection_uses_two_nearest_anchors() -> None:
    candidates = temporal_fixture(lags=(-4, -2, -1), bands=(1, 0, 0))

    selection = select_temporal_candidates(candidates, torch.ones(1, 1, 1, 3))

    assert selection.local_mode.item() == 2
    assert selection.local_selected.flatten().tolist() == [False, True, True]


def test_right_only_local_selection_backcasts_from_two_nearest_anchors() -> None:
    candidates = temporal_fixture(lags=(1, 2, 4), bands=(0, 0, 1))

    selection = select_temporal_candidates(candidates, torch.ones(1, 1, 1, 3))

    assert selection.local_mode.item() == 3
    assert selection.local_selected.flatten().tolist() == [True, True, False]


def test_single_sided_anchor_has_no_local_mode() -> None:
    candidates = temporal_fixture(lags=(-3, -1), bands=(1, 0))

    selection = select_temporal_candidates(candidates, torch.ones(1, 1, 1, 2))

    assert selection.local_mode.item() == 0
    assert not selection.local_selected.any()


def test_global_quota_keeps_far_band_despite_lower_score() -> None:
    candidates = temporal_fixture(lags=(-3, -6, -10), bands=(1, 2, 3))
    score = torch.tensor([[[[0.9, 0.8, 0.1]]]])

    selection = select_temporal_candidates(candidates, score)

    assert selection.global_selected.flatten().tolist() == [True, True, True]


def _forward_candidates(rho: torch.Tensor) -> TemporalCandidateBatch:
    patch_count = rho.shape[-1]
    values = torch.zeros(1, 1, patch_count, 2)
    mask = torch.ones_like(values, dtype=torch.bool)
    starts = (torch.arange(patch_count) * 8).unsqueeze(0)
    return build_temporal_candidates(values, mask, rho, starts, (1, 2, 4))


def test_dynamic_relation_without_admissible_candidates_is_exact_zero() -> None:
    nodes = torch.randn(1, 1, 3, 4)
    rho = torch.zeros(1, 1, 3)
    output = DynamicTemporalRelation(4, 12, 4)(nodes, _forward_candidates(rho), rho)

    assert not output.available.any()
    assert torch.equal(output.message, torch.zeros_like(nodes))
    assert torch.equal(output.predicted_gain, torch.zeros_like(rho))
    assert torch.equal(output.candidate_selected, torch.zeros_like(output.candidate_selected))


def test_dynamic_relation_uses_remote_sources_and_has_finite_source_gradients() -> None:
    torch.manual_seed(7)
    nodes = torch.randn(1, 1, 3, 4, requires_grad=True)
    rho = torch.tensor([[[1.0, 0.0, 1.0]]])
    candidates = _forward_candidates(rho)
    relation = DynamicTemporalRelation(4, 12, 4)

    output = relation(nodes, candidates, rho)
    output.message.square().mean().backward()

    changed_nodes = nodes.detach().clone()
    changed_nodes[0, 0, 0] += 10.0
    changed = relation(changed_nodes, candidates, rho)

    assert torch.isfinite(output.message).all()
    assert nodes.grad is not None
    assert torch.isfinite(nodes.grad).all()
    # Target patch 1 is completely missing but has left/right observed anchors.
    assert torch.isfinite(output.message[0, 0, 1]).all()
    assert output.available[0, 0, 1]
    assert not torch.allclose(output.message[0, 0, 1], changed.message[0, 0, 1])


def test_dynamic_relation_replays_its_local_and_global_aggregation_plan() -> None:
    """A plan must retain local/global membership and reproduce fused messages."""

    nodes = torch.randn(1, 1, 3, 4)
    rho = torch.tensor([[[1.0, 0.0, 1.0]]])
    candidates = _forward_candidates(rho)
    operator = DynamicTemporalRelation(4, 12, 4)

    output, plan = operator.forward_with_plan(nodes, candidates, rho)
    projected = operator.project_candidate_messages(nodes, candidates)
    replayed = operator.replay_message(nodes, projected, plan)

    torch.testing.assert_close(replayed, output.message)
    assert torch.equal(plan.local_selected, output.diagnostics["local_selected"])
    assert torch.equal(plan.global_selected, output.diagnostics["global_selected"])
    assert torch.equal(plan.available, output.available)


def _temporal_plan_mode_fixture() -> tuple[torch.Tensor, TemporalCandidateBatch, torch.Tensor]:
    """Five targets: bilateral, left, right, global-only, and local+global."""

    nodes = torch.arange(20, dtype=torch.float32).reshape(1, 1, 5, 4) / 10.0
    rho = torch.ones(1, 1, 5)
    offsets = torch.tensor([[-8, 8, 0, 0], [-8, -16, 0, 0], [8, 16, 0, 0], [-24, 24, 0, 0], [-8, 8, -24, 0]])
    eligible = torch.tensor([[True, True, False, False], [True, True, False, False], [True, True, False, False], [True, True, False, False], [True, True, True, False]]).reshape(1, 1, 5, 4)
    is_local = torch.tensor([[True, True, False, False], [True, True, False, False], [True, True, False, False], [False, False, False, False], [True, True, False, False]]).reshape(1, 1, 5, 4)
    source_patch = torch.tensor([[0, 1, 0, 0], [0, 1, 0, 0], [3, 4, 0, 0], [0, 4, 0, 0], [0, 4, 2, 0]]).reshape(1, 1, 5, 4)
    offset = offsets.reshape(1, 1, 5, 4)
    return nodes, TemporalCandidateBatch(
        source_patch=source_patch,
        signed_patch_lag=offset.sign().long(),
        signed_time_offset=offset,
        direction=offset.sign().long(),
        distance_band=torch.where(is_local, torch.zeros_like(offset), torch.ones_like(offset)),
        eligible=eligible,
        is_local=is_local,
        features=torch.zeros(1, 1, 5, 4, 12),
        candidate_count=eligible.sum(dim=-1),
    ), rho


def test_temporal_plan_independently_reproduces_all_aggregation_modes() -> None:
    nodes, candidates, rho = _temporal_plan_mode_fixture()
    operator = DynamicTemporalRelation(4, 12, 4)
    with torch.no_grad():
        operator.gain_head[-1].weight.zero_()
        operator.gain_head[-1].bias.fill_(1.0)
        operator.uncertainty_head[-1].weight.zero_()
        operator.uncertainty_head[-1].bias.zero_()
        for parameter in operator.fusion_gate.parameters():
            parameter.zero_()

    output, plan = operator.forward_with_plan(nodes, candidates, rho)
    projected = operator.project_candidate_messages(nodes, candidates)
    local_coefficients = torch.tensor([[[[0.5, 0.5, 0.0, 0.0], [2.0, -1.0, 0.0, 0.0], [2.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]]]])
    global_coefficients = torch.tensor([[[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]]])
    local_available = local_coefficients.ne(0).any(dim=-1)
    global_available = global_coefficients.ne(0).any(dim=-1)
    expected_local = (local_coefficients.unsqueeze(-1) * projected).sum(dim=-2)
    expected_global = (global_coefficients.unsqueeze(-1) * projected).sum(dim=-2)
    expected_message = torch.where(
        (local_available & global_available).unsqueeze(-1),
        0.5 * expected_local + 0.5 * expected_global,
        torch.where(local_available.unsqueeze(-1), expected_local, expected_global),
    )

    torch.testing.assert_close(plan.local_coefficients, local_coefficients)
    torch.testing.assert_close(plan.global_coefficients, global_coefficients)
    assert plan.local_mode.flatten().tolist() == [1, 2, 3, 0, 1]
    assert torch.equal(plan.available, local_available | global_available)
    torch.testing.assert_close(output.message, expected_message)
    torch.testing.assert_close(operator.replay_message(nodes, projected, plan), expected_message)


def test_temporal_forward_preserves_legacy_global_weighting_gradients() -> None:
    nodes, candidates, rho = _temporal_plan_mode_fixture()
    operator = DynamicTemporalRelation(4, 12, 4)
    with torch.no_grad():
        operator.gain_head[-1].weight.zero_()
        operator.gain_head[-1].bias.fill_(1.0)
        operator.uncertainty_head[-1].weight.zero_()
        operator.uncertainty_head[-1].bias.zero_()
        for parameter in operator.fusion_gate.parameters():
            parameter.zero_()
    reference = DynamicTemporalRelation(4, 12, 4)
    reference.load_state_dict(operator.state_dict())

    operator(nodes, candidates, rho).message.square().mean().backward()
    messages = reference.project_candidate_messages(nodes, candidates)
    source_index = candidates.source_patch.long()
    source_rho = rho.unsqueeze(2).expand(-1, -1, nodes.shape[2], -1).gather(3, source_index)
    edge = torch.cat((nodes.unsqueeze(-2).expand_as(messages), messages, candidates.features, rho.unsqueeze(-1).expand_as(source_rho).unsqueeze(-1), source_rho.unsqueeze(-1)), dim=-1)
    gain = reference.gain_head(edge).squeeze(-1)
    uncertainty = torch.nn.functional.softplus(reference.uncertainty_head(edge).squeeze(-1)) + torch.finfo(nodes.dtype).eps
    local = torch.tensor([[[[0.5, 0.5, 0.0, 0.0], [2.0, -1.0, 0.0, 0.0], [2.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]]]])
    selected_global = torch.tensor([[[[False, False, False, False], [False, False, False, False], [False, False, False, False], [True, True, False, False], [False, False, True, False]]]])
    global_weight = torch.softmax((gain - uncertainty).masked_fill(~selected_global, -torch.inf), dim=-1).masked_fill(~selected_global, 0.0)
    local_message = (local.unsqueeze(-1) * messages).sum(dim=-2)
    global_message = (global_weight.unsqueeze(-1) * messages).sum(dim=-2)
    local_available = local.ne(0).any(dim=-1)
    global_available = selected_global.any(dim=-1)
    expected = torch.where((local_available & global_available).unsqueeze(-1), 0.5 * local_message + 0.5 * global_message, torch.where(local_available.unsqueeze(-1), local_message, global_message))
    expected.square().mean().backward()

    for actual_head, expected_head in ((operator.gain_head, reference.gain_head), (operator.uncertainty_head, reference.uncertainty_head)):
        for actual, expected_parameter in zip(actual_head.parameters(), expected_head.parameters()):
            assert actual.grad is not None
            torch.testing.assert_close(actual.grad, expected_parameter.grad, rtol=1e-5, atol=1e-8)
