import torch

from malr.models.historical_relation import HistoricalRelationOperator
from malr.models.online_history import HistoricalCandidateBatch


def _legacy_historical(
    operator: HistoricalRelationOperator, target: torch.Tensor, candidates: HistoricalCandidateBatch
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pre-plan H aggregation, intentionally independent of replay methods."""

    candidate_messages = operator.message_head(torch.cat((candidates.source_nodes, candidates.features), dim=-1))
    edge_input = torch.cat((target.unsqueeze(-2).expand_as(candidate_messages), candidate_messages, candidates.features), dim=-1)
    gain = operator.gain_head(edge_input).squeeze(-1)
    uncertainty = torch.nn.functional.softplus(operator.uncertainty_head(edge_input).squeeze(-1)) + 1e-6
    eligible = candidates.eligible.to(device=target.device, dtype=torch.bool)
    score = gain - uncertainty
    admissible = eligible & gain.gt(0.0)
    selected = torch.zeros_like(admissible)
    take = min(operator.max_items, score.shape[-1])
    if take:
        order = torch.argsort(score.masked_fill(~admissible, -torch.inf), dim=-1, descending=True, stable=True)
        chosen = order[..., :take]
        selected.scatter_(-1, chosen, admissible.gather(-1, chosen))
    coefficients = torch.softmax(score.masked_fill(~selected, -torch.inf), dim=-1).masked_fill(~selected, 0.0)
    available = selected.any(dim=-1)
    message = (coefficients.unsqueeze(-1) * candidate_messages).sum(dim=-2)
    return torch.where(available.unsqueeze(-1), message, torch.zeros_like(message)), coefficients, selected, available


def empty_historical_batch(target: torch.Tensor, width: int, feature_dim: int) -> HistoricalCandidateBatch:
    shape = (*target.shape[:3], width)
    return HistoricalCandidateBatch(
        source_nodes=target.new_zeros(*shape, target.shape[-1]),
        features=target.new_zeros(*shape, feature_dim),
        eligible=torch.zeros(*shape, dtype=torch.bool, device=target.device),
        candidate_count=torch.zeros(target.shape[:3], dtype=torch.long, device=target.device),
    )


def test_historical_relation_without_candidates_is_exact_zero() -> None:
    target = torch.randn(1, 2, 3, 4)
    candidates = empty_historical_batch(target, width=2, feature_dim=11)
    output = HistoricalRelationOperator(4, 11, max_items=2)(target, candidates)
    assert not output.available.any()
    assert torch.equal(output.message, torch.zeros_like(target))
    assert torch.equal(output.candidate_selected, torch.zeros_like(output.candidate_selected))


def test_historical_relation_candidate_messages_have_source_gradients() -> None:
    target = torch.randn(1, 1, 1, 4)
    source = torch.randn(1, 1, 1, 2, 4, requires_grad=True)
    candidates = HistoricalCandidateBatch(
        source_nodes=source,
        features=torch.randn(1, 1, 1, 2, 11),
        eligible=torch.ones(1, 1, 1, 2, dtype=torch.bool),
        candidate_count=torch.full((1, 1, 1), 2, dtype=torch.long),
    )
    output = HistoricalRelationOperator(4, 11, max_items=2)(target, candidates)
    output.candidate_messages.sum().backward()
    assert source.grad is not None
    assert torch.isfinite(source.grad).all()


def test_historical_relation_replays_its_live_aggregation_plan() -> None:
    """A missing plan or detached scoring weights would break this replay."""

    target = torch.randn(1, 1, 1, 4)
    candidates = HistoricalCandidateBatch(
        source_nodes=torch.randn(1, 1, 1, 2, 4),
        features=torch.randn(1, 1, 1, 2, 11),
        eligible=torch.ones(1, 1, 1, 2, dtype=torch.bool),
        candidate_count=torch.full((1, 1, 1), 2, dtype=torch.long),
    )
    operator = HistoricalRelationOperator(4, 11, max_items=2)
    with torch.no_grad():
        operator.gain_head[-1].weight.zero_()
        operator.gain_head[-1].bias.fill_(1.0)
        operator.uncertainty_head[-1].weight.zero_()
        operator.uncertainty_head[-1].bias.zero_()

    output, plan = operator.forward_with_plan(target, candidates)
    projected = operator.project_candidate_messages(target, candidates)
    replayed = operator.replay_message(projected, plan)
    expected_message, expected_coefficients, expected_selected, expected_available = _legacy_historical(operator, target, candidates)

    torch.testing.assert_close(replayed, output.message)
    torch.testing.assert_close(output.message, expected_message)
    torch.testing.assert_close(plan.coefficients, expected_coefficients)
    assert torch.equal(plan.selected, expected_selected)
    assert torch.equal(plan.available, expected_available)
    torch.testing.assert_close(
        plan.coefficients.sum(dim=-1)[plan.available],
        torch.ones_like(plan.available, dtype=target.dtype)[plan.available],
    )


def test_historical_forward_keeps_gain_and_uncertainty_gradients_live() -> None:
    """Detaching the ordinary plan would cut the Stage-1 weighting gradients."""

    target = torch.randn(1, 1, 1, 4)
    candidates = HistoricalCandidateBatch(
        source_nodes=torch.randn(1, 1, 1, 2, 4),
        features=torch.randn(1, 1, 1, 2, 11),
        eligible=torch.ones(1, 1, 1, 2, dtype=torch.bool),
        candidate_count=torch.full((1, 1, 1), 2, dtype=torch.long),
    )
    operator = HistoricalRelationOperator(4, 11, max_items=2)
    with torch.no_grad():
        operator.gain_head[-1].weight.zero_()
        operator.gain_head[-1].bias.fill_(1.0)
        operator.uncertainty_head[-1].weight.zero_()
        operator.uncertainty_head[-1].bias.zero_()

    reference = HistoricalRelationOperator(4, 11, max_items=2)
    reference.load_state_dict(operator.state_dict())

    operator(target, candidates).message.square().mean().backward()
    expected, _, _, _ = _legacy_historical(reference, target, candidates)
    expected.square().mean().backward()

    for actual_head, expected_head in ((operator.gain_head, reference.gain_head), (operator.uncertainty_head, reference.uncertainty_head)):
        for actual, expected_parameter in zip(actual_head.parameters(), expected_head.parameters()):
            assert actual.grad is not None
            torch.testing.assert_close(actual.grad, expected_parameter.grad, rtol=0.0, atol=0.0)
