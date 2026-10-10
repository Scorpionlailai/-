import pytest
import torch

from malr.models.filtering import (
    BilateralContextEncoder,
    EdgeConfidenceFilter,
    hard_threshold_topk,
)


def test_bilateral_summary_uses_visible_sources_on_each_side() -> None:
    module = BilateralContextEncoder(d_model=1, hidden_multiplier=2)
    visible = torch.tensor([[[[2.0], [100.0], [8.0], [14.0]]]])
    mask_embedding = torch.zeros_like(visible)
    prior = torch.zeros_like(visible)
    rho = torch.tensor([[[1.0, 0.0, 0.5, 1.0]]])
    target = torch.tensor([[[False, True, False, False]]])

    output = module(visible, mask_embedding, prior, rho, target)

    torch.testing.assert_close(output.left[0, 0, 1], torch.tensor([2.0]))
    expected_right = ((0.5 / 2.0) * 8.0 + (1.0 / 3.0) * 14.0) / (
        0.5 / 2.0 + 1.0 / 3.0
    )
    torch.testing.assert_close(
        output.right[0, 0, 1], torch.tensor([expected_right])
    )
    assert output.left_available[0, 0, 1]
    assert output.right_available[0, 0, 1]


def test_bilateral_summary_marks_an_absent_side() -> None:
    module = BilateralContextEncoder(d_model=2, hidden_multiplier=2)
    visible = torch.randn(1, 1, 3, 2)
    rho = torch.tensor([[[0.0, 0.0, 1.0]]])
    target = torch.tensor([[[True, False, False]]])

    output = module(
        visible,
        torch.zeros_like(visible),
        torch.zeros_like(visible),
        rho,
        target,
    )

    torch.testing.assert_close(output.left[0, 0, 0], torch.zeros(2))
    assert not output.left_available[0, 0, 0]
    assert output.right_available[0, 0, 0]


def test_context_encoder_zeroes_queries_for_non_target_nodes() -> None:
    torch.manual_seed(3)
    module = BilateralContextEncoder(d_model=4, hidden_multiplier=2)
    visible = torch.randn(1, 1, 3, 4)
    rho = torch.tensor([[[1.0, 0.0, 1.0]]])
    target = torch.tensor([[[False, True, False]]])
    output = module(
        visible,
        torch.zeros_like(visible),
        torch.zeros_like(visible),
        rho,
        target,
    )

    torch.testing.assert_close(output.query[:, :, 0], torch.zeros(1, 1, 4))
    torch.testing.assert_close(output.query[:, :, 2], torch.zeros(1, 1, 4))


@pytest.mark.parametrize(
    ("rho", "match"),
    [
        (torch.tensor([[[float("nan")]]]), "finite"),
        (torch.tensor([[[1.1]]]), r"\[0, 1\]"),
    ],
)
def test_context_encoder_rejects_invalid_observed_fraction(
    rho: torch.Tensor, match: str
) -> None:
    module = BilateralContextEncoder(d_model=2)
    visible = torch.zeros(1, 1, 1, 2)

    with pytest.raises(ValueError, match=match):
        module(
            visible,
            torch.zeros_like(visible),
            torch.zeros_like(visible),
            rho,
            torch.ones(1, 1, 1, dtype=torch.bool),
        )


def test_hard_threshold_topk_rejects_low_confidence_and_allows_zero() -> None:
    probabilities = torch.tensor([[0.9, 0.49, 0.8, 0.1]])
    base_scores = torch.tensor([[1.0, 4.0, 2.0, 3.0]])
    eligible = torch.ones_like(probabilities, dtype=torch.bool)

    gate, selected = hard_threshold_topk(
        probabilities, base_scores, eligible, threshold=0.5, max_edges=1
    )
    empty_gate, empty_selected = hard_threshold_topk(
        torch.full_like(probabilities, 0.1),
        base_scores,
        eligible,
        threshold=0.5,
        max_edges=2,
    )

    assert gate.tolist() == [[True, False, False, False]]
    assert selected.tolist() == [[0]]
    assert not empty_gate.any()
    assert empty_selected.tolist() == [[-1, -1]]


def test_equal_probabilities_use_base_score_then_index() -> None:
    probabilities = torch.full((1, 4), 0.9)
    base_scores = torch.tensor([[2.0, 3.0, 3.0, 1.0]])
    eligible = torch.ones_like(probabilities, dtype=torch.bool)

    _, selected = hard_threshold_topk(
        probabilities, base_scores, eligible, threshold=0.5, max_edges=2
    )

    assert selected.tolist() == [[1, 2]]


def test_filter_initialization_reproduces_base_topk_and_is_binary() -> None:
    module = EdgeConfidenceFilter(
        d_model=2,
        auxiliary_dim=1,
        max_edges=2,
        threshold=0.5,
        hidden_multiplier=2,
        dropout=0.0,
        temperature=1.0,
        initial_keep_probability=0.9,
    )
    result = module(
        torch.zeros(1, 2),
        torch.zeros(1, 4, 2),
        torch.zeros(1, 4, 1),
        torch.tensor([[1.0, 4.0, 3.0, 2.0]]),
        torch.ones(1, 4, dtype=torch.bool),
    )

    assert result.selected.tolist() == [[1, 2]]
    assert set(result.hard_gate.unique().tolist()) <= {False, True}
    torch.testing.assert_close(
        result.probabilities,
        torch.full_like(result.probabilities, 0.9),
    )
    torch.testing.assert_close(result.weights.sum(-1), torch.ones(1))
    assert result.keep_loss.item() == pytest.approx(0.9)


def test_filter_with_no_eligible_candidate_is_finite_zero() -> None:
    module = EdgeConfidenceFilter(
        d_model=2,
        auxiliary_dim=1,
        max_edges=4,
        threshold=0.5,
        hidden_multiplier=2,
        dropout=0.0,
        temperature=1.0,
        initial_keep_probability=0.9,
    )
    result = module(
        torch.zeros(1, 2),
        torch.zeros(1, 2, 2),
        torch.zeros(1, 2, 1),
        torch.zeros(1, 2),
        torch.zeros(1, 2, dtype=torch.bool),
    )

    torch.testing.assert_close(result.weights, torch.zeros(1, 2))
    torch.testing.assert_close(result.probabilities, torch.zeros(1, 2))
    assert result.selected.tolist() == [[-1, -1, -1, -1]]
    assert result.keep_loss.item() == 0.0
    assert torch.isfinite(result.keep_loss)


def test_straight_through_filter_reaches_confidence_parameters() -> None:
    torch.manual_seed(5)
    module = EdgeConfidenceFilter(
        d_model=2,
        auxiliary_dim=1,
        max_edges=1,
        threshold=0.5,
        hidden_multiplier=2,
        dropout=0.0,
        temperature=1.0,
        initial_keep_probability=0.9,
    ).train()
    result = module(
        torch.randn(1, 2),
        torch.randn(1, 3, 2),
        torch.randn(1, 3, 1),
        torch.tensor([[3.0, 2.0, 1.0]]),
        torch.ones(1, 3, dtype=torch.bool),
    )

    loss = (result.weights * torch.tensor([[1.0, -1.0, 2.0]])).sum()
    loss.backward()

    gradient = module.network[-1].weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0
