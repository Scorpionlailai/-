from __future__ import annotations

import pytest
import torch

from malr.models.branch_recovery import BranchRoutedResidualRecovery


def test_recovery_materializes_all_eight_counterfactuals() -> None:
    """A selected triple route must be one of eight independently fused outputs."""

    module = BranchRoutedResidualRecovery(d_model=2, hidden_dim=4)
    nodes = torch.zeros(1, 1, 1, 2)
    messages = torch.tensor([[[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]]])
    scores = torch.tensor([[[[0.2, 0.3, 0.4]]]])
    available = torch.ones(1, 1, 1, 3, dtype=torch.bool)

    output = module(
        nodes,
        messages,
        scores,
        available,
        torch.tensor([[[7]]]),
        torch.ones(1, 1, 1, dtype=torch.bool),
    )

    assert output.route_latents.shape == (1, 1, 1, 8, 2)
    assert output.route_messages.shape == (1, 1, 1, 8, 2)
    assert output.route_weights.shape == (1, 1, 1, 8, 3)
    assert output.route_weights[0, 0, 0, 0].sum().item() == 0.0
    assert output.route_weights[0, 0, 0, 7].sum().item() == pytest.approx(1.0)
    torch.testing.assert_close(output.latents, output.route_latents[..., 7, :])
    torch.testing.assert_close(output.message, output.route_messages[..., 7, :])


def test_recovery_excludes_unavailable_relations_from_route_weights() -> None:
    """A masked temporal branch cannot receive weight even with the largest score."""

    module = BranchRoutedResidualRecovery(d_model=2, hidden_dim=4)
    output = module(
        torch.zeros(1, 1, 1, 2),
        torch.tensor([[[[[1.0, 0.0], [0.0, 1.0], [2.0, 0.0]]]]]),
        torch.tensor([[[[0.1, 1000.0, 0.2]]]]),
        torch.tensor([[[[True, False, True]]]]),
        torch.tensor([[[7]]]),
        torch.ones(1, 1, 1, dtype=torch.bool),
    )

    assert output.active_relations[0, 0, 0].tolist() == [True, False, True]
    torch.testing.assert_close(output.weights[0, 0, 0, 1], torch.tensor(0.0))
    torch.testing.assert_close(output.weights.sum(dim=-1), torch.ones(1, 1, 1))


def test_recovery_writes_no_residual_for_a_clean_target() -> None:
    """The recovery branch may only change targets marked damaged."""

    module = BranchRoutedResidualRecovery(d_model=2, hidden_dim=4)
    nodes = torch.tensor([[[[3.0, -2.0]]]])
    output = module(
        nodes,
        torch.randn(1, 1, 1, 3, 2),
        torch.ones(1, 1, 1, 3),
        torch.ones(1, 1, 1, 3, dtype=torch.bool),
        torch.tensor([[[7]]]),
        torch.zeros(1, 1, 1, dtype=torch.bool),
    )

    assert torch.equal(output.residual, torch.zeros_like(output.residual))
    assert torch.equal(output.latents, nodes)


def test_candidate_recovery_uses_the_same_residual_mlp() -> None:
    """Candidate counterfactuals retain their candidate axis after recovery."""

    module = BranchRoutedResidualRecovery(2, hidden_dim=4)
    nodes = torch.zeros(1, 1, 1, 2)
    candidates = torch.randn(1, 1, 1, 3, 2)

    recovered = module.recover_candidates(
        nodes,
        candidates,
        torch.ones(1, 1, 1, dtype=torch.bool),
    )

    assert recovered.shape == candidates.shape


def test_inactive_extreme_branch_score_has_finite_backward_gradient() -> None:
    """Masked huge scores must not poison the fused route softmax with NaNs."""

    module = BranchRoutedResidualRecovery(2, hidden_dim=4)
    scores = torch.tensor([[[[0.1, 1_000.0, 1_000.0]]]], requires_grad=True)
    output = module(
        torch.zeros(1, 1, 1, 2),
        torch.ones(1, 1, 1, 3, 2),
        scores,
        torch.tensor([[[[True, False, False]]]]),
        torch.tensor([[[1]]]),
        torch.ones(1, 1, 1, dtype=torch.bool),
    )
    (output.latents.sum() + output.weights.sum()).backward()

    assert torch.isfinite(output.latents).all()
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()
