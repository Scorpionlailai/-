from __future__ import annotations

import pytest
import torch

from malr.models.graph_recovery import RelationAwareResidualRecovery


def _inputs() -> tuple[torch.Tensor, ...]:
    nodes = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    source_nodes = torch.tensor(
        [[[[[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]], [[1.0, 0.0], [1.0, 2.0], [1.0, 3.0]]]]]
    )
    edge_features = torch.zeros(1, 1, 2, 3, 2)
    edge_type = torch.tensor([[[[0, 1, 2], [0, 1, 2]]]])
    predicted_gain = torch.tensor([[[[0.1, 0.2, 0.9], [0.5, 0.4, 0.3]]]])
    admitted = torch.ones(1, 1, 2, 3, dtype=torch.bool)
    route_index = torch.tensor([[[4, 0]]])  # historical + temporal, then none
    target_damaged = torch.ones(1, 1, 2, dtype=torch.bool)
    return (
        nodes,
        source_nodes,
        edge_features,
        edge_type,
        predicted_gain,
        admitted,
        route_index,
        target_damaged,
    )


def test_graph_recovery_uses_one_unified_softmax_and_route_type_mask() -> None:
    module = RelationAwareResidualRecovery(d_model=2, edge_feature_dim=2, hidden_dim=4)
    inputs = _inputs()
    source_before = inputs[1].clone()

    output = module(*inputs)

    assert output.active_edges[0, 0, 0].tolist() == [True, True, False]
    assert not output.active_edges[0, 0, 1].any()
    torch.testing.assert_close(output.weights[0, 0, 0, 2], torch.tensor(0.0))
    torch.testing.assert_close(
        output.weights[0, 0, 0, :2].sum(), torch.tensor(1.0), atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(output.weights[0, 0, 1], torch.zeros(3))
    torch.testing.assert_close(output.message[0, 0, 1], torch.zeros(2))
    torch.testing.assert_close(output.residual[0, 0, 1], torch.zeros(2))
    torch.testing.assert_close(inputs[1], source_before)


def test_graph_recovery_has_exact_zero_residual_without_admitted_edges() -> None:
    inputs = list(_inputs())
    inputs[5] = torch.zeros_like(inputs[5])
    module = RelationAwareResidualRecovery(d_model=2, edge_feature_dim=2)

    output = module(*inputs)

    assert not output.active_edges.any()
    assert torch.equal(output.message, torch.zeros_like(output.message))
    assert torch.equal(output.residual, torch.zeros_like(output.residual))
    assert torch.equal(output.latents, inputs[0])


def test_graph_recovery_rejects_explicit_self_loop_edges() -> None:
    inputs = list(_inputs())
    self_edge = torch.zeros_like(inputs[5])
    self_edge[..., 0] = True
    with pytest.raises(ValueError, match="self-loop"):
        RelationAwareResidualRecovery(d_model=2, edge_feature_dim=2)(*inputs, self_edge=self_edge)


def test_inactive_extreme_gain_has_finite_backward_gradient() -> None:
    inputs = list(_inputs())
    predicted_gain = torch.zeros_like(inputs[4])
    predicted_gain[0, 0, 0] = torch.tensor([0.1, 1_000.0, 1_000.0])
    predicted_gain.requires_grad_()
    admitted = torch.zeros_like(inputs[5])
    admitted[0, 0, 0, 0] = True
    inputs[4] = predicted_gain
    inputs[5] = admitted
    module = RelationAwareResidualRecovery(d_model=2, edge_feature_dim=2)

    output = module(*inputs)
    (output.latents.sum() + output.weights.sum()).backward()

    assert torch.isfinite(output.latents).all()
    assert predicted_gain.grad is not None
    assert torch.isfinite(predicted_gain.grad).all()
