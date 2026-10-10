import pytest
import torch

from malr.models.e23.scale_guard import compress_hidden_normalized_tails


def test_only_hidden_tails_change_and_extreme_gradients_remain_finite():
    x = torch.tensor([-1000., -16., -1., 0., 1., 16., 1000., 500., float('nan')],
                     dtype=torch.float64, requires_grad=True)
    observed = torch.tensor([False]*7+[True, False])
    valid = torch.tensor([True]*8+[False])
    y = compress_hidden_normalized_tails(x, observed, valid)
    assert torch.equal(y[1:6], x[1:6])
    assert y[7] == x[7]
    assert torch.isnan(y[8])
    assert 16 < y[6] < 24
    assert torch.equal(y[0], -y[6])
    y[:8].sum().backward()
    assert torch.isfinite(x.grad).all()
    assert (x.grad[:8] > 0).all()
    assert x.grad[8] == 0


def test_value_and_first_derivative_are_continuous_at_boundary():
    x = torch.tensor([16-1e-7, 16., 16+1e-7], dtype=torch.float64, requires_grad=True)
    y = compress_hidden_normalized_tails(x, torch.zeros_like(x, dtype=torch.bool),
                                        torch.ones_like(x, dtype=torch.bool))
    y.sum().backward()
    assert torch.allclose(y, x, atol=1e-12, rtol=0)
    assert torch.allclose(x.grad, torch.ones_like(x), atol=1e-12, rtol=0)


def test_hidden_truth_is_not_an_input_and_no_parameters_are_added():
    from malr.models.e23.recent_mean import RecentMeanProjector
    old = RecentMeanProjector(21, mode='global_dense_time_v4')
    new = RecentMeanProjector(21, mode='global_dense_guard_v5')
    assert new.normalized_output_guard and not old.normalized_output_guard
    assert sum(p.numel() for p in old.parameters()) == sum(p.numel() for p in new.parameters()) == 109719
    new.load_state_dict(old.state_dict(), strict=True)


@pytest.mark.parametrize('threshold', [0., -1., float('inf'), float('nan')])
def test_invalid_threshold_rejected(threshold):
    x = torch.zeros(2)
    with pytest.raises(ValueError):
        compress_hidden_normalized_tails(x, x.bool(), x.bool(), threshold=threshold)
