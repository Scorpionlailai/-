"""Hand-derived segment moments catch bad units, merging, and weighting."""
import pytest
import torch
from integrations.e23.train import stage1_objective


def moments(candidate, target, hidden, std):
    try:
        from integrations.e23.long_gap import long_gap_moments
    except ImportError:
        pytest.fail('long-gap moment supervision has not been implemented')
    return long_gap_moments(candidate, target, hidden, std)


def test_affine_residual_has_correct_global_level_and_trend_amplitude():
    hidden = torch.ones(1, 16, 1, dtype=torch.bool)
    u = torch.linspace(-1, 1, 16, dtype=torch.float64)[None, :, None]
    candidate = ((1 + 2*u)/5).requires_grad_()
    std = torch.tensor([[5.]], dtype=torch.float64, requires_grad=True)
    level, trend, count, segments = moments(candidate, torch.zeros_like(candidate), hidden, std)
    torch.testing.assert_close(level, torch.tensor(1., dtype=torch.float64))
    torch.testing.assert_close(trend, torch.tensor(4., dtype=torch.float64))
    assert count.item() == 16 and segments.item() == 1
    (level + trend).backward()
    assert candidate.grad.abs().sum() > 0 and std.grad is None


def test_short_runs_are_excluded_and_visible_points_separate_long_runs():
    hidden = torch.ones(1, 33, 1, dtype=torch.bool)
    hidden[:, 15] = False  # left run15 excluded, right run17 included
    candidate = torch.ones(1, 33, 1)
    candidate[:, :15] = 1000
    level, trend, count, segments = moments(candidate, torch.zeros_like(candidate), hidden, torch.ones(1, 1))
    assert level.item() == pytest.approx(1.) and trend.item() == pytest.approx(0., abs=1e-12)
    assert count.item() == 17 and segments.item() == 1


def test_segment_moments_are_weighted_by_hidden_coordinate_count():
    hidden = torch.ones(1, 49, 1, dtype=torch.bool)
    hidden[:, 16] = False
    candidate = torch.ones(1, 49, 1)
    candidate[:, 17:] = 3
    level, trend, count, segments = moments(candidate, torch.zeros_like(candidate), hidden, torch.ones(1, 1))
    assert level.item() == pytest.approx(19/3)  # (16*1 + 32*9)/48
    assert trend.item() == pytest.approx(0., abs=1e-12)
    assert count.item() == 48 and segments.item() == 2


def test_observed_invalid_nan_coordinates_do_not_enter_segment_fitting():
    hidden = torch.zeros(1, 96, 1, dtype=torch.bool)
    hidden[:, 64:80] = True
    candidate = torch.full((1, 96, 1), float('nan'))
    target = torch.full_like(candidate, float('inf'))
    candidate[:, 64:80] = 2
    target[:, 64:80] = 0
    level, trend, count, segments = moments(candidate, target, hidden, torch.tensor([[.5]]))
    assert level.item() == pytest.approx(1.) and trend.item() == pytest.approx(0., abs=1e-12)
    assert count.item() == 16 and segments.item() == 1


def test_no_eligible_gap_has_finite_zero_loss_and_zero_gradient():
    candidate = torch.randn(2, 96, 3, requires_grad=True)
    hidden = torch.zeros_like(candidate, dtype=torch.bool)
    hidden[:, :15] = True
    level, trend, count, segments = moments(candidate, torch.zeros_like(candidate), hidden, torch.ones(2, 3))
    assert level.item() == trend.item() == count.item() == segments.item() == 0
    (level + trend).backward()
    assert torch.equal(candidate.grad, torch.zeros_like(candidate))


def test_ineligible_poisoned_runs_cannot_contaminate_loss_or_gradients():
    hidden = torch.ones(1, 33, 1, dtype=torch.bool)
    hidden[:, 15] = False
    candidate = torch.full((1, 33, 1), 2.)
    target = torch.zeros_like(candidate)
    candidate[:, :15] = float('nan')
    target[:, :15] = float('inf')
    candidate.requires_grad_()
    level, trend, count, segments = moments(candidate, target, hidden, torch.ones(1, 1))
    assert level.item() == pytest.approx(4.) and trend.item() == pytest.approx(0., abs=1e-12)
    assert count.item() == 17 and segments.item() == 1
    (level + trend).backward()
    assert torch.isfinite(candidate.grad).all() and candidate.grad[:, :16].eq(0).all()
    short = torch.full((1, 15, 1), float('inf'), requires_grad=True)
    a, b, count, segments = moments(short, torch.full_like(short, float('nan')), torch.ones_like(short, dtype=torch.bool), torch.ones(1, 1))
    assert a.item() == b.item() == count.item() == segments.item() == 0
    (a+b).backward()
    assert short.grad.eq(0).all()


def test_stage1_adds_only_declared_moments_to_the_original_objective():
    candidate = torch.zeros(1, 96, 1, dtype=torch.float64)
    candidate[:, :16] = 1 + 2*torch.linspace(-1, 1, 16, dtype=torch.float64)[None, :, None]
    observed = torch.ones_like(candidate, dtype=torch.bool)
    observed[:, :16] = False
    kwargs = dict(retrieval_weight=.1, loss_mode='global_mse_v1', query_std=torch.ones(1, 1), structure_mode='recent_mean_v2')
    old, old_parts = stage1_objective(candidate, torch.zeros_like(candidate), observed, torch.tensor(7.), **kwargs)
    new, parts = stage1_objective(candidate, torch.zeros_like(candidate), observed, torch.tensor(7.), **kwargs, gap_structure_mode='long_gap_moments_v1')
    torch.testing.assert_close(new - old, torch.tensor(1., dtype=torch.float64))
    for key, value in old_parts.items():
        torch.testing.assert_close(parts[key], value, rtol=0, atol=0)
    assert parts['gap_hidden_coordinates'].item() == 16
    assert parts['gap_segments'].item() == 1


def test_invalid_policy_and_non_global_objective_are_rejected():
    x = torch.zeros(1, 96, 1)
    for mode, loss in [('unknown', 'global_mse_v1'), ('long_gap_moments_v1', 'window_smooth_l1')]:
        with pytest.raises(ValueError):
            stage1_objective(x, x, torch.zeros_like(x, dtype=torch.bool), torch.tensor(0.), retrieval_weight=.1, query_std=torch.ones(1, 1), loss_mode=loss, gap_structure_mode=mode)
