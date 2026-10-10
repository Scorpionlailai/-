from dataclasses import replace

import torch

from integrations.e23.train import stage1_objective, _trial_identity
from integrations.e23.config import E23Settings


def test_weaker_mean_term_changes_only_mean_supervision_and_gradient():
    torch.manual_seed(11)
    x = torch.randn(2, 96, 7, requires_grad=True)
    target = torch.randn_like(x)
    observed = torch.rand_like(x) > .7
    scales = torch.rand(2, 7) + .1
    args = dict(retrieval_weight=.1, loss_mode='global_mse_v1', query_std=scales)
    zero, _ = stage1_objective(x, target, observed, torch.tensor(.5), structure_mode='none', **args)
    strong, _ = stage1_objective(x, target, observed, torch.tensor(.5), structure_mode='recent_mean_v2', **args)
    weak, _ = stage1_objective(x, target, observed, torch.tensor(.5), structure_mode='recent_mean_weak_v3', **args)
    torch.testing.assert_close(weak - zero, .25 * (strong - zero))
    g0 = torch.autograd.grad(zero, x, retain_graph=True)[0]
    gs = torch.autograd.grad(strong, x, retain_graph=True)[0]
    gw = torch.autograd.grad(weak, x)[0]
    torch.testing.assert_close(gw - g0, .25 * (gs - g0), atol=1e-8, rtol=1e-5)
    assert torch.count_nonzero(gw[observed]) == 0
    torch.testing.assert_close(gw[:, :64], g0[:, :64])


def test_no_hidden_values_gives_zero_mean_gradient():
    x = torch.randn(2, 96, 7, requires_grad=True)
    loss, _ = stage1_objective(x, torch.zeros_like(x), torch.ones_like(x, dtype=torch.bool),
        torch.tensor(0.), retrieval_weight=.1, loss_mode='global_mse_v1', query_std=torch.ones(2, 7),
        structure_mode='recent_mean_weak_v3')
    loss.backward()
    assert loss == 0 and torch.count_nonzero(x.grad) == 0
