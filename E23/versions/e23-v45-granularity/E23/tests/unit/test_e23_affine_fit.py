from types import SimpleNamespace

import torch

from integrations.e23 import train


def test_pairwise_affine_fit_ignores_hidden_training_values(monkeypatch):
    values = torch.tensor([[
        [0., 5., 1.], [1., 3., 2.], [2., 1., 1.],
        [3., -1., 3.], [4., -3., 2.], [5., -5., 4.],
    ]])
    observed = torch.ones_like(values, dtype=torch.bool)
    observed[:, 2, 1] = False
    batch = SimpleNamespace(normalized_target=values, observed_mask=observed)
    protocol = SimpleNamespace(config=SimpleNamespace(data=SimpleNamespace(
        datasets={"ETTh1": SimpleNamespace(n_vars=3)})))
    monkeypatch.setattr(train, "iter_p4_batches", lambda *args, **kwargs: [batch])
    slope, intercept = train._train_observed_pairwise_affine(protocol)
    assert slope[1, 0] < 0
    changed_values = values.clone()
    changed_values[~observed] = 100_000
    batch.normalized_target = changed_values
    changed_slope, changed_intercept = train._train_observed_pairwise_affine(protocol)
    torch.testing.assert_close(slope, changed_slope)
    torch.testing.assert_close(intercept, changed_intercept)
