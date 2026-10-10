import copy

import pytest
import torch
from torch import nn


@pytest.mark.parametrize('kind', ['linear', 'norm', 'mlp'])
def test_grouped_operations_do_not_alias_or_detach_parameters(kind):
    from malr.models.e23.group_ops import grouped_linear, grouped_layer_norm, grouped_mlp
    torch.manual_seed(6)
    constructors = {'linear': lambda: nn.Linear(5, 3), 'norm': lambda: nn.LayerNorm(5),
                    'mlp': lambda: nn.Sequential(nn.Linear(5, 7), nn.GELU(), nn.Linear(7, 3))}
    modules = [constructors[kind]().double() for _ in range(2)]
    references = copy.deepcopy(modules)
    functions = {'linear': grouped_linear, 'norm': grouped_layer_norm, 'mlp': grouped_mlp}
    x = torch.randn(2, 3, 2, 5, dtype=torch.double, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    actual = functions[kind](x, modules)
    expected = torch.stack([m(y[..., i, :]) for i, m in enumerate(references)], -2)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-10)
    actual.square().sum().backward(); expected.square().sum().backward()
    torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=1e-10)
    for module, reference in zip(modules, references):
        for p, q in zip(module.parameters(), reference.parameters()):
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=1e-10)
    first, second = next(modules[0].parameters()), next(modules[1].parameters())
    assert first.data_ptr() != second.data_ptr()
    with torch.no_grad(): first.add_(1)
    after = functions[kind](x, modules)
    torch.testing.assert_close(after[..., 1, :], actual[..., 1, :], rtol=0, atol=0)
    assert not torch.allclose(after[..., 0, :], actual[..., 0, :])
