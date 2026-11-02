"""Batched group math using the original independent module parameters."""
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def grouped_linear(x: Tensor, layers: Sequence[nn.Linear]) -> Tensor:
    if not layers or x.shape[-2] != len(layers):
        raise ValueError('linear group count mismatch')
    weights = torch.stack([layer.weight for layer in layers])
    y = torch.einsum('...gi,goi->...go', x, weights)
    if layers[0].bias is not None:
        y = y + torch.stack([layer.bias for layer in layers])
    elif any(layer.bias is not None for layer in layers):
        raise ValueError('inconsistent grouped linear bias')
    return y


def grouped_layer_norm(x: Tensor, layers: Sequence[nn.LayerNorm]) -> Tensor:
    if not layers or x.shape[-2] != len(layers) or any(layer.eps != layers[0].eps for layer in layers):
        raise ValueError('layer norm group geometry mismatch')
    y = F.layer_norm(x, layers[0].normalized_shape, eps=layers[0].eps)
    if layers[0].elementwise_affine:
        y = y * torch.stack([layer.weight for layer in layers])
        if layers[0].bias is not None:
            y = y + torch.stack([layer.bias for layer in layers])
    return y


def grouped_mlp(x: Tensor, layers: Sequence[nn.Sequential]) -> Tensor:
    if not layers or any(len(layer) != len(layers[0]) for layer in layers):
        raise ValueError('MLP group structure mismatch')
    for index in range(len(layers[0])):
        modules = [layer[index] for layer in layers]
        module = modules[0]
        if any(type(m) is not type(module) for m in modules):
            raise ValueError('MLP group operation mismatch')
        if isinstance(module, nn.Linear): x = grouped_linear(x, modules)
        elif isinstance(module, nn.LayerNorm): x = grouped_layer_norm(x, modules)
        elif isinstance(module, (nn.GELU, nn.Tanh, nn.ReLU, nn.Identity)): x = module(x)
        else: raise TypeError(f'unsupported grouped operation {type(module).__name__}')
    return x
