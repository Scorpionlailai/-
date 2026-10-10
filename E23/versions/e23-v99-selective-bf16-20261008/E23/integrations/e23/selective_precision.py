"""Autocast only expensive repair modules during a delimited training forward.

Weights/Adam/gradients and the outer reconstruction/loss stay FP32. Validation,
repair audits, historical lookup and the forecast model keep original precision.
"""
from contextlib import contextmanager

import torch
from torch.utils._pytree import tree_map


class SelectiveRepairPrecision:
    def __init__(self, model, *, compile_readout=False):
        if not next(model.parameters()).is_cuda or not torch.cuda.is_bf16_supported():
            raise ValueError('Selective BF16 requires a compatible CUDA device')
        if getattr(model, '_selective_precision_installed', False):
            raise ValueError('Selective precision already installed')
        if model.front_context is None or model.point_candidate_readout is None:
            raise ValueError('Selective BF16 requires window context and point readout')
        self.active = False
        model._selective_precision_installed = True
        for module in (model.front_context, model.point_candidate_readout):
            original = module.forward

            def forward(*args, _original=original, **kwargs):
                if not self.active:
                    return _original(*args, **kwargs)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    result = _original(*args, **kwargs)
                return tree_map(lambda value: value.float() if isinstance(value, torch.Tensor)
                                and value.dtype == torch.bfloat16 else value, result)

            module.forward = forward
        if compile_readout:
            # Module.compile preserves parameter names and state_dict geometry.
            # First compilation and FP32 validation compilation count in timing.
            model.point_candidate_readout.compile(fullgraph=True, dynamic=False)

    @contextmanager
    def training(self):
        if self.active:
            raise RuntimeError('Nested selective precision scope')
        self.active = True
        try:
            yield
        finally:
            self.active = False
