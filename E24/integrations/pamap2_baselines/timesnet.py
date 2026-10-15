"""TimesNet author imputation branch, with a sealed empty-variable fallback."""
import importlib
import inspect
from pathlib import Path
import sys
import textwrap
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch import nn

from .data import ROOT


class TimesNetAdapter(nn.Module):
    def __init__(self, data):
        super().__init__()
        self.data = data
        source = ROOT / 'TimesNet'
        sys.path.insert(0, str(source))
        module = importlib.import_module('models.TimesNet')
        if Path(module.__file__).resolve() != source / 'models/TimesNet.py':
            raise RuntimeError('wrong TimesNet source')
        self.config = dict(task_name='imputation', seq_len=96, label_len=0, pred_len=0,
            e_layers=2, top_k=3, num_kernels=6, enc_in=data.n_vars, c_out=data.n_vars,
            d_model=16 if data.dataset == 'ETTh1' else 64,
            d_ff=32 if data.dataset == 'ETTh1' else 64, dropout=.1, embed='timeF', freq='h')
        self.core = module.Model(SimpleNamespace(**self.config))
        # Counts alone change; original finite-observation behavior is identical.
        # Empty variables have zero mean and variance floor 1e-5, using no truth.
        method = textwrap.dedent(inspect.getsource(module.Model.imputation))
        if method.count('torch.sum(mask == 1, dim=1)') != 2:
            raise ValueError('TimesNet normalization patch no longer matches upstream')
        method = method.replace('torch.sum(mask == 1, dim=1)', 'torch.sum(mask == 1, dim=1).clamp_min(1)')
        namespace = dict(vars(module))
        exec(compile(method, str(source / 'models/TimesNet.py'), 'exec'), namespace)
        self.core.imputation = namespace['imputation'].__get__(self.core)
        from utils.timefeatures import time_features
        # Match author script default freq=h, including Weather. Actual CSV
        # timestamps and row order are retained; no data-dependent features.
        self.time_marks = np.zeros((len(data.values),4),dtype=np.float32)

    def forward(self, values, observed, starts):
        marks = torch.from_numpy(np.stack([self.time_marks[s:s+96] for s in starts])).to(values.device)
        candidate = self.core(values, marks, None, None, observed.to(values.dtype))
        return torch.where(observed, values, candidate)

    def loss(self, truth, observed, starts):
        values = torch.where(observed, truth, 0.)
        repaired = self(values, observed, starts)
        from .losses import hidden_error
        valid = torch.from_numpy(self.data.valid_for(starts)).to(truth.device)
        return hidden_error(repaired, truth, observed, valid).square().mean()
