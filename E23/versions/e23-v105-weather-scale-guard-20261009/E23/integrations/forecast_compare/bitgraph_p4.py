"""Adapter for the official BiTGraph model on locked ETTh1 P4 windows."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys

import torch

from .protocol import P4Batch


def bitgraph_inputs(batch: P4Batch) -> tuple[torch.Tensor, torch.Tensor]:
    mask = batch.observed_mask.to(dtype=batch.normalized_observed.dtype)
    values = batch.normalized_observed * batch.query_std[:, None, :] + batch.query_mean[:, None, :]
    return (values * mask).unsqueeze(-1), mask.unsqueeze(-1)


def load_official_bitgraph(root: Path, *, device: torch.device) -> torch.nn.Module:
    """Instantiate the exact small ETTh1 model used by upstream main.py."""
    if device.type != 'cuda' or device.index not in (None, 0):
        raise ValueError('official BiTGraph hardcodes CUDA device 0')
    root = Path(root).resolve()
    source = root / 'models' / 'BiaTCGNet' / 'BiaTCGNet.py'
    if not source.is_file():
        raise FileNotFoundError(f'official BiTGraph source missing: {source}')
    sys.path.insert(0, str(root))
    model_class = importlib.import_module('models.BiaTCGNet.BiaTCGNet').Model
    return model_class(
        True, True, 2, 7, [2, 3, 6, 7], 'cuda:0', predefined_A=None,
        dropout=0.3, subgraph_size=5, node_dim=3, dilation_exponential=1,
        conv_channels=8, residual_channels=8, skip_channels=16, end_channels=32,
        seq_length=96, in_dim=1, out_len=96, out_dim=1, layers=2,
        propalpha=0.05, tanhalpha=3, layer_norm_affline=True,
    ).to(device)
