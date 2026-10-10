"""Per-sample mask adapter for official ChannelTokenFormer_missing on P4."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

from .protocol import P4Batch


def ctf_inputs(batch: P4Batch) -> tuple[torch.Tensor, torch.Tensor]:
    mask = batch.observed_mask.to(dtype=batch.normalized_observed.dtype)
    values = batch.normalized_observed * batch.query_std[:, None, :] + batch.query_mean[:, None, :]
    return values * mask, mask


def group_by_full_patch_visibility(mask: torch.Tensor, *, patch_len: int = 24) -> list[torch.Tensor]:
    """Group samples so upstream's first-sample patch deletion is correct for all."""
    batch_size, length, variables = mask.shape
    if length % patch_len:
        raise ValueError('CTF lookback length must divide into patches')
    available = mask.bool().reshape(batch_size, length // patch_len, patch_len, variables).any(dim=2)
    groups: dict[tuple[bool, ...], list[int]] = {}
    for index, pattern in enumerate(available):
        key = tuple(pattern.flatten().tolist())
        groups.setdefault(key, []).append(index)
    return [torch.tensor(indices, dtype=torch.long, device=mask.device) for indices in groups.values()]


def load_official_ctf(root: Path, *, device: torch.device) -> torch.nn.Module:
    """Use upstream missing-aware model with its regular ETTh1 96→96 settings."""
    root = Path(root).resolve()
    source = root / 'models' / 'ChannelTokenFormer_missing.py'
    if not source.is_file():
        raise FileNotFoundError(f'official CTF source missing: {source}')
    sys.path.insert(0, str(root))
    module_name = '_p4_official_ctf_missing'
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError('cannot load official CTF missing model')
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    args = SimpleNamespace(
        task_name='long_term_forecast', features='M', seq_len=96, pred_len=96,
        use_norm=1, patch_lens=[24] * 7, sampling_periods=[1] * 7,
        enc_in=7, num_global_tokens=2, keep_prob=0.6, factor=3,
        dropout=0.1, d_model=256, n_heads=8, d_ff=1024,
        activation='gelu', batch_size=128, e_layers=2,
    )
    return module.Model(args).to(device)
