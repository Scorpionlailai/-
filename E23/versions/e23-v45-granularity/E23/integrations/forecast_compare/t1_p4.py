"""Official T1 imputation followed by the locked P4 PatchTST forecast head."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from torch import Tensor, nn

from malr.data.patching import patchify_preserving_values

from .protocol import P4Batch


T1_SETTINGS = {
    'task_name': 'imputation', 'seq_len': 96, 'pred_len': 96, 'enc_in': 7,
    'n_heads': 128, 'patch_size': 2, 'patch_stride': 1,
    'n_blocks': [2, 2], 'kernel_size_large': [71, 31],
    'kernel_size_small': 5, 'ffn_ratio': 1.0, 'downsample_ratio': 2,
    'positional_encoding': True, 'qkv_bias': True,
    'cosine_attention': False, 'use_head_reconstruction': True,
    'head_params_shared': True, 'imputation_use_mask_embedding': False,
    'drop_attn': 0.1, 'drop_ffn': 0.0, 'drop_proj': 0.0,
    'drop_path': 0.1, 'drop_head': 0.0, 'SACA': False,
}


def load_official_t1(root: Path, *, device: torch.device) -> nn.Module:
    source = Path(root).resolve() / 'models' / 'T1.py'
    if not source.is_file():
        raise FileNotFoundError(f'official T1 source missing: {source}')
    name = '_p4_official_t1'
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise ImportError('cannot load official T1 source')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.Model(SimpleNamespace(**T1_SETTINGS)).to(device)


def t1_inputs(batch: P4Batch) -> tuple[Tensor, Tensor]:
    """Pass observed train-standardized values and observed=1 mask only."""
    mask = batch.observed_mask.to(batch.normalized_observed.dtype)
    standardized = batch.normalized_observed * batch.query_std[:, None, :] + batch.query_mean[:, None, :]
    return standardized * mask, mask


def t1_clean_history(batch: P4Batch) -> Tensor:
    """Training target only; never use this tensor as a model input."""
    return batch.normalized_target * batch.query_std[:, None, :] + batch.query_mean[:, None, :]


def t1_repair_loss(model: nn.Module, batch: P4Batch) -> Tensor:
    values, mask = t1_inputs(batch)
    candidate = model(values, mask=mask)
    hidden = ~batch.observed_mask
    if not hidden.any():
        raise ValueError('T1 repair training requires hidden coordinates')
    return (candidate - t1_clean_history(batch)).square()[hidden].mean()


class T1ForecastModel(nn.Module):
    def __init__(self, imputer: nn.Module, predictor: nn.Module) -> None:
        super().__init__()
        self.imputer = imputer
        self.predictor = predictor

    def forward(self, batch: P4Batch) -> Tensor:
        values, mask = t1_inputs(batch)
        candidate = self.imputer(values, mask=mask)
        repaired_standardized = torch.where(batch.observed_mask, values, candidate)
        repaired_normalized = (repaired_standardized - batch.query_mean[:, None, :]) / batch.query_std[:, None, :]
        # Preserve observed coordinates bit for bit in the predictor input.
        repaired_normalized = torch.where(batch.observed_mask, batch.normalized_observed, repaired_normalized)
        patches = patchify_preserving_values(repaired_normalized, batch.observed_mask, length=16, stride=8)
        return self.predictor(
            patches.values, patches.effective_mask, batch.query_mean, batch.query_std,
            patch_validity=patches.validity,
        )
