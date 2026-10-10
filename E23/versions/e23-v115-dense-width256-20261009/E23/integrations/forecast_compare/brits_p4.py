"""SAITS repository's BRITS implementation adapted to locked P4 windows."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys

import torch
from torch import Tensor, nn

from malr.data.patching import patchify_preserving_values

from .protocol import P4Batch


BRITS_SETTINGS = {'seq_len': 96, 'feature_num': 7, 'rnn_hidden_size': 128, 'MIT': True}


def load_brits(root: Path, *, device: torch.device) -> nn.Module:
    source = Path(root).resolve() / 'modeling' / 'brits.py'
    if not source.is_file():
        raise FileNotFoundError(f'BRITS source missing: {source}')
    sys.path.insert(0, str(root.resolve()))
    module = importlib.import_module('modeling.brits')
    if Path(module.__file__).resolve() != source:
        raise RuntimeError(f'loaded BRITS from wrong source: {module.__file__}')
    return module.BRITS(**BRITS_SETTINGS, device=device).to(device)


def brits_deltas(mask: Tensor) -> Tensor:
    """Same recursion as SAITS modeling.unified_dataloader.parse_delta."""
    deltas = [torch.zeros_like(mask[:, 0])]
    for step in range(1, mask.shape[1]):
        deltas.append(torch.ones_like(mask[:, step]) + (1 - mask[:, step]) * deltas[-1])
    return torch.stack(deltas, dim=1)


def brits_inputs(batch: P4Batch, *, supervised: bool) -> dict:
    mask = batch.observed_mask.to(batch.normalized_observed.dtype)
    values = (batch.normalized_observed * batch.query_std[:, None, :]
              + batch.query_mean[:, None, :]) * mask
    backwards_mask = torch.flip(mask, dims=(1,))
    data = {
        'forward': {'X': values, 'missing_mask': mask,
                    'deltas': brits_deltas(mask)},
        'backward': {'X': torch.flip(values, dims=(1,)),
                     'missing_mask': backwards_mask,
                     'deltas': brits_deltas(backwards_mask)},
    }
    if supervised:
        data['X_holdout'] = (batch.normalized_target * batch.query_std[:, None, :]
                             + batch.query_mean[:, None, :])
        data['indicating_mask'] = 1 - mask
    return data


def brits_repair_loss(model: nn.Module, batch: P4Batch) -> Tensor:
    result = model(brits_inputs(batch, supervised=True), 'train')
    return result['consistency_loss'] + result['reconstruction_loss'] + result['imputation_loss']


class BritsForecastModel(nn.Module):
    def __init__(self, imputer: nn.Module, predictor: nn.Module) -> None:
        super().__init__()
        self.imputer = imputer
        self.predictor = predictor

    def forward(self, batch: P4Batch) -> Tensor:
        candidate, _ = self.imputer.impute(brits_inputs(batch, supervised=False))
        normalized_candidate = ((candidate - batch.query_mean[:, None, :])
                                / batch.query_std[:, None, :])
        repaired = torch.where(batch.observed_mask, batch.normalized_observed,
                               normalized_candidate)
        patches = patchify_preserving_values(repaired, batch.observed_mask, length=16, stride=8)
        return self.predictor(
            patches.values, patches.effective_mask, batch.query_mean, batch.query_std,
            patch_validity=patches.validity,
        )
