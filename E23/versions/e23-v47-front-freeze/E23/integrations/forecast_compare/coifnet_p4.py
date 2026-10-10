"""Adapter for the unmodified official CoIFNet on locked ETTh1 P4 windows."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import torch
from omegaconf import OmegaConf
import pandas as pd

from .protocol import P4Batch


def coifnet_inputs(batch: P4Batch, time_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return observed standardized values, visibility, and known history timestamps."""
    mask = batch.observed_mask.to(dtype=batch.normalized_observed.dtype)
    values = batch.normalized_observed * batch.query_std[:, None, :] + batch.query_mean[:, None, :]
    values = values * mask
    if len(batch.refs) != values.shape[0]:
        raise ValueError('each CoIFNet sample needs a window reference')
    features = torch.stack([time_features[ref.input_start:ref.input_end] for ref in batch.refs])
    if features.shape[:2] != values.shape[:2]:
        raise ValueError('CoIFNet timestamp features do not align with history')
    return values, mask, features.to(values.device)


def _load_coifnet_class(root: Path):
    model_dir = Path(root).resolve() / 'model'
    source = model_dir / 'CoIFNet.py'
    if not source.is_file():
        raise FileNotFoundError(f'official CoIFNet source missing: {source}')
    package_name = '_p4_official_coifnet'
    package = ModuleType(package_name)
    package.__path__ = [str(model_dir)]
    sys.modules[package_name] = package
    module_name = f'{package_name}.CoIFNet'
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError('cannot load official CoIFNet module')
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.CoIFNet


def load_official_coifnet(root: Path, *, device: torch.device) -> torch.nn.Module:
    model_cfg = OmegaConf.load(Path(root) / 'conf' / 'model' / 'CoIFNet.yaml')
    cfg = OmegaConf.create({
        'model': OmegaConf.to_container(model_cfg, resolve=True),
        'dataset': {'channels': 7},
        'device': str(device),
    })
    return _load_coifnet_class(root)(cfg, seq_len=96, pred_len=192).to(device)


def load_timestamp_features(csv_path: Path) -> torch.Tensor:
    """Match upstream ETTh1 simple timestamps: day-of-month mod 7, hour."""
    dates = pd.to_datetime(pd.read_csv(csv_path, usecols=['date'])['date'])
    return torch.tensor(
        list(zip((dates.dt.day % 7).tolist(), dates.dt.hour.tolist())),
        dtype=torch.long,
    )


def coifnet_training_loss(
    prediction: torch.Tensor,
    observed_values: torch.Tensor,
    observed_mask: torch.Tensor,
    forecast_target: torch.Tensor,
) -> torch.Tensor:
    """Official CoIFNet 0.1 forecast MAE + 0.9 observed-history reconstruction MAE."""
    history_length = observed_values.shape[1]
    history_prediction = prediction[:, :history_length]
    future_prediction = prediction[:, history_length:]
    if history_prediction.shape != observed_values.shape or future_prediction.shape != forecast_target.shape:
        raise ValueError('CoIFNet output does not match P4 history and horizon')
    reconstruction = (observed_mask * (history_prediction - observed_values)).abs().sum()
    reconstruction = reconstruction / observed_mask.sum().clamp_min(1)
    forecast_mae = torch.nn.functional.l1_loss(future_prediction, forecast_target)
    return 0.1 * forecast_mae + 0.9 * reconstruction
