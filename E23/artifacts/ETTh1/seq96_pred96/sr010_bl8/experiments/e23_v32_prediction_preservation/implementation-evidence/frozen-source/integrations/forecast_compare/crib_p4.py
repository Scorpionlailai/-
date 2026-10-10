"""Thin adapter for the official CRIB model on the locked ETTh1 P4 protocol."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch

from .protocol import P4Batch


def crib_input(batch: P4Batch, *, patch_len: int = 8) -> torch.Tensor:
    """Return zero-filled, train-standardized observations in CRIB patch order."""
    observed = batch.observed_mask
    values = batch.normalized_observed * batch.query_std[:, None, :] + batch.query_mean[:, None, :]
    values = torch.where(observed, values, torch.zeros_like(values))
    batch_size, length, variables = values.shape
    if length % patch_len:
        raise ValueError("CRIB requires the lookback length to divide into patches")
    return values.reshape(batch_size, length // patch_len, patch_len, variables).permute(0, 1, 3, 2).contiguous()


def _load_crib_class(root: Path):
    model_dir = Path(root).resolve() / "TSL_models"
    if not (model_dir / "CRIB.py").is_file():
        raise FileNotFoundError(f"official CRIB source missing: {model_dir / 'CRIB.py'}")
    package_name = "_p4_official_crib"
    package = ModuleType(package_name)
    package.__path__ = [str(model_dir)]
    sys.modules[package_name] = package
    module_name = f"{package_name}.CRIB"
    spec = importlib.util.spec_from_file_location(module_name, model_dir / "CRIB.py")
    if spec is None or spec.loader is None:
        raise ImportError("cannot load official CRIB module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.CRIB


def load_official_crib(root: Path, *, device: torch.device, batch_size: int = 128) -> torch.nn.Module:
    """Build unmodified upstream CRIB at its default capacity for P4's 96→96 task."""
    args = SimpleNamespace(
        patch_num=12, patch_len=8, pred_len=96, var_num=7,
        model_dim=32, heads_num=4, enc_num=3, dropout=0.1,
        output_attention=True, activation="relu", batch_size=batch_size,
        device=device,
    )
    return _load_crib_class(root)(args).to(device)


def crib_training_loss(output: tuple, target: torch.Tensor) -> torch.Tensor:
    """Upstream CRIB default: MAE prediction + consistency + 1e-6 KL."""
    encoding, _, augmented_encoding, _, prediction, _, kl = output
    return (
        torch.nn.functional.l1_loss(prediction, target)
        + torch.nn.functional.mse_loss(encoding, augmented_encoding)
        + 1e-6 * kl
    )


def verify_crib_selection(
    selection: dict,
    *,
    source_sha256: str,
    validation_mask_sha256: str,
    test_mask_sha256: str,
) -> None:
    """Refuse test access unless a full P4 run selected a matching checkpoint."""
    fields = (
        ("source_csv_sha256", source_sha256, "source CSV"),
        ("validation_mask_sha256", validation_mask_sha256, "validation mask"),
        ("test_mask_sha256", test_mask_sha256, "test mask"),
    )
    for field, expected, label in fields:
        if selection.get(field) != expected:
            raise ValueError(f"selected CRIB {label} SHA-256 mismatch")
    if selection.get("epochs_completed") != 100 or not 1 <= int(selection.get("best_epoch", 0)) <= 100:
        raise ValueError("selected CRIB run did not complete 100 epochs")
