"""Adapter for pinned upstream SAITS; upstream source is never copied or edited."""

from __future__ import annotations

import importlib
from hashlib import sha256
from pathlib import Path

import torch
from torch import Tensor, nn

from malr.data.patching import patchify_preserving_values

from .protocol import P4Batch


SAITS_COMMIT = "660b87f19c1277065f314f24134f646229e89ca9"
SAITS_SOURCE_SHA256 = "7c52cbcff6d2a223f1e788efabf8d46ebc84f79d9d480d59ae76619b7d31a556"
SAITS_SETTINGS = {
    "n_groups": 2,
    "n_group_inner_layers": 1,
    "d_time": 96,
    "d_feature": 7,
    "d_model": 256,
    "d_inner": 128,
    "n_head": 4,
    "d_k": 64,
    "d_v": 64,
    "dropout": 0.1,
    "input_with_mask": True,
    "param_sharing_strategy": "inner_group",
    "MIT": True,
    "diagonal_attention_mask": True,
}


def build_official_saits(repo_root: Path, *, device: torch.device) -> nn.Module:
    expected = Path(repo_root).resolve() / "modeling" / "saits.py"
    if not expected.is_file():
        raise ValueError(f"official SAITS source missing: {expected}")
    if sha256(expected.read_bytes()).hexdigest() != SAITS_SOURCE_SHA256:
        raise ValueError(f"official SAITS source SHA-256 mismatch: {expected}")
    try:
        module = importlib.import_module("modeling.saits")
    except ImportError as exc:
        raise RuntimeError(
            "SAITS dependencies are unavailable. Launch in an isolated SAITS environment "
            "with the pinned external/SAITS root on PYTHONPATH; do not alter malr CUDA/PyTorch."
        ) from exc
    if Path(module.__file__).resolve() != expected:
        raise RuntimeError(f"loaded SAITS from wrong source: {module.__file__}")
    return module.SAITS(**SAITS_SETTINGS, device=device).to(device)


def saits_inputs(batch: P4Batch, *, supervised: bool) -> dict[str, Tensor]:
    observed = batch.observed_mask.float()
    result = {"X": batch.normalized_observed, "missing_mask": observed}
    if supervised:
        holdout = (~batch.observed_mask).float()
        if torch.any(observed * holdout):
            raise AssertionError("SAITS observed and holdout masks overlap")
        result["X_holdout"] = batch.normalized_target
        result["indicating_mask"] = holdout
    return result


def saits_repair_loss(model: nn.Module, batch: P4Batch) -> Tensor:
    result = model(saits_inputs(batch, supervised=True), "train")
    return result["reconstruction_loss"] + result["imputation_loss"]


def impute_with_observed_clamp(model: nn.Module, batch: P4Batch) -> Tensor:
    inputs = saits_inputs(batch, supervised=False)
    if hasattr(model, "impute"):
        candidate, _ = model.impute(inputs)
    else:
        candidate = model(inputs, "test")["imputed_data"]
    if candidate.shape != batch.normalized_observed.shape:
        raise ValueError("SAITS imputation shape mismatch")
    return torch.where(batch.observed_mask, batch.normalized_observed, candidate)


class SaitsForecastModel(nn.Module):
    """Jointly trainable SAITS imputer followed by the E22-structure predictor."""

    def __init__(self, imputer: nn.Module, predictor: nn.Module) -> None:
        super().__init__()
        self.imputer = imputer
        self.predictor = predictor

    def forward(self, batch: P4Batch) -> Tensor:
        repaired = impute_with_observed_clamp(self.imputer, batch)
        patches = patchify_preserving_values(repaired, batch.observed_mask, length=16, stride=8)
        return self.predictor(
            patches.values,
            patches.effective_mask,
            batch.query_mean,
            batch.query_std,
            patch_validity=patches.validity,
        )
