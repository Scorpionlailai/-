from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch

from malr.data.repaired import RepairedSplit
from malr.models.online_stage1_v2 import RelationV2OnlineStage1Model
from malr.models.reconstruction import aggregate_patch_predictions
from malr.training.trainer import PreparedBatch


@torch.no_grad()
def reconstruct_split(
    model: RelationV2OnlineStage1Model,
    batches: Iterable[PreparedBatch],
    *,
    seq_len: int,
    patch_len: int,
    patch_stride: int,
    epoch_index: int,
    device: str | torch.device = "cpu",
) -> RepairedSplit:
    """Materialize Stage-1 outputs as a read-only Stage-2 dataset."""

    reconstruction_device = torch.device(device)
    model.to(reconstruction_device).eval()
    before = model.online_state()
    values: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    means: list[np.ndarray] = []
    stds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    starts: list[np.ndarray] = []
    for batch in batches:
        current = batch.to(reconstruction_device)
        if current.patch_starts is None:
            raise ValueError("repaired export requires absolute patch starts")
        if current.patch_validity is None:
            raise ValueError("repaired export requires coordinate validity")
        output = model(
            current.patch_values,
            current.patch_mask,
            current.rho,
            current.state,
            current.query_mean,
            current.query_std,
            patch_starts=current.patch_starts,
            epoch_index=epoch_index,
            patch_valid=current.patch_valid,
            patch_validity=current.patch_validity,
        )
        repaired, _ = aggregate_patch_predictions(
            output.reconstructed_patches,
            window_len=seq_len,
            patch_len=patch_len,
            stride=patch_stride,
            validity=current.patch_validity,
        )
        # ``reconstructed_patches`` is query-normalized.  Observed values were
        # copied with torch.where in Stage 1; only originally missing entries
        # contain decoded values.
        values.append(repaired.cpu().numpy().astype(np.float32, copy=False))
        masks.append(current.context_mask.cpu().numpy().astype(np.bool_, copy=False))
        means.append(current.query_mean.cpu().numpy().astype(np.float32, copy=False))
        stds.append(current.query_std.cpu().numpy().astype(np.float32, copy=False))
        targets.append(
            current.forecast_target.cpu().numpy().astype(np.float32, copy=False)
        )
        starts.append(
            current.patch_starts[:, 0].cpu().numpy().astype(np.int64, copy=False)
        )
    if not values:
        raise ValueError("repaired export batch iterator produced no batches")
    after = model.online_state()
    for key in ("memory_hash", "ema_hash", "relation_prior_hash"):
        if before[key] != after[key]:
            raise RuntimeError("repaired export changed Stage-1 online state")
    return RepairedSplit(
        values=np.concatenate(values, axis=0),
        observed_mask=np.concatenate(masks, axis=0),
        query_mean=np.concatenate(means, axis=0),
        query_std=np.concatenate(stds, axis=0),
        targets=np.concatenate(targets, axis=0),
        input_starts=np.concatenate(starts, axis=0),
    )


__all__ = ["reconstruct_split"]
