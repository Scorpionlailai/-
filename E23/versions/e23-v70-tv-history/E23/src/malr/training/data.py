from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import numpy as np
import torch

from malr.data.normalization import TrainVariableStats, masked_window_normalize
from malr.data.patching import patchify_with_state
from malr.data.windows import WindowRef
from malr.models.stage1 import ModelVariant

from .trainer import PreparedBatch


def iter_prepared_batches(
    series: np.ndarray,
    refs: Sequence[WindowRef],
    timeline_mask: np.ndarray | Callable[[int, WindowRef], np.ndarray],
    *,
    mask_support_start: int,
    train_stats: TrainVariableStats,
    variant: ModelVariant | str,
    batch_size: int,
    patch_len: int,
    patch_stride: int,
    shuffle_seed: int | None = None,
    normalization_eps: float = 1e-5,
    drop_last: bool = False,
    include_imputation_target: bool = False,
) -> Iterable[PreparedBatch]:
    values = np.asarray(series, dtype=np.float32)
    observed_timeline = (
        None
        if callable(timeline_mask)
        else np.asarray(timeline_mask, dtype=bool)
    )
    selected_variant = ModelVariant(variant)
    if values.ndim != 2 or (
        observed_timeline is not None and observed_timeline.ndim != 2
    ):
        raise ValueError("series and timeline_mask must be two-dimensional")
    if observed_timeline is not None and values.shape[1] != observed_timeline.shape[1]:
        raise ValueError("timeline mask variable count does not match series")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    indices = np.arange(len(refs))
    if shuffle_seed is not None:
        np.random.default_rng(shuffle_seed).shuffle(indices)
    usable = (
        len(indices) - len(indices) % batch_size if drop_last else len(indices)
    )
    for offset in range(0, usable, batch_size):
        batch_refs = [refs[int(index)] for index in indices[offset : offset + batch_size]]
        clean_numpy = np.stack(
            [values[ref.input_start : ref.input_end] for ref in batch_refs]
        )
        target_numpy = np.stack(
            [values[ref.target_start : ref.target_end] for ref in batch_refs]
        )
        masks = []
        for ref in batch_refs:
            if callable(timeline_mask):
                local_mask = np.asarray(
                    timeline_mask(ref.target_start, ref), dtype=bool
                )
                expected_shape = (
                    ref.input_end - ref.input_start,
                    values.shape[1],
                )
                if local_mask.shape != expected_shape:
                    raise ValueError("sample mask factory returned an invalid shape")
                masks.append(local_mask)
            else:
                assert observed_timeline is not None
                local_start = ref.input_start - mask_support_start
                local_end = ref.input_end - mask_support_start
                if local_start < 0 or local_end > observed_timeline.shape[0]:
                    raise ValueError("window input lies outside mask support")
                masks.append(observed_timeline[local_start:local_end])
        observed_numpy = np.stack(masks)
        if selected_variant is ModelVariant.CLEAN:
            observed_numpy = np.ones_like(observed_numpy, dtype=bool)

        clean = torch.from_numpy(clean_numpy)
        observed = torch.from_numpy(observed_numpy)
        normalized = masked_window_normalize(
            clean,
            observed,
            train_stats,
            eps=normalization_eps,
        )
        patches = patchify_with_state(
            normalized.student,
            normalized.effective_mask,
            length=patch_len,
            stride=patch_stride,
        )
        clean_patches = patchify_with_state(
            clean,
            torch.ones_like(clean, dtype=torch.bool),
            length=patch_len,
            stride=patch_stride,
        )
        raw_context = torch.where(
            normalized.effective_mask, clean, torch.zeros_like(clean)
        )
        yield PreparedBatch(
            patch_values=patches.values,
            patch_mask=patches.effective_mask,
            rho=patches.rho,
            state=patches.state,
            query_mean=normalized.mean,
            query_std=normalized.std,
            forecast_target=torch.from_numpy(target_numpy).permute(0, 2, 1).contiguous(),
            clean_raw_patches=clean_patches.values,
            patch_valid=patches.validity.any(dim=-1),
            raw_context=raw_context,
            context_mask=normalized.effective_mask,
            train_std=train_stats.std,
            imputation_target=(normalized.target if include_imputation_target else None),
            patch_starts=torch.from_numpy(
                np.stack(
                    [
                        ref.input_start
                        + np.arange(patches.values.shape[2], dtype=np.int64) * patch_stride
                        for ref in batch_refs
                    ],
                    axis=0,
                )
            ),
            patch_validity=patches.validity,
        )
