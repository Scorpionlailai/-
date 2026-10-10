"""Locked ETTh1 P4 data identity and leak-free SAITS-ready batches."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterator, Literal

import numpy as np
import torch
from torch import Tensor

from malr.config import Stage1Config, load_config
from malr.data.datasets import load_clean_csv
from malr.data.masks import EpochMaskManager, MaskExpectation, load_fixed_mask
from malr.data.normalization import (
    TrainVariableStats, fit_train_variable_stats, masked_window_normalize,
    standardize_values,
)
from malr.data.splits import DatasetProtocol, protocol_from_bounds
from malr.data.windows import WindowRef, build_window_index
from malr.evaluation.report import file_sha256

SOURCE_SHA256 = "f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066"
MASK_SHA256 = {
    "validation": "cbe7c520afded9002f714c5ed8e67d62c92b97fa7aa9b1e6e4711db785fd846c",
    "test": "f7974ff9b4228d69fe33d940e3722cd1a069766e2fa1387c6240757d5a8762d8",
}
Split = Literal["train", "validation", "test"]


@dataclass(frozen=True)
class P4Batch:
    normalized_observed: Tensor  # [B, 96, 7], zero at masked coordinates
    normalized_target: Tensor  # [B, 96, 7], repair supervision only
    observed_mask: Tensor  # True means observed
    query_mean: Tensor  # [B, 7]
    query_std: Tensor  # [B, 7]
    forecast_target: Tensor  # [B, 7, 96], never part of model inputs
    refs: tuple[WindowRef, ...] = ()

    def to(self, device: torch.device | str) -> "P4Batch":
        return replace(
            self,
            normalized_observed=self.normalized_observed.to(device),
            normalized_target=self.normalized_target.to(device),
            observed_mask=self.observed_mask.to(device),
            query_mean=self.query_mean.to(device),
            query_std=self.query_std.to(device),
            forecast_target=self.forecast_target.to(device),
        )


@dataclass(frozen=True)
class P4Protocol:
    config: Stage1Config
    bounds: DatasetProtocol
    standardized: np.ndarray
    fallback: TrainVariableStats
    refs: dict[str, tuple[WindowRef, ...]]
    fixed_masks: dict[str, np.ndarray]
    source_sha256: str
    mask_sha256: dict[str, str]
    config_path: Path
    artifacts: Path


def prepare_p4_batch(
    clean_history: Tensor,
    observed_mask: Tensor,
    forecast_target: Tensor,
    fallback: TrainVariableStats,
    *,
    refs: tuple[WindowRef, ...] = (),
) -> P4Batch:
    normalized = masked_window_normalize(clean_history, observed_mask, fallback, eps=1e-5)
    return P4Batch(
        normalized_observed=normalized.student,
        normalized_target=normalized.target,
        observed_mask=normalized.effective_mask,
        query_mean=normalized.mean,
        query_std=normalized.std,
        forecast_target=forecast_target,
        refs=refs,
    )


def load_p4_protocol(config_path: Path, artifacts: Path) -> P4Protocol:
    config_path = Path(config_path).resolve()
    artifacts = Path(artifacts).resolve()
    config = load_config(config_path)
    dataset = config.data.datasets["ETTh1"]
    if (
        config.data.seq_len, config.data.pred_len, dataset.n_vars,
        dataset.train_start, dataset.train_end, dataset.validation_end, dataset.test_end,
        config.training.seed, config.missing.start_rate, config.missing.block_len,
        config.missing.allow_block_overlap,
    ) != (96, 96, 7, 0, 8640, 11520, 14400, 2021, 0.10, 8, True):
        raise ValueError("config does not match locked ETTh1 P4 protocol")
    series = load_clean_csv(
        Path(dataset.path), dataset.n_vars, dataset.expected_length,
        allow_length_override=dataset.allow_length_override,
    )
    source_hash = file_sha256(series.source_path)
    if source_hash != SOURCE_SHA256:
        raise ValueError("source CSV SHA-256 does not match locked P4 identity")
    bounds = protocol_from_bounds(
        "ETTh1", len(series.values), train_start=dataset.train_start,
        train_end=dataset.train_end, validation_end=dataset.validation_end,
        test_end=dataset.test_end,
    )
    raw_stats = fit_train_variable_stats(torch.from_numpy(series.values[:8640]), eps=0.0)
    standardized = np.ascontiguousarray(
        standardize_values(torch.from_numpy(series.values), raw_stats).numpy(), dtype=np.float32
    )
    fallback = fit_train_variable_stats(torch.from_numpy(standardized[:8640]), eps=0.0)
    fixed_masks: dict[str, np.ndarray] = {}
    mask_sha256: dict[str, str] = {}
    for split in ("validation", "test"):
        mask_path = artifacts / "masks" / f"ETTh1_{split}.npz"
        if not mask_path.is_file():
            raise ValueError(f"{split} fixed mask file is missing: {mask_path}")
        digest = file_sha256(mask_path)
        if digest != MASK_SHA256[split]:
            raise ValueError(f"{split} mask SHA-256 does not match locked P4 identity")
        split_bounds = bounds.bounds_for(split)
        support_start = split_bounds.start - 96
        fixed_masks[split], _ = load_fixed_mask(
            mask_path,
            MaskExpectation(
                dataset="ETTh1", split=split, seed=2021,
                start_rate=0.10, block_len=8,
                support_start=support_start, support_end=split_bounds.end,
                shape=(split_bounds.end - support_start, 7),
            ),
        )
        mask_sha256[split] = digest
    refs = {split: tuple(build_window_index(bounds, split, 96, 96))
            for split in ("train", "validation", "test")}
    return P4Protocol(config, bounds, standardized, fallback, refs, fixed_masks,
                      source_hash, mask_sha256, config_path, artifacts)


def iter_p4_batches(
    protocol: P4Protocol,
    split: Split,
    *,
    batch_size: int = 128,
    epoch: int | None = None,
) -> Iterator[P4Batch]:
    if split not in protocol.refs or batch_size <= 0:
        raise ValueError("invalid split or batch size")
    if split == "train" and epoch is None:
        raise ValueError("train batches require an epoch for deterministic masks")
    indices = np.arange(len(protocol.refs[split]))
    if split == "train":
        np.random.default_rng(2021 + int(epoch)).shuffle(indices)
        manager = EpochMaskManager(8640, 7, 0.10, 8, 2021)
    usable = len(indices) // batch_size * batch_size
    support_start = protocol.bounds.bounds_for(split).start - 96 if split != "train" else 0
    for offset in range(0, usable, batch_size):
        refs = tuple(protocol.refs[split][int(i)] for i in indices[offset:offset + batch_size])
        clean = torch.from_numpy(np.stack([
            protocol.standardized[ref.input_start:ref.input_end] for ref in refs
        ]))
        future = torch.from_numpy(np.stack([
            protocol.standardized[ref.target_start:ref.target_end] for ref in refs
        ])).permute(0, 2, 1).contiguous()
        if split == "train":
            masks = np.stack([
                manager.mask_for_sample(int(epoch), ref.target_start, length=96)
                for ref in refs
            ])
        else:
            timeline = protocol.fixed_masks[split]
            masks = np.stack([
                timeline[ref.input_start - support_start:ref.input_end - support_start]
                for ref in refs
            ])
        yield prepare_p4_batch(clean, torch.from_numpy(masks), future,
                               protocol.fallback, refs=refs)
