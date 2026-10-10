from __future__ import annotations

from argparse import Namespace
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from malr.config import config_sha256, load_config
from malr.data.datasets import load_clean_csv
from malr.data.masks import EpochMaskManager, MaskExpectation, load_fixed_mask
from malr.data.normalization import fit_train_variable_stats, standardize_values
from malr.data.splits import protocol_for
from malr.data.windows import WindowRef, build_window_index
from malr.evaluation.report import file_sha256


S4MSample = tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]


def build_s4m_sample(
    clean_input: np.ndarray,
    clean_target: np.ndarray,
    observed_mask: np.ndarray,
) -> S4MSample:
    clean = np.asarray(clean_input, dtype=np.float32)
    target = np.asarray(clean_target, dtype=np.float32)
    observed = np.asarray(observed_mask, dtype=bool)
    if clean.ndim != 2 or target.ndim != 2:
        raise ValueError("S4M clean input and target must be [time, variables]")
    if observed.shape != clean.shape:
        raise ValueError("S4M input mask must match the clean input")
    if clean.shape[1] != target.shape[1]:
        raise ValueError("S4M input and target variable counts must match")
    if not np.isfinite(clean).all() or not np.isfinite(target).all():
        raise ValueError("S4M clean values must be finite")

    seq_x_mask = observed.astype(np.float32)
    seq_x = np.where(observed, clean, 0.0).astype(np.float32)
    positions = np.arange(seq_x.shape[0], dtype=np.int64)[:, None]
    max_position = np.argmax(seq_x, axis=0)
    min_position = np.argmin(seq_x, axis=0)
    max_idx = np.abs(max_position[None, :] - positions).astype(np.float32)
    min_idx = np.abs(min_position[None, :] - positions).astype(np.float32)
    max_value = np.broadcast_to(np.max(seq_x, axis=0), seq_x.shape).copy()
    min_value = np.broadcast_to(np.min(seq_x, axis=0), seq_x.shape).copy()
    seq_true_y = np.ascontiguousarray(target, dtype=np.float32)
    return (
        np.ascontiguousarray(seq_x),
        np.ascontiguousarray(seq_x_mask),
        seq_true_y.copy(),
        np.ones_like(seq_true_y, dtype=np.float32),
        np.ascontiguousarray(clean, dtype=np.float32),
        seq_true_y,
        np.ascontiguousarray(max_idx),
        np.ascontiguousarray(min_idx),
        np.ascontiguousarray(max_value, dtype=np.float32),
        np.ascontiguousarray(min_value, dtype=np.float32),
    )


def _project_root(config_path: Path) -> Path:
    resolved = config_path.resolve()
    return resolved.parent.parent if resolved.parent.name == "configs" else Path.cwd()


def _refs_checksum(refs: tuple[WindowRef, ...]) -> str:
    payload = json.dumps(
        [asdict(ref) for ref in refs],
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(payload.encode("utf-8")).hexdigest()


class FairS4MDataset(Dataset[S4MSample]):
    def __init__(
        self,
        args: Namespace,
        flag: Literal["train", "val", "test"],
    ) -> None:
        if flag not in {"train", "val", "test"}:
            raise ValueError(f"unsupported S4M split flag: {flag}")
        self.args = args
        self.flag = flag
        self.split = "validation" if flag == "val" else flag
        self.config_path = Path(args.malr_config).resolve()
        self.config = load_config(self.config_path)
        self.dataset_name = str(args.malr_dataset)
        if self.dataset_name not in self.config.data.datasets:
            raise ValueError(f"dataset {self.dataset_name!r} is not configured")
        dataset_config = self.config.data.datasets[self.dataset_name]
        source_path = Path(dataset_config.path)
        if not source_path.is_absolute():
            source_path = _project_root(self.config_path) / source_path
        self.series = load_clean_csv(
            source_path,
            expected_n_vars=dataset_config.n_vars,
            expected_length=dataset_config.expected_length,
            allow_length_override=dataset_config.allow_length_override,
        )
        self.protocol = protocol_for(self.dataset_name, len(self.series.values))
        train_values = torch.from_numpy(
            self.series.values[
                self.protocol.train.start : self.protocol.train.end
            ]
        )
        self.train_stats = fit_train_variable_stats(train_values, eps=0.0)
        standardized = standardize_values(
            torch.from_numpy(self.series.values), self.train_stats
        )
        self.standardized = np.ascontiguousarray(
            standardized.numpy(), dtype=np.float32
        )

        refs = build_window_index(
            self.protocol,
            self.split,  # type: ignore[arg-type]
            self.config.data.seq_len,
            self.config.data.pred_len,
        )
        if self.split in {"validation", "test"}:
            reference_batch = int(
                getattr(args, "fair_eval_reference_batch", 128)
            )
            if reference_batch <= 0:
                raise ValueError("fair_eval_reference_batch must be positive")
            usable = len(refs) - len(refs) % reference_batch
            refs = refs[:usable]
        self.refs = tuple(refs)
        if not self.refs:
            raise ValueError(f"S4M {self.split} split has no forecast windows")

        self.clean_variant = str(getattr(args, "fair_variant", "missing")) == "clean"
        self.mask_manager: EpochMaskManager | None = None
        self.mask_support_start = 0
        self.mask_path: Path | None = None
        self.mask_sha256: str | None = None
        self.final_rate = 0.0
        if self.split == "train":
            self.mask_manager = EpochMaskManager(
                length=self.protocol.train.end,
                n_vars=self.series.values.shape[1],
                start_rate=self.config.missing.start_rate,
                block_len=self.config.missing.block_len,
                base_seed=self.config.training.seed,
            )
            self.observed_timeline = np.ones(
                (self.protocol.train.end, self.series.values.shape[1]), dtype=bool
            )
            self.set_epoch(0)
        else:
            bounds = self.protocol.bounds_for(self.split)  # type: ignore[arg-type]
            self.mask_support_start = bounds.start - self.config.data.seq_len
            support_end = bounds.end
            shape = (
                support_end - self.mask_support_start,
                self.series.values.shape[1],
            )
            if self.clean_variant:
                self.observed_timeline = np.ones(shape, dtype=bool)
            else:
                expected = MaskExpectation(
                    dataset=self.dataset_name,
                    split=self.split,
                    seed=self.config.training.seed,
                    start_rate=self.config.missing.start_rate,
                    block_len=self.config.missing.block_len,
                    support_start=self.mask_support_start,
                    support_end=support_end,
                    shape=shape,
                )
                self.mask_path = (
                    Path(args.malr_artifacts).resolve()
                    / "masks"
                    / f"{self.dataset_name}_{self.split}.npz"
                )
                self.observed_timeline, metadata = load_fixed_mask(
                    self.mask_path, expected
                )
                self.mask_sha256 = file_sha256(self.mask_path)
                self.final_rate = metadata.final_rate

    def set_epoch(self, epoch: int) -> None:
        if self.split != "train":
            raise ValueError("set_epoch is only valid for the training split")
        if self.clean_variant:
            self.observed_timeline = np.ones(
                (self.protocol.train.end, self.series.values.shape[1]), dtype=bool
            )
        else:
            if self.mask_manager is None:
                raise AssertionError("training mask manager is unavailable")
            self.observed_timeline = self.mask_manager.mask_for_epoch(epoch)
        self.final_rate = float(1.0 - self.observed_timeline.mean(dtype=np.float64))

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int) -> S4MSample:
        ref = self.refs[index]
        local_start = ref.input_start - self.mask_support_start
        local_end = ref.input_end - self.mask_support_start
        if local_start < 0 or local_end > len(self.observed_timeline):
            raise ValueError("S4M input window lies outside mask support")
        clean_input = self.standardized[ref.input_start : ref.input_end]
        clean_target = self.standardized[ref.target_start : ref.target_end]
        observed = self.observed_timeline[local_start:local_end]
        return build_s4m_sample(clean_input, clean_target, observed)

    def identity(self) -> dict[str, object]:
        bounds = self.protocol.bounds_for(self.split)  # type: ignore[arg-type]
        return {
            "dataset": self.dataset_name,
            "split": self.split,
            "split_start": bounds.start,
            "split_end": bounds.end,
            "seq_len": self.config.data.seq_len,
            "pred_len": self.config.data.pred_len,
            "sample_count": len(self.refs),
            "window_indices_sha256": _refs_checksum(self.refs),
            "mask_sha256": self.mask_sha256,
            "final_rate": self.final_rate,
            "source_csv_sha256": file_sha256(self.series.source_path),
            "config_hash": config_sha256(self.config),
            "metric_space": "train_standardized",
        }


def fair_data_provider(
    args: Namespace,
    flag: Literal["train", "val", "test"],
) -> tuple[FairS4MDataset, DataLoader[S4MSample]]:
    dataset = FairS4MDataset(args, flag)
    generator = torch.Generator()
    generator.manual_seed(int(getattr(args, "seed", 2021)))
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=flag == "train",
        num_workers=int(getattr(args, "num_workers", 0)),
        drop_last=flag == "train",
        generator=generator,
    )
    return dataset, loader
