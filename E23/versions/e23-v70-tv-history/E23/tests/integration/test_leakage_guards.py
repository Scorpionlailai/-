from pathlib import Path

import numpy as np
import pytest
import torch

from malr.data.masks import (
    MaskExpectation,
    generate_variable_missing_mask,
    load_fixed_mask,
    save_fixed_mask,
)
from malr.data.splits import DatasetProtocol, SplitBounds
from malr.data.windows import build_window_index
from malr.prototypes.bank import build_prototype_bank
from tests.unit.test_teacher_and_losses import _tiny_ours


def test_bank_cannot_extend_beyond_declared_train_boundary():
    values = np.zeros((96, 1), dtype=np.float64)
    with pytest.raises(ValueError, match="train boundary"):
        build_prototype_bank(
            values,
            train_end=96,
            allowed_train_end=64,
        )


def test_fixed_mask_from_another_support_interval_is_rejected(tmp_path: Path):
    observed, metadata = generate_variable_missing_mask(
        64, 2, 0.06, 5, 2021, dataset="fixture", split="validation", support_start=32
    )
    path = tmp_path / "mask.npz"
    save_fixed_mask(path, observed, metadata)
    expected = MaskExpectation(
        dataset="fixture",
        split="test",
        seed=2021,
        start_rate=0.06,
        block_len=5,
        support_start=64,
        support_end=128,
        shape=(64, 2),
    )
    with pytest.raises(ValueError, match="metadata mismatch"):
        load_fixed_mask(path, expected)


def test_teacher_clean_target_is_rejected_during_validation():
    model = _tiny_ours().eval()
    patch_count = model.core.patch_num
    with pytest.raises(ValueError, match="training-only"):
        model(
            torch.zeros(1, 1, patch_count, 16),
            torch.zeros(1, 1, patch_count, 16, dtype=torch.bool),
            torch.zeros(1, 1, patch_count),
            torch.full((1, 1, patch_count), 2, dtype=torch.long),
            torch.zeros(1, 1),
            torch.ones(1, 1),
            clean_raw_patches=torch.zeros(1, 1, patch_count, 16),
        )


def test_window_target_never_crosses_split_boundary():
    protocol = DatasetProtocol(
        "fixture",
        source_length=160,
        train=SplitBounds(0, 80),
        validation=SplitBounds(80, 120),
        test=SplitBounds(120, 160),
    )
    refs = build_window_index(protocol, "validation", seq_len=16, pred_len=8)
    assert refs
    assert all(ref.target_start >= 80 and ref.target_end <= 120 for ref in refs)
