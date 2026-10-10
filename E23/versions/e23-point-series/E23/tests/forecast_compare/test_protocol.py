from pathlib import Path
import shutil

import numpy as np
import pytest
import torch

from integrations.forecast_compare.protocol import (
    load_p4_protocol,
    iter_p4_batches,
    prepare_p4_batch,
)
from malr.data.masks import EpochMaskManager
from malr.data.normalization import TrainVariableStats


CONFIG = Path("configs/stage1_seq96_dynamic_temporal_repair.yaml")
MASKS = Path("../E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context")


@pytest.fixture(scope="module")
def protocol():
    return load_p4_protocol(CONFIG, MASKS)


def test_p4_real_identity(protocol):
    assert protocol.source_sha256 == "f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066"
    assert protocol.mask_sha256["validation"] == "cbe7c520afded9002f714c5ed8e67d62c92b97fa7aa9b1e6e4711db785fd846c"
    assert protocol.mask_sha256["test"] == "f7974ff9b4228d69fe33d940e3722cd1a069766e2fa1387c6240757d5a8762d8"
    assert (protocol.bounds.train.start, protocol.bounds.train.end) == (0, 8640)
    assert (protocol.bounds.validation.start, protocol.bounds.validation.end) == (8640, 11520)
    assert (protocol.bounds.test.start, protocol.bounds.test.end) == (11520, 14400)
    assert {split: len(refs) // 128 * 128 for split, refs in protocol.refs.items()} == {
        "train": 8448, "validation": 2688, "test": 2688
    }


def test_p4_mask_rejects_wrong_hash(tmp_path):
    dest = tmp_path / "masks"
    dest.mkdir()
    for split in ("validation", "test"):
        shutil.copyfile(MASKS / "masks" / f"ETTh1_{split}.npz", dest / f"ETTh1_{split}.npz")
    path = dest / "ETTh1_validation.npz"
    payload = bytearray(path.read_bytes())
    payload[-10] ^= 1
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="SHA-256"):
        load_p4_protocol(CONFIG, tmp_path)


def test_train_mask_matches_sample_manager(protocol):
    epoch = 3
    batch = next(iter_p4_batches(protocol, "train", batch_size=128, epoch=epoch))
    manager = EpochMaskManager(8640, 7, 0.10, 8, 2021)
    expected = manager.mask_for_sample(epoch, batch.refs[0].target_start, length=96)
    np.testing.assert_array_equal(batch.observed_mask[0].numpy(), expected)


def test_hidden_history_does_not_change_student_or_query_stats():
    clean = torch.arange(42, dtype=torch.float32).reshape(1, 6, 7)
    observed = torch.ones_like(clean, dtype=torch.bool)
    observed[:, 2:4] = False
    changed = clean.clone()
    changed[~observed] = 99999
    stats = TrainVariableStats(torch.zeros(7), torch.ones(7))
    target = torch.zeros(1, 7, 96)
    a = prepare_p4_batch(clean, observed, target, stats)
    b = prepare_p4_batch(changed, observed, target, stats)
    assert torch.equal(a.normalized_observed, b.normalized_observed)
    assert torch.equal(a.query_mean, b.query_mean)
    assert torch.equal(a.query_std, b.query_std)
