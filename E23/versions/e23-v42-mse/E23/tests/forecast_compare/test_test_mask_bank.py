"""The extra missing draws change test inputs only, with verifiable identities."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from integrations.forecast_compare.protocol import iter_p4_batches, load_p4_protocol


CONFIG = Path('configs/stage1_seq96_dynamic_temporal_repair.yaml')
MASKS = Path('../E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context')


@pytest.fixture(scope='module')
def protocol():
    return load_p4_protocol(CONFIG, MASKS)


def test_bank_preserves_original_draw_and_changes_only_test_inputs(protocol, tmp_path):
    from integrations.forecast_compare.test_mask_bank import create_test_mask_bank, load_test_mask_bank

    manifest = create_test_mask_bank(protocol, tmp_path / 'bank')
    groups = load_test_mask_bank(protocol, manifest)
    assert len(groups) == 3
    np.testing.assert_array_equal(groups[0].fixed_masks['test'], protocol.fixed_masks['test'])
    original = next(iter_p4_batches(protocol, 'test'))
    for group in groups:
        assert group.standardized is protocol.standardized
        assert group.refs is protocol.refs
        assert group.fixed_masks['validation'] is protocol.fixed_masks['validation']
        assert group.mask_sha256['validation'] == protocol.mask_sha256['validation']
        batch = next(iter_p4_batches(group, 'test'))
        torch.testing.assert_close(batch.forecast_target, original.forecast_target)
        assert batch.refs == original.refs
    for left in range(3):
        for right in range(left + 1, 3):
            assert not np.array_equal(groups[left].fixed_masks['test'], groups[right].fixed_masks['test'])
    train_original = next(iter_p4_batches(protocol, 'train', epoch=2))
    train_extra = next(iter_p4_batches(groups[2], 'train', epoch=2))
    torch.testing.assert_close(train_extra.observed_mask, train_original.observed_mask)


def test_bank_reuses_same_seeds_and_refuses_to_replace_a_frozen_bank(protocol, tmp_path):
    from integrations.forecast_compare.test_mask_bank import create_test_mask_bank

    manifest = create_test_mask_bank(protocol, tmp_path / 'bank')
    original = manifest.read_bytes()
    assert create_test_mask_bank(protocol, tmp_path / 'bank') == manifest
    assert manifest.read_bytes() == original
    with pytest.raises(ValueError, match='seed'):
        create_test_mask_bank(protocol, tmp_path / 'bank', seeds=(13, 14))
    with pytest.raises(ValueError, match='distinct'):
        create_test_mask_bank(protocol, tmp_path / 'another-bank', seeds=(13, 13))


def test_bank_rejects_changed_mask_file(protocol, tmp_path):
    from integrations.forecast_compare.test_mask_bank import create_test_mask_bank, load_test_mask_bank

    manifest = create_test_mask_bank(protocol, tmp_path / 'bank')
    record = json.loads(manifest.read_text())
    mask = manifest.parent / record['groups'][1]['path']
    mask.write_bytes(mask.read_bytes() + b'tampered')
    with pytest.raises(ValueError, match='SHA-256'):
        load_test_mask_bank(protocol, manifest)


def test_bank_requires_original_validation_identity(protocol, tmp_path):
    from dataclasses import replace
    from integrations.forecast_compare.test_mask_bank import create_test_mask_bank, load_test_mask_bank

    manifest = create_test_mask_bank(protocol, tmp_path / 'bank')
    changed = replace(protocol, mask_sha256={**protocol.mask_sha256, 'validation': 'different'})
    with pytest.raises(ValueError, match='validation'):
        load_test_mask_bank(changed, manifest)
