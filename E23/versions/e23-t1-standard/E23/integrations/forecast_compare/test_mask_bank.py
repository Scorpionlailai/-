"""Frozen P4 test draws; training and the original validation mask stay fixed."""

from __future__ import annotations

import csv
from dataclasses import replace
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from malr.data.masks import (
    MaskExpectation, generate_variable_missing_mask, load_fixed_mask, save_fixed_mask,
)
from malr.evaluation.report import file_sha256

from .protocol import MASK_SHA256, SOURCE_SHA256, P4Protocol


EXTRA_TEST_SEEDS = (2026100201, 2026100202)


def _require_base_identity(protocol: P4Protocol) -> None:
    if protocol.source_sha256 != SOURCE_SHA256:
        raise ValueError('test bank source CSV identity mismatch')
    if protocol.mask_sha256.get('validation') != MASK_SHA256['validation']:
        raise ValueError('test bank requires the original validation mask identity')
    if protocol.mask_sha256.get('test') != MASK_SHA256['test']:
        raise ValueError('test bank requires the original test mask identity')


def _missing_fractions(protocol: P4Protocol, mask: np.ndarray) -> tuple[float, float, int]:
    refs = protocol.refs['test']
    refs = refs[:len(refs) // 128 * 128]
    support_start = protocol.bounds.test.start - 96
    missing = sum(int((~mask[ref.input_start - support_start:ref.input_end - support_start]).sum())
                  for ref in refs)
    return float((~mask).mean()), missing / (len(refs) * 96 * 7), len(refs)


def create_test_mask_bank(
    protocol: P4Protocol, output: Path, *, seeds: tuple[int, int] = EXTRA_TEST_SEEDS,
) -> Path:
    """Save the original test draw and exactly two new, predeclared draws."""
    _require_base_identity(protocol)
    if len(seeds) != 2 or len(set((2021, *seeds))) != 3 or any(seed < 0 for seed in seeds):
        raise ValueError('exactly two non-negative seeds distinct from each other and 2021 are required')
    output = Path(output).resolve()
    manifest = output / 'manifest.json'
    if output.exists():
        load_test_mask_bank(protocol, manifest)
        saved = json.loads(manifest.read_text())
        if tuple(group['seed'] for group in saved['groups'][1:]) != tuple(seeds):
            raise ValueError('refusing to replace a frozen test bank with different seeds')
        return manifest
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.test-mask-bank-', dir=output.parent))
    try:
        original = protocol.artifacts / 'masks' / 'ETTh1_test.npz'
        if file_sha256(original) != MASK_SHA256['test']:
            raise ValueError('original test mask SHA-256 mismatch')
        original_path = staging / 'test_group_0.npz'
        shutil.copyfile(original, original_path)
        support_start = protocol.bounds.test.start - 96
        length = protocol.bounds.test.end - support_start
        groups = []
        for index, seed in enumerate((2021, *seeds)):
            path = staging / f'test_group_{index}.npz'
            if index == 0:
                mask = protocol.fixed_masks['test']
            else:
                mask, metadata = generate_variable_missing_mask(
                    length, 7, 0.10, 8, seed, dataset='ETTh1', split='test',
                    support_start=support_start,
                )
                save_fixed_mask(path, mask, metadata)
            timeline_fraction, window_fraction, windows = _missing_fractions(protocol, mask)
            groups.append({
                'group_id': f'group_{index}', 'split': 'test', 'seed': seed,
                'path': path.name, 'sha256': file_sha256(path),
                'missing_timeline_fraction': timeline_fraction,
                'missing_evaluation_fraction': window_fraction,
                'evaluation_windows': windows,
            })
        record = {
            'schema_version': 1, 'dataset': 'ETTh1', 'scenario': 'P4',
            'source_csv_sha256': protocol.source_sha256,
            'validation_mask_sha256': protocol.mask_sha256['validation'],
            'training_seed': 2021, 'seq_len': 96, 'pred_len': 96, 'n_vars': 7,
            'start_rate': 0.10, 'block_len': 8, 'allow_block_overlap': True,
            'support_start': support_start, 'support_end': protocol.bounds.test.end,
            'group_count': 3, 'groups': groups,
            'purpose': 'test-only missing-position robustness; no training or checkpoint selection',
        }
        staged_manifest = staging / 'manifest.json'
        staged_manifest.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
        (staging / 'manifest.sha256').write_text(file_sha256(staged_manifest) + '\n')
        with (staging / 'mask_groups.csv').open('w', newline='') as stream:
            fieldnames = ['group_id', 'split', 'seed', 'start_rate', 'block_len',
                          'missing_timeline_fraction', 'missing_evaluation_fraction',
                          'evaluation_windows', 'sha256', 'path']
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for group in groups:
                writer.writerow({**group, 'start_rate': 0.10, 'block_len': 8})
        load_test_mask_bank(protocol, staged_manifest)
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


def load_test_mask_bank(protocol: P4Protocol, manifest: Path) -> tuple[P4Protocol, ...]:
    """Verify the frozen bank and replace only the test mask for each draw."""
    _require_base_identity(protocol)
    manifest = Path(manifest).resolve()
    if not manifest.is_file():
        raise ValueError(f'test mask bank manifest is missing: {manifest}')
    seal = manifest.with_suffix('.sha256')
    if not seal.is_file() or file_sha256(manifest) != seal.read_text().strip():
        raise ValueError('test mask bank manifest SHA-256 mismatch')
    record = json.loads(manifest.read_text())
    expected = {
        'schema_version': 1, 'dataset': 'ETTh1', 'scenario': 'P4',
        'source_csv_sha256': protocol.source_sha256,
        'validation_mask_sha256': protocol.mask_sha256['validation'],
        'training_seed': 2021, 'seq_len': 96, 'pred_len': 96, 'n_vars': 7,
        'start_rate': 0.10, 'block_len': 8, 'allow_block_overlap': True,
        'support_start': protocol.bounds.test.start - 96,
        'support_end': protocol.bounds.test.end, 'group_count': 3,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f'test mask bank {key} mismatch')
    entries = record.get('groups', [])
    if len(entries) != 3 or entries[0].get('seed') != 2021:
        raise ValueError('test bank must include the original group and exactly two extra groups')
    if len({entry['seed'] for entry in entries}) != 3:
        raise ValueError('test bank seeds must be distinct')
    protocols = []
    for index, entry in enumerate(entries):
        if entry.get('group_id') != f'group_{index}' or entry.get('split') != 'test':
            raise ValueError('test mask group identity mismatch')
        path = (manifest.parent / entry['path']).resolve()
        if not path.is_relative_to(manifest.parent):
            raise ValueError('test mask path must stay inside its bank')
        if not path.is_file() or file_sha256(path) != entry['sha256']:
            raise ValueError('test mask group SHA-256 mismatch')
        if index == 0 and entry['sha256'] != MASK_SHA256['test']:
            raise ValueError('original test group SHA-256 mismatch')
        mask, _ = load_fixed_mask(path, MaskExpectation(
            dataset='ETTh1', split='test', seed=entry['seed'], start_rate=0.10,
            block_len=8, support_start=record['support_start'], support_end=record['support_end'],
            shape=(record['support_end'] - record['support_start'], 7),
        ))
        if any(np.array_equal(mask, group.fixed_masks['test']) for group in protocols):
            raise ValueError('test bank missing positions must be distinct')
        protocols.append(replace(
            protocol, fixed_masks={**protocol.fixed_masks, 'test': mask},
            mask_sha256={**protocol.mask_sha256, 'test': entry['sha256']},
        ))
    return tuple(protocols)
