"""Version 2: any predeclared number of new, test-only P4 missing masks."""

from dataclasses import replace
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from malr.data.masks import MaskExpectation, generate_variable_missing_mask, load_fixed_mask, save_fixed_mask
from malr.evaluation.report import file_sha256
from .test_mask_bank import _require_base_identity, _missing_fractions

TEST_SEEDS = tuple(range(2026100401, 2026100421))


def _identity(protocol):
    return dict(schema_version=2, dataset='ETTh1', scenario='P4',
                source_csv_sha256=protocol.source_sha256,
                validation_mask_sha256=protocol.mask_sha256['validation'],
                seq_len=96, pred_len=96, n_vars=7, start_rate=.10, block_len=8,
                allow_block_overlap=True, support_start=protocol.bounds.test.start-96,
                support_end=protocol.bounds.test.end)


def create_test_bank(protocol, output, *, seeds=TEST_SEEDS):
    _require_base_identity(protocol)
    seeds = tuple(seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(
            isinstance(s, bool) or not isinstance(s, int) or not 0 <= s < 2**32 or s == 2021
            for s in seeds):
        raise ValueError('new mask seeds must be distinct nonnegative integers excluding 2021')
    output = Path(output).resolve()
    if output.exists():
        load_test_bank(protocol, output/'manifest.json', expected_seeds=seeds)
        return output/'manifest.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.mask-bank-v2-', dir=output.parent))
    try:
        identity = _identity(protocol)
        entries = []
        for index, seed in enumerate(seeds):
            path = staging/f'test_group_{index:02d}.npz'
            mask, metadata = generate_variable_missing_mask(
                identity['support_end']-identity['support_start'], 7, .10, 8, seed,
                dataset='ETTh1', split='test', support_start=identity['support_start'])
            save_fixed_mask(path, mask, metadata)
            timeline, window, count = _missing_fractions(protocol, mask)
            entries.append(dict(group_id=f'group_{index:02d}', split='test', seed=seed,
                                path=path.name, sha256=file_sha256(path),
                                missing_timeline_fraction=timeline,
                                missing_evaluation_fraction=window, evaluation_windows=count))
        record = {**identity, 'group_count': len(seeds), 'seeds': list(seeds), 'groups': entries,
                  'purpose': 'test-only; fixed before training; never used for selection',
                  'training_seeds': 'independent of mask seeds'}
        manifest = staging/'manifest.json'
        manifest.write_text(json.dumps(record, indent=2, sort_keys=True)+'\n')
        manifest.with_suffix('.sha256').write_text(file_sha256(manifest)+'\n')
        load_test_bank(protocol, manifest, expected_seeds=seeds)
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return output/'manifest.json'


def load_test_bank(protocol, manifest, *, expected_seeds=TEST_SEEDS):
    _require_base_identity(protocol)
    manifest = Path(manifest).resolve(strict=True)
    if manifest.with_suffix('.sha256').read_text().strip() != file_sha256(manifest):
        raise ValueError('mask bank seal mismatch')
    record = json.loads(manifest.read_text())
    for key, value in _identity(protocol).items():
        if record.get(key) != value:
            raise ValueError(f'mask bank {key} mismatch')
    seeds = tuple(record['seeds'])
    if seeds != tuple(expected_seeds) or record['group_count'] != len(seeds):
        raise ValueError('mask bank seeds/count mismatch')
    if len(record['groups']) != len(seeds):
        raise ValueError('mask bank group count mismatch')
    views = []
    for index, (seed, entry) in enumerate(zip(seeds, record['groups'])):
        if (entry['seed'], entry['group_id'], entry['split']) != (seed, f'group_{index:02d}', 'test'):
            raise ValueError('mask group identity mismatch')
        path = (manifest.parent/entry['path']).resolve(strict=True)
        if not path.is_relative_to(manifest.parent) or file_sha256(path) != entry['sha256']:
            raise ValueError('mask group path/hash mismatch')
        mask, _ = load_fixed_mask(path, MaskExpectation(
            dataset='ETTh1', split='test', seed=seed, start_rate=.10, block_len=8,
            support_start=record['support_start'], support_end=record['support_end'],
            shape=(record['support_end']-record['support_start'], 7)))
        if np.array_equal(mask, protocol.fixed_masks['test']) or any(
                np.array_equal(mask, view.fixed_masks['test']) for view in views):
            raise ValueError('duplicate test mask positions')
        timeline, window, count = _missing_fractions(protocol, mask)
        if (timeline, window, count) != (entry['missing_timeline_fraction'],
                                      entry['missing_evaluation_fraction'], entry['evaluation_windows']):
            raise ValueError('mask evaluation support mismatch')
        views.append(replace(protocol, fixed_masks={**protocol.fixed_masks, 'test': mask},
                             mask_sha256={**protocol.mask_sha256, 'test': entry['sha256']}))
    return tuple(views)
