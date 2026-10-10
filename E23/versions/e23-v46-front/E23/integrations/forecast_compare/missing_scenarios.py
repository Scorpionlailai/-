"""Sealed high-missing scenarios derived from the unchanged, catalog-checked P4 data."""
from dataclasses import replace
import json
import os
from pathlib import Path
import numpy as np
from malr.config import load_config
from malr.data.masks import MaskExpectation, load_fixed_mask
from malr.evaluation.report import file_sha256
from .test_mask_bank import _missing_fractions

SCENARIOS = (("sr015_bl8", .15, 8), ("sr020_bl8", .20, 8), ("sr010_bl16", .10, 16))
TEST_SEEDS = tuple(range(2026100401, 2026100421))

def sealed(path):
    path = Path(path).resolve(strict=True)
    if path.with_suffix('.sha256').read_text().strip() != file_sha256(path):
        raise ValueError('scenario manifest seal mismatch')
    return path, json.loads(path.read_text())

def load_mask(root, entry, *, split, start_rate, block_len, bounds):
    path = (root / entry['path']).resolve(strict=True)
    if not path.is_relative_to(root) or file_sha256(path) != entry['sha256']:
        raise ValueError('scenario mask path/hash mismatch')
    return load_fixed_mask(path, MaskExpectation(dataset='ETTh1', split=split,
        seed=entry['seed'], start_rate=start_rate, block_len=block_len,
        support_start=bounds.start-96, support_end=bounds.end,
        shape=(bounds.end-bounds.start+96,7)))[0]

def apply_scenario(base, manifest):
    path, record = sealed(manifest)
    key = (record['scenario'], record['start_rate'], record['block_len'])
    if key not in SCENARIOS or record['base_config_sha256'] != file_sha256(base.config_path):
        raise ValueError('unsupported scenario/base configuration')
    if record['source_csv_sha256'] != base.source_sha256:
        raise ValueError('scenario source identity mismatch')
    config_path = (path.parent / record['config_path']).resolve(strict=True)
    if not config_path.is_relative_to(path.parent) or file_sha256(config_path) != record['config_sha256']:
        raise ValueError('scenario config path/hash mismatch')
    config = load_config(config_path)
    expected = replace(base.config, missing=replace(base.config.missing,
        start_rate=record['start_rate'], block_len=record['block_len']))
    if config != expected:
        raise ValueError('scenario changed settings beyond missing parameters')
    masks = {split: load_mask(path.parent,record['masks'][split],split=split,
        start_rate=config.missing.start_rate,block_len=config.missing.block_len,
        bounds=base.bounds.bounds_for(split)) for split in ('validation','test')}
    return replace(base, config=config, config_path=config_path, artifacts=path.parent,
        fixed_masks=masks, mask_sha256={k:record['masks'][k]['sha256'] for k in masks})

def load_scenario_test_bank(protocol, manifest, *, expected_seeds=TEST_SEEDS):
    path, record = sealed(manifest)
    for key,value in dict(source_csv_sha256=protocol.source_sha256,
        validation_mask_sha256=protocol.mask_sha256['validation'],
        start_rate=protocol.config.missing.start_rate,block_len=protocol.config.missing.block_len,
        seq_len=96,pred_len=96,n_vars=7).items():
        if record.get(key) != value: raise ValueError(f'scenario test bank {key} mismatch')
    if tuple(record['seeds']) != tuple(expected_seeds) or len(record['groups']) != len(expected_seeds):
        raise ValueError('scenario test mask count/seeds mismatch')
    views=[]
    for i,(seed,entry) in enumerate(zip(expected_seeds,record['groups'])):
        if entry['seed'] != seed or entry['group_id'] != f'group_{i:02d}':
            raise ValueError('scenario test group order mismatch')
        mask=load_mask(path.parent,entry,split='test',start_rate=record['start_rate'],
            block_len=record['block_len'],bounds=protocol.bounds.test)
        if np.array_equal(mask,protocol.fixed_masks['test']) or any(np.array_equal(mask,v.fixed_masks['test']) for v in views):
            raise ValueError('duplicate scenario test mask')
        if _missing_fractions(protocol,mask) != (entry['missing_timeline_fraction'],entry['missing_evaluation_fraction'],entry['evaluation_windows']):
            raise ValueError('scenario evaluation support mismatch')
        views.append(replace(protocol,fixed_masks={**protocol.fixed_masks,'test':mask},
            mask_sha256={**protocol.mask_sha256,'test':entry['sha256']}))
    return tuple(views)
