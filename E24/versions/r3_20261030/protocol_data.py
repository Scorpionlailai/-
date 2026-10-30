"""Portable assets, independent of the checkpoint-free migration snapshot."""
import json
import os
from pathlib import Path

import numpy as np
import torch

from malr.data.datasets import load_clean_csv
from malr.data.masks import (EpochMaskManager, MaskExpectation,
                             generate_variable_missing_mask, load_fixed_mask, save_fixed_mask)
from malr.data.normalization import fit_train_variable_stats, standardize_values
from malr.data.splits import protocol_for
from malr.data.windows import build_window_index

ROOT = Path(__file__).resolve().parent / 'protocol_references'
DATA_ROOT = Path(os.environ.get('R2_DATA_ROOT',str(Path(__file__).resolve().parent)))
SCENES = {'sr010_bl8': (.10, 8), 'sr015_bl8': (.15, 8),
          'sr020_bl8': (.20, 8), 'sr010_bl16': (.10, 16)}


def file_sha256(path):
    digest = __import__('hashlib').sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


class SharedData:
    def __init__(self, dataset, scene, assets):
        self.dataset, self.scene = dataset, scene
        self.rate, self.block = SCENES[scene]
        self.n_vars = 7 if dataset == 'ETTh1' else 21
        source = DATA_ROOT / 'data/raw' / ('ETTh1.csv' if dataset == 'ETTh1' else 'weather.csv')
        series = load_clean_csv(source, self.n_vars)
        self.bounds = protocol_for(dataset, len(series.values))
        stats = fit_train_variable_stats(torch.from_numpy(series.values[:self.bounds.train.end]), eps=0.)
        self.values = np.ascontiguousarray(standardize_values(torch.from_numpy(series.values), stats).numpy(), dtype=np.float32)
        self.timestamps = series.timestamps
        self.refs = {split: build_window_index(self.bounds, split, 96, 96)
                     for split in ('train', 'validation', 'test')}
        # Fixed legacy support: truncation uses the shared 128-window rule,
        # independently of each author's inference batch size.
        if dataset == 'ETTh1':
            for split in ('validation', 'test'):
                self.refs[split] = self.refs[split][:len(self.refs[split]) // 128 * 128]
        expected = (2688, 2688) if dataset == 'ETTh1' else (5175, 10444)
        assert tuple(len(self.refs[s]) for s in ('validation', 'test')) == expected
        self.starts = {s: np.array([r.input_start for r in refs], dtype=np.int64)
                       for s, refs in self.refs.items()}
        self.assets = Path(assets).resolve() / dataset / scene
        self.assets.mkdir(parents=True, exist_ok=True)
        first_seed = 2026100401 if dataset == 'ETTh1' else 2026100801
        self.test_seeds = list(range(first_seed, first_seed + 20))
        self.masks = {}
        mask_records = []
        for split, seeds in (('validation', [2021]), ('test', self.test_seeds)):
            bounds = self.bounds.bounds_for(split)
            start = bounds.start - 96
            length = bounds.end - start
            for seed in seeds:
                path = self.assets / f'{split}_{seed}.npz'
                expectation = MaskExpectation(dataset, split, seed, self.rate, self.block,
                                              start, bounds.end, (length, self.n_vars))
                if not path.exists():
                    mask, meta = generate_variable_missing_mask(length, self.n_vars, self.rate,
                        self.block, seed, dataset=dataset, split=split, support_start=start)
                    save_fixed_mask(path, mask, meta)
                mask, meta = load_fixed_mask(path, expectation)
                self.masks[split, seed] = mask
                mask_records.append(dict(split=split, seed=seed, path=path.name,
                    sha256=file_sha256(path), content_checksum=__import__('malr.data.masks', fromlist=['_mask_checksum'])._mask_checksum(mask, meta.canonical_json()), final_missing_rate=meta.final_rate,
                    support=[start, bounds.end]))
        self.manifest = dict(schema_version=1, protocol_id='repair-shared-legacy-v1',
            dataset=dataset, scene=scene, source_sha256=file_sha256(source),
            columns=list(series.columns), source_rows=len(series.values), seq_len=96, pred_len=96,
            splits={s: [self.bounds.bounds_for(s).start, self.bounds.bounds_for(s).end] for s in self.refs},
            windows={s: dict(count=len(self.starts[s]), first=int(self.starts[s][0]),
                            last=int(self.starts[s][-1]), sha256=__import__('hashlib').sha256(self.starts[s].tobytes()).hexdigest()) for s in self.refs},
            scaler=dict(mean=stats.mean.tolist(), std=stats.std.tolist(), fit_rows=[0, self.bounds.train.end]),
            start_rate=self.rate, block_len=self.block, masks=mask_records,
            training_masks='EpochMaskManager.mask_for_sample(epoch,target_start,length=96)',
            generator_sha256=file_sha256(Path(__import__('malr.data.masks', fromlist=['x']).__file__)),
            prior_history_at_split_boundary=True, test_seeds=self.test_seeds,
            status='exploratory_until_original_mask_identity_verified')
        # The migration snapshot excludes binary masks. Verify regenerated
        # Weather masks against every available original manifest entry.
        legacy = ROOT / 'E24/artifacts/Weather/seq96_pred96' / scene / 'shared_protocol_v1/dataset_manifest.json'
        if dataset == 'Weather' and legacy.exists():
            original = json.loads(legacy.read_text())
            assert original['source_sha256'] == self.manifest['source_sha256']
            for item in mask_records:
                prior = next(x for x in original['masks'] if (x['split'], x['seed']) == (item['split'], item['seed']))
                assert prior['checksum'] == item['content_checksum'], 'original Weather mask content differs'
                item['legacy_file_sha256'] = prior['sha256']
                item['legacy_file_sha_matches'] = prior['sha256'] == item['sha256']
            self.manifest['status'] = 'original_weather_mask_content_identity_verified'
        if dataset == 'ETTh1':
            base = ROOT / 'E23/artifacts/ETTh1/seq96_pred96'
            if scene == 'sr010_bl8':
                bank = base / 'sr010_bl8/experiments/e23_t1_five_seed_standard_20261004/test_masks/manifest.json'
                val_sha = 'cbe7c520afded9002f714c5ed8e67d62c92b97fa7aa9b1e6e4711db785fd846c'
            else:
                previous = base / 'experiments/t1_v41_higher_missing_seed502_20261004' / scene
                bank = previous / 'test_masks/manifest.json'
                val_sha = json.loads((previous / 'scenario.json').read_text())['masks']['validation']['sha256']
            prior = json.loads(bank.read_text())
            assert prior['source_csv_sha256'] == self.manifest['source_sha256']
            for item in mask_records:
                expected_sha = val_sha if item['split'] == 'validation' else next(x['sha256'] for x in prior['groups'] if x['seed'] == item['seed'])
                item['legacy_file_sha256'] = expected_sha
                item['legacy_file_sha_matches'] = item['sha256'] == expected_sha
                if item['split'] == 'test':
                    old = next(x for x in prior['groups'] if x['seed'] == item['seed'])
                    mask = self.masks['test', item['seed']]
                    support = self.bounds.test.start - 96
                    missing = sum(int((~mask[s-support:s-support+96]).sum()) for s in self.starts['test'])
                    assert item['final_missing_rate'] == old['missing_timeline_fraction']
                    assert missing / (2688 * 96 * 7) == old['missing_evaluation_fraction']
            self.manifest['status'] = ('original_etth1_mask_file_identity_verified'
                if all(x['legacy_file_sha_matches'] for x in mask_records)
                else 'rebuilt_etth1_bank_all_20_legacy_statistics_match_content_identity_unverified')
            if 'unverified' in self.manifest['status']:
                self.manifest['protocol_id'] = 'repair-shared-rebuilt-etth1-v1'
        manifest_path = self.assets / 'manifest.json'
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != self.manifest:
                raise ValueError('refusing to overwrite changed shared assets')
        else:
            write_json(manifest_path, self.manifest)
        self.manifest_sha = file_sha256(manifest_path)

    def batches(self, split, batch_size, *, epoch=0, seed=2023, mask_seed=None, max_batches=None, fixed_training=False, ordered=False):
        indices = np.arange(len(self.starts[split]))
        if split == 'train':
            if not ordered:
                np.random.default_rng(seed + epoch).shuffle(indices)
                indices = indices[:len(indices) // batch_size * batch_size]
            manager = EpochMaskManager(self.bounds.train.length, self.n_vars, self.rate, self.block, seed)
        else:
            timeline = self.masks[split, 2021 if split == 'validation' else mask_seed]
            support = self.bounds.bounds_for(split).start - 96
        for batch_no, offset in enumerate(range(0, len(indices), batch_size)):
            if max_batches is not None and batch_no >= max_batches:
                break
            starts = self.starts[split][indices[offset:offset + batch_size]]
            truth = torch.from_numpy(np.stack([self.values[s:s+96] for s in starts]))
            if split == 'train':
                masks = np.stack([manager.mask_for_sample(0 if fixed_training else epoch, int(s+96), length=96) for s in starts])
            else:
                masks = np.stack([timeline[s-support:s-support+96] for s in starts])
            observed = torch.from_numpy(masks)
            # Only history enters the adapter: future targets are never loaded.
            yield truth, observed, starts
