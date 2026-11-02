"""Sealed UCI 12-station PM2.5 protocol with native observation validity."""
from pathlib import Path
import hashlib
import json

import numpy as np
import torch

from malr.data.datasets import load_clean_csv
from malr.data.masks import EpochMaskManager, MaskExpectation, generate_variable_missing_mask, load_fixed_mask, save_fixed_mask, _mask_checksum
from malr.data.normalization import fit_train_variable_stats, standardize_values
from malr.data.splits import protocol_from_bounds
from malr.data.windows import build_window_index
from integrations.repair_baselines.data import file_sha256, write_json, SCENES

ROOT = Path(__file__).resolve().parents[3]


class SharedData:
    def __init__(self, dataset, scene, assets):
        if dataset != 'BeijingAirQuality':
            raise ValueError('this sealed pipeline supports BeijingAirQuality only')
        self.dataset, self.scene, self.n_vars = dataset, scene, 12
        self.rate, self.block = SCENES[scene]
        source = ROOT / 'data/processed/beijing_pm25_12stations.csv'
        sidecar = source.with_suffix('.natural_missing.npz')
        series = load_clean_csv(source, 12, 35064)
        with np.load(sidecar, allow_pickle=False) as z:
            self.real_observed = z['observed_mask'].copy()
            assert tuple(z['stations']) == series.columns
            assert np.array_equal(z['dates'].astype('datetime64[ns]'), series.timestamps)
        assert self.real_observed.dtype == bool and self.real_observed.shape == series.values.shape
        self.bounds = protocol_from_bounds(dataset, 35064, train_start=0,
            train_end=24544, validation_end=28050, test_end=35064)
        # The filled series is a finite container. Only original observations
        # enter losses, metrics, and the observed input channels.
        stats = fit_train_variable_stats(torch.from_numpy(series.values[:24544]), eps=0.)
        self.values = np.ascontiguousarray(standardize_values(torch.from_numpy(series.values), stats).numpy(), dtype=np.float32)
        self.timestamps = series.timestamps
        self.refs = {s: build_window_index(self.bounds, s, 96, 96) for s in ('train','validation','test')}
        self.starts = {s:np.array([r.input_start for r in refs],dtype=np.int64) for s,refs in self.refs.items()}
        assert [len(self.starts[s]) for s in self.refs] == [24353, 3411, 6919]
        self.assets = Path(assets).resolve() / dataset / scene
        self.assets.mkdir(parents=True, exist_ok=True)
        self.test_seeds = list(range(2026101301, 2026101321))
        self.masks, records = {}, []
        for split,seeds in (('validation',[2021]),('test',self.test_seeds)):
            b=self.bounds.bounds_for(split)
            start=b.start-96
            length=b.end-start
            for seed in seeds:
                path=self.assets/f'{split}_{seed}.npz'
                expected=MaskExpectation(dataset,split,seed,self.rate,self.block,start,b.end,(length,12))
                if not path.exists():
                    mask,meta=generate_variable_missing_mask(length,12,self.rate,self.block,seed,dataset=dataset,split=split,support_start=start)
                    save_fixed_mask(path,mask,meta)
                mask,meta=load_fixed_mask(path,expected)
                self.masks[split,seed]=mask
                records.append(dict(split=split,seed=seed,path=path.name,sha256=file_sha256(path),
                    content_checksum=_mask_checksum(mask,meta.canonical_json()),final_missing_rate=meta.final_rate,support=[start,b.end]))
        self.manifest=dict(schema_version=1,protocol_id='beijing-pm25-native-valid-ffill-v1',dataset=dataset,
            scene=scene,source_sha256=file_sha256(source),native_observation_sidecar_sha256=file_sha256(sidecar),
            columns=list(series.columns),source_rows=35064,seq_len=96,pred_len=96,
            splits={s:[self.bounds.bounds_for(s).start,self.bounds.bounds_for(s).end] for s in self.refs},
            windows={s:dict(count=len(x),first=int(x[0]),last=int(x[-1]),sha256=hashlib.sha256(x.tobytes()).hexdigest()) for s,x in self.starts.items()},
            scaler=dict(mean=stats.mean.tolist(),std=stats.std.tolist(),fit_rows=[0,24544],fit_series='train_only_causal_forward_filled'),
            native_missing=dict(count=int((~self.real_observed).sum()),fraction=float((~self.real_observed).mean()),
                finite_container='causal_forward_fill_then_train_median',model_input='natural_mask_AND_artificial_mask',
                supervision='originally_observed_coordinates_only',evaluation='originally_observed_coordinates_only'),
            start_rate=self.rate,block_len=self.block,masks=records,test_seeds=self.test_seeds,
            training_masks='dynamic_epoch_mask_AND_original_observation',stage2_training_masks='fixed_epoch0_cached',
            generator_sha256=file_sha256(Path(__import__('malr.data.masks',fromlist=['x']).__file__)),
            prior_history_at_split_boundary=True,status='new_common_protocol_for_all_six_models')
        path=self.assets/'manifest.json'
        if path.exists():
            if json.loads(path.read_text()) != self.manifest:
                raise ValueError('refusing to overwrite changed shared assets')
        else:
            write_json(path,self.manifest)
        self.manifest_sha=file_sha256(path)

    def valid_for(self, starts, offset=0):
        return np.stack([self.real_observed[s+offset:s+offset+96] for s in starts])

    def batches(self,split,batch_size,*,epoch=0,seed=2023,mask_seed=None,max_batches=None,fixed_training=False,ordered=False):
        indices=np.arange(len(self.starts[split]))
        if split=='train':
            if not ordered:
                np.random.default_rng(seed+epoch).shuffle(indices)
                indices=indices[:len(indices)//batch_size*batch_size]
            manager=EpochMaskManager(24544,12,self.rate,self.block,seed)
        else:
            timeline=self.masks[split,2021 if split=='validation' else mask_seed]
            support=self.bounds.bounds_for(split).start-96
        for batch_no,offset in enumerate(range(0,len(indices),batch_size)):
            if max_batches is not None and batch_no>=max_batches:
                break
            starts=self.starts[split][indices[offset:offset+batch_size]]
            truth=torch.from_numpy(np.stack([self.values[s:s+96] for s in starts]))
            if split=='train':
                masks=np.stack([manager.mask_for_sample(0 if fixed_training else epoch,int(s+96),length=96) for s in starts])
            else:
                masks=np.stack([timeline[s-support:s-support+96] for s in starts])
            observed=torch.from_numpy(masks & self.valid_for(starts))
            yield truth,observed,starts
