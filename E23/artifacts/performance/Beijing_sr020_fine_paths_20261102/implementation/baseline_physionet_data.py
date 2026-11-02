"""Shared record-disjoint PhysioNet2012 windows and record-local missing-mask banks."""
import fcntl
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from malr.data.masks import EpochMaskManager, generate_variable_missing_mask, _mask_checksum
from malr.data.normalization import TrainVariableStats, standardize_values
from malr.data.splits import protocol_from_bounds
from malr.data.windows import WindowRef
from integrations.repair_baselines.data import ROOT, SCENES, file_sha256, write_json
from integrations.physionet_baselines.prepare import record_window_starts


def record_seed(seed, record):
    return int.from_bytes(hashlib.sha256((str(seed)+'|'+record).encode()).digest()[:4],'little')


class SharedData:
    def __init__(self,dataset,scene,assets):
        if dataset!='PhysioNet2012':raise ValueError('this sealed pipeline supports PhysioNet2012 only')
        self.dataset,self.scene=dataset,scene
        self.rate,self.block=SCENES[scene]
        prepared=ROOT/'data/processed/physionet_15min_recordwise'
        prepared_path=prepared/'manifest.json'
        source=json.loads(prepared_path.read_text())
        self.prepared_sha=file_sha256(prepared_path)
        for name,digest in source['files'].items():
            if file_sha256(prepared/name)!=digest:raise ValueError('prepared PhysioNet2012 file changed: '+name)
        self.n_vars=source['n_vars']
        self.segments=source['segments']
        self.segment_ends=np.array([s['end'] for s in self.segments],dtype=np.int64)
        self.real_observed=np.load(prepared/'native_observed.npy',mmap_mode='r',allow_pickle=False)
        self.elapsed_seconds=np.load(prepared/'elapsed_seconds.npy',mmap_mode='r',allow_pickle=False)
        finite=np.load(prepared/'values.npy',mmap_mode='r',allow_pickle=False)
        self.bounds=protocol_from_bounds('PhysioNet2012',len(finite),train_start=0,
            train_end=source['splits']['train'][1],validation_end=source['splits']['validation'][1],test_end=len(finite))
        tensor=torch.from_numpy(np.array(finite,dtype=np.float64))
        stats=TrainVariableStats(torch.tensor(source['scaler_mean'],dtype=torch.float64),
            torch.tensor(source['scaler_std'],dtype=torch.float64))
        self._fallback=TrainVariableStats(torch.zeros(self.n_vars),torch.ones(self.n_vars))
        self._reference_fallback=self._fallback
        self.values=np.ascontiguousarray(standardize_values(tensor,stats).numpy(),dtype=np.float32)
        # PhysioNet2012 supplies relative seconds, not calendar timestamps. The adapters
        # use zero calendar covariates; elapsed time is retained for boundary audit.
        self.timestamps=np.datetime64('1970-01-01')+(np.asarray(self.elapsed_seconds)*1e9).astype('timedelta64[ns]')
        self.starts=record_window_starts(self.segments)
        self.refs={split:[WindowRef(int(s),int(s+96),int(s+96),int(s+192),split) for s in starts]
                   for split,starts in self.starts.items()}
        for split,starts in self.starts.items():
            if not len(starts):raise ValueError('empty PhysioNet2012 subject split '+split)
            ids=self.segment_indices(starts)
            for start,index in zip(starts,ids):
                segment=self.segments[int(index)]
                if segment['split']!=split or start+192>segment['end']:
                    raise AssertionError('window crosses a recording or split')
        self.assets=Path(assets).resolve()/dataset/scene
        self.assets.mkdir(parents=True,exist_ok=True)
        self.test_seeds=list(range(2026101501,2026101521))
        self.masks={};records=[]
        generator=Path(__import__('malr.data.masks',fromlist=['x']).__file__)
        self.generator_sha=file_sha256(generator)
        with (self.assets/'asset.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            for split,seeds in (('validation',[2021]),('test',self.test_seeds)):
                for seed in seeds:
                    mask,metadata,path=self.bank(split,seed)
                    self.masks[split,seed]=mask
                    records.append(dict(split=split,seed=seed,path=path.name,sha256=file_sha256(path),
                        content_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),support=metadata['support'],
                        record_local=True,actual_artificial_missing_fraction=float((~mask).mean())))
            self.manifest=dict(schema_version=1,protocol_id=source['protocol_id'],dataset=dataset,scene=scene,
                prepared_manifest_sha256=self.prepared_sha,columns=source['columns'],n_vars=self.n_vars,
                source_rows=len(finite),seq_len=96,pred_len=96,window_stride=192,
                splits=source['splits'],subjects=source['subjects'],
                segment_window_counts={s['id']:s['window_count'] for s in self.segments},
                windows={s:dict(count=len(x),first=int(x[0]),last=int(x[-1]),sha256=hashlib.sha256(x.tobytes()).hexdigest()) for s,x in self.starts.items()},
                scaler=dict(mean=stats.mean.tolist(),std=stats.std.tolist(),fit_rows=[0,self.bounds.train.end],fit_series='training_only_original_observations'),
                native_missing_count=source['native_missing_count'],native_observation_sha256=source['files']['native_observed.npy'],
                missing_input='native_observation_AND_record_local_artificial_mask',
                labels='originally_observed_only_for_repair_and_forecast',
                start_rate=self.rate,block_len=self.block,masks=records,test_seeds=self.test_seeds,
                training_masks='per_segment_EpochMaskManager_local_target_seeded_by_segment_id',
                record_seed_rule='little_endian_first4bytes_SHA256(str(seed)+pipe+segment_id)',
                stage2_training_masks='fixed_epoch0_cached',prior_history_at_split_boundary=False,
                cross_record_windows=False,generator_sha256=self.generator_sha,
                status='record_ID_disjoint_ICU_boundaries_verified')
            path=self.assets/'manifest.json'
            if path.exists() and json.loads(path.read_text())!=self.manifest:
                raise ValueError('refusing to overwrite changed PhysioNet2012 shared assets')
            if not path.exists():write_json(path,self.manifest)
            self.manifest_sha=file_sha256(path)

    def segment_indices(self,starts):
        return np.searchsorted(self.segment_ends,np.asarray(starts),side='right')

    def bank(self,split,seed):
        bounds=self.bounds.bounds_for(split)
        path=self.assets/f'{split}_{seed}.npz'
        expected=dict(dataset=self.dataset,split=split,seed=seed,start_rate=self.rate,block_len=self.block,
            n_vars=self.n_vars,prepared_sha256=self.prepared_sha,generator_sha256=self.generator_sha,
            support=[bounds.start,bounds.end])
        if path.exists():
            with np.load(path,allow_pickle=False) as saved:
                mask=saved['observed'].copy();metadata=json.loads(str(saved['metadata']))
            if metadata['binding']!=expected:raise ValueError('record-local mask binding differs')
        else:
            mask=np.ones((bounds.length,self.n_vars),dtype=bool);parts=[]
            for segment in self.segments:
                if segment['split']!=split or not segment['window_count']:continue
                seed_local=record_seed(seed,segment['id'])
                length=segment['end']-segment['start']
                local,meta=generate_variable_missing_mask(length,self.n_vars,self.rate,self.block,seed_local,
                    dataset=self.dataset,split=split,support_start=segment['start'])
                a,b=segment['start']-bounds.start,segment['end']-bounds.start
                mask[a:b]=local
                parts.append(dict(segment_id=segment['id'],seed=seed_local,metadata_checksum=_mask_checksum(local,meta.canonical_json())))
            metadata=dict(binding=expected,records=parts,support=[bounds.start,bounds.end])
            temporary=path.with_suffix('.npz.tmp')
            with temporary.open('wb') as stream:
                np.savez_compressed(stream,observed=mask,metadata=np.asarray(json.dumps(metadata,sort_keys=True)))
            temporary.replace(path)
        if mask.shape!=(bounds.length,self.n_vars) or mask.dtype!=bool:
            raise ValueError('record-local mask has wrong shape or dtype')
        return mask,metadata,path

    def valid_for(self,starts,offset=0):
        for start,index in zip(starts,self.segment_indices(starts)):
            segment=self.segments[int(index)]
            if start<segment['start'] or start+offset+96>segment['end']:
                raise AssertionError('requested supervision crosses a recording')
        return np.stack([self.real_observed[s+offset:s+offset+96] for s in starts])

    def batches(self,split,batch_size,*,epoch=0,seed=2023,mask_seed=None,max_batches=None,fixed_training=False,ordered=False):
        indices=np.arange(len(self.starts[split]))
        if split=='train':
            if not ordered:
                np.random.default_rng(seed+epoch).shuffle(indices)
                indices=indices[:len(indices)//batch_size*batch_size]
            managers={}
            for i,segment in enumerate(self.segments):
                if segment['split']=='train' and segment['window_count']:
                    managers[i]=EpochMaskManager(segment['end']-segment['start'],self.n_vars,self.rate,self.block,record_seed(seed,segment['id']))
        else:
            timeline=self.masks[split,2021 if split=='validation' else mask_seed]
            support=self.bounds.bounds_for(split).start
        for batch_no,offset in enumerate(range(0,len(indices),batch_size)):
            if max_batches is not None and batch_no>=max_batches:break
            starts=self.starts[split][indices[offset:offset+batch_size]]
            truth=torch.from_numpy(np.stack([self.values[s:s+96] for s in starts]))
            if split=='train':
                masks=np.stack([managers[int(i)].mask_for_sample(0 if fixed_training else epoch,
                    int(s-self.segments[int(i)]['start']+96),length=96) for s,i in zip(starts,self.segment_indices(starts))])
            else:
                masks=np.stack([timeline[s-support:s-support+96] for s in starts])
            yield truth,torch.from_numpy(masks & self.valid_for(starts)),starts
