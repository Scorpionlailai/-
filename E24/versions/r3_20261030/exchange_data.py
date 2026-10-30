"""Exchange row-index protocol, matching the sealed 2026-10-10 split/masks."""
import os,json
from pathlib import Path
import numpy as np
import torch
import bootstrap
from protocol_data import SharedData as ContinuousData, SCENES, file_sha256, write_json
from malr.data.masks import MaskExpectation,generate_variable_missing_mask,save_fixed_mask,load_fixed_mask
from malr.data.normalization import fit_train_variable_stats,standardize_values
from malr.data.splits import protocol_from_bounds
from malr.data.windows import build_window_index

class SharedData(ContinuousData):
    def __init__(self,dataset,scene,assets):
        if dataset!='Exchange-Rate':raise ValueError(dataset)
        self.dataset,self.scene=dataset,scene
        self.rate,self.block=SCENES[scene];self.n_vars=8
        source=Path(os.environ['R2_DATA_ROOT'])/'data/raw/Exchange-Rate/exchange_rate.txt'
        raw=np.loadtxt(source,delimiter=',',dtype=np.float64)
        if raw.shape!=(7588,8) or not np.isfinite(raw).all():raise ValueError('Expected finite 7588 x 8 raw Exchange-Rate matrix')
        self.bounds=protocol_from_bounds(dataset,7588,train_start=0,train_end=5311,validation_end=6070,test_end=7588)
        stats=fit_train_variable_stats(torch.from_numpy(raw[:5311]),eps=0.)
        self.values=np.ascontiguousarray(standardize_values(torch.from_numpy(raw),stats).numpy(),dtype=np.float32)
        self.timestamps=np.arange(7588)
        self.refs={s:build_window_index(self.bounds,s,96,96) for s in ['train','validation','test']}
        self.starts={s:np.array([r.input_start for r in rr],dtype=np.int64) for s,rr in self.refs.items()}
        self.assets=Path(assets).resolve()/dataset/scene;self.assets.mkdir(parents=True,exist_ok=True)
        self.test_seeds=list(range(2026101001,2026101021));self.masks={};records=[]
        for split,seeds in [('validation',[2021]),('test',self.test_seeds)]:
            bounds=self.bounds.bounds_for(split);start=bounds.start-96;length=bounds.end-start
            for seed in seeds:
                path=self.assets/f'{split}_{seed}.npz'
                expect=MaskExpectation(dataset,split,seed,self.rate,self.block,start,bounds.end,(length,8))
                if not path.exists():
                    mask,meta=generate_variable_missing_mask(length,8,self.rate,self.block,seed,dataset=dataset,split=split,support_start=start)
                    save_fixed_mask(path,mask,meta)
                mask,meta=load_fixed_mask(path,expect);self.masks[split,seed]=mask
                records.append(dict(split=split,seed=seed,path=path.name,sha256=file_sha256(path),final_missing_rate=meta.final_rate))
        self.manifest=dict(protocol_id='Exchange-Rate-R3-row-index-v1',dataset=dataset,scene=scene,source_sha256=file_sha256(source),rows=7588,n_vars=8,seq_len=96,pred_len=96,start_rate=self.rate,block_len=self.block,splits={s:[self.bounds.bounds_for(s).start,self.bounds.bounds_for(s).end] for s in self.refs},windows={s:len(x) for s,x in self.starts.items()},train_scaler=dict(mean=stats.mean.tolist(),std=stats.std.tolist(),fit_rows=[0,5311]),test_seeds=self.test_seeds,masks=records,status='row_index_protocol_regenerated_masks',timestamp_policy='row index only, no actual trading-day dates')
        path=self.assets/'manifest.json'
        if path.exists() and json.loads(path.read_text())!=self.manifest:raise ValueError('changed Exchange assets')
        if not path.exists():write_json(path,self.manifest)
        self.manifest_sha=file_sha256(path)
