"""Record-disjoint, causal quarter-hour PhysioNet 2012 forecasting protocol."""
import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

from integrations.repair_baselines.data import ROOT, file_sha256, write_json


def author_columns():
    tree = ast.parse((ROOT/'CSDI/dataset_physio.py').read_text())
    return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='attributes' for t in n.targets))


def record_splits(record_ids, seed=2023):
    ids = np.array(sorted(map(str,record_ids)))
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate ICU record IDs')
    np.random.default_rng(seed).shuffle(ids)
    train, validation = int(len(ids)*.7), int(len(ids)*.1)
    return {str(v): ('train' if i<train else 'validation' if i<train+validation else 'test')
            for i,v in enumerate(ids)}


def aggregate_record(frame, columns):
    selected = frame.loc[frame.Parameter.isin(columns)].copy()
    selected['Value'] = pd.to_numeric(selected.Value,errors='coerce')
    selected = selected.loc[np.isfinite(selected.Value) & selected.Value.ge(0)]
    selected['_minute'] = selected.Time.map(lambda s: int(s.split(':')[0])*60+int(s.split(':')[1]))
    selected = selected.loc[selected._minute.between(0,2880)].sort_values('_minute',kind='stable')
    selected['_bin'] = np.maximum(0,(selected._minute.to_numpy(dtype=np.int64)+14)//15-1)
    grouped = selected.groupby(['_bin','Parameter'],sort=False).Value.last()
    values = np.full((192,len(columns)),np.nan,dtype=np.float64)
    lookup = {v:i for i,v in enumerate(columns)}
    for (index,name),value in grouped.items():
        values[int(index),lookup[name]] = value
    return values,np.isfinite(values),np.arange(1,193,dtype=np.float64)*900.


def fill_segments(values, segments, train_medians):
    if not np.isfinite(train_medians).all():
        raise ValueError('a channel has no training observations')
    filled = np.empty_like(values,dtype=np.float64)
    for segment in segments:
        a,b = segment['start'],segment['end']
        filled[a:b] = pd.DataFrame(values[a:b]).ffill().fillna(dict(enumerate(train_medians))).to_numpy()
    return filled


def observed_training_stats(values, valid, train_end):
    if values.shape != valid.shape or not 0<train_end<=len(values):
        raise ValueError('invalid training statistics support')
    means,stds = [],[]
    for column in range(values.shape[1]):
        observed = values[:train_end,column][valid[:train_end,column]]
        if not len(observed) or not np.isfinite(observed).all():
            raise ValueError('a channel has no finite training observations')
        means.append(float(observed.mean()))
        deviation=float(observed.std(ddof=0))
        stds.append(deviation if deviation>0 else 1.)
    return np.asarray(means),np.asarray(stds)


def record_window_starts(segments, span=192, stride=192):
    result={s:[] for s in ('train','validation','test')}
    for segment in segments:
        result[segment['split']].extend(range(segment['start'],segment['end']-span+1,stride))
    return {s:np.asarray(v,dtype=np.int64) for s,v in result.items()}


def prepare(source,destination):
    source,destination=Path(source),Path(destination)
    manifest_path=destination/'manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if manifest['preprocessing_sha256']!=file_sha256(Path(__file__)):
            raise ValueError('prepared PhysioNet implementation changed')
        for name,sha in manifest['files'].items():
            if file_sha256(destination/name)!=sha:raise ValueError('corrupt prepared '+name)
        return manifest
    paths=sorted(source.glob('set-*/*.txt'))
    if len(paths)!=12000:raise ValueError('requires all 12000 official ICU records')
    identity=json.loads((ROOT/'data/metadata/PhysioNet2012_download_status.json').read_text())
    expected={str(Path(v['path']).relative_to('raw/PhysioNet2012')):v['sha256'] for v in identity['raw_files']}
    split=record_splits([p.stem for p in paths])
    order={'train':0,'validation':1,'test':2}
    paths.sort(key=lambda p:(order[split[p.stem]],p.stem))
    columns=author_columns();arrays=[];segments=[];records=[];cursor=0
    for number,path in enumerate(paths,1):
        relative=str(path.relative_to(source));sha=file_sha256(path)
        if sha!=expected[relative]:raise ValueError('raw source identity differs: '+relative)
        frame=pd.read_csv(path)
        values,valid,_=aggregate_record(frame,columns)
        group=split[path.stem]
        arrays.append(values)
        segments.append(dict(id=path.stem,record_id=path.stem,subject=path.stem,split=group,
            start=cursor,end=cursor+192,window_count=1,source_set=path.parent.name))
        records.append(dict(id=path.stem,split=group,source_set=path.parent.name,sha256=sha,
            observed_history=int(valid[:96].sum()),observed_future=int(valid[96:].sum())))
        cursor+=192
        if number%2000==0:print(json.dumps(dict(records_processed=number,total=12000)),flush=True)
    raw=np.concatenate(arrays);del arrays
    native=np.isfinite(raw)
    train_end=max(s['end'] for s in segments if s['split']=='train')
    validation_end=max(s['end'] for s in segments if s['split']=='validation')
    medians=np.nanmedian(raw[:train_end],axis=0)
    mean,std=observed_training_stats(raw,native,train_end)
    finite=fill_segments(raw,segments,medians).astype(np.float32)
    if not np.isfinite(finite).all():raise ValueError('nonfinite prepared placeholders')
    destination.mkdir(parents=True,exist_ok=True)
    outputs={'values.npy':finite,'native_observed.npy':native,
        'elapsed_seconds.npy':np.tile(np.arange(1,193,dtype=np.float64)*900.,12000)}
    for name,value in outputs.items():
        temporary=destination/(name+'.tmp')
        with temporary.open('wb') as stream:np.save(stream,value,allow_pickle=False)
        temporary.replace(destination/name)
    starts=record_window_starts(segments)
    manifest=dict(schema=1,protocol_id='physionet-record-15min-96to96-v1',dataset='PhysioNet2012',
        source_url='https://physionet.org/content/challenge-2012/1.0.0/',
        columns=columns,n_vars=len(columns),rows=len(raw),seq_len=96,pred_len=96,window_stride=192,
        aggregation='right_closed_15min_last_finite_nonnegative_observation; minute0_in_first_bin',
        interval_minutes=15,history_hours=24,forecast_hours=24,
        splits=dict(train=[0,train_end],validation=[train_end,validation_end],test=[validation_end,len(raw)]),
        subjects={s:sorted(k for k,v in split.items() if v==s) for s in starts},
        grouping_unit='unique_deidentified_ICU_RecordID_not_reidentified_person',
        records=records,segments=segments,windows={s:len(v) for s,v in starts.items()},
        record_split_seed=2023,record_split_ratios=[.7,.1,.2],
        native_missing_count=int((~native).sum()),native_missing_fraction=float((~native).mean()),
        empty_dynamic_record_ids=[r['id'] for r in records if r['observed_history']+r['observed_future']==0],
        records_without_history=[r['id'] for r in records if r['observed_history']==0],
        records_without_future=[r['id'] for r in records if r['observed_future']==0],
        empty_record_policy='retain_every_record; absent_truth_has_zero_loss_weight_and_no_score_coordinates',
        train_medians=medians.tolist(),scaler_mean=mean.tolist(),scaler_std=std.tolist(),
        scaler_fit='originally_observed_training_records_only; no_filled_or_validation_or_test_values',
        filling='within_record_forward_fill_then_observed_training_median; placeholders_never_truth',
        boundaries='RecordID,48h_ICU_record; one_window_per_record; no_cross_record_history',
        author_protocol_difference='15min96to96_all12000_fixed_split; BRITS_CSDI_author_hourly48_setA_imputation',
        outcomes_used=False,preprocessing_sha256=file_sha256(Path(__file__)),
        files={name:file_sha256(destination/name) for name in outputs})
    write_json(manifest_path,manifest)
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=ROOT/'data/raw/PhysioNet2012')
    parser.add_argument('--output',type=Path,default=ROOT/'data/processed/physionet_15min_recordwise')
    args=parser.parse_args();result=prepare(args.source,args.output)
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','segments','subjects')},indent=2))
