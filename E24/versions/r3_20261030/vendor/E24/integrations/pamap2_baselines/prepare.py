"""PAMAP2: subject isolation and causal bins inside uninterrupted activity runs."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from integrations.repair_baselines.data import ROOT, file_sha256, write_json

SENSOR_COLUMNS = [2, *range(3,16), *range(20,33), *range(37,50)]
SENSOR_NAMES = ['heart_rate'] + [site+'_'+field for site in ('hand','chest','ankle')
    for field in ('temperature','acc16_x','acc16_y','acc16_z','acc6_x','acc6_y','acc6_z',
                  'gyro_x','gyro_y','gyro_z','mag_x','mag_y','mag_z')]


def subject_split(subject):
    if 101 <= subject <= 106:return 'train'
    if subject == 107:return 'validation'
    if subject in (108,109):return 'test'
    raise ValueError('unregistered PAMAP2 subject')


def segment_runs(timestamps, labels):
    timestamps,labels=np.asarray(timestamps),np.asarray(labels)
    if timestamps.ndim!=1 or timestamps.shape!=labels.shape:
        raise ValueError('timestamps and activity labels must align')
    if not np.isfinite(timestamps).all() or not np.isfinite(labels).all():
        raise ValueError('PAMAP2 timestamp/activity cannot be missing')
    if not np.equal(labels,labels.astype(np.int64)).all():
        raise ValueError('activity labels must be integers')
    if not len(labels):return []
    delta=np.diff(timestamps)
    boundary=np.r_[True,(labels[1:]!=labels[:-1]) | (delta<=0.) | (delta>.015)]
    starts=np.flatnonzero(boundary)
    ends=np.r_[starts[1:],len(labels)]
    return [(int(a),int(b),int(labels[a])) for a,b in zip(starts,ends) if labels[a]!=0]


def aggregate_10hz(timestamps, values):
    timestamps,values=np.asarray(timestamps),np.asarray(values)
    if len(timestamps)!=len(values) or values.ndim!=2 or not len(values):
        raise ValueError('nonempty, aligned sensor observations required')
    if np.any(np.diff(timestamps)<=0):
        raise ValueError('aggregate one monotonic record segment at a time')
    # Right-closed bins end at k/10 seconds; samples after that endpoint cannot
    # enter the bin. Source timestamps have hundredth-second precision.
    bins=np.ceil(timestamps*10.-1e-6).astype(np.int64)
    frame=pd.DataFrame(values).groupby(bins,sort=True).mean()
    result=frame.to_numpy(dtype=np.float64)
    return result,np.isfinite(result),frame.index.to_numpy(dtype=np.float64)/10.


def fill_segments(values, segments, train_medians):
    medians=np.asarray(train_medians)
    if not np.isfinite(medians).all():raise ValueError('training median has a missing channel')
    filled=np.empty_like(values,dtype=np.float64)
    for segment in segments:
        a,b=segment['start'],segment['end']
        part=pd.DataFrame(values[a:b]).ffill().fillna(dict(enumerate(medians)))
        filled[a:b]=part.to_numpy()
    return filled


def record_window_starts(segments, span=192, stride=16):
    if span<=0 or stride<=0:raise ValueError('positive window span and stride required')
    result={s:[] for s in ('train','validation','test')}
    for segment in segments:
        result[segment['split']].extend(range(segment['start'],segment['end']-span+1,stride))
    return {s:np.asarray(starts,dtype=np.int64) for s,starts in result.items()}


def prepare(source, destination):
    source,destination=Path(source),Path(destination)
    manifest_path=destination/'manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if manifest['preprocessing_sha256']!=file_sha256(Path(__file__)):
            raise ValueError('prepared PAMAP2 assets belong to a different implementation')
        for name,sha in manifest['files'].items():
            if file_sha256(destination/name)!=sha:raise ValueError('corrupt prepared PAMAP2 '+name)
        return manifest
    files=sorted([*source.glob('Protocol/subject*.dat'),*source.glob('Optional/subject*.dat')],
        key=lambda p:({'train':0,'validation':1,'test':2}[subject_split(int(re.search(r'subject(\d+)',p.name).group(1)))],p.name,p.parent.name))
    if len(files)!=14:raise ValueError('PAMAP2 requires all 14 Protocol/Optional source recordings')
    expected=json.loads((ROOT/'data/metadata/PAMAP2_download_status.json').read_text())
    expected={Path(x['path']).name+'@'+Path(x['path']).parent.name:x['sha256'] for x in expected['raw_files'] if x['path'].endswith('.dat')}
    arrays,times,segments,records=[],[],[],[]
    cursor=0
    for path in files:
        digest=file_sha256(path)
        if digest!=expected[path.name+'@'+path.parent.name]:raise ValueError('source SHA differs: '+str(path))
        subject=int(re.search(r'subject(\d+)',path.name).group(1))
        split=subject_split(subject)
        frame=pd.read_csv(path,sep=r'\s+',header=None,usecols=[0,1,*SENSOR_COLUMNS],dtype=np.float64)
        timestamp=frame[0].to_numpy();activity=frame[1].to_numpy()
        values=frame.loc[:,SENSOR_COLUMNS].to_numpy()
        spans=segment_runs(timestamp,activity)
        record_id=path.parent.name+'/'+path.name
        for ordinal,(a,b,label) in enumerate(spans):
            aggregated,_,ends=aggregate_10hz(timestamp[a:b],values[a:b])
            length=len(aggregated)
            arrays.append(aggregated);times.append(ends)
            segments.append(dict(id=record_id+f'/segment{ordinal:04d}',record_id=record_id,
                subject=subject,split=split,activity_id=label,start=cursor,end=cursor+length,
                raw_start=a,raw_end=b,source_start_seconds=float(timestamp[a]),source_end_seconds=float(timestamp[b-1]),
                first_bin_end_seconds=float(ends[0]),last_bin_end_seconds=float(ends[-1]),
                window_count=max(0,(length-192)//16+1)))
            cursor+=length
        records.append(dict(id=record_id,subject=subject,split=split,raw_rows=len(frame),
            discarded_activity0_rows=int((activity==0).sum()),activity_segments=len(spans),sha256=digest))
        print(json.dumps(records[-1]),flush=True)
    raw=np.concatenate(arrays)
    native=np.isfinite(raw)
    train_end=max(s['end'] for s in segments if s['split']=='train')
    validation_end=max(s['end'] for s in segments if s['split']=='validation')
    train_medians=np.nanmedian(raw[:train_end],axis=0)
    finite=fill_segments(raw,segments,train_medians).astype(np.float32)
    if not np.isfinite(finite).all():raise ValueError('prepared sensor container must be finite')
    destination.mkdir(parents=True,exist_ok=True)
    outputs={'values.npy':finite,'native_observed.npy':native,'elapsed_seconds.npy':np.concatenate(times)}
    for name,value in outputs.items():
        temporary=destination/(name+'.tmp')
        with temporary.open('wb') as f:np.save(f,value,allow_pickle=False)
        temporary.replace(destination/name)
    starts=record_window_starts(segments)
    if any(not len(x) for x in starts.values()):raise ValueError('a subject split has no complete 96+96 windows')
    manifest=dict(schema=1,protocol_id='pamap2-subject-record-10hz-stride16-v1',dataset='PAMAP2',
        source_url='https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring',
        original_rate_hz=100,resampled_rate_hz=10,aggregation='right_closed_causal_100ms_mean_observed_only',
        raw_columns_zero_based=SENSOR_COLUMNS,columns=SENSOR_NAMES,n_vars=40,rows=len(raw),
        seq_len=96,pred_len=96,window_stride=16,
        splits=dict(train=[0,train_end],validation=[train_end,validation_end],test=[validation_end,len(raw)]),
        subjects=dict(train=list(range(101,107)),validation=[107],test=[108,109]),records=records,segments=segments,
        windows={s:len(x) for s,x in starts.items()},native_missing_count=int((~native).sum()),
        native_missing_fraction=float((~native).mean()),train_medians=train_medians.tolist(),
        filling='per_segment_forward_fill_then_observed_training_median; never labels or scoring',
        boundaries='file,subject,activity_change,activity0_removal,nonmonotonic_time,raw_gap_gt_0.015s',
        preprocessing_sha256=file_sha256(Path(__file__)),files={name:file_sha256(destination/name) for name in outputs})
    write_json(manifest_path,manifest)
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=ROOT/'data/raw/PAMAP2/PAMAP2_Dataset')
    parser.add_argument('--output',type=Path,default=ROOT/'data/processed/pamap2_10hz_recordwise')
    args=parser.parse_args()
    result=prepare(args.source,args.output)
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','segments')},indent=2))
