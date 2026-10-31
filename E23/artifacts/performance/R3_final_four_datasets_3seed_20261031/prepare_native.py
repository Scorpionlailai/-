from pathlib import Path
import sys,os,json,hashlib,shutil
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'implementation'));import bootstrap
from support import write_json,sha
DATA=ROOT/'data_root';BASE=ROOT/'baseline_references'
# Canonicalize only date serialization, preserving every observation bit.
original=Path('/home/xyw/paper/data/processed/beijing_pm25_12stations.natural_missing.npz')
assert sha(original)=='7d3ffbb9fca649df6d982029bd0d00a1954f51f5d54e53b36398bf6a7cf89cf8'
with np.load(original,allow_pickle=True) as z:arrays={k:z[k].copy() for k in z.files}
# This is the pinned existing local file; no remote/untrusted pickle is read.
arrays['dates']=pd.to_datetime(arrays['dates']).to_numpy(dtype='datetime64[ns]')
raw=Path('/home/xyw/paper/data/raw/Beijing_AirQuality/PRSA_Data_20130301-20170228')
for i,station in enumerate(arrays['stations']):
 f=next(raw.glob(f'PRSA_Data_{station}_*.csv'));frame=pd.read_csv(f)
 assert np.array_equal(frame.PM2_5.notna().to_numpy(),arrays['observed_mask'][:,i]) if 'PM2_5' in frame else np.array_equal(frame['PM2.5'].notna().to_numpy(),arrays['observed_mask'][:,i])
 dates=pd.to_datetime(frame[['year','month','day','hour']]).to_numpy(dtype='datetime64[ns]');assert np.array_equal(dates,arrays['dates'])
out=DATA/'data/processed/beijing_pm25_12stations.natural_missing.npz'
with out.open('wb') as f:np.savez_compressed(f,**arrays)
with np.load(out,allow_pickle=False) as z:assert np.array_equal(z['observed_mask'],arrays['observed_mask'])
write_json(ROOT/'preparation/BeijingAirQuality.json',dict(status='ready',original_sidecar_sha256=sha(original),canonical_sidecar_sha256=sha(out),native_mask_bitwise_same=True,native_mask_matches_all_12_raw_station_series=True,dates_match_raw=True,model_changed=False))
print('Beijing native observation and dates verified against raw 12 station files',flush=True)
# Exact original PhysioNet preparation, with raw source hashes checked by its preparer.
from integrations.physionet_baselines import prepare
prepare.ROOT=Path('/home/xyw/paper')
expected=json.loads((BASE/'PhysioNet2012/prepared_manifest.json').read_text())
out=DATA/'data/processed/physionet_15min_recordwise'
r=prepare.prepare(Path('/home/xyw/paper/data/raw/PhysioNet2012'),out)
for name,h in expected['files'].items():assert sha(out/name)==h,(name,'baseline prepared file mismatch')
for name in ['subjects','segments','records','columns','splits','windows','scaler_mean','scaler_std','train_medians']:assert r[name]==expected[name],name
# Byte-identical baseline manifest gives the original bank binding identity.
shutil.copy2(BASE/'PhysioNet2012/prepared_manifest.json',out/'manifest.json')
assert sha(out/'manifest.json')==json.loads((BASE/'PhysioNet2012/sr015_bl8/manifest.json').read_text())['prepared_manifest_sha256']
write_json(ROOT/'preparation/PhysioNet2012.json',dict(status='ready',prepared_manifest_sha256=sha(out/'manifest.json'),all_prepared_file_hashes_match_baseline=True,subjects_splits_scaler_exact=True,files=expected['files'],model_changed=False))
print('PhysioNet prepared files, records, split, stats and manifest all match original baseline',flush=True)
