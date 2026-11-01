"""Exact existing baseline masks, windows, scales, source and training-mask checks."""
import os,json,hashlib,gc
from pathlib import Path
import numpy as np
from adapter import ROOT,install
os.environ['R2_DATA_ROOT']='/home/xyw/paper'
data=install()
from support import write_json,sha
BASE={'Weather':Path('/home/xyw/paper/E24/artifacts/repair_reproduction_20261010/shared_assets'),'PAMAP2':Path('/home/xyw/paper/E24/artifacts/pamap2_reproduction_20261014/shared_assets')}

def run():
 rows=[]
 for dataset in ['Weather','PAMAP2']:
  for scene in ['sr010_bl8','sr015_bl8','sr020_bl8','sr010_bl16']:
   d=data.load_data(dataset,scene,ROOT/'outputs/assets');base=BASE[dataset]/dataset/scene;b=json.loads((base/'manifest.json').read_text());checks=[]
   assert d.test_seeds==b['test_seeds']
   for split,seed in d.masks:
    p=base/f'{split}_{seed}.npz';npz=np.load(p,allow_pickle=False)
    key='observed' if 'observed' in npz.files else 'observed_mask'
    assert np.array_equal(d.masks[split,seed],npz[key]),str(p)
    copy=ROOT/'outputs/assets'/dataset/scene/p.name;assert sha(copy)==sha(p)
    checks.append({'split':split,'seed':seed,'file_sha256':sha(p),'array_sha256':hashlib.sha256(d.masks[split,seed].tobytes()).hexdigest()})
   for split in ['train','validation','test']:
    assert hashlib.sha256(d.starts[split].tobytes()).hexdigest()==b['windows'][split]['sha256']
   for name in ['source_sha256','prepared_manifest_sha256','splits','scaler']:
    if name in b:assert d.manifest[name]==b[name],(dataset,scene,name)
   # Exact common EpochMaskManager source and training batches policy is unchanged.
   assert d.manifest['generator_sha256']==b['generator_sha256']
   streams=[]
   for epoch in [0,49,99]:
    _,seen,starts=next(d.batches('train',128,epoch=epoch,seed=2023))
    streams.append({'epoch':epoch,'starts_sha256':hashlib.sha256(starts.tobytes()).hexdigest(),'mask_sha256':hashlib.sha256(seen.numpy().tobytes()).hexdigest()})
   rows.append({'dataset':dataset,'scene':scene,'baseline_manifest':str(base/'manifest.json'),'baseline_manifest_sha256':sha(base/'manifest.json'),'mask_files':checks,'all_21_masks_byte_and_array_exact':True,'source_splits_scaler_windows_match':True,'training_generator_sha_matches':True,'training_stream_samples':streams,'updates_per_epoch':len(d.starts['train'])//128})
   del d;gc.collect()
 write_json(ROOT/'baseline_alignment.json',{'datasets':['Weather','PAMAP2'],'rows':rows,'commit':'59419e9bbc8f2e8833c8265505d0e84fa2b04de4'});print('ALL_EIGHT_SCENES_BASELINE_PROTOCOLS_MATCH',flush=True)
if __name__=='__main__':run()
