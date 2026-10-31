from pathlib import Path
import sys,os,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'implementation'))
from data import load_data
from support import sha,write_json
os.environ['R2_DATA_ROOT']=str(ROOT/'data_root')
PAPER=Path('/home/xyw/paper');scenes=['sr015_bl8','sr020_bl8','sr010_bl8','sr010_bl16']
for ds in sys.argv[1:] or ['ETTh1','BeijingAirQuality','PhysioNet2012']:
 rows=[]
 for scene in scenes:
  d=load_data(ds,scene,ROOT/'outputs/assets');r=d.manifest
  if ds=='ETTh1':
   base=PAPER/'E23/artifacts/ETTh1/seq96_pred96'
   if scene=='sr010_bl8':
    bank=base/'sr010_bl8/experiments/e23_t1_five_seed_standard_20261004/test_masks'
    val=PAPER/'E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context/masks/ETTh1_validation.npz'
    expected_val='cbe7c520afded9002f714c5ed8e67d62c92b97fa7aa9b1e6e4711db785fd846c'
   else:
    bank=base/'experiments/t1_v41_higher_missing_seed502_20261004'/scene/'test_masks'
    parent=bank.parent;val=parent/'masks/ETTh1_validation.npz'
    expected_val=json.loads((parent/'scenario.json').read_text())['masks']['validation']['sha256']
   old=json.loads((bank/'manifest.json').read_text());assert sha(val)==expected_val
   with np.load(val,allow_pickle=False) as z:assert np.array_equal(z['observed_mask'],d.masks['validation',2021])
   for item in old['groups']:
    p=bank/item['path'];assert sha(p)==item['sha256']
    with np.load(p,allow_pickle=False) as z:assert np.array_equal(z['observed_mask'],d.masks['test',item['seed']])
   n=len(old['groups'])+1
  else:
   old=json.loads((ROOT/'baseline_references'/ds/scene/'manifest.json').read_text())
   for key in ['splits','windows','scaler','test_seeds','start_rate','block_len','columns','source_rows','generator_sha256']:
    assert r[key]==old[key],(ds,scene,key)
   if ds=='BeijingAirQuality':
    assert r['source_sha256']==old['source_sha256'];assert r['native_missing']==old['native_missing']
   else:
    assert r['prepared_manifest_sha256']==old['prepared_manifest_sha256'];assert r['native_observation_sha256']==old['native_observation_sha256']
   for item in old['masks']:
    generated=next(x for x in r['masks'] if (x['split'],x['seed'])==(item['split'],item['seed']))
    if ds=='BeijingAirQuality':assert generated['content_checksum']==item['content_checksum']
    else:assert generated['content_sha256']==item['content_sha256']
   n=len(old['masks'])
  rows.append(dict(scene=scene,validation_and_test_masks_equal=n,test_seeds=d.test_seeds,windows={s:len(v) for s,v in d.starts.items()},manifest_sha256=d.manifest_sha))
 write_json(ROOT/'audit'/f'{ds}_mask_alignment.json',dict(status='passed',dataset=ds,scenes=rows,training_masks='Inherited dynamic epoch masks, same generator and seed2023; no model-specific artificial masking',original_native_labels_only=True,model_changed=False))
 print(ds,'all four scenes: original baseline validation/test masks aligned',flush=True)
