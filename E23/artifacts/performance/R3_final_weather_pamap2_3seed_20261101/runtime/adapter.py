"""Run the immutable published R3 with this server's baseline mask policy."""
from pathlib import Path
import sys,os,json,hashlib
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'snapshot/E24/versions/r3_20261030'
sys.path.insert(0,str(SOURCE))

def install():
 import data,support
 import baseline_pamap_data
 original=data.load_data
 def aligned_data(dataset,scene,assets):
  if dataset=='PAMAP2':
   baseline_pamap_data.ROOT=Path(os.environ['R2_DATA_ROOT'])
   return baseline_pamap_data.SharedData(dataset,scene,assets)
  return original(dataset,scene,assets)
 data.load_data=aligned_data
 original_binding=support.files_binding
 def bound():
  return {**original_binding(),'execution_lock_sha256':support.sha(ROOT/'execution_lock.json'),'mask_comparison_protocol_sha256':support.sha(ROOT/'protocol.json')}
 support.files_binding=bound
 return data
