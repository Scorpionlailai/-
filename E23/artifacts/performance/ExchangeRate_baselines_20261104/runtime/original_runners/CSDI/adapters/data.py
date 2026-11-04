from pathlib import Path
import json
from exchange_v125_protocol import read_manifest,with_test_mask,protocol_for
def load_dataset_protocol(manifest,*,config_path,artifacts):
 r=read_manifest(manifest)
 assert Path(config_path)==Path(r['config_path']) and Path(artifacts)==Path(manifest).parent
 return protocol_for({'dataset_manifest':str(manifest)},dict(start_rate=r['start_rate'],block_len=r['block_len']))
