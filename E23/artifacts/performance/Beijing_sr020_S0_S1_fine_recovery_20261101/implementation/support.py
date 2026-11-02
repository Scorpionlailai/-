import hashlib,json,random,os
from pathlib import Path
import numpy as np
import torch
from bootstrap import ROOT
TASK=ROOT/'README.md'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def save(path,state):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');torch.save(state,tmp);tmp.replace(path)
def files_binding():
    return {'files':{p.name:sha(p) for p in sorted(ROOT.glob('*.py'))},'source_lock':sha(ROOT/'vendor_lock.json'),'task':sha(TASK),'protocol':sha(ROOT/'protocol.json')}
def rng_state():return {'python':random.getstate(),'numpy':np.random.get_state(),'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
def restore_rng(s):
    random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['cpu'].cpu())
    if s['cuda']:torch.cuda.set_rng_state_all([x.cpu() for x in s['cuda']])
def digest(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):
        h.update(k.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def recover(folder,epoch):
    rows=[json.loads((folder/'epochs'/f'{i:03}.json').read_text()) for i in range(1,epoch+1)]
    assert [x['epoch'] for x in rows]==list(range(1,epoch+1))
    tmp=folder/'curve.jsonl.tmp';tmp.write_text(''.join(json.dumps(x)+'\n' for x in rows));tmp.replace(folder/'curve.jsonl')
def artifact_root():return Path(os.environ.get('R2_OUTPUT_ROOT',str(ROOT/'outputs'))).resolve()
def folder(scene,seed,variant,stage,smoke=False):return artifact_root()/('smoke' if smoke else 'runs')/os.environ['A64_DATASET']/scene/f'seed_{seed}'/variant/stage
