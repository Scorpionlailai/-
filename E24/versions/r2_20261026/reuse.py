"""Local-only stage loading: never depends on the original experiment server."""
import json
import torch
from support import folder,sha
from variants import make_repair,make_calibrator,restore_retrieval,library_hash
from single import FrozenRepair,SingleBranchModel

def expected_updates(data,smoke=False):return 2*smoke if smoke else 100*(len(data.starts['train'])//128)
def load_main_checkpoint(data,scene,seed,variant,device,smoke=False):
    from train import binding
    path=folder(scene,seed,variant,'main',smoke)/'best.pt'
    done=json.loads((path.parent/'complete.json').read_text())
    if done['binding']!=binding(data,variant,seed,'main',smoke):raise RuntimeError('local main binding differs')
    if done['updates']!=expected_updates(data,smoke) or sha(path)!=done['checkpoints']['mse']:raise RuntimeError('main budget or hash differs')
    state=torch.load(path,map_location=device,weights_only=False)
    repair=make_repair(data,variant,seed).to(device);repair.load_state_dict(state['model'],strict=True);restore_retrieval(repair,state['retrieval']);repair.epoch=state['epoch'];repair.eval()
    if library_hash(repair)!=state['library_sha256']:raise RuntimeError('library differs')
    return repair,{'file':str(path),'sha256':sha(path),'epoch':state['epoch'],'validation':state['validation'],'reused':False,'training_seconds':done['seconds']}

def load_model_checkpoint(data,scene,seed,variant,device,metric='mae',smoke=False):
    from train import binding
    repair,coarse=load_main_checkpoint(data,scene,seed,variant,device,smoke)
    path=folder(scene,seed,variant,'calibration',smoke);file=path/f'best_{metric}.pt';done=json.loads((path/'complete.json').read_text())
    if done['binding']!=binding(data,variant,seed,'calibration',smoke,coarse['sha256']):raise RuntimeError('local calibration binding differs')
    if done['updates']!=expected_updates(data,smoke) or done['frozen_before']!=done['frozen_after'] or sha(file)!=done['checkpoints'][metric]:raise RuntimeError('calibration budget/freeze/hash differs')
    state=torch.load(file,map_location=device,weights_only=False);head=make_calibrator(data.n_vars,variant,seed).to(device);head.load_state_dict(state['head'],strict=True)
    return SingleBranchModel(FrozenRepair(repair),head).eval(),{'coarse':coarse,'head':str(file),'head_sha256':sha(file),'epoch':state['epoch'],'validation':state['validation'],'calibration_seconds':done['seconds']}
