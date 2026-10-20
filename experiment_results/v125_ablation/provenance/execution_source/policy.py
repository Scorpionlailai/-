"""Source, freeze, initialization and summary invariants for the approved protocol."""
import hashlib
import json
import numpy as np
import torch
from integrations.v125_ablation.common import seed_all
from integrations.v125_ablation.run import configure_phase
from integrations.physionet_baselines.losses import hidden_error,mean_or_zero

MEMBERS='core.recent_mean_projector.members.'

def tensor_hash(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        h.update(k.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def frozen_hash(state,active_index=None):
    exclude=MEMBERS if active_index is None else MEMBERS+str(active_index)+'.'
    return tensor_hash({k:v for k,v in state.items() if not k.startswith(exclude)})

def object_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def source_matches(expected,actual):
    fields=('coarse_sha256','library_sha256','protocol_sha256','training_config_sha256','arm','model_seed')
    return all(k in expected and k in actual and expected[k]==actual[k] for k in fields)

def initialize_branch(model,index,seed):
    member=model.core.recent_mean_projector.members[index]
    inherited=model.coarse_projector.state_dict();before=member.state_dict()
    novel={k:v for k,v in before.items() if k not in inherited}
    result=member.load_state_dict(inherited,strict=False)
    assert not result.unexpected_keys
    assert set(result.missing_keys)==set(novel)
    params=configure_phase(model,'calibration'+str((64,256)[index]));seed_all(seed);model.epoch=100
    for p in model.parameters():p.grad=None
    allstate=member.state_dict()
    return params,dict(inherited_tensor_hash=tensor_hash({k:allstate[k] for k in inherited}),
        source_coarse_mean_tensor_hash=tensor_hash(inherited),novel_tensor_hash=tensor_hash(novel),
        complete_initial_tensor_hash=tensor_hash(allstate),inherited_keys=sorted(inherited),novel_keys=sorted(novel),
        trainable_parameters=sum(p.numel() for p in params),model_seed=seed)

def calibration_loss(candidate,truth,observed,valid):
    return mean_or_zero(hidden_error(candidate,truth,observed,valid).abs())+.25*mean_or_zero(
        hidden_error(candidate[:,-32:],truth[:,-32:],observed[:,-32:],valid[:,-32:]).abs())

def paired_summary(rows):
    out={}
    for arm in ('A0','A1'):
        take=sorted([x for x in rows if x['arm']==arm],key=lambda x:x['model_seed'])
        means=[float(np.mean(x['values'])) for x in take]
        out[arm]=dict(training_seeds=[x['model_seed'] for x in take],training_means=means,
            n_training_seeds=len(means),mean=float(np.mean(means)) if means else None,
            training_std_ddof1=float(np.std(means,ddof=1)) if len(means)>1 else None)
    maps={arm:dict(zip(out[arm]['training_seeds'],out[arm]['training_means'])) for arm in ('A0','A1')}
    common=sorted(set(maps['A0']) & set(maps['A1']))
    out['paired_A1_minus_A0']=[maps['A1'][s]-maps['A0'][s] for s in common]
    return out
