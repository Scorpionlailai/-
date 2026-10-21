"""Exact state comparisons, isolated complete RNG streams and original outer coordinates."""
import hashlib
import json
import numpy as np
import torch
from integrations.v125_ablation.common import rng_state,restore_rng
from integrations.v125_ablation.model.coordinates import repair_scale


def tree_hash(value):
    h=hashlib.sha256()
    def walk(x):
        if isinstance(x,torch.Tensor):
            h.update(b'tensor');h.update(str(x.dtype).encode());h.update(str(tuple(x.shape)).encode());h.update(x.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(x,np.ndarray):
            h.update(b'ndarray');h.update(str(x.dtype).encode());h.update(str(x.shape).encode());h.update(x.tobytes())
        elif isinstance(x,dict):
            h.update(b'dict')
            for key in sorted(x,key=lambda k:(type(k).__name__,str(k))):walk(key);walk(x[key])
        elif isinstance(x,(tuple,list)):
            h.update(type(x).__name__.encode());h.update(str(len(x)).encode())
            for a in x:walk(a)
        else:h.update(type(x).__name__.encode());h.update(repr(x).encode())
    walk(value);return h.hexdigest()


def stream_call(state,function):
    outer=rng_state();restore_rng(state)
    try:return function()
    finally:
        after=rng_state();state.clear();state.update(after);restore_rng(outer)


def finish_calibration(explicit,norm,visible,observed,scaling):
    # Same operation order as the sealed core.forward wrapper and V125Adapter.forward.
    if scaling=='legacy_v1':candidate=explicit.standardized_sequence
    else:
        scale=repair_scale(norm.std)
        normalized=explicit.normalized_sequence*(scale/norm.std)[:,None]
        safe=torch.where(observed,norm.student,0.)
        normalized=torch.where(explicit.sequence_observed,safe,normalized).masked_fill(~explicit.sequence_validity,0)
        candidate=normalized*norm.std[:,None]+norm.mean[:,None]
    return torch.where(observed,visible,candidate)


def compare_trees(reference,candidate):
    failures=[];maximum=0.
    def walk(a,b,path):
        nonlocal maximum
        if isinstance(a,torch.Tensor):
            if not isinstance(b,torch.Tensor) or a.dtype!=b.dtype or a.shape!=b.shape:failures.append(path+':tensor type/shape');return
            a,b=a.detach().cpu(),b.detach().cpu()
            if not torch.equal(a,b):
                delta=float((a.double()-b.double()).abs().max()) if a.numel() else 0.;maximum=max(maximum,delta);failures.append(path)
        elif isinstance(a,np.ndarray):
            if not isinstance(b,np.ndarray) or a.dtype!=b.dtype or not np.array_equal(a,b):failures.append(path)
        elif isinstance(a,dict):
            if not isinstance(b,dict) or a.keys()!=b.keys():failures.append(path+':dict keys');return
            for k in a:walk(a[k],b[k],path+'.'+str(k))
        elif isinstance(a,(tuple,list)):
            if type(a)!=type(b) or len(a)!=len(b):failures.append(path+':sequence');return
            for i,(x,y) in enumerate(zip(a,b)):walk(x,y,path+f'[{i}]')
        elif a!=b:failures.append(path)
    walk(reference,candidate,'root')
    return dict(exact_equal=not failures,max_abs_difference=maximum,failures=failures[:20])
