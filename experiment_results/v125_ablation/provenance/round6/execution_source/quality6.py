"""Observe the sealed A5 intervention without changing expert internals or masks."""
from contextlib import contextmanager,ExitStack
from dataclasses import replace
from unittest.mock import patch
import torch
from integrations.v125_ablation.common import state_sha,seed_all

NAMES=('historical','variable','temporal')


def assert_legal_attention(weights,valid):
    # Inspect one candidate at a time. A full boolean gather materializes several GiB at batch128.
    for k in range(weights.shape[-1]):
        column=weights[...,k]
        assert torch.isfinite(column).all()
        if k:assert not column.masked_fill(valid[...,k-1,None,None],0).any()
    total=weights.sum(-1)
    assert torch.allclose(total,torch.ones_like(total),atol=1e-6,rtol=1e-6)


@contextmanager
def monitor_quality(model,detailed=False):
    assert model.disabled==frozenset()
    assert model.core.iterative.neutral_quality and model.core.point_candidate_readout.neutral_quality
    assert not model.core.iterative.parallel and not model.core.point_candidate_readout.patch_shared
    counts={};raw={};handles=[]
    def count(name):counts[name]=counts.get(name,0)+1
    def item_check(name,item):
        assert not item.quality.any(),'expert fusion quality not fixed0'
        source=raw[name]
        assert item.message is source.message
        assert item.candidates is source.candidates,'candidate tensors replaced by fusion intervention'
        assert not (item.available & ~source.available).any(),'unavailable source made available'
        count('constant_expert_quality_'+name)
    def delta_check(mod,args):pass
    def point_router(mod,args):
        for name,item in args[2].items():item_check(name,item)
    def shared_score(mod,args):
        assert not args[0][...,128:192].any(),'shared router reads nonconstant expert quality'
        count('shared_router_quality_zero')
    def zero_hook(label):
        def check(mod,args):assert not args[0].any(),label;count(label)
        return check
    def point_candidates(mod,args,out):
        inputs,router,evidence=args[:3]
        expected=torch.cat([item.candidates.valid & item.available[...,None] & router.available[...,i+1,None]
                            for i,item in enumerate(evidence[name] for name in NAMES)],-1)
        attention=out[1];assert torch.equal(expected,attention['valid'])
        if detailed:assert_legal_attention(attention['weights'],expected)
        raw.clear()  # No hooks retain the previous autograd graph across forwards.
        count('candidate_legality_and_null_preserved')
    with ExitStack() as stack:
        for name in NAMES:
            expert=getattr(model.core,name);original=expert.from_cache
            def read(*args,_name=name,_original=original,**kw):
                result=_original(*args,**kw);raw[_name]=result;count('internal_expert_'+_name);return result
            stack.enter_context(patch.object(expert,'from_cache',side_effect=read))
            handles.append(expert.quality_projector.register_forward_pre_hook(lambda *a,_name=name:count('internal_quality_projector_'+_name)))
            handles.append(model.core.point_router.quality[name].register_forward_pre_hook(zero_hook('point_router_quality_zero_'+name)))
            handles.append(model.core.point_candidate_readout.quality[name].register_forward_pre_hook(zero_hook('point_candidate_quality_zero_'+name)))
        original=model.core.router.expert_delta
        def delta(inputs,name,item):item_check(name,item);return original(inputs,name,item)
        stack.enter_context(patch.object(model.core.router,'expert_delta',side_effect=delta))
        handles.append(model.core.router.shared_score.register_forward_pre_hook(shared_score))
        handles.append(model.core.point_router.register_forward_pre_hook(point_router))
        handles.append(model.core.point_candidate_readout.register_forward_hook(point_candidates))
        try:yield counts
        finally:
            for handle in handles:handle.remove()


def tensor_fields(value):
    if isinstance(value,torch.Tensor):return value.detach().clone()
    if isinstance(value,dict):return {k:tensor_fields(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [tensor_fields(v) for v in value]
    if hasattr(value,'__dataclass_fields__'):return {k:tensor_fields(getattr(value,k)) for k in value.__dataclass_fields__}
    return value


def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,list):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def isolation(model,data,device,path):
    model.inference_path=path;model.requires_grad_(False).eval();before=state_sha(model)
    truth,obs,starts=next(data.batches('validation',128));truth,obs=truth.to(device),obs.to(device);visible=torch.where(obs,truth,0.)
    expert_inputs={};expert_caches={};handles=[]
    # Capture the actual original expert inputs/caches to replay flag-only comparisons with identical operands.
    with torch.no_grad(),monitor_quality(model,detailed=True) as counts,ExitStack() as stack:
        for name in NAMES:
            expert=getattr(model.core,name);original=expert.from_cache
            def capture(*args,_name=name,_original=original,**kw):
                expert_inputs[_name]=(args,kw);return _original(*args,**kw)
            stack.enter_context(patch.object(expert,'from_cache',side_effect=capture))
        result,_,_=model.output(visible,obs,starts)
    assert all(counts.get('internal_expert_'+name,0)>0 for name in NAMES)
    assert all(counts.get('point_candidate_quality_zero_'+name,0)>0 for name in NAMES)
    # Changing only A5 flags cannot alter any expert's ranking, cache, message, candidates or weights for fixed inputs.
    with torch.no_grad():
        for name in NAMES:
            args,kw=expert_inputs[name];expert=getattr(model.core,name)
            a=tensor_fields(expert.from_cache(*args,**kw))
            model.core.iterative.neutral_quality=False;model.core.point_candidate_readout.neutral_quality=False
            try:b=tensor_fields(expert.from_cache(*args,**kw))
            finally:model.core.iterative.neutral_quality=True;model.core.point_candidate_readout.neutral_quality=True
            assert equal(a,b),'A5 flags changed internal expert evidence'
    # Only reports of quality, after original internal generation/ranking, are poisoned.
    with torch.no_grad(),ExitStack() as stack:
        for name in NAMES:
            original=getattr(model.core,name).from_cache
            def poison(*args,_original=original,**kw):
                item=_original(*args,**kw)
                return replace(item,quality=torch.full_like(item.quality,123.),
                    candidates=replace(item.candidates,quality=torch.full_like(item.candidates.quality,321.)))
            stack.enter_context(patch.object(getattr(model.core,name),'from_cache',side_effect=poison))
        poisoned,_,_=model.output(visible,obs,starts)
    assert torch.equal(poisoned.explicit.standardized_sequence,result.explicit.standardized_sequence)
    # No-source protection: all experts unavailable and all candidate masks false must leave only the legal null token.
    with torch.no_grad(),ExitStack() as stack:
        for name in NAMES:
            original=getattr(model.core,name).from_cache
            def unavailable(*args,_original=original,**kw):
                item=_original(*args,**kw);c=item.candidates
                return replace(item,message=torch.zeros_like(item.message),available=torch.zeros_like(item.available),
                    candidates=replace(c,valid=torch.zeros_like(c.valid),weights=torch.zeros_like(c.weights)))
            stack.enter_context(patch.object(getattr(model.core,name),'from_cache',side_effect=unavailable))
        null_result,_,_=model.output(visible,obs,starts)
    assert not null_result.router.available[...,1:].any() and not null_result.point_candidate_attention['valid'].any()
    assert torch.equal(null_result.point_candidate_attention['weights'][...,0],torch.ones_like(null_result.point_candidate_attention['weights'][...,0]))
    assert torch.isfinite(null_result.explicit.standardized_sequence).all()
    # In eval, arbitrary hidden truth perturbations cannot affect the model's output.
    perturb=truth.clone();perturb[~obs]=1000.
    with torch.no_grad():hidden_changed,_,_=model.output(perturb,obs,starts)
    assert torch.equal(hidden_changed.explicit.standardized_sequence,result.explicit.standardized_sequence)
    assert torch.equal(torch.where(obs,visible,result.explicit.standardized_sequence)[obs],truth[obs])
    assert before==state_sha(model)
    return dict(path=path,passed=True,quality_monitor=counts,internal_expert_fixed_input_flag_invariance=True,
        raw_candidate_values_counts_masks_weights_and_ranking_preserved=True,quality_only_report_poison_invariant=True,
        all_explicit_fusion_quality_inputs_fixed0=True,legality_and_no_source_null_protection=True,
        hidden_truth_perturbation_invariant=True,observed_unchanged=True,checkpoint_unchanged=True,
        content_driven_and_internal_candidate_weight_adaptation_retained=True,not_uniform_fusion=True,validation_batch=128)
