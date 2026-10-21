"""No-H runtime sentinels; never change model/data/optimizer definitions."""
from contextlib import ExitStack,contextmanager
from unittest.mock import patch
import torch
from integrations.v125_ablation.common import state_sha


H_PREFIXES=('core.historical.','ema_encoder.','frozen_encoder.',
    'core.router.residual.historical.','core.router.scales.historical.',
    'core.point_router.quality.historical.','core.point_candidate_readout.quality.historical.')


@contextmanager
def forbid_history(model):
    assert model.disabled==frozenset(('historical',))
    calls={}
    def blocked(label):
        calls[label]=0
        def fail(*a,**kw):
            calls[label]+=1
            raise AssertionError('A2 forbidden historical computation/mutation: '+label)
        return fail
    from malr.prototypes.tensor_index import TensorPrototypeIndex
    import malr.prototypes.retrieval as retrieval_module
    import integrations.physionet_baselines.losses as loss_module
    with ExitStack() as stack:
        targets=[(model.core.historical,n,'H.'+n) for n in ('prepare_candidates','from_cache','forward')]
        targets += [(model.core.historical.key_encoder,'forward','H.key_encoder.forward'),
                    (model.ema_encoder,'forward','H.ema_encoder.forward'),(model.frozen_encoder,'forward','H.frozen_encoder.forward')]
        targets += [(model.core.historical.retrieval,n,'H.retrieval.'+n) for n in ('begin_epoch','update_ema','encode_frozen','encode_ema')]
        targets += [(TensorPrototypeIndex,n,'H.index.'+n) for n in ('query','query_two_groups')]
        targets += [(model.memory,n,'H.memory.'+n) for n in ('stage_patch','rekey','rekey_batch','publish_epoch')]
        targets += [(retrieval_module,'masked_pair_loss','H.masked_pair_loss'),
                    (loss_module,'chunked_contrastive_loss','H.chunked_contrastive_loss')]
        for obj,name,label in targets:stack.enter_context(patch.object(obj,name,side_effect=blocked(label)))
        yield calls
    assert not any(calls.values())


class InaccessibleHistory:
    def __getattr__(self,name):raise AssertionError('A2 accessed poisoned historical store/index: '+name)


def isolation(model,data,device,path):
    model.inference_path=path;model.requires_grad_(False).eval()
    assert model.core.history_adapter is None and model.core.stacked is None
    truth,obs,starts=next(data.batches('validation',128));truth,obs=truth.to(device),obs.to(device)
    visible=torch.where(obs,truth,0.);before=state_sha(model);positive={};handles=[]
    def hook(name):
        def count(*args):positive[name]=positive.get(name,0)+1
        return count
    for name,module in [('front',model.core.front_context),('V_prepare',model.core.variable),('T_prepare',model.core.temporal)]:
        if name=='front':handles.append(module.register_forward_pre_hook(hook(name)))
    with torch.no_grad(),forbid_history(model) as blocked,ExitStack() as stack:
        for obj,n,label in [(model.core.variable,'prepare_sources','V_prepare'),(model.core.variable,'from_cache','V_message'),
                            (model.core.temporal,'prepare_sources','T_prepare'),(model.core.temporal,'from_cache','T_message')]:
            original=getattr(obj,n)
            def record(*args,_original=original,_label=label,**kw):
                positive[_label]=positive.get(_label,0)+1
                return _original(*args,**kw)
            stack.enter_context(patch.object(obj,n,side_effect=record))
        result,_,_=model.output(visible,obs,starts)
        h=result.evidence['historical']
        assert not h.message.any() and not h.quality.any() and not h.available.any()
        assert not h.candidates.valid.any() and not h.candidates.weights.any() and not h.candidates.messages.any()
        assert not result.router.deltas['historical'].any() and not result.router.available[...,1].any() and not result.router.weights[...,1].any()
        assert not result.router.point_weights[...,1].any()
        attention=result.point_candidate_attention;count=attention['candidate_counts'][0]
        assert not attention['weights'][...,1:1+count].any() and not attention['valid'][...,:count].any()
        for iteration in result.iterations:assert not iteration.allowed[...,0].any() and not iteration.weights[...,1].any()
        assert all(positive.get(k,0)>0 for k in ('front','V_prepare','V_message','T_prepare','T_message'))
        assert torch.equal(result.explicit.sequence_observed,obs)
        actual=torch.where(obs,visible,result.explicit.standardized_sequence)
        assert torch.equal(actual[obs],truth[obs])
        saved={n:p.detach().clone() for n,p in model.named_parameters() if n.startswith(H_PREFIXES)}
        memory,index=model.memory,model.core.historical.index
        try:
            for n,p in model.named_parameters():
                if n in saved:p.fill_(123.)
            model.memory=InaccessibleHistory();model.core.historical.index=InaccessibleHistory()
            poisoned,_,_=model.output(visible,obs,starts)
            assert torch.equal(poisoned.explicit.standardized_sequence,result.explicit.standardized_sequence)
        finally:
            model.memory=memory;model.core.historical.index=index
            for n,p in model.named_parameters():
                if n in saved:p.copy_(saved[n])
    for handle in handles:handle.remove()
    assert before==state_sha(model)
    return dict(path=path,passed=True,forbidden_H_calls=blocked,public_front_VT_calls=positive,
        H_messages_quality_candidates_route_weights_deltas_zero=True,H_point_candidate_and_point_router_weights_zero=True,
        H_serial_and_final_state_contribution_zero=True,H_parameters_and_inaccessible_memory_index_poison_output_unchanged=True,
        retrieval_and_key_encoders_not_executed=True,checkpoint_state_unchanged=True,observed_unchanged=True,validation_batch=128)
