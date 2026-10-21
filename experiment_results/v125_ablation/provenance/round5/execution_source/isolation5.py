"""No-V computation sentinels; preserve public cross-variable context and H/T."""
from contextlib import ExitStack,contextmanager
from unittest.mock import patch
import torch
from integrations.v125_ablation.common import state_sha

V_PREFIXES=('core.variable.','core.router.residual.variable.','core.router.scales.variable.',
    'core.point_router.quality.variable.','core.point_candidate_readout.quality.variable.')


@contextmanager
def forbid_variable(model):
    assert model.disabled==frozenset(('variable',))
    calls={}
    def blocked(label):
        calls[label]=0
        def fail(*a,**kw):
            calls[label]+=1
            raise AssertionError('A3 forbidden V source/message computation: '+label)
        return fail
    with ExitStack() as stack:
        for name in ('prepare_sources','from_cache','forward'):
            stack.enter_context(patch.object(model.core.variable,name,side_effect=blocked('V.'+name)))
        for name,module in model.core.variable.named_modules():
            if name:stack.enter_context(patch.object(module,'forward',side_effect=blocked('V.module.'+name)))
        yield calls
    assert not any(calls.values())


def isolation(model,data,device,path):
    model.inference_path=path;model.requires_grad_(False).eval()
    assert model.core.history_adapter is None and model.core.stacked is None
    assert model.core.point_candidate_readout.variable_readout is None
    assert model.core.front_context.variable_context is not None
    truth,obs,starts=next(data.batches('validation',128));truth,obs=truth.to(device),obs.to(device)
    visible=torch.where(obs,truth,0.);before=state_sha(model);positive={};states={};handles=[]
    def hook(name):
        def count(*args):positive[name]=positive.get(name,0)+1
        return count
    for name,module in [('front',model.core.front_context),('front_cross_variable',model.core.front_context.variable_context)]:
        handles.append(module.register_forward_pre_hook(hook(name)))
    with torch.no_grad(),forbid_variable(model) as blocked,ExitStack() as stack:
        for obj,n,label in [(model.core.historical,'prepare_candidates','H_prepare'),(model.core.historical,'from_cache','H_message'),
                            (model.core.temporal,'prepare_sources','T_prepare'),(model.core.temporal,'from_cache','T_message'),
                            (model.core.router,'expert_delta','expert_delta'),(model.core.router,'preview_gate','preview_gate')]:
            original=getattr(obj,n)
            def record(*args,_original=original,_label=label,**kw):
                positive[_label]=positive.get(_label,0)+1
                if _label=='T_message':states['before_T']=args[0].base_latents.detach().clone()
                if _label=='H_message':states['before_H']=args[0].base_latents.detach().clone()
                result=_original(*args,**kw)
                if _label in ('expert_delta','preview_gate'):
                    assert args[1]!='variable','V router update unexpectedly executed'
                    if args[1]=='temporal':states[_label]=result.detach().clone()
                return result
            stack.enter_context(patch.object(obj,n,side_effect=record))
        result,_,_=model.output(visible,obs,starts)
        v=result.evidence['variable']
        assert not v.message.any() and not v.quality.any() and not v.available.any()
        assert not v.candidates.valid.any() and not v.candidates.weights.any() and not v.candidates.messages.any()
        assert not result.router.deltas['variable'].any() and not result.router.available[...,2].any() and not result.router.weights[...,2].any()
        assert not result.router.point_weights[...,2].any()
        attention=result.point_candidate_attention;offset=attention['candidate_counts'][0];count=attention['candidate_counts'][1]
        assert not attention['weights'][...,1+offset:1+offset+count].any() and not attention['valid'][...,offset:offset+count].any()
        for iteration in result.iterations:
            assert not iteration.allowed[...,1].any() and not iteration.weights[...,2].any() and not iteration.preview_gates[...,1].any()
        assert torch.equal(states['before_H'],states['before_T']+states['preview_gate'][...,None]*states['expert_delta'])
        assert all(positive.get(k,0)>0 for k in ('front','front_cross_variable','H_prepare','H_message','T_prepare','T_message'))
        assert torch.equal(result.explicit.sequence_observed,obs)
        assert torch.equal(torch.where(obs,visible,result.explicit.standardized_sequence)[obs],truth[obs])
        saved={n:p.detach().clone() for n,p in model.named_parameters() if n.startswith(V_PREFIXES)}
        # V rows in shared type embeddings are also V-specific operands; masked rows cannot affect other experts.
        embedding_rows=[model.core.router.type_embedding.weight,model.core.point_candidate_readout.kind.weight]
        row_copies=[p[1].detach().clone() for p in embedding_rows]
        try:
            for n,p in model.named_parameters():
                if n in saved:p.fill_(123.)
            for p in embedding_rows:p[1].fill_(123.)
            poisoned,_,_=model.output(visible,obs,starts)
            assert torch.equal(poisoned.explicit.standardized_sequence,result.explicit.standardized_sequence)
        finally:
            for n,p in model.named_parameters():
                if n in saved:p.copy_(saved[n])
            for p,value in zip(embedding_rows,row_copies):p[1].copy_(value)
    for handle in handles:handle.remove()
    assert before==state_sha(model)
    return dict(path=path,passed=True,forbidden_V_calls=blocked,public_front_HT_calls=positive,
        public_cross_variable_signature_context_preserved=True,V_source_preparation_and_message_computation_skipped=True,
        V_messages_quality_candidates_route_weights_deltas_zero=True,V_point_candidate_and_point_router_weights_zero=True,
        V_serial_update_absent_H_receives_exact_afterT_state=True,V_parameters_and_shared_type_rows_poison_output_unchanged=True,
        checkpoint_state_unchanged=True,observed_unchanged=True,validation_batch=128)
