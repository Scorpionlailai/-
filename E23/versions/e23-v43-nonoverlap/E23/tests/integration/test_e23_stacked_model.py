from dataclasses import replace
import copy

import pytest
import torch

from malr.models.e23.model import MultiExpertRepairEncoder
from tests.integration.test_e23_model import _inputs


def model(layers=2,groups=2,checkpoint=False,schedule='stacked_parallel_tv_h_v1'):
    return MultiExpertRepairEncoder(2,24,8,4,d_model=8,retrieval_dim=8,
        historical_threshold=-1,variable_selection='window_context_topk_v1',
        recovery_schedule=schedule,recovery_rounds=1,recovery_layers=layers,
        expert_groups=groups,activation_checkpointing=checkpoint)


def arguments():
    values,mask,valid,starts,mean,std=_inputs()
    return (values,mask,mean,std,starts),{'valid_mask':valid}


def test_stacked_model_preserves_observed_values_and_ignores_hidden_inputs():
    torch.manual_seed(22); repair=model(); args,kwargs=arguments()
    before=repair(*args,**kwargs)
    values=args[0].clone(); values[~(args[1] & kwargs['valid_mask'])]=999999
    after=repair(values,*args[1:],**kwargs)
    torch.testing.assert_close(before.router.latents,after.router.latents,rtol=0,atol=0)
    observed=args[1] & kwargs['valid_mask']
    torch.testing.assert_close(before.explicit.normalized_sequence[observed],args[0][observed],rtol=0,atol=0)
    assert len(before.stack_layers)==2
    assert not hasattr(repair,'variable') and not hasattr(repair,'router')
    assert all(k.startswith('key_encoder.') for k in repair.historical.state_dict())


def test_v29_single_group_reference_matches_output_and_gradients():
    from malr.models.e23.stacked_reference import copy_v29_r1_into_stack_reference
    torch.manual_seed(12)
    source=model(1,1,schedule='iterative_tv_h_v1')
    target=model(1,1)
    copy_v29_r1_into_stack_reference(source,target)
    args,kwargs=arguments()
    left=source(*args,**kwargs); right=target(*args,**kwargs)
    torch.testing.assert_close(left.router.latents,right.router.latents,rtol=0,atol=1e-6)
    torch.testing.assert_close(left.explicit.normalized_sequence,right.explicit.normalized_sequence,rtol=0,atol=1e-6)
    left.explicit.normalized_sequence.square().mean().backward()
    right.explicit.normalized_sequence.square().mean().backward()
    torch.testing.assert_close(source.base_encoder.weight.grad,target.base_encoder.weight.grad,rtol=0,atol=1e-6)
    group=target.stacked.layers[0].groups[0]
    for name in ('variable','temporal','router'):
        for p,q in zip(getattr(source,name).parameters(),getattr(group,name).parameters()):
            assert (p.grad is None)==(q.grad is None)
            if p.grad is not None: torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=1e-6)
    with pytest.raises(ValueError): copy_v29_r1_into_stack_reference(source,model())


def test_checkpointed_layers_match_normal_output_and_gradients_without_extra_h_queries(monkeypatch):
    repair=model(); recomputed=copy.deepcopy(repair); recomputed.stacked.activation_checkpointing=True
    calls=[]; method=recomputed.historical.prepare_candidates
    def counted(*args,**kwargs): calls.append(1); return method(*args,**kwargs)
    monkeypatch.setattr(recomputed.historical,'prepare_candidates',counted)
    args,kwargs=arguments()
    left=repair(*args,**kwargs); right=recomputed(*args,**kwargs)
    torch.testing.assert_close(left.router.latents,right.router.latents,rtol=0,atol=0)
    left.explicit.normalized_sequence.square().mean().backward()
    right.explicit.normalized_sequence.square().mean().backward()
    assert calls==[1]
    for p,q in zip(repair.parameters(),recomputed.parameters()):
        assert (p.grad is None)==(q.grad is None)
        if p.grad is not None: torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)


def test_legacy_model_has_no_stack_state_keys():
    repair=model(1,1,schedule='iterative_tv_h_v1')
    assert not any(k.startswith('stacked.') for k in repair.state_dict())


def test_stacked_public_candidate_counts_are_finite_descriptive_group_means():
    repair=model(); args,kwargs=arguments(); output=repair(*args,**kwargs)
    for item in output.evidence.values():
        count=item.provenance['candidate_count']
        assert count.shape==output.router.latents.shape[:-1]
        assert torch.isfinite(count).all() and (count>=0).all()
