from dataclasses import replace
import pytest
import torch

from malr.models.e23.feature_tv import SharedTemporalFeatureBank, FeatureTVHController
from tests.unit.test_e23_iterative import _parts


def _bank():
    controller, inputs, parts = _parts(1)
    return SharedTemporalFeatureBank(3, 8, 6, 3, 9), inputs, parts


def test_channel_attention_excludes_self_and_unobserved_sources():
    bank, inputs, _ = _bank()
    observed = inputs.patch_mask.clone()
    observed[:, 2] = False
    inputs = replace(inputs, patch_mask=observed)
    _, _, valid, weights = bank.variable_transfer(inputs, inputs.base_latents, None)
    assert not valid[..., 2].any()
    assert not weights[..., 2, :].any()
    for target in range(3):
        assert not weights[:, target, :, target].any()
    torch.testing.assert_close(weights.sum(-2), valid.any(-1)[..., None].expand_as(weights.sum(-2)).float())
    assert (weights.var(-1)>0).any()  # Feature channels actually make different choices.


def test_hidden_point_values_cannot_change_feature_messages():
    bank, inputs, _ = _bank()
    polluted = replace(inputs, patch_values=inputs.patch_values.masked_fill(~inputs.patch_mask, 10000))
    first = bank.variable_transfer(inputs, inputs.base_latents, None)
    second = bank.variable_transfer(polluted, inputs.base_latents, None)
    for a, b in zip(first, second):torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_all_missing_feature_sources_have_finite_zero_attention():
    bank, inputs, _ = _bank()
    inputs = replace(inputs, patch_mask=torch.zeros_like(inputs.patch_mask))
    message, quality, valid, weights = bank.variable_transfer(inputs, inputs.base_latents, None)
    assert torch.isfinite(message).all() and torch.isfinite(quality).all()
    assert not valid.any() and not weights.any()


def test_feature_projection_and_channel_queries_receive_gradients():
    bank, inputs, _ = _bank()
    message, _, _, weights = bank.variable_transfer(inputs, inputs.base_latents, None)
    (message.square().sum()+weights.square().sum()).backward()
    for layer in [bank.input_projection, bank.long_filter, bank.query, bank.key, bank.value]:
        assert layer.weight.grad is not None
        assert torch.isfinite(layer.weight.grad).all()
        assert layer.weight.grad.abs().sum()>0


def test_second_layer_reencodes_updated_state(monkeypatch):
    _, inputs, parts = _parts(1)
    controller = FeatureTVHController(3, 8, 6, 3, 9, 2)
    seen=[]
    method = controller.feature_banks[1].temporal_evidence
    def capture(current,*args):
        seen.append(current.base_latents)
        return method(current,*args)
    monkeypatch.setattr(controller.feature_banks[1], 'temporal_evidence', capture)
    out=controller(inputs, **parts)
    assert len(out.iterations)==2
    torch.testing.assert_close(seen[0], out.iterations[0].latents, rtol=0, atol=0)
    assert not torch.equal(seen[0],inputs.base_latents)
    assert seen[0].grad_fn is not None


@pytest.mark.parametrize('disabled',['temporal','variable'])
def test_disabled_branch_cannot_leak_through_preview(monkeypatch,disabled):
    _, inputs, parts = _parts(1)
    controller = FeatureTVHController(3, 8, 6, 3, 9, 2)
    first=controller(inputs,**parts,disabled_experts=frozenset([disabled]))
    method=parts['router'].expert_delta
    def corrupt(arg,name,item):
        value=method(arg,name,item)
        return value+1000 if name==disabled else value
    monkeypatch.setattr(parts['router'],'expert_delta',corrupt)
    second=controller(inputs,**parts,disabled_experts=frozenset([disabled]))
    torch.testing.assert_close(first.router.latents,second.router.latents,rtol=0,atol=0)
