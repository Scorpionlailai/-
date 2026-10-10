from dataclasses import replace

import pytest
import torch

from malr.models.e23.historical import HistoricalExpert
from tests.unit.test_e23_historical import _inputs, _memory


def _pair(**kwargs):
    from malr.models.e23.historical_candidates import HistoricalCandidateProvider
    from malr.models.e23.historical_messages import HistoricalMessageEncoder

    old = HistoricalExpert(4, 8, 8, 1, contextual=True,
                           similarity_threshold=-1, **kwargs)
    provider = HistoricalCandidateProvider(4, 8, 8, 1,
                                          similarity_threshold=-1, **kwargs)
    message = HistoricalMessageEncoder(4, 8, contextual=True)
    provider.load_state_dict({k: v for k, v in old.state_dict().items()
                              if k.startswith('key_encoder.')}, strict=True)
    message.load_state_dict({k: v for k, v in old.state_dict().items()
                             if not k.startswith('key_encoder.')}, strict=True)
    provider.retrieval.load_state_dict(old.retrieval.state_dict())
    old.refresh_snapshot(_memory())
    provider.refresh_snapshot(_memory())
    return old, provider, message


def test_provider_has_only_retrieval_parameters():
    _, provider, message = _pair()
    assert provider.state_dict()
    assert all(k.startswith('key_encoder.') for k in provider.state_dict())
    assert 'key_encoder' not in dict(message.named_modules())
    assert not hasattr(message, 'retrieval') and not hasattr(message, 'index')


@pytest.mark.parametrize('options', [{}, {'top_p': .9, 'candidate_cap': 8}])
@pytest.mark.parametrize('observed', [False, True])
def test_provider_cache_matches_existing_historical_cache(options, observed):
    old, provider, _ = _pair(**options)
    expected = old.prepare_candidates(_inputs(observed=observed))
    actual = provider.prepare_candidates(_inputs(observed=observed))
    for key in ('normalized_raw', 'raw_quality', 'base_logits', 'valid', 'available'):
        torch.testing.assert_close(getattr(actual, key), getattr(expected, key), rtol=0, atol=0)
    assert actual.empty == expected.empty
    assert actual.provenance.keys() == expected.provenance.keys()
    for key in actual.provenance:
        torch.testing.assert_close(actual.provenance[key], expected.provenance[key], rtol=0, atol=0)
    starts = actual.provenance['candidate_starts'][actual.valid]
    assert starts.tolist() == ([0] if observed else [])


def test_message_encoder_matches_existing_message_and_gradients():
    old, provider, message = _pair()
    # Nonzero query exercises the contextual path as well as the MLPs.
    with torch.no_grad():
        old.context_query.weight.normal_(std=.1)
        message.context_query.weight.copy_(old.context_query.weight)
    original = _inputs()
    left = original.base_latents.clone().requires_grad_()
    right = original.base_latents.clone().requires_grad_()
    expected = old.from_cache(replace(original, base_latents=left), old.prepare_candidates(original))
    actual = message.from_cache(replace(original, base_latents=right), provider.prepare_candidates(original))
    for field in ('message', 'quality', 'available'):
        torch.testing.assert_close(getattr(actual, field), getattr(expected, field), rtol=0, atol=0)
    torch.testing.assert_close(actual.candidates.weights, expected.candidates.weights, rtol=0, atol=0)
    (expected.message.square().sum() + expected.quality.square().sum()).backward()
    (actual.message.square().sum() + actual.quality.square().sum()).backward()
    torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
    old_params = dict(old.named_parameters())
    for name, parameter in message.named_parameters():
        torch.testing.assert_close(parameter.grad, old_params[name].grad, rtol=0, atol=0)


def test_message_empty_bank_retains_candidate_geometry():
    _, provider, message = _pair(top_p=.9, candidate_cap=8)
    provider.index = None
    actual = message.from_cache(_inputs(), provider.prepare_candidates(_inputs()))
    assert not actual.available.any()
    assert actual.candidates.messages.shape == (1, 1, 1, 8, 8)
    assert torch.equal(actual.message, torch.zeros_like(actual.message))
