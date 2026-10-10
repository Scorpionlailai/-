from dataclasses import replace

import torch

from malr.models.e23.historical import HistoricalExpert
from tests.unit.test_e23_historical import _inputs, _memory


def _expert():
    return HistoricalExpert(4, 8, 8, 1, similarity_threshold=-1., contextual=True)


def test_h_cache_queries_once_and_messages_depend_on_current_state(monkeypatch):
    # Catches repeated retrieval, a detached message graph, or stale conditioning.
    torch.manual_seed(8)
    expert = _expert()
    expert.refresh_snapshot(_memory())
    calls = []
    query = expert.index.query
    def counted(*args, **kwargs):
        calls.append(1)
        return query(*args, **kwargs)
    monkeypatch.setattr(expert.index, "query", counted)
    inputs = _inputs()
    cache = expert.prepare_candidates(inputs)
    first = expert.from_cache(inputs, cache)
    changed = replace(inputs, base_latents=inputs.base_latents * 2)
    second = expert.from_cache(changed, cache)
    third = expert.from_cache(inputs, cache)
    assert calls == [1]
    assert not torch.allclose(first.message, second.message)
    torch.testing.assert_close(first.message, third.message)
    assert first.provenance["candidate_starts"][cache.valid].tolist() == [0]
    assert first.candidates.weights[~cache.valid].sum() == 0
    first.message.square().mean().backward()
    grad = expert.message_projector[0].weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_h_cache_rebuilds_after_mask_or_snapshot_change():
    expert = _expert()
    expert.refresh_snapshot(_memory())
    present = expert.prepare_candidates(_inputs())
    absent = expert.prepare_candidates(_inputs(observed=False))
    assert present.valid.any() and not absent.valid.any()
    expert.index = None
    empty = expert.prepare_candidates(_inputs())
    assert not empty.valid.any()
    # Preparing a new call must not mutate the previous call's cache.
    assert present.valid.any()


def test_h_cache_empty_and_unobserved_queries_stay_unavailable():
    expert = _expert()
    inputs = _inputs()
    cache = expert.prepare_candidates(inputs)
    assert cache.normalized_raw.shape == (1, 1, 1, 4, 4)
    assert not expert.from_cache(inputs, cache).available.any()
    expert.refresh_snapshot(_memory())
    no_obs = _inputs(observed=False)
    assert not expert.from_cache(no_obs, expert.prepare_candidates(no_obs)).available.any()


def test_h_context_scores_cannot_restore_invalid_candidates():
    expert = _expert()
    expert.refresh_snapshot(_memory())
    inputs = _inputs()
    cache = expert.prepare_candidates(inputs)
    with torch.no_grad():
        expert.context_query.weight.fill_(1.)
    out = expert.from_cache(replace(inputs, base_latents=torch.randn_like(inputs.base_latents)), cache)
    assert torch.equal(out.candidates.valid, cache.valid)
    assert out.candidates.weights[~cache.valid].sum() == 0
    torch.testing.assert_close(out.candidates.weights.sum(-1), torch.ones_like(out.available, dtype=torch.float))
