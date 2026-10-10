"""Behavioral checks for TV-conditioned retrieval of fully hidden patches."""
from dataclasses import replace
import numpy as np
import torch

from malr.models.e23.contracts import ExpertInputs
from malr.models.e23.historical import HistoricalExpert
from malr.prototypes.online_bank import OnlinePrototypeMemory


def inputs():
    mask = torch.tensor([[[[True, True, True, True], [False, False, False, False]]]])
    values = torch.tensor([[[[1., 2., 3., 4.], [0., 0., 0., 0.]]]])
    state = torch.zeros(1, 1, 2, 8)
    state[0, 0, 1, 0] = 1.
    state[0, 0, 1, 7] = -1.
    return ExpertInputs(state, values, mask, torch.ones_like(mask), mask.float().mean(-1),
                        torch.tensor([[28, 32]]), torch.zeros(1, 1), torch.ones(1, 1),
                        mask.float().mean(-1).lt(1.))


def expert_and_cache():
    h = HistoricalExpert(4, 8, 8, 1, contextual=True, tv_query='fully_hidden_v1')
    with torch.no_grad():
        h.tv_query.project.weight.zero_()
        h.tv_query.project.weight[0, 0] = 1.
        h.tv_query.project.weight[1, 1] = 1.
    memory = OnlinePrototypeMemory(patch_len=4, patch_stride=4, train_end=64, variable_count=1)
    memory.stage_patch(0, 0, np.array([1., 2., 3., 4.]), np.eye(8)[0], observed=True)
    memory.stage_patch(0, 4, np.array([9., 8., 7., 6.]), np.eye(8)[1], observed=True)
    memory.stage_patch(0, 40, np.ones(4)*100., np.eye(8)[0], observed=True)
    h.refresh_snapshot(memory.publish_epoch(1))
    arg = inputs()
    return h, arg, h.prepare_candidates(arg)


def test_tv_state_changes_history_candidate_and_excludes_future_member():
    # Catches caching H before TV or querying with the original masked patch.
    h, arg, legacy = expert_and_cache()
    assert not legacy.available.any()
    first = h.prepare_tv_candidates(arg, legacy)
    assert first.available[0, 0, 1]
    assert first.provenance['candidate_starts'][0, 0, 1][first.valid[0, 0, 1]].tolist() == [0]
    state = arg.base_latents.clone()
    state[0, 0, 1, 0] = 0.
    state[0, 0, 1, 1] = 1.
    second = h.prepare_tv_candidates(replace(arg, base_latents=state), legacy)
    assert second.provenance['candidate_starts'][0, 0, 1][second.valid[0, 0, 1]].tolist() == [4]
    assert first.valid.shape[-1] == 4
    assert not torch.equal(first.normalized_raw[0, 0, 1], second.normalized_raw[0, 0, 1])


def test_tv_query_ignores_hidden_values_and_preserves_partial_patch_candidates():
    # Catches treating generated/hidden values as newly observed data.
    h, arg, _ = expert_and_cache()
    mask = arg.patch_mask.clone(); mask[0, 0, 0, 3] = False
    arg = replace(arg, patch_mask=mask, rho=mask.float().mean(-1), damaged=mask.float().mean(-1).lt(1.))
    legacy = h.prepare_candidates(arg)
    first = h.prepare_tv_candidates(arg, legacy)
    corrupted = arg.patch_values.masked_fill(~mask, float('nan'))
    second = h.prepare_tv_candidates(replace(arg, patch_values=corrupted), legacy)
    torch.testing.assert_close(first.normalized_raw, second.normalized_raw, rtol=0, atol=0)
    torch.testing.assert_close(first.normalized_raw[0, 0, 0], legacy.normalized_raw[0, 0, 0], rtol=0, atol=0)
    assert torch.equal(first.valid[0, 0, 0], legacy.valid[0, 0, 0])
    torch.testing.assert_close(first.base_logits[0, 0, 0], legacy.base_logits[0, 0, 0], rtol=0, atol=0)


def test_tv_query_rejects_windows_without_any_real_observation():
    # Catches using a learned prior as reliable observed context.
    h, arg, _ = expert_and_cache()
    mask = torch.zeros_like(arg.patch_mask)
    arg = replace(arg, patch_mask=mask, rho=torch.zeros_like(arg.rho), damaged=torch.ones_like(arg.damaged))
    result = h.prepare_tv_candidates(arg, h.prepare_candidates(arg))
    assert not result.available.any()
    assert not result.provenance['tv_query_eligible'].any()


def test_tv_query_alignment_has_gradient_without_updating_frozen_key_encoder():
    # Catches detaching TV query or teaching against a key space different from the bank.
    from integrations.e23.tv_history import tv_query_alignment_loss
    h, arg, legacy = expert_and_cache()
    state = arg.base_latents.clone().requires_grad_()
    cache = h.prepare_tv_candidates(replace(arg, base_latents=state), legacy)
    clean_raw = torch.tensor([[[[1., 2., 3., 4.], [2., -1., 4., 3.]]]])
    clean_validity = torch.ones_like(arg.patch_mask)
    loss = tv_query_alignment_loss(h, cache.provenance, clean_raw, clean_validity)
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    assert state.grad is not None and state.grad.abs().sum() > 0
    assert h.tv_query.project.weight.grad is not None and h.tv_query.project.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in h.retrieval.frozen_encoder.parameters())


def test_tv_query_new_parameters_do_not_change_legacy_initialization_or_rng():
    # Catches consuming the original forecast head's initialization RNG stream.
    torch.manual_seed(2023)
    legacy = HistoricalExpert(4, 8, 8, 1, contextual=True)
    legacy_rng = torch.get_rng_state()
    torch.manual_seed(2023)
    actual = HistoricalExpert(4, 8, 8, 1, contextual=True, tv_query='fully_hidden_v1')
    assert torch.equal(torch.get_rng_state(), legacy_rng)
    for name, value in legacy.state_dict().items():
        torch.testing.assert_close(actual.state_dict()[name], value, rtol=0, atol=0)
