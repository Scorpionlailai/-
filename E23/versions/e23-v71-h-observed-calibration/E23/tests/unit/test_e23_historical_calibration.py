from dataclasses import replace
import numpy as np
import pytest
import torch
from malr.models.e23.contracts import ExpertInputs
from malr.models.e23.historical import HistoricalExpert
from malr.prototypes.online_bank import OnlinePrototypeMemory


def inputs(mask=(True, False, True, False), std=1.):
    observed = torch.tensor(mask)[None, None, None]
    return ExpertInputs(torch.ones(1, 1, 1, 8), torch.tensor([3., 0., 5., 0.])[None, None, None],
        observed, torch.ones_like(observed), observed.float().mean(-1), torch.tensor([[8]]),
        torch.zeros(1, 1), torch.full((1, 1), std), torch.ones(1, 1, 1, dtype=torch.bool))


def run(inp, raw):
    from malr.models.e23.historical_calibration import calibrate_historical_candidates
    return calibrate_historical_candidates(inp, raw, torch.ones(raw.shape[:-1], dtype=torch.bool))


def test_two_visible_points_give_shrunk_offset_and_out_of_sample_error():
    raw = torch.tensor([1., 2., 3., 4.])[None, None, None, None]
    adjusted, error, offset = run(inputs(), raw)
    torch.testing.assert_close(offset, torch.full((1, 1, 1, 1), 2/3))
    torch.testing.assert_close(adjusted, torch.tensor([5/3, 8/3, 11/3, 14/3])[None, None, None, None])
    torch.testing.assert_close(error, torch.full((1, 1, 1, 1), 1.6))


def test_single_visible_point_cannot_improve_its_own_confidence():
    raw = torch.tensor([1., 2., 3., 4.])[None, None, None, None]
    _, error, offset = run(inputs((True, False, False, False)), raw)
    torch.testing.assert_close(offset, torch.full((1, 1, 1, 1), .4))
    torch.testing.assert_close(error, torch.full((1, 1, 1, 1), 2.))


def test_hidden_nan_and_invalid_padding_never_enter_fit():
    inp = inputs()
    validity = inp.patch_validity.clone(); validity[..., 2:] = False
    inp = replace(inp, patch_validity=validity)
    changed = inp.patch_values.clone(); changed[..., 1:] = float('nan')
    raw = torch.tensor([1., 2., 3., 4.])[None, None, None, None]
    a = run(inp, raw); b = run(replace(inp, patch_values=changed), raw)
    for left, right in zip(a, b):
        assert torch.isfinite(right).all()
        torch.testing.assert_close(left, right)


def test_near_constant_scale_has_finite_bounded_global_offset_and_empty_fit_stays_identity():
    raw = torch.full((1, 1, 1, 1, 4), -1e6)
    adjusted, error, offset = run(inputs(std=.003), raw)
    assert torch.isfinite(adjusted).all() and torch.isfinite(error).all()
    assert (offset * .003).abs().max() < .201
    adjusted, error, offset = run(inputs((False,)*4), raw)
    assert torch.equal(adjusted, raw) and not error.any() and not offset.any()


def test_calibration_changes_values_without_changing_members_keys_or_initial_parameters():
    torch.manual_seed(7)
    old = HistoricalExpert(4, 8, 8, 1, similarity_threshold=-1.)
    state = torch.random.get_rng_state()
    torch.manual_seed(7)
    new = HistoricalExpert(4, 8, 8, 1, similarity_threshold=-1., calibration='ridge_offset_loo_v1')
    assert torch.equal(state, torch.random.get_rng_state())
    assert old.state_dict().keys() == new.state_dict().keys()
    assert all(torch.equal(value, new.state_dict()[key]) for key, value in old.state_dict().items())
    memory = OnlinePrototypeMemory(patch_len=4, patch_stride=2, train_end=32, variable_count=1)
    for start in (0, 12):
        memory.stage_patch(0, start, np.array([1., 2., 3., 4.]), np.array([1., 0., 0., 0., 0., 0., 0., 0.]), observed=True)
    snapshot = memory.publish_epoch(1)
    old.refresh_snapshot(snapshot); new.refresh_snapshot(snapshot)
    a = old.prepare_candidates(inputs()); b = new.prepare_candidates(inputs())
    assert torch.equal(a.valid, b.valid)
    assert b.provenance['candidate_starts'][b.valid].tolist() == [0]
    assert torch.equal(a.provenance['candidate_raw'], b.provenance['candidate_raw'])
    torch.testing.assert_close(b.normalized_raw[..., 0, :] - a.normalized_raw[..., 0, :], torch.full((1, 1, 1, 4), 2/3))
    torch.testing.assert_close(b.provenance['visible_error'][b.valid], torch.tensor([1.6]))
    assert not new.prepare_candidates(inputs((False,)*4)).available.any()


def test_unknown_or_incompatible_calibration_is_rejected():
    with pytest.raises(ValueError):
        HistoricalExpert(4, 8, 8, 1, calibration='wrong')
    with pytest.raises(ValueError):
        HistoricalExpert(4, 8, 8, 1, calibration='ridge_offset_loo_v1', top_p=.9)
