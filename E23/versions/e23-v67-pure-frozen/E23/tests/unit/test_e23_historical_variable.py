from dataclasses import replace

import pytest
import torch

from malr.models.e23.contracts import ExpertEvidence, ExpertInputs
from malr.models.e23.historical_variable import HistoricalVariableExpert


def _case(*, anchor=0, valid=True, query_start=8):
    archive = torch.stack((torch.arange(12).float(), 10 + torch.arange(12).float()), dim=-1)
    values = torch.zeros(1, 2, 1, 4)
    mask = torch.tensor([[[[True, False, False, False]], [[True, False, False, False]]]])
    inputs = ExpertInputs(
        base_latents=torch.ones(1, 2, 1, 6), patch_values=values,
        patch_mask=mask, patch_validity=torch.ones_like(mask),
        rho=torch.full((1, 2, 1), 0.25),
        patch_starts=torch.tensor([[query_start]]),
        query_mean=torch.zeros(1, 2), query_std=torch.ones(1, 2),
        damaged=torch.ones(1, 2, 1, dtype=torch.bool),
    )
    candidate_valid = torch.full((1, 2, 1, 1), valid)
    candidate_raw = torch.zeros(1, 2, 1, 1, 4)
    if anchor + 4 <= len(archive):
        candidate_raw[0, :, 0, 0] = archive[anchor:anchor + 4].T
    historical = ExpertEvidence(
        message=torch.ones(1, 2, 1, 6), quality=torch.ones(1, 2, 1, 6),
        available=candidate_valid.any(-1),
        provenance={
            "candidate_starts": torch.full((1, 2, 1, 1), anchor),
            "candidate_valid": candidate_valid,
            "candidate_raw": candidate_raw,
            "weights": torch.ones(1, 2, 1, 1),
        },
    )
    return archive, inputs, historical


def test_historical_variable_reads_aligned_archive_and_ignores_current_source():
    archive, inputs, historical = _case()
    expert = HistoricalVariableExpert(2, 4, 6, topk=1, archive=archive)
    first = expert(inputs, historical, epoch=1, explore=False)
    changed = ExpertInputs(**{**vars(inputs), "patch_values": inputs.patch_values + 50})
    second = expert(changed, historical, epoch=1, explore=False)
    assert first.available.all()
    torch.testing.assert_close(first.provenance["source_values"][0, 0, 0, 0, 0], archive[:4, 1])
    torch.testing.assert_close(first.message, second.message)
    assert "archive" not in expert.state_dict()


def test_historical_variable_rejects_invalid_future_and_out_of_archive_anchors():
    for anchor, valid, query_start in [(0, False, 8), (5, True, 8), (9, True, 20)]:
        archive, inputs, historical = _case(anchor=anchor, valid=valid, query_start=query_start)
        expert = HistoricalVariableExpert(2, 4, 6, topk=1, archive=archive)
        result = expert(inputs, historical, epoch=0, explore=False)
        assert not result.available.any()
        torch.testing.assert_close(result.message, torch.zeros_like(result.message))
        assert result.provenance["candidate_count"].sum() == 0


def test_historical_variable_accepts_exact_causal_boundary_and_backpropagates():
    archive, inputs, historical = _case(query_start=4)
    expert = HistoricalVariableExpert(2, 4, 6, topk=1, archive=archive)
    output = expert(inputs, historical, epoch=0, explore=False)
    assert output.available.all()
    output.message.square().sum().backward()
    assert expert.message_projector[0].weight.grad.abs().sum() > 0
    assert expert.source_embedding.weight.grad.abs().sum() > 0
    assert expert.target_embedding.weight.grad.abs().sum() > 0


def test_historical_variable_rejects_a_prototype_from_inside_current_window():
    archive, inputs, historical = _case(anchor=6, query_start=12)
    inputs = replace(
        inputs,
        base_latents=inputs.base_latents.repeat(1, 1, 2, 1),
        patch_values=inputs.patch_values.repeat(1, 1, 2, 1),
        patch_mask=inputs.patch_mask.repeat(1, 1, 2, 1),
        patch_validity=inputs.patch_validity.repeat(1, 1, 2, 1),
        rho=inputs.rho.repeat(1, 1, 2),
        patch_starts=torch.tensor([[8, 12]]),
        damaged=inputs.damaged.repeat(1, 1, 2),
    )
    historical = replace(
        historical,
        message=historical.message.repeat(1, 1, 2, 1),
        quality=historical.quality.repeat(1, 1, 2, 1),
        available=torch.tensor([[[False, True], [False, True]]]),
        provenance={
            "candidate_starts": historical.provenance["candidate_starts"].repeat(1, 1, 2, 1),
            "candidate_valid": torch.tensor([[[[False], [True]], [[False], [True]]]]),
            "candidate_raw": historical.provenance["candidate_raw"].repeat(1, 1, 2, 1, 1),
            "weights": historical.provenance["weights"].repeat(1, 1, 2, 1),
        },
    )
    expert = HistoricalVariableExpert(2, 4, 6, topk=1, archive=archive)
    result = expert(inputs, historical, epoch=0, explore=False)
    assert not result.available.any()


def test_historical_variable_rejects_mismatched_target_prototype():
    archive, inputs, historical = _case()
    provenance = dict(historical.provenance)
    provenance["candidate_raw"] = provenance["candidate_raw"] + 1
    expert = HistoricalVariableExpert(2, 4, 6, topk=1, archive=archive)
    with pytest.raises(ValueError, match="prototype target differs"):
        expert(inputs, replace(historical, provenance=provenance), epoch=0, explore=False)


def test_bounded_history_values_limit_extremes_and_preserve_small_inputs():
    archive, inputs, historical = _case()
    tiny_std = replace(inputs, query_std=torch.full((1, 2), 0.0031622776))
    bounded = HistoricalVariableExpert(2, 4, 6, topk=1, archive=archive,
                                       bounded_values=True)
    extreme = bounded(tiny_std, historical, epoch=0, explore=False)
    source = extreme.provenance["source_values"]
    assert source.abs().max() <= 8
    assert source.abs().max() > 7.9
    small_std = replace(inputs, query_std=torch.full((1, 2), 100.0))
    small = bounded(small_std, historical, epoch=0, explore=False)
    raw_source = archive[:4, 1] / 100.0
    torch.testing.assert_close(small.provenance["source_values"][0, 0, 0, 0, 0],
                               8 * torch.tanh(raw_source / 8))
    torch.testing.assert_close(small.provenance["source_values"][0, 0, 0, 0, 0],
                               raw_source, atol=5e-5, rtol=0)
    assert torch.isfinite(extreme.message).all()


def test_unbounded_v1_historical_values_remain_exact():
    archive, inputs, historical = _case()
    old = HistoricalVariableExpert(2, 4, 6, topk=1, archive=archive)
    new_default = HistoricalVariableExpert(2, 4, 6, topk=1, archive=archive,
                                           bounded_values=False)
    new_default.load_state_dict(old.state_dict())
    first = old(inputs, historical, epoch=0, explore=False)
    second = new_default(inputs, historical, epoch=0, explore=False)
    assert torch.equal(first.message, second.message)
    assert torch.equal(first.provenance["source_values"], second.provenance["source_values"])
