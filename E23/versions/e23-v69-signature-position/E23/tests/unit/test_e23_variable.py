from math import e

import pytest
import torch

from malr.models.e23.contracts import ExpertInputs
from malr.models.e23.variable import SparseVariableExpert


def _inputs(variables=7, patches=5, length=6):
    torch.manual_seed(12)
    values = torch.randn(2, variables, patches, length)
    mask = torch.ones_like(values, dtype=torch.bool)
    mask[..., 2:4] = False
    return ExpertInputs(
        base_latents=torch.randn(2, variables, patches, 8),
        patch_values=values, patch_mask=mask,
        patch_validity=torch.ones_like(mask),
        rho=mask.float().mean(-1),
        patch_starts=torch.arange(patches).repeat(2, 1) * 3,
        query_mean=torch.zeros(2, variables),
        query_std=torch.ones(2, variables),
        damaged=torch.ones(2, variables, patches, dtype=torch.bool),
    )


def test_v_constructs_only_selected_local_edges_and_ignores_unselected_source():
    inputs = _inputs()
    expert = SparseVariableExpert(variable_count=7, patch_length=6, d_model=8)
    result = expert(inputs, epoch=0, explore=False)
    provenance = result.provenance
    source = provenance["source_variables"]
    assert source.shape == (7, 4)
    assert not source.eq(torch.arange(7)[:, None]).any()
    assert provenance["candidate_valid"].shape == (2, 7, 5, 4, 3)
    assert provenance["candidate_count"].max() <= 12
    assert result.message.shape == inputs.base_latents.shape
    assert result.available.all()

    unselected = next(i for i in range(1, 7) if i not in source[0].tolist())
    modified = inputs.patch_values.clone()
    modified[:, unselected] += 1000
    changed = ExpertInputs(**{**vars(inputs), "patch_values": modified})
    after = expert(changed, epoch=0, explore=False)
    torch.testing.assert_close(result.message[:, 0], after.message[:, 0])
    torch.testing.assert_close(result.quality[:, 0], after.quality[:, 0])


def test_v_uses_only_visible_coordinates_and_unavailable_when_sources_empty():
    inputs = _inputs(variables=3)
    expert = SparseVariableExpert(variable_count=3, patch_length=6, d_model=8)
    original = expert(inputs, epoch=0, explore=False)
    hidden_changed = inputs.patch_values.clone()
    hidden_changed[~inputs.patch_mask] = 10000
    after = expert(ExpertInputs(**{**vars(inputs), "patch_values": hidden_changed}), epoch=0, explore=False)
    torch.testing.assert_close(original.message, after.message)
    torch.testing.assert_close(original.quality, after.quality)

    no_source = inputs.patch_mask.clone()
    no_source[:, 1:] = False
    empty = expert(ExpertInputs(**{**vars(inputs), "patch_mask": no_source}), epoch=0, explore=False)
    assert not empty.available[:, 0].any()
    assert torch.equal(empty.message[:, 0], torch.zeros_like(empty.message[:, 0]))
    assert torch.isfinite(empty.quality).all()


def test_v_one_variable_and_rotating_exploration_slot():
    inputs = _inputs(variables=1)
    expert = SparseVariableExpert(variable_count=1, patch_length=6, d_model=8)
    output = expert(inputs, epoch=0, explore=True)
    assert not output.available.any()
    assert output.provenance["candidate_count"].max() == 0

    expert = SparseVariableExpert(variable_count=7, patch_length=6, d_model=8)
    a = expert(_inputs(), epoch=0, explore=True).provenance["source_variables"]
    b = expert(_inputs(), epoch=1, explore=True).provenance["source_variables"]
    assert (a[:, -1] != b[:, -1]).all()
    torch.testing.assert_close(a[:, :3], b[:, :3])
    assert not a.eq(torch.arange(7)[:, None]).any()


def test_v_relation_and_message_paths_receive_finite_gradients():
    expert = SparseVariableExpert(variable_count=7, patch_length=6, d_model=8)
    result = expert(_inputs(), epoch=0, explore=False)
    (result.message.square().mean() + result.quality.square().mean()).backward()
    for parameter in (expert.source_embedding.weight, expert.target_embedding.weight,
                      expert.message_projector[0].weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_channelwise_v_starts_at_scalar_aggregation_and_can_split_feature_weights():
    inputs = _inputs()
    torch.manual_seed(7)
    scalar = SparseVariableExpert(variable_count=7, patch_length=6, d_model=8,
                                  selection="window_context_topk_v1")
    channelwise = SparseVariableExpert(variable_count=7, patch_length=6, d_model=8,
                                       selection="window_context_topk_v1",
                                       aggregation="channelwise_v1")
    channelwise.load_state_dict(scalar.state_dict(), strict=False)
    first = scalar(inputs, epoch=0, explore=False)
    equal_start = channelwise(inputs, epoch=0, explore=False)
    torch.testing.assert_close(equal_start.message, first.message, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(equal_start.quality, first.quality)
    assert "channel_projector.weight" not in scalar.state_dict()

    with torch.no_grad():
        channelwise.channel_projector.weight[0, 0] = 10
        channelwise.channel_projector.weight[1, 0] = -10
    output = channelwise(inputs, epoch=0, explore=False)
    channel_weights = output.provenance["channel_weights"]
    valid = output.provenance["candidate_valid"]
    assert torch.equal(channel_weights.masked_select(~valid[..., None].expand_as(channel_weights)),
                       torch.zeros_like(channel_weights).masked_select(
                           ~valid[..., None].expand_as(channel_weights)))
    torch.testing.assert_close(channel_weights.sum(dim=(-3, -2)),
                               output.available[..., None].expand_as(output.message).float())
    assert (channel_weights[..., 0] - channel_weights[..., 1]).abs().max() > 1e-4
    assert not torch.allclose(output.message, first.message)


def test_channelwise_v_preserves_observation_boundary_and_trains_projection():
    inputs = _inputs()
    expert = SparseVariableExpert(variable_count=7, patch_length=6, d_model=8,
                                  aggregation="channelwise_v1")
    baseline = expert(inputs, epoch=0, explore=False)
    changed_values = inputs.patch_values.clone()
    changed_values[~inputs.patch_mask] = 1e6
    changed = expert(ExpertInputs(**{**vars(inputs), "patch_values": changed_values}),
                     epoch=0, explore=False)
    torch.testing.assert_close(baseline.message, changed.message)
    (baseline.message.square().mean()).backward()
    assert expert.channel_projector.weight.grad is not None
    assert expert.channel_projector.weight.grad.abs().sum() > 0


def test_bounded_channelwise_v_limits_weight_shift_and_keeps_v12_start():
    inputs = _inputs()
    torch.manual_seed(7)
    scalar = SparseVariableExpert(variable_count=7, patch_length=6, d_model=8,
                                  selection="window_context_topk_v1")
    bounded = SparseVariableExpert(variable_count=7, patch_length=6, d_model=8,
                                   selection="window_context_topk_v1",
                                   aggregation="channelwise_bounded_v1")
    bounded.load_state_dict(scalar.state_dict(), strict=False)
    torch.testing.assert_close(bounded(inputs, epoch=0, explore=False).message,
                               scalar(inputs, epoch=0, explore=False).message,
                               atol=1e-6, rtol=1e-6)
    with torch.no_grad():
        bounded.channel_projector.weight.fill_(1000)
        bounded.channel_projector.weight[0].neg_()
    output = bounded(inputs, epoch=0, explore=False)
    prior = output.provenance["weights"][..., None].expand_as(
        output.provenance["channel_weights"])
    ratio = output.provenance["channel_weights"] / prior.clamp_min(1e-12)
    valid = output.provenance["candidate_valid"][..., None].expand_as(ratio)
    assert (ratio[valid] <= e + 1e-5).all()
    assert (ratio[valid] >= 1 / e - 1e-5).all()
    assert (output.provenance["channel_weights"][~valid] == 0).all()


def test_v_top_p_uses_stable_six_source_prefix_and_exploration():
    expert = SparseVariableExpert(variable_count=7, patch_length=6, d_model=8,
                                  top_p=0.5)
    with torch.no_grad():
        expert.source_embedding.weight.zero_()
        expert.target_embedding.weight.zero_()
    inputs = _inputs()
    selected = expert(inputs, epoch=0, explore=False).provenance
    assert selected["source_variables"].shape == (7, 6)
    assert not selected["source_variables"].eq(torch.arange(7)[:, None]).any()
    assert selected["source_selected"].sum(-1).tolist() == [3] * 7
    assert selected["candidate_count"].max() <= 9
    explored = expert(inputs, epoch=0, explore=True).provenance
    assert explored["source_selected"].sum(-1).tolist() == [4] * 7
    assert explored["candidate_count"].max() <= 12


def test_v_top_p_empty_local_sources_are_unavailable():
    expert = SparseVariableExpert(variable_count=7, patch_length=6, d_model=8,
                                  top_p=0.9)
    inputs = _inputs()
    mask = torch.zeros_like(inputs.patch_mask)
    output = expert(ExpertInputs(**{**vars(inputs), "patch_mask": mask}),
                    epoch=0, explore=False)
    assert not output.available.any()
    assert output.provenance["candidate_count"].sum() == 0


def test_contextual_v_changes_sources_by_window_and_patch_without_self_edges():
    inputs = _inputs()
    mask = torch.zeros_like(inputs.patch_mask)
    mask[:, 0] = True
    mask[0, 1, 0] = True
    mask[0, 6, 4] = True
    mask[1, 6, 0] = True
    mask[1, 1, 4] = True
    inputs = ExpertInputs(**{**vars(inputs), "patch_mask": mask})
    expert = SparseVariableExpert(variable_count=7, patch_length=6, d_model=8,
                                  selection="window_context_topk_v1")
    with torch.no_grad():
        expert.source_embedding.weight.zero_()
        expert.target_embedding.weight.zero_()
    provenance = expert(inputs, epoch=0, explore=False).provenance
    source = provenance["source_variables"]
    assert source.shape == (2, 7, 5, 4)
    assert source[0, 0, 0, 0] == 1
    assert source[0, 0, 4, 0] == 6
    assert source[1, 0, 0, 0] == 6
    assert source[1, 0, 4, 0] == 1
    assert not source.eq(torch.arange(7)[None, :, None, None]).any()


@pytest.mark.parametrize("selection", ("window_context_topk_v1", "window_context_topk_v2",
                                        "window_affine_v1"))
def test_contextual_v_stable_ties_explore_and_ignore_hidden_values(selection):
    inputs = _inputs()
    values = torch.zeros_like(inputs.patch_values)
    inputs = ExpertInputs(**{**vars(inputs), "patch_values": values})
    expert = SparseVariableExpert(variable_count=7, patch_length=6, d_model=8,
                                  selection=selection)
    with torch.no_grad():
        expert.source_embedding.weight.zero_()
        expert.target_embedding.weight.zero_()
    base = expert(inputs, epoch=0, explore=False)
    assert base.provenance["source_variables"][0, 0, 2].tolist() == [1, 2, 3, 4]
    explored0 = expert(inputs, epoch=0, explore=True).provenance["source_variables"]
    explored1 = expert(inputs, epoch=1, explore=True).provenance["source_variables"]
    torch.testing.assert_close(explored0[..., :3], explored1[..., :3])
    assert (explored0[..., -1] != explored1[..., -1]).all()
    hidden_changed = values.clone()
    hidden_changed[~inputs.patch_mask] = 100000.
    changed = expert(ExpertInputs(**{**vars(inputs), "patch_values": hidden_changed}),
                     epoch=0, explore=False)
    torch.testing.assert_close(base.message, changed.message)
    torch.testing.assert_close(base.quality, changed.quality)
    torch.testing.assert_close(base.provenance["source_variables"],
                               changed.provenance["source_variables"])


def test_contextual_v_is_unavailable_when_all_sources_are_hidden():
    inputs = _inputs()
    mask = torch.zeros_like(inputs.patch_mask)
    mask[:, 0] = True
    expert = SparseVariableExpert(variable_count=7, patch_length=6, d_model=8,
                                  selection="window_context_topk_v1")
    output = expert(ExpertInputs(**{**vars(inputs), "patch_mask": mask}),
                    epoch=0, explore=False)
    assert not output.available[:, 0].any()
    assert output.provenance["candidate_count"][:, 0].sum() == 0


@pytest.mark.parametrize("selection", ("window_context_topk_v1", "window_context_topk_v2",
                                        "window_affine_v1"))
def test_contextual_v_ignores_invalid_padding_values_and_bounds_candidates(selection):
    inputs = _inputs()
    validity = inputs.patch_validity.clone()
    validity[..., -2:] = False
    inputs = ExpertInputs(**{**vars(inputs), "patch_validity": validity})
    expert = SparseVariableExpert(variable_count=7, patch_length=6, d_model=8,
                                  selection=selection)
    original = expert(inputs, epoch=0, explore=False)
    changed_values = inputs.patch_values.clone()
    changed_values[~validity] = 1e6
    changed = expert(ExpertInputs(**{**vars(inputs), "patch_values": changed_values}),
                     epoch=0, explore=False)
    torch.testing.assert_close(original.message, changed.message)
    torch.testing.assert_close(original.quality, changed.quality)
    torch.testing.assert_close(original.provenance["source_variables"],
                               changed.provenance["source_variables"])
    assert original.provenance["candidate_count"].max() <= 12


def test_stronger_context_v_can_override_moderate_global_prior():
    inputs = _inputs()
    mask = torch.ones_like(inputs.patch_mask)
    mask[:, 0] = False
    mask[:, 1, :, 1:] = False
    inputs = ExpertInputs(**{**vars(inputs), "patch_mask": mask})
    v1 = SparseVariableExpert(variable_count=7, patch_length=6, d_model=8,
                              selection="window_context_topk_v1")
    v2 = SparseVariableExpert(variable_count=7, patch_length=6, d_model=8,
                              selection="window_context_topk_v2")
    for expert in (v1, v2):
        with torch.no_grad():
            expert.source_embedding.weight.zero_()
            expert.target_embedding.weight.zero_()
            expert.source_embedding.weight[1, 0] = 0.6 * 8 ** 0.5
            expert.target_embedding.weight[0, 0] = 1.
    first = v1(inputs, epoch=0, explore=False).provenance["source_variables"]
    second = v2(inputs, epoch=0, explore=False).provenance["source_variables"]
    assert first[0, 0, 2, 0] == 1
    assert second[0, 0, 2, 0] == 2
    assert not second.eq(torch.arange(7)[None, :, None, None]).any()


def test_affine_v_changes_visible_evidence_and_preserves_observation_boundary():
    inputs = _inputs()
    expert = SparseVariableExpert(variable_count=7, patch_length=6, d_model=8,
                                  selection="window_affine_v1")
    baseline = expert(inputs, epoch=0, explore=False)
    slope = torch.full((7, 7), -0.5)
    intercept = torch.full((7, 7), 0.25)
    expert.set_affine_map(slope, intercept)
    aligned = expert(inputs, epoch=0, explore=False)
    assert not torch.allclose(aligned.message, baseline.message)
    assert not torch.allclose(aligned.quality, baseline.quality)
    hidden_changed = inputs.patch_values.clone()
    hidden_changed[~inputs.patch_mask] = 10000
    changed = expert(ExpertInputs(**{**vars(inputs), "patch_values": hidden_changed}),
                     epoch=0, explore=False)
    torch.testing.assert_close(aligned.message, changed.message)
    torch.testing.assert_close(aligned.quality, changed.quality)
    torch.testing.assert_close(aligned.provenance["source_variables"],
                               changed.provenance["source_variables"])
    assert "pair_slope" in expert.state_dict()
    assert "pair_intercept" in expert.state_dict()
