import torch
from math import sqrt

from malr.models.e23.contracts import ExpertInputs
from malr.models.e23.temporal import MultiScaleTemporalExpert


def _inputs(patches, *, visible_patches=None):
    torch.manual_seed(23)
    batch, variables, length, width = 2, 3, 6, 8
    values = torch.randn(batch, variables, patches, length)
    mask = torch.zeros_like(values, dtype=torch.bool)
    if visible_patches is None:
        mask[:] = True
        mask[..., 2:4] = False
    else:
        mask[:, :, visible_patches] = True
    return ExpertInputs(
        base_latents=torch.randn(batch, variables, patches, width),
        patch_values=values, patch_mask=mask,
        patch_validity=torch.ones_like(mask),
        rho=mask.float().mean(-1),
        patch_starts=torch.arange(patches).repeat(batch, 1) * 3,
        query_mean=torch.zeros(batch, variables), query_std=torch.ones(batch, variables),
        damaged=torch.ones(batch, variables, patches, dtype=torch.bool),
    )


def test_t_uses_remote_source_and_encodes_each_source_once():
    inputs = _inputs(20, visible_patches=[0])
    expert = MultiScaleTemporalExpert(patch_length=6, d_model=8)
    calls = []
    handle = expert.source_encoder.register_forward_hook(lambda *_: calls.append(1))
    try:
        result = expert(inputs)
    finally:
        handle.remove()
    assert calls == [1]
    assert result.available[:, :, 19].all()
    assert 0 in result.provenance["source_indices"][0, 0, 19][
        result.provenance["candidate_valid"][0, 0, 19]
    ].tolist()
    assert result.provenance["candidate_count"].max() <= 10
    assert result.message.shape == inputs.base_latents.shape
    assert torch.isfinite(result.message).all()


def test_t_p4_has_no_beyond_sixteen_source_and_respects_direction():
    inputs = _inputs(11)
    expert = MultiScaleTemporalExpert(patch_length=6, d_model=8)
    result = expert(inputs)
    provenance = result.provenance
    assert not provenance["candidate_valid"][..., 4].any()
    assert not provenance["candidate_valid"][..., 9].any()
    source = provenance["source_indices"]
    valid = provenance["candidate_valid"]
    target = torch.arange(11)[None, None, :, None]
    assert (source[..., :5][valid[..., :5]] < target.expand_as(source[..., :5])[valid[..., :5]]).all()
    assert (source[..., 5:][valid[..., 5:]] > target.expand_as(source[..., 5:])[valid[..., 5:]]).all()


def test_t_one_patch_and_no_visible_source_are_unavailable():
    expert = MultiScaleTemporalExpert(patch_length=6, d_model=8)
    single = expert(_inputs(1))
    assert not single.available.any()
    assert torch.equal(single.message, torch.zeros_like(single.message))

    inputs = _inputs(5, visible_patches=[])
    empty = expert(inputs)
    assert not empty.available.any()
    assert empty.provenance["candidate_count"].max() == 0
    assert torch.isfinite(empty.quality).all()


def test_t_ignores_hidden_source_values_and_has_finite_gradients():
    inputs = _inputs(11)
    expert = MultiScaleTemporalExpert(patch_length=6, d_model=8)
    first = expert(inputs)
    modified = inputs.patch_values.clone()
    modified[~inputs.patch_mask] = 5000
    changed = expert(ExpertInputs(**{**vars(inputs), "patch_values": modified}))
    torch.testing.assert_close(first.message, changed.message)
    torch.testing.assert_close(first.quality, changed.quality)
    (first.message.square().mean() + first.quality.square().mean()).backward()
    for parameter in (expert.source_encoder[0].weight, expert.message_projector[0].weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_t_batched_band_selection_matches_scalar_reference():
    expert = MultiScaleTemporalExpert(patch_length=6, d_model=8)
    for patch_count in (1, 11, 20):
        inputs = _inputs(patch_count)
        result = expert(inputs)
        observed = inputs.patch_mask & inputs.patch_validity
        source = expert.source_encoder(torch.cat((
            inputs.patch_values * observed, observed.float(),
        ), dim=-1))
        query = expert.query_projection(inputs.base_latents)
        key = expert.key_projection(source)
        similarity = torch.einsum("bcnd,bcmd->bcnm", query, key) / sqrt(8)
        positions = torch.arange(patch_count)
        separation = positions[:, None] - positions[None, :]
        eligible = observed.any(-1)[:, :, None, :] & inputs.damaged[..., None]
        expected_indices = []
        expected_valid = []
        for direction in (1, -1):
            signed = direction * separation
            for band_index, (lower, high) in enumerate(zip(
                (0, *expert.boundaries), (*expert.boundaries, None),
            )):
                allowed = eligible & (signed > lower)[None, None]
                if high is not None:
                    allowed = allowed & (signed <= high)[None, None]
                rank = -signed.float().expand_as(similarity) if band_index == 0 else similarity
                expected_indices.append(rank.masked_fill(~allowed, -torch.inf).argmax(-1))
                expected_valid.append(allowed.any(-1))
        assert torch.equal(result.provenance["source_indices"], torch.stack(expected_indices, -1))
        assert torch.equal(result.provenance["candidate_valid"], torch.stack(expected_valid, -1))
