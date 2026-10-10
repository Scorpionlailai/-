import torch

from malr.models.e23.contracts import ExpertInputs
from malr.models.e23.temporal_attention import VisibleTemporalAttentionExpert


def _inputs(*, any_visible: bool = True) -> ExpertInputs:
    torch.manual_seed(18)
    values = torch.randn(2, 3, 6, 8)
    mask = torch.zeros_like(values, dtype=torch.bool)
    if any_visible:
        mask[..., ::2] = True
        mask[..., 2, :] = False
    return ExpertInputs(
        base_latents=torch.randn(2, 3, 6, 8),
        patch_values=values, patch_mask=mask,
        patch_validity=torch.ones_like(mask),
        rho=mask.float().mean(-1),
        patch_starts=torch.arange(6).repeat(2, 1) * 3,
        query_mean=torch.zeros(2, 3), query_std=torch.ones(2, 3),
        damaged=torch.ones(2, 3, 6, dtype=torch.bool),
    )


def test_visible_temporal_attention_uses_only_observed_other_patches():
    inputs = _inputs()
    expert = VisibleTemporalAttentionExpert(8, 8)
    output = expert(inputs)
    weights = output.provenance["weights"]
    assert output.message.shape == inputs.base_latents.shape
    assert output.available.all()
    assert not output.provenance["candidate_valid"][..., 2].any()
    assert torch.equal(weights.diagonal(dim1=-2, dim2=-1), torch.zeros_like(
        weights.diagonal(dim1=-2, dim2=-1),
    ))
    torch.testing.assert_close(weights.sum(-1), torch.ones_like(weights.sum(-1)))
    changed_values = inputs.patch_values.clone()
    changed_values[~inputs.patch_mask] = 1e4
    changed = expert(ExpertInputs(**{**vars(inputs), "patch_values": changed_values}))
    torch.testing.assert_close(output.message, changed.message)
    torch.testing.assert_close(output.quality, changed.quality)


def test_visible_temporal_attention_empty_source_and_gradients():
    expert = VisibleTemporalAttentionExpert(8, 8)
    empty = expert(_inputs(any_visible=False))
    assert not empty.available.any()
    assert torch.equal(empty.message, torch.zeros_like(empty.message))
    assert torch.equal(empty.provenance["weights"], torch.zeros_like(
        empty.provenance["weights"],
    ))
    assert torch.isfinite(empty.quality).all()

    output = expert(_inputs())
    (output.message.square().mean() + output.quality.square().mean()).backward()
    for parameter in (expert.source_encoder[0].weight, expert.query_projection.weight,
                      expert.key_projection.weight, expert.value_projection.weight,
                      expert.relative_bias.weight, expert.message_projector[0].weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
