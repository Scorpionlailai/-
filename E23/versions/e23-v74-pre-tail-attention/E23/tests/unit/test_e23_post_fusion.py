import torch

from malr.models.e23.contracts import RouterOutput
from malr.models.e23.post_fusion import EvidenceAwareFusionRefiner


def _routed() -> tuple[RouterOutput, torch.Tensor]:
    torch.manual_seed(20)
    shape = (2, 3, 6, 8)
    latents = torch.randn(shape, requires_grad=True)
    weights = torch.softmax(torch.randn(*shape[:-1], 4), dim=-1)
    deltas = {name: torch.randn(shape) for name in
              ("historical", "variable", "temporal")}
    available = torch.ones(*shape[:-1], 4, dtype=torch.bool)
    rho = torch.rand(*shape[:-1])
    rho[..., 0] = 1.0
    return RouterOutput(latents, weights, deltas, available), rho


def test_post_fusion_starts_at_identity_and_preserves_complete_patches():
    routed, rho = _routed()
    refiner = EvidenceAwareFusionRefiner(8, 6)
    torch.testing.assert_close(refiner(routed, rho), routed.latents)
    with torch.no_grad():
        refiner.output_projection.weight.normal_(std=0.01)
    refined = refiner(routed, rho)
    torch.testing.assert_close(refined[..., 0, :], routed.latents[..., 0, :])
    assert (refined[..., 1:, :] - routed.latents[..., 1:, :]).abs().sum() > 0
    refined.square().mean().backward()
    for parameter in (refiner.output_projection.weight,
                      refiner.temporal_attention.in_proj_weight,
                      refiner.variable_attention.in_proj_weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_post_fusion_rejects_invalid_geometry():
    routed, rho = _routed()
    refiner = EvidenceAwareFusionRefiner(8, 6)
    try:
        refiner(routed, rho[..., :-1])
    except ValueError as error:
        assert "geometry" in str(error)
    else:
        raise AssertionError("invalid visible fraction geometry was accepted")
