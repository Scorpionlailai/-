from dataclasses import replace

import pytest
import torch

from malr.models.e23.contracts import ExpertEvidence, ExpertInputs
from malr.models.e23.router import ExpertRouter


def _inputs():
    base = torch.randn(2, 3, 4, 8, requires_grad=True)
    values = torch.randn(2, 3, 4, 6)
    mask = torch.ones_like(values, dtype=torch.bool)
    mask[..., 2:4] = False
    return ExpertInputs(
        base_latents=base, patch_values=values, patch_mask=mask,
        patch_validity=torch.ones_like(mask), rho=mask.float().mean(-1),
        patch_starts=torch.arange(4).repeat(2, 1) * 3,
        query_mean=torch.zeros(2, 3), query_std=torch.ones(2, 3),
        damaged=torch.ones(2, 3, 4, dtype=torch.bool),
    )


def _evidence(inputs, enabled=None, message=None):
    result = {}
    for name in ("historical", "variable", "temporal"):
        result[name] = ExpertEvidence(
            message=message if name == enabled else torch.zeros_like(inputs.base_latents),
            quality=torch.zeros_like(inputs.base_latents),
            available=torch.full(inputs.damaged.shape, name == enabled),
            provenance={},
        )
    return result


def test_router_masks_unavailable_experts_and_preserves_base():
    torch.manual_seed(42)
    inputs = _inputs()
    router = ExpertRouter(d_model=8)
    output = router(inputs, _evidence(inputs))
    assert output.weights.shape == (2, 3, 4, 4)
    assert torch.equal(output.weights[..., 0], torch.ones(2, 3, 4))
    assert torch.equal(output.latents, inputs.base_latents)
    assert torch.equal(output.available[..., 0], torch.ones_like(inputs.damaged))
    assert not output.available[..., 1:].any()
    output.latents.square().mean().backward()
    assert torch.isfinite(inputs.base_latents.grad).all()


def test_enabled_expert_receives_gradient_and_normalized_weight():
    inputs = _inputs()
    message = torch.randn_like(inputs.base_latents, requires_grad=True)
    router = ExpertRouter(d_model=8)
    output = router(inputs, _evidence(inputs, enabled="variable", message=message))
    torch.testing.assert_close(output.weights.sum(-1), torch.ones_like(inputs.damaged, dtype=torch.float))
    assert torch.equal(output.weights[..., 1], torch.zeros_like(output.weights[..., 1]))
    assert torch.equal(output.weights[..., 3], torch.zeros_like(output.weights[..., 3]))
    assert output.weights[..., 2].gt(0).all()
    assert output.weights[..., 2].lt(1).all()
    output.latents.square().mean().backward()
    assert message.grad is not None and message.grad.abs().sum() > 0
    assert inputs.base_latents.grad is not None and inputs.base_latents.grad.abs().sum() > 0


def test_router_rejects_nonfinite_quality_and_wrong_shape():
    inputs = _inputs()
    router = ExpertRouter(d_model=8)
    evidence = _evidence(inputs)
    evidence["historical"].quality[..., 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        router(inputs, evidence)
    evidence = _evidence(inputs)
    evidence["variable"] = replace(evidence["variable"], message=torch.zeros(2, 3, 4, 7))
    with pytest.raises(ValueError, match="shape"):
        router(inputs, evidence)
    with pytest.raises(ValueError, match="exactly"):
        router(inputs, {"historical": evidence["historical"]})


def test_none_can_suppress_an_available_expert_without_erasing_availability():
    inputs = _inputs()
    message = torch.ones_like(inputs.base_latents)
    router = ExpertRouter(d_model=8)
    with torch.no_grad():
        router.none_bias.fill_(20)
    output = router(inputs, _evidence(inputs, enabled="historical", message=message))
    assert output.available[..., 1].all()
    assert output.weights[..., 1].max() < 1e-6
    assert torch.isfinite(output.latents).all()


def test_router_scores_experts_in_one_batched_call():
    torch.manual_seed(19)
    inputs = _inputs()
    evidence = {
        name: ExpertEvidence(
            message=torch.randn_like(inputs.base_latents),
            quality=torch.randn_like(inputs.base_latents),
            available=torch.ones_like(inputs.damaged), provenance={},
        ) for name in ("historical", "variable", "temporal")
    }
    router = ExpertRouter(d_model=8)
    calls = []
    handle = router.shared_score.register_forward_hook(lambda _, args, __: calls.append(args[0].shape))
    try:
        output = router(inputs, evidence)
    finally:
        handle.remove()
    assert calls == [(2, 3, 4, 3, 35)]
    assert torch.isfinite(output.latents).all()
