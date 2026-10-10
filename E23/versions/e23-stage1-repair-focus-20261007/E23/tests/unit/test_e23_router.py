from dataclasses import replace
from copy import deepcopy

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


def test_precomputed_router_matches_legacy_output_and_gradients():
    # Catches different arbitration or a second application of the residual.
    torch.manual_seed(42)
    inputs = _inputs()
    evidence = {name: ExpertEvidence(torch.randn_like(inputs.base_latents),
                                     torch.randn_like(inputs.base_latents),
                                     torch.ones_like(inputs.damaged), {})
                for name in ("historical", "variable", "temporal")}
    legacy = ExpertRouter(8)
    current = deepcopy(legacy)
    other = replace(inputs, base_latents=inputs.base_latents.detach().clone().requires_grad_())
    expected = legacy(inputs, evidence)
    deltas = {name: current.expert_delta(other, name, item) for name, item in evidence.items()}
    actual = current.route_precomputed(other, evidence, deltas)
    torch.testing.assert_close(actual.latents, expected.latents, rtol=0, atol=0)
    torch.testing.assert_close(actual.weights, expected.weights, rtol=0, atol=0)
    first = torch.autograd.grad(expected.latents.square().mean(),
                                (inputs.base_latents, *legacy.parameters()))
    second = torch.autograd.grad(actual.latents.square().mean(),
                                 (other.base_latents, *current.parameters()))
    for a, b in zip(first, second):
        torch.testing.assert_close(a, b)


def test_preview_gate_is_zero_when_disallowed_or_undamaged():
    inputs = _inputs()
    item = ExpertEvidence(torch.ones_like(inputs.base_latents),
                          torch.zeros_like(inputs.base_latents),
                          torch.ones_like(inputs.damaged), {})
    router = ExpertRouter(8)
    delta = router.expert_delta(inputs, "temporal", item)
    zero = router.preview_gate(inputs, "temporal", item, delta,
                               allowed=torch.zeros_like(inputs.damaged))
    assert torch.equal(zero, torch.zeros_like(zero))
    clean = replace(inputs, damaged=torch.zeros_like(inputs.damaged))
    assert not router.preview_gate(clean, "temporal", item, delta,
                                    allowed=torch.ones_like(inputs.damaged)).any()
    active = router.preview_gate(inputs, "temporal", item, delta,
                                  allowed=torch.ones_like(inputs.damaged))
    assert ((active > 0) & (active < 1)).all()


def test_round_condition_does_not_change_no_evidence_base():
    inputs = _inputs()
    router = ExpertRouter(8)
    evidence = _evidence(inputs)
    deltas = {name: router.expert_delta(inputs, name, item) for name, item in evidence.items()}
    actual = router.route_precomputed(inputs, evidence, deltas, round_embedding=torch.ones(8))
    assert torch.equal(actual.latents, inputs.base_latents)
