from dataclasses import replace

import pytest
import torch

from malr.models.e23.contracts import CandidateEvidence, ExpertEvidence, ExpertInputs
from malr.models.e23.multihead_router import CandidateMultiHeadRouter


def _inputs():
    base = torch.randn(2, 3, 4, 128, requires_grad=True)
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


def _evidence(inputs):
    evidence = {}
    for name, count, quality_dim in (("historical", 4, 5), ("variable", 12, 4),
                                     ("temporal", 10, 5)):
        shape = (*inputs.damaged.shape, count)
        valid = torch.ones(shape, dtype=torch.bool)
        valid[..., -1] = False
        weights = valid.float() / valid.sum(-1, keepdim=True)
        messages = torch.randn(*shape, 128, requires_grad=True)
        quality = torch.randn(*shape, quality_dim)
        evidence[name] = ExpertEvidence(
            message=torch.zeros_like(inputs.base_latents),
            quality=torch.zeros_like(inputs.base_latents),
            available=torch.ones_like(inputs.damaged), provenance={},
            candidates=CandidateEvidence(messages, quality, weights, valid),
        )
    return evidence


def test_multihead_reads_candidates_and_backpropagates_to_them():
    torch.manual_seed(11)
    inputs = _inputs()
    evidence = _evidence(inputs)
    router = CandidateMultiHeadRouter(128, head_count=8)
    output = router(inputs, evidence)
    assert output.latents.shape == inputs.base_latents.shape
    assert output.head_weights.shape == (*inputs.damaged.shape, 8, 4)
    torch.testing.assert_close(output.weights, output.head_weights.mean(-2))
    torch.testing.assert_close(output.head_weights.sum(-1), torch.ones_like(output.head_weights[..., 0]))
    for name, count in (("historical", 4), ("variable", 12), ("temporal", 10)):
        attention = output.candidate_attention[name]
        assert attention.shape == (*inputs.damaged.shape, 8, count)
        torch.testing.assert_close(attention.sum(-1), torch.ones_like(attention[..., 0]))
        assert attention[..., -1].count_nonzero() == 0
    output.latents.square().mean().backward()
    for item in evidence.values():
        assert item.candidates.messages.grad is not None
        assert item.candidates.messages.grad.abs().sum() > 0


def test_candidate_change_matters_even_with_same_aggregated_message():
    torch.manual_seed(17)
    inputs = _inputs()
    evidence = _evidence(inputs)
    router = CandidateMultiHeadRouter(128, head_count=8)
    initial = router(inputs, evidence).latents
    item = evidence["historical"]
    changed = item.candidates.messages.detach().clone()
    changed[..., 0, :] += 2.0
    evidence["historical"] = replace(
        item, candidates=replace(item.candidates, messages=changed),
    )
    later = router(inputs, evidence).latents
    assert not torch.allclose(initial, later)
    assert torch.equal(evidence["historical"].message, item.message)


def test_multihead_none_path_and_invalid_candidate_mask():
    inputs = _inputs()
    evidence = _evidence(inputs)
    router = CandidateMultiHeadRouter(128, head_count=8)
    for name, item in evidence.items():
        evidence[name] = replace(item, available=torch.zeros_like(item.available))
    output = router(inputs, evidence)
    assert torch.equal(output.latents, inputs.base_latents)
    assert output.head_weights[..., 0].eq(1).all()
    assert output.head_weights[..., 1:].eq(0).all()
    assert torch.isfinite(output.latents).all()


def test_multihead_rejects_missing_candidates_and_wrong_width():
    inputs = _inputs()
    evidence = _evidence(inputs)
    router = CandidateMultiHeadRouter(128, head_count=8)
    evidence["historical"] = replace(evidence["historical"], candidates=None)
    with pytest.raises(ValueError, match="candidate"):
        router(inputs, evidence)
    with pytest.raises(ValueError, match="divisible"):
        CandidateMultiHeadRouter(127, head_count=8)


def test_eight_heads_can_concat_to_width64():
    inputs = _inputs()
    inputs = replace(inputs, base_latents=inputs.base_latents.detach()[..., :64].requires_grad_())
    evidence = _evidence(inputs)
    for name, item in evidence.items():
        candidates = item.candidates
        evidence[name] = replace(
            item, candidates=replace(
                candidates, messages=candidates.messages.detach()[..., :64].requires_grad_(),
            ),
        )
    router = CandidateMultiHeadRouter(64, head_count=8)
    output = router(inputs, evidence)
    assert output.latents.shape == (*inputs.damaged.shape, 64)
    assert output.head_weights.shape == (*inputs.damaged.shape, 8, 4)
    output.latents.square().mean().backward()
    assert evidence["variable"].candidates.messages.grad.abs().sum() > 0


def _width64_evidence():
    inputs = _inputs()
    inputs = replace(inputs, base_latents=inputs.base_latents.detach()[..., :64].requires_grad_())
    evidence = _evidence(inputs)
    for name, item in evidence.items():
        candidates = item.candidates
        evidence[name] = replace(
            item, candidates=replace(
                candidates, messages=candidates.messages.detach()[..., :64].requires_grad_(),
            ),
        )
    return inputs, evidence


def test_full_width_head_gate_selects_eight_64_dimensional_proposals():
    torch.manual_seed(31)
    inputs, evidence = _width64_evidence()
    router = CandidateMultiHeadRouter(64, head_count=8, fusion_mode="gated_sum")
    output = router(inputs, evidence)
    assert output.latents.shape == (*inputs.damaged.shape, 64)
    assert output.head_deltas.shape == (*inputs.damaged.shape, 8, 64)
    assert output.head_gate_weights.shape == (*inputs.damaged.shape, 9)
    torch.testing.assert_close(output.head_gate_weights.sum(-1),
                               torch.ones_like(output.head_gate_weights[..., 0]))
    torch.testing.assert_close(output.weights.sum(-1), torch.ones_like(output.weights[..., 0]))
    expected = inputs.base_latents + (
        output.head_gate_weights[..., 1:, None] * output.head_deltas
    ).sum(-2)
    torch.testing.assert_close(output.latents, expected)
    output.latents.square().mean().backward()
    assert router.head_gate[0].weight.grad.abs().sum() > 0
    for item in evidence.values():
        assert item.candidates.messages.grad.abs().sum() > 0


def test_full_width_head_gate_keeps_base_when_no_expert_can_help():
    inputs, evidence = _width64_evidence()
    for name, item in evidence.items():
        evidence[name] = replace(item, available=torch.zeros_like(item.available))
    router = CandidateMultiHeadRouter(64, head_count=8, fusion_mode="gated_sum")
    output = router(inputs, evidence)
    assert torch.equal(output.latents, inputs.base_latents)
    assert output.head_gate_weights[..., 0].eq(1).all()
    assert output.head_gate_weights[..., 1:].eq(0).all()
    assert output.weights[..., 0].eq(1).all()
