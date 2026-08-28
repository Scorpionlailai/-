import torch

from malr.data.patching import PATCH_MISSING, PATCH_OBSERVED, PATCH_PARTIAL
from malr.models.recovery import MissingAwareRecovery


def test_state_fusion_is_exact_and_observed_bypasses_prior():
    module = MissingAwareRecovery(c_in=1, patch_num=3, patch_len=16, d_model=4)
    visible = torch.full((1, 1, 3, 4), 2.0)
    prior = torch.full_like(visible, 3.0)
    recovered = torch.full_like(visible, 5.0)
    state = torch.tensor([[[PATCH_OBSERVED, PATCH_PARTIAL, PATCH_MISSING]]])

    output = module.fuse(visible, prior, recovered, state)

    torch.testing.assert_close(output[:, :, 0], visible[:, :, 0])
    torch.testing.assert_close(
        output[:, :, 1], visible[:, :, 1] + prior[:, :, 1] + recovered[:, :, 1]
    )
    torch.testing.assert_close(output[:, :, 2], prior[:, :, 2] + recovered[:, :, 2])


def test_frontend_shape_and_disabled_messages():
    module = MissingAwareRecovery(
        c_in=2,
        patch_num=3,
        patch_len=4,
        d_model=8,
        temporal_topk=2,
        variable_topk=1,
        use_temporal=False,
        use_variable=False,
        use_historical=False,
    )
    values = torch.randn(2, 2, 3, 4)
    mask = torch.ones_like(values, dtype=torch.bool)
    rho = torch.ones(2, 2, 3)
    state = torch.zeros(2, 2, 3, dtype=torch.long)
    state[:, :, 1] = PATCH_PARTIAL

    result = module(values, mask, rho, state)

    assert result.latents.shape == (2, 2, 3, 8)
    torch.testing.assert_close(result.messages.temporal, torch.zeros_like(result.latents))
    torch.testing.assert_close(result.messages.variable, torch.zeros_like(result.latents))
    torch.testing.assert_close(result.messages.historical, torch.zeros_like(result.latents))
    torch.testing.assert_close(result.latents[:, :, 0], result.visible[:, :, 0])


def test_recovery_relations_use_immutable_pre_recovery_visible_sources():
    module = MissingAwareRecovery(
        c_in=1,
        patch_num=3,
        patch_len=2,
        d_model=2,
        temporal_topk=1,
        use_variable=False,
        use_historical=False,
    )
    with torch.no_grad():
        module.W_x.weight.copy_(torch.eye(2))
        module.W_x.bias.zero_()
        module.W_m.weight.zero_()
        module.W_m.bias.zero_()
        module.W_q.weight.copy_(torch.eye(2))
        module.W_q.bias.zero_()
        module.temporal.W_k.weight.copy_(torch.eye(2))
        module.temporal.W_v.weight.copy_(torch.eye(2))
        module.fusion.weight.zero_()
        module.fusion.bias.zero_()
        module.fusion.weight[:, :2].copy_(torch.eye(2))
        module.variable_embedding.zero_()
        module.position_embedding.zero_()
        module.state_embedding.zero_()

    values = torch.tensor([[[[1.0, 0.0], [0.0, 0.0], [2.0, 0.0]]]])
    mask = torch.tensor([[[[True, True], [False, False], [True, True]]]])
    rho = torch.tensor([[[1.0, 0.0, 1.0]]])
    state = torch.tensor([[[PATCH_OBSERVED, PATCH_MISSING, PATCH_OBSERVED]]])

    result = module(values, mask, rho, state)

    # Patch 1 recovers from patch 0 or 2, but cannot then feed its recovered value
    # back into any target in the same layer because relation sources are `visible`.
    assert result.diagnostics["temporal_candidate_count"][0, 0, 1].item() == 2
    assert result.diagnostics["temporal_candidate_count"][0, 0, 0].item() == 0
