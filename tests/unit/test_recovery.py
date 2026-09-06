import pytest
import torch

from malr.data.patching import PATCH_MISSING, PATCH_OBSERVED, PATCH_PARTIAL
from malr.models.filtering import MissingAwareEdgeFilter
from malr.models.recovery import MissingAwareRecovery


def _filtered_replacement_recovery() -> MissingAwareRecovery:
    edge_filter = MissingAwareEdgeFilter(
        d_model=4,
        temporal_topk=2,
        variable_topk=1,
        historical_topk=1,
        temporal_threshold=0.5,
        variable_threshold=0.5,
        historical_threshold=0.5,
        hidden_multiplier=2,
        dropout=0.0,
        temperature=1.0,
        initial_keep_probability=0.9,
    )
    return MissingAwareRecovery(
        c_in=2,
        patch_num=3,
        patch_len=4,
        d_model=4,
        temporal_topk=2,
        variable_topk=1,
        use_historical=False,
        fusion="replacement_mlp",
        edge_filter=edge_filter,
        relation_selection="missing_aware_filter",
    )


def _force_all_filter_logits(module: MissingAwareRecovery, value: float) -> None:
    assert module.edge_filter is not None
    with torch.no_grad():
        for head in (
            module.edge_filter.temporal,
            module.edge_filter.variable,
            module.edge_filter.historical,
        ):
            head.network[-1].weight.zero_()
            head.network[-1].bias.fill_(value)


def _mixed_state_recovery_inputs() -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    values = torch.tensor(
        [
            [
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [2.0, 3.0, 4.0, 5.0],
                    [3.0, 4.0, 5.0, 6.0],
                    [4.0, 5.0, 6.0, 7.0],
                ],
            ]
        ]
    )
    mask = torch.tensor(
        [
            [
                [
                    [True, True, True, True],
                    [True, False, False, False],
                    [False, False, False, False],
                ],
                [
                    [True, True, True, True],
                    [True, True, True, True],
                    [True, True, True, True],
                ],
            ]
        ]
    )
    rho = mask.to(torch.float32).mean(dim=-1)
    state = torch.tensor(
        [
            [
                [PATCH_OBSERVED, PATCH_PARTIAL, PATCH_MISSING],
                [PATCH_OBSERVED, PATCH_OBSERVED, PATCH_OBSERVED],
            ]
        ]
    )
    return values, mask, rho, state, torch.randn(1, 2, 3, 4)


def test_all_rejected_relations_fall_back_without_changing_observed_nodes() -> None:
    module = _filtered_replacement_recovery()
    _force_all_filter_logits(module, value=-20.0)
    values, mask, rho, state, base = _mixed_state_recovery_inputs()

    result = module(values, mask, rho, state, base_latents=base)

    assert torch.equal(
        result.latents[state == PATCH_OBSERVED],
        base[state == PATCH_OBSERVED],
    )
    assert torch.isfinite(result.latents).all()
    assert torch.isfinite(result.filter_keep_loss)
    assert result.diagnostics["temporal_filter_retained_count"].sum() == 0
    assert result.diagnostics["variable_filter_retained_count"].sum() == 0


def test_recovery_ignores_raw_values_at_masked_positions() -> None:
    module = _filtered_replacement_recovery()
    values, mask, rho, state, base = _mixed_state_recovery_inputs()
    changed = values.clone()
    changed[~mask] = 1e6

    first = module(values, mask, rho, state, base_latents=base)
    second = module(changed, mask, rho, state, base_latents=base)

    torch.testing.assert_close(first.messages.temporal, second.messages.temporal)
    torch.testing.assert_close(first.messages.variable, second.messages.variable)
    torch.testing.assert_close(first.latents, second.latents)


def test_filter_keep_loss_averages_nonempty_relation_types_equally() -> None:
    module = _filtered_replacement_recovery()
    assert module.edge_filter is not None
    with torch.no_grad():
        for head, probability in (
            (module.edge_filter.temporal, 0.8),
            (module.edge_filter.variable, 0.4),
        ):
            head.network[-1].weight.zero_()
            head.network[-1].bias.copy_(torch.logit(torch.tensor(probability)))
    values, mask, rho, state, base = _mixed_state_recovery_inputs()

    result = module(values, mask, rho, state, base_latents=base)

    assert result.filter_keep_loss.item() == pytest.approx(0.6)


@pytest.mark.parametrize("hidden_multiplier", [1.5, True])
def test_residual_hidden_multiplier_requires_a_non_boolean_integer(
    hidden_multiplier: object,
) -> None:
    with pytest.raises(ValueError, match="positive non-boolean integer"):
        MissingAwareRecovery(
            c_in=1,
            patch_num=3,
            patch_len=4,
            d_model=8,
            residual_hidden_multiplier=hidden_multiplier,  # type: ignore[arg-type]
        )


def test_state_fusion_is_exact_and_observed_bypasses_prior():
    module = MissingAwareRecovery(c_in=1, patch_num=3, patch_len=16, d_model=4)
    visible = torch.full((1, 1, 3, 4), 2.0)
    prior = torch.full_like(visible, 3.0)
    recovered = torch.full_like(visible, 5.0)
    rho = torch.tensor([[[1.0, 0.5, 0.0]]])
    state = torch.tensor([[[PATCH_OBSERVED, PATCH_PARTIAL, PATCH_MISSING]]])

    output = module.fuse(visible, prior, recovered, state, rho)

    torch.testing.assert_close(output[:, :, 0], visible[:, :, 0])
    torch.testing.assert_close(
        output[:, :, 1], visible[:, :, 1] + prior[:, :, 1] + recovered[:, :, 1]
    )
    torch.testing.assert_close(output[:, :, 2], prior[:, :, 2] + recovered[:, :, 2])


def test_residual_mlp_is_exact_identity_at_initialization() -> None:
    module = MissingAwareRecovery(
        c_in=2,
        patch_num=3,
        patch_len=4,
        d_model=8,
        use_temporal=False,
        use_variable=False,
        use_historical=False,
        fusion="residual_mlp",
    )
    base = torch.randn(2, 2, 3, 8)
    rho = torch.tensor([[[1.0, 0.5, 0.0], [0.0, 0.75, 1.0]]]).expand(2, -1, -1)
    state = torch.tensor(
        [
            [
                [PATCH_OBSERVED, PATCH_PARTIAL, PATCH_MISSING],
                [PATCH_MISSING, PATCH_PARTIAL, PATCH_OBSERVED],
            ]
        ]
    ).expand(2, -1, -1)
    values = torch.zeros(2, 2, 3, 4)
    mask = torch.ones_like(values, dtype=torch.bool)

    out = module(values, mask, rho, state, base_latents=base)

    assert torch.equal(out.latents, base)
    assert torch.count_nonzero(module.residual_mlp[-1].weight) == 0
    assert torch.count_nonzero(module.residual_mlp[-1].bias) == 0


def test_residual_mlp_scales_recovery_by_missing_fraction() -> None:
    module = MissingAwareRecovery(
        c_in=2,
        patch_num=3,
        patch_len=4,
        d_model=8,
        fusion="residual_mlp",
    )
    base = torch.randn(2, 2, 3, 8)
    prior = torch.randn_like(base)
    recovered = torch.full_like(base, 2.0)
    rho = torch.tensor([[[1.0, 0.5, 0.0], [0.0, 0.75, 1.0]]]).expand(2, -1, -1)
    state = torch.tensor(
        [
            [
                [PATCH_OBSERVED, PATCH_PARTIAL, PATCH_MISSING],
                [PATCH_MISSING, PATCH_PARTIAL, PATCH_OBSERVED],
            ]
        ]
    ).expand(2, -1, -1)

    output = module.fuse(base, prior, recovered, state, rho)
    expected = base + (1.0 - rho).unsqueeze(-1) * 2.0

    assert torch.equal(output[rho == 1.0], base[rho == 1.0])
    assert torch.allclose(output, expected)


def test_replacement_mlp_interpolates_base_and_candidate_by_observation_ratio() -> None:
    module = MissingAwareRecovery(
        c_in=1,
        patch_num=3,
        patch_len=4,
        d_model=8,
        fusion="replacement_mlp",
    )
    base = torch.full((1, 1, 3, 8), 2.0)
    candidate = torch.full_like(base, 10.0)
    prior = torch.zeros_like(base)
    rho = torch.tensor([[[1.0, 0.5, 0.0]]])
    state = torch.tensor([[[PATCH_OBSERVED, PATCH_PARTIAL, PATCH_MISSING]]])

    output = module.fuse(base, prior, candidate, state, rho)

    expected = rho.unsqueeze(-1) * base + (1.0 - rho).unsqueeze(-1) * candidate
    torch.testing.assert_close(output, expected)


def test_fully_missing_replacement_does_not_retain_base_latent() -> None:
    module = MissingAwareRecovery(
        c_in=1,
        patch_num=1,
        patch_len=4,
        d_model=8,
        fusion="replacement_mlp",
    )
    first_base = torch.full((1, 1, 1, 8), -100.0)
    second_base = torch.full_like(first_base, 100.0)
    candidate = torch.full_like(first_base, 3.0)
    prior = torch.zeros_like(first_base)
    rho = torch.zeros(1, 1, 1)
    state = torch.full((1, 1, 1), PATCH_MISSING, dtype=torch.long)

    first = module.fuse(first_base, prior, candidate, state, rho)
    second = module.fuse(second_base, prior, candidate, state, rho)

    torch.testing.assert_close(first, candidate)
    torch.testing.assert_close(second, candidate)


def test_replacement_can_apply_identity_initialized_global_refiner() -> None:
    module = MissingAwareRecovery(
        c_in=2,
        patch_num=3,
        patch_len=4,
        d_model=8,
        use_temporal=False,
        use_variable=False,
        use_historical=False,
        fusion="replacement_mlp",
        global_refiner_enabled=True,
        global_refiner_heads=2,
        global_refiner_ff_multiplier=2,
        global_refiner_dropout=0.0,
        global_refiner_output_init="zero",
    )
    values = torch.zeros(1, 2, 3, 4)
    mask = torch.ones_like(values, dtype=torch.bool)
    base = torch.randn(1, 2, 3, 8)
    rho = torch.tensor([[[1.0, 0.5, 0.0], [0.0, 0.75, 1.0]]])
    state = torch.tensor(
        [
            [
                [PATCH_OBSERVED, PATCH_PARTIAL, PATCH_MISSING],
                [PATCH_MISSING, PATCH_PARTIAL, PATCH_OBSERVED],
            ]
        ]
    )

    output = module(values, mask, rho, state, base_latents=base)

    expected = rho.unsqueeze(-1) * base
    torch.testing.assert_close(output.latents, expected)
    assert module.global_refiner is not None
    assert "global_refiner_delta_norm" in output.diagnostics
    assert "global_refiner_applied_norm" in output.diagnostics
    assert output.diagnostics["global_refiner_delta_norm"].requires_grad is False


def test_residual_mlp_output_gate_opens_before_earlier_layers_learn() -> None:
    module = MissingAwareRecovery(
        c_in=2,
        patch_num=3,
        patch_len=4,
        d_model=8,
        use_temporal=False,
        use_variable=False,
        use_historical=False,
        fusion="residual_mlp",
    )
    base = torch.randn(2, 2, 3, 8)
    rho = torch.tensor([[[1.0, 0.5, 0.0], [0.0, 0.75, 1.0]]]).expand(2, -1, -1)
    state = torch.tensor(
        [
            [
                [PATCH_OBSERVED, PATCH_PARTIAL, PATCH_MISSING],
                [PATCH_MISSING, PATCH_PARTIAL, PATCH_OBSERVED],
            ]
        ]
    ).expand(2, -1, -1)
    values = torch.zeros(2, 2, 3, 4)
    mask = torch.ones_like(values, dtype=torch.bool)

    loss = module(values, mask, rho, state, base_latents=base).latents.square().mean()
    loss.backward()

    assert module.residual_mlp[-1].weight.grad is not None
    assert module.residual_mlp[-1].weight.grad.abs().sum() > 0
    assert module.residual_mlp[1].weight.grad is not None
    assert torch.count_nonzero(module.residual_mlp[1].weight.grad) == 0

    with torch.no_grad():
        module.residual_mlp[-1].weight.fill_(0.001)
    module.zero_grad()
    loss = module(values, mask, rho, state, base_latents=base).latents.square().mean()
    loss.backward()

    assert module.residual_mlp[1].weight.grad.abs().sum() > 0


def test_residual_mlp_reports_detached_damage_diagnostics() -> None:
    module = MissingAwareRecovery(
        c_in=2,
        patch_num=3,
        patch_len=4,
        d_model=8,
        use_temporal=False,
        use_variable=False,
        use_historical=False,
        fusion="residual_mlp",
    )
    values = torch.zeros(1, 2, 3, 4)
    mask = torch.ones_like(values, dtype=torch.bool)
    rho = torch.tensor([[[1.0, 0.5, 0.0], [0.0, 0.75, 1.0]]])
    state = torch.tensor(
        [
            [
                [PATCH_OBSERVED, PATCH_PARTIAL, PATCH_MISSING],
                [PATCH_MISSING, PATCH_PARTIAL, PATCH_OBSERVED],
            ]
        ]
    )
    base = torch.randn(1, 2, 3, 8)
    with torch.no_grad():
        module.residual_mlp[-1].bias.fill_(0.25)

    output = module(values, mask, rho, state, base_latents=base)
    diagnostics = output.diagnostics

    keys = {
        "residual_delta_norm",
        "residual_applied_norm",
        "residual_partial_mean_norm",
        "residual_missing_mean_norm",
    }
    assert keys <= diagnostics.keys()
    assert all(diagnostics[key].requires_grad is False for key in keys)
    expected_delta_norm = output.recovered.norm(dim=-1)
    expected_applied_norm = ((1.0 - rho).unsqueeze(-1) * output.recovered).norm(
        dim=-1
    )
    torch.testing.assert_close(diagnostics["residual_delta_norm"], expected_delta_norm)
    torch.testing.assert_close(diagnostics["residual_applied_norm"], expected_applied_norm)


def test_residual_mlp_reports_detached_zero_means_without_damaged_patches() -> None:
    module = MissingAwareRecovery(
        c_in=2,
        patch_num=3,
        patch_len=4,
        d_model=8,
        use_temporal=False,
        use_variable=False,
        use_historical=False,
        fusion="residual_mlp",
    )
    values = torch.randn(1, 2, 3, 4)
    mask = torch.ones_like(values, dtype=torch.bool)
    rho = torch.ones(1, 2, 3)
    state = torch.full((1, 2, 3), PATCH_OBSERVED, dtype=torch.long)
    base = torch.randn(1, 2, 3, 8)

    diagnostics = module(values, mask, rho, state, base_latents=base).diagnostics

    for key in ("residual_partial_mean_norm", "residual_missing_mean_norm"):
        assert torch.equal(diagnostics[key], torch.zeros(()))
        assert diagnostics[key].requires_grad is False


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
    base_latents = torch.randn(2, 2, 3, 8)

    result = module(values, mask, rho, state, base_latents=base_latents)

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
    base_latents = values.clone()

    result = module(values, mask, rho, state, base_latents=base_latents)

    # Patch 1 recovers from patch 0 or 2, but cannot then feed its recovered value
    # back into any target in the same layer because relation sources are `visible`.
    assert result.diagnostics["temporal_candidate_count"][0, 0, 1].item() == 2
    assert result.diagnostics["temporal_candidate_count"][0, 0, 0].item() == 0
