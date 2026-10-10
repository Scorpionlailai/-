import torch

from malr.models.refinement import GlobalLatentRefiner


def test_global_refiner_is_identity_at_zero_initialization() -> None:
    module = GlobalLatentRefiner(
        d_model=8,
        n_heads=2,
        ff_multiplier=2,
        dropout=0.0,
        output_init="zero",
    )
    latents = torch.randn(2, 3, 4, 8)
    prior = torch.randn_like(latents)
    rho = torch.rand(2, 3, 4)

    output = module(latents, prior, rho)

    assert torch.equal(output.latents, latents)
    assert torch.count_nonzero(module.output_projection.weight) == 0
    assert torch.count_nonzero(module.output_projection.bias) == 0


def test_global_refiner_never_changes_fully_observed_nodes() -> None:
    module = GlobalLatentRefiner(
        d_model=4,
        n_heads=1,
        ff_multiplier=1,
        dropout=0.0,
        output_init="zero",
    )
    with torch.no_grad():
        module.output_projection.bias.fill_(2.0)
    latents = torch.randn(1, 1, 3, 4)
    prior = torch.zeros_like(latents)
    rho = torch.tensor([[[1.0, 0.5, 0.0]]])

    output = module(latents, prior, rho)

    assert torch.equal(output.latents[:, :, 0], latents[:, :, 0])
    torch.testing.assert_close(
        output.latents[:, :, 1], latents[:, :, 1] + 1.0
    )
    torch.testing.assert_close(
        output.latents[:, :, 2], latents[:, :, 2] + 2.0
    )


def test_global_refiner_jointly_uses_other_nodes_after_gate_opens() -> None:
    module = GlobalLatentRefiner(
        d_model=4,
        n_heads=1,
        ff_multiplier=1,
        dropout=0.0,
        output_init="zero",
    )
    with torch.no_grad():
        module.attention.in_proj_weight.zero_()
        module.attention.in_proj_bias.zero_()
        module.attention.in_proj_weight[8:12].copy_(torch.eye(4))
        module.attention.out_proj.weight.copy_(torch.eye(4))
        module.attention.out_proj.bias.zero_()
        module.feed_forward[0].weight.zero_()
        module.feed_forward[0].bias.zero_()
        module.feed_forward[3].weight.zero_()
        module.feed_forward[3].bias.zero_()
        module.output_projection.weight.copy_(torch.eye(4))
        module.output_projection.bias.zero_()

    first = torch.tensor([[[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]]])
    second = first.clone()
    second[0, 0, 1] = torch.tensor([0.0, 0.0, 1.0, 0.0])
    prior = torch.zeros_like(first)
    rho = torch.tensor([[[0.0, 1.0]]])

    first_output = module(first, prior, rho).latents[0, 0, 0]
    second_output = module(second, prior, rho).latents[0, 0, 0]

    assert not torch.allclose(first_output, second_output)


def test_global_refiner_output_gate_receives_first_step_gradient() -> None:
    module = GlobalLatentRefiner(
        d_model=8,
        n_heads=2,
        ff_multiplier=2,
        dropout=0.0,
        output_init="zero",
    )
    latents = torch.randn(2, 2, 3, 8)
    prior = torch.randn_like(latents)
    rho = torch.zeros(2, 2, 3)
    target = torch.randn_like(latents)

    loss = (module(latents, prior, rho).latents - target).square().mean()
    loss.backward()

    gradient = module.output_projection.weight.grad
    assert gradient is not None
    assert gradient.abs().sum() > 0
