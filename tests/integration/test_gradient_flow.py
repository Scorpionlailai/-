import torch

from malr.data.patching import PATCH_MISSING, PATCH_OBSERVED
from tests.unit.test_teacher_and_losses import _tiny_ours


def test_ours_joint_training_updates_patch_projection_encoder_and_head():
    torch.manual_seed(7)
    model = _tiny_ours().train()
    batch_size = 2
    patch_count = model.core.patch_num
    values = torch.randn(batch_size, 1, patch_count, 16)
    mask = torch.ones_like(values, dtype=torch.bool)
    mask[:, :, 1] = False
    values = torch.where(mask, values, torch.zeros_like(values))
    rho = mask.float().mean(dim=-1)
    state = torch.zeros(batch_size, 1, patch_count, dtype=torch.long)
    state[:, :, 1] = 2
    clean_raw = torch.randn_like(values)
    targets = torch.randn(batch_size, 1, 4)
    patch_valid = torch.ones_like(state, dtype=torch.bool)

    result = model.training_objective(
        values,
        mask,
        rho,
        state,
        torch.zeros(batch_size, 1),
        torch.ones(batch_size, 1),
        targets,
        clean_raw,
        patch_valid,
        lambda_rec=1.0,
    )
    result.loss.total.backward()

    assert not result.teacher_target.requires_grad
    assert not result.prototype_latents.requires_grad
    assert model.core.W_P.weight.grad is not None
    assert model.core.encoder.layers[0].self_attn.W_Q.weight.grad is not None
    assert model.core.head.linear.weight.grad is not None


def test_forecast_loss_trains_residual_output_then_hidden_layer():
    torch.manual_seed(11)
    model = _tiny_ours(fusion="residual_mlp").train()
    batch_size = 2
    patch_count = model.core.patch_num
    values = torch.randn(batch_size, 1, patch_count, 16)
    mask = torch.ones_like(values, dtype=torch.bool)
    mask[:, :, 1] = False
    values = torch.where(mask, values, torch.zeros_like(values))
    rho = mask.float().mean(dim=-1)
    state = torch.zeros(batch_size, 1, patch_count, dtype=torch.long)
    state[:, :, 1] = 2
    targets = torch.randn(batch_size, 1, 4)
    patch_valid = torch.ones_like(state, dtype=torch.bool)

    result = model.training_objective(
        values,
        mask,
        rho,
        state,
        torch.zeros(batch_size, 1),
        torch.ones(batch_size, 1),
        targets,
        None,
        patch_valid,
        use_recovery_loss=False,
    )
    result.loss.total.backward()

    output_projection = model.recovery.residual_mlp[-1]
    assert result.loss.total.item() == result.loss.forecast_mse.item()
    assert output_projection.weight.grad is not None
    assert torch.isfinite(output_projection.weight.grad).all()
    assert torch.count_nonzero(output_projection.weight.grad) > 0

    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        output_projection.weight.fill_(0.01)
    fresh_result = model.training_objective(
        values,
        mask,
        rho,
        state,
        torch.zeros(batch_size, 1),
        torch.ones(batch_size, 1),
        targets,
        None,
        patch_valid,
        use_recovery_loss=False,
    )
    fresh_result.loss.total.backward()

    earlier_linear = model.recovery.residual_mlp[1]
    assert earlier_linear.weight.grad is not None
    assert torch.isfinite(earlier_linear.weight.grad).all()
    assert torch.count_nonzero(earlier_linear.weight.grad) > 0
    temporal_gradients = [
        parameter.grad for parameter in model.recovery.temporal.parameters()
    ]
    assert any(
        gradient is not None
        and torch.isfinite(gradient).all()
        and torch.count_nonzero(gradient) > 0
        for gradient in temporal_gradients
    )


def test_forecast_loss_reaches_replacement_and_global_refiner_output_gates() -> None:
    torch.manual_seed(17)
    model = _tiny_ours(
        fusion="replacement_mlp",
        global_refiner_enabled=True,
    ).train()
    batch_size = 2
    patch_count = model.core.patch_num
    values = torch.randn(batch_size, 1, patch_count, 16)
    mask = torch.ones_like(values, dtype=torch.bool)
    mask[:, :, 1] = False
    values = torch.where(mask, values, torch.zeros_like(values))
    rho = mask.float().mean(dim=-1)
    state = torch.zeros(batch_size, 1, patch_count, dtype=torch.long)
    state[:, :, 1] = 2
    targets = torch.randn(batch_size, 1, 4)
    patch_valid = torch.ones_like(state, dtype=torch.bool)

    result = model.training_objective(
        values,
        mask,
        rho,
        state,
        torch.zeros(batch_size, 1),
        torch.ones(batch_size, 1),
        targets,
        None,
        patch_valid,
        use_recovery_loss=False,
    )
    result.loss.total.backward()

    replacement_gate = model.recovery.replacement_mlp[-1]
    refiner_gate = model.recovery.global_refiner.output_projection
    assert replacement_gate.weight.grad is not None
    assert replacement_gate.weight.grad.abs().sum() > 0
    assert refiner_gate.weight.grad is not None
    assert refiner_gate.weight.grad.abs().sum() > 0


def test_keep_penalty_reaches_filter_while_forecast_trains_patchtst() -> None:
    torch.manual_seed(23)
    model = _tiny_ours(fusion="replacement_mlp", filtered=True).train()
    patch_count = model.core.patch_num
    values = torch.randn(2, 1, patch_count, 16)
    mask = torch.ones_like(values, dtype=torch.bool)
    mask[:, :, 1] = False
    values = torch.where(mask, values, torch.zeros_like(values))
    rho = mask.float().mean(dim=-1)
    state = torch.full((2, 1, patch_count), PATCH_OBSERVED, dtype=torch.long)
    state[:, :, 1] = PATCH_MISSING

    result = model.training_objective(
        values,
        mask,
        rho,
        state,
        torch.zeros(2, 1),
        torch.ones(2, 1),
        torch.randn(2, 1, 4),
        None,
        torch.ones_like(state, dtype=torch.bool),
        use_recovery_loss=False,
        lambda_keep=1e-4,
    )
    result.loss.total.backward()

    filter_gradient = model.recovery.edge_filter.temporal.network[-1].bias.grad
    assert filter_gradient is not None
    assert torch.isfinite(filter_gradient).all()
    assert torch.count_nonzero(filter_gradient) > 0
    assert model.core.encoder.layers[0].self_attn.W_Q.weight.grad is not None
    assert model.core.head.linear.weight.grad is not None
