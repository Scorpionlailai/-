import torch

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
