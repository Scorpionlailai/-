import pytest
import torch

from malr.models.patchtst import PatchTSTCore, load_clean_initialization


def build_tiny_core() -> PatchTSTCore:
    return PatchTSTCore(
        c_in=7,
        seq_len=96,
        pred_len=96,
        patch_len=16,
        stride=8,
        d_model=32,
        n_heads=4,
        d_ff=64,
        n_layers=2,
        dropout=0.0,
        head_dropout=0.0,
    )


def test_clean_and_latent_entry_paths_match() -> None:
    torch.manual_seed(7)
    model = build_tiny_core().eval()
    patches = torch.randn(2, 7, 12, 16)

    latent = model.project_clean_patches(patches)
    direct = model.forecast_from_latents(latent)
    manual = model.head(model.encode_latents(latent))

    torch.testing.assert_close(direct, manual)
    torch.testing.assert_close(model.forward_clean(patches), direct)
    assert latent.shape == (2, 7, 12, 32)
    assert direct.shape == (2, 7, 96)


def test_clean_path_updates_projection_encoder_and_head() -> None:
    model = build_tiny_core().train()
    prediction = model.forward_clean(torch.randn(2, 7, 12, 16))

    prediction.square().mean().backward()

    assert model.W_P.weight.grad is not None
    assert model.encoder.layers[0].self_attn.W_Q.weight.grad is not None
    assert model.head.linear.weight.grad is not None


def test_latent_entry_rejects_wrong_patch_count() -> None:
    model = build_tiny_core()

    with pytest.raises(ValueError, match="expected 12 patches"):
        model.encode_latents(torch.randn(2, 7, 11, 32))


def test_clean_initialization_copies_matching_weights() -> None:
    source = build_tiny_core()
    target = build_tiny_core()
    with torch.no_grad():
        source.W_P.weight.fill_(0.25)
        source.head.linear.bias.fill_(0.75)

    load_clean_initialization(target, source.state_dict())

    torch.testing.assert_close(target.W_P.weight, source.W_P.weight)
    torch.testing.assert_close(target.head.linear.bias, source.head.linear.bias)
