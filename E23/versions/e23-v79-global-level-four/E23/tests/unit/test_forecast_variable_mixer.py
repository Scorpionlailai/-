from copy import deepcopy

import torch

from malr.models.patchtst import PatchTSTCore
from malr.models.repaired_forecast import MaskAwareForecastModel


def _core():
    return PatchTSTCore(
        c_in=3, seq_len=24, pred_len=6, patch_len=8, stride=4,
        d_model=8, n_heads=2, d_ff=16, n_layers=1, dropout=0.0,
    )


def test_patch_variable_mixer_starts_as_old_path_then_exchanges_variables():
    torch.manual_seed(41)
    core = _core()
    old = MaskAwareForecastModel(core).eval()
    mixed = MaskAwareForecastModel(deepcopy(core), variable_mixer=True).eval()
    assert not any(key.startswith("variable_mixer.") for key in old.state_dict())
    assert mixed.variable_mixer is not None
    patches = torch.randn(2, 3, 6, 8)
    observed = torch.ones_like(patches, dtype=torch.bool)
    mean = torch.zeros(2, 3)
    std = torch.ones(2, 3)
    original = old(patches, observed, mean, std)
    initial = mixed(patches, observed, mean, std)
    torch.testing.assert_close(initial, original, rtol=0, atol=0)
    initial.square().mean().backward()
    assert mixed.variable_mixer.residual_scale.grad.abs() > 0
    mixed.zero_grad(set_to_none=True)

    with torch.no_grad():
        mixed.variable_mixer.residual_scale.fill_(1.0)
    changed = patches.clone()
    changed[:, 1, :, 0] += 2.0
    torch.testing.assert_close(old(changed, observed, mean, std)[:, 0],
                               original[:, 0], rtol=0, atol=0)
    mixed_before = mixed(patches, observed, mean, std)
    mixed_after = mixed(changed, observed, mean, std)
    assert (mixed_after[:, 0] - mixed_before[:, 0]).abs().max() > 1e-6

    mixed_before.square().mean().backward()
    gradient = mixed.variable_mixer.attention.in_proj_weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0
