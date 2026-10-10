"""Catch hard-coded channel dimensions and changes to observed inputs."""
from types import SimpleNamespace

import pytest
import torch

from malr.data.patching import patchify_preserving_values
from malr.models.explicit_repair import build_explicit_repair
from malr.models.e23.recent_mean import RecentMeanProjector


@pytest.mark.parametrize('variables', [7, 21, 64])
@pytest.mark.parametrize('mode', ['global_level_mean_v2', 'global_dense_time_v4', 'global_recent_bands_v3'])
def test_refinement_accepts_variable_count_and_preserves_observed_values(variables, mode):
    torch.manual_seed(31)
    values = torch.randn(2, 96, variables)
    mask = torch.rand_like(values) > .7
    mask[1] = False  # no source anywhere in this window
    visible = values.masked_fill(~mask, 0)
    patches = patchify_preserving_values(visible, mask, 16, 8)
    explicit = build_explicit_repair(torch.randn_like(patches.values), patches.values,
                                    patches.effective_mask, patches.validity,
                                    torch.zeros(2, variables), torch.ones(2, variables),
                                    window_len=96, patch_len=16, stride=8)
    model = RecentMeanProjector(variables, mode=mode)
    # Activate the zero-initialized outputs so preservation cannot pass by
    # accident merely because all repair residuals are zero.
    for name, parameter in model.named_parameters():
        if name.endswith('output.weight') or name in ['network.2.weight', 'global_level_network.2.weight', 'recent_band_network.2.weight']:
            torch.nn.init.normal_(parameter, std=.01)
    result, _ = model(explicit, SimpleNamespace(latents=torch.randn(2, variables, explicit.validity_patches.shape[2], 64)),
                      torch.randn(2*variables, 64, 96), torch.zeros(2, variables), torch.ones(2, variables))
    assert result.normalized_sequence.shape == values.shape
    assert torch.isfinite(result.normalized_sequence).all()
    assert torch.equal(result.normalized_sequence[mask], visible[mask])
    if mode == 'global_dense_time_v4':
        # The dense residual must not hallucinate when no true observations exist.
        assert torch.equal(result.normalized_sequence[1, :64], explicit.normalized_sequence[1, :64])
    result.normalized_sequence.square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_seven_variable_state_geometry_remains_compatible():
    model = RecentMeanProjector(7, mode='global_dense_time_v4')
    assert sum(p.numel() for p in model.global_dense_refiner.parameters()) == 80647
    clone = RecentMeanProjector(7, mode='global_dense_time_v4')
    clone.load_state_dict(model.state_dict(), strict=True)
