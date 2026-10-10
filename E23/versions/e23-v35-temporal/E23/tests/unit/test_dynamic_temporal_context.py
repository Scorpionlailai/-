from __future__ import annotations

import torch

from malr.models.candidates_v2 import build_temporal_candidates
from malr.models.dynamic_temporal_context import DynamicTemporalContextRepair


def _fixture() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    object,
]:
    torch.manual_seed(23)
    batch, channels, patches, patch_len, d_model = 1, 1, 11, 4, 4
    values = torch.randn(batch, channels, patches, patch_len)
    mask = torch.ones_like(values, dtype=torch.bool)
    mask[:, :, 5] = False
    validity = torch.ones_like(values, dtype=torch.bool)
    rho = torch.ones(batch, channels, patches)
    rho[:, :, 5] = 0.0
    damaged = torch.zeros_like(rho, dtype=torch.bool)
    damaged[:, :, 5] = True
    starts = torch.arange(patches).view(1, patches) * patch_len
    candidates = build_temporal_candidates(
        values,
        mask,
        rho,
        starts,
        distance_boundaries=(2, 4, 8),
        target_eligible=damaged,
    )
    hv_latents = torch.randn(batch, channels, patches, d_model)
    return values, mask, validity, rho, damaged, candidates, hv_latents


def test_full_lookback_temporal_context_separates_local_and_global_edges() -> None:
    values, mask, validity, rho, damaged, candidates, hv_latents = _fixture()
    repair = DynamicTemporalContextRepair(
        d_model=hv_latents.shape[-1],
        patch_len=values.shape[-1],
        feature_dim=candidates.features.shape[-1],
        residual_limit=1.5,
    )

    output = repair(
        hv_latents,
        values,
        values,
        mask,
        validity,
        rho,
        candidates,
        damaged,
    )

    target_local = output.local_selected[0, 0, 5]
    target_global = output.global_selected[0, 0, 5]
    assert target_local.sum().item() == 2
    assert target_global.sum().item() == 8
    expected_local = candidates.signed_patch_lag[0, 0, 5].abs().eq(1)
    assert torch.equal(
        target_local,
        expected_local,
    )
    assert torch.equal(target_local & target_global, torch.zeros_like(target_local))
    assert torch.allclose(output.residual_patches, torch.zeros_like(output.residual_patches))
    assert torch.equal(
        output.residual_patches[0, 0, 0], torch.zeros(values.shape[-1])
    )


def test_temporal_context_masks_unreliable_sources_and_bounds_residuals() -> None:
    values, mask, validity, rho, damaged, candidates, hv_latents = _fixture()
    # Source patch 4 is the nearest left anchor; make it unreliable.
    rho[:, :, 4] = 0.0
    candidates = build_temporal_candidates(
        values,
        mask,
        rho,
        torch.arange(values.shape[2]).view(1, -1) * values.shape[-1],
        distance_boundaries=(2, 4, 8),
        target_eligible=damaged,
    )
    repair = DynamicTemporalContextRepair(
        d_model=hv_latents.shape[-1],
        patch_len=values.shape[-1],
        feature_dim=candidates.features.shape[-1],
        residual_limit=0.25,
    )
    with torch.no_grad():
        repair.residual_head[-1].weight.fill_(2.0)
        repair.residual_head[-1].bias.fill_(2.0)

    output = repair(
        hv_latents,
        values,
        values,
        mask,
        validity,
        rho,
        candidates,
        damaged,
    )

    target_candidates = candidates.eligible[0, 0, 5]
    target_sources = candidates.source_patch[0, 0, 5]
    assert not target_candidates[target_sources.eq(4)].any()
    assert torch.isfinite(output.residual_patches).all()
    assert output.residual_patches.abs().max().item() <= 0.25 + 1e-6
    assert output.gate_weights.shape[-1] == 3
    assert torch.allclose(
        output.gate_weights[..., 0] + output.gate_weights[..., 1] + output.gate_weights[..., 2],
        torch.ones_like(output.gate_weights[..., 0]),
        atol=1e-6,
    )
