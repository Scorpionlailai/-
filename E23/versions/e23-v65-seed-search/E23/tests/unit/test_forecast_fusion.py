from __future__ import annotations

import pytest
import torch

from malr.models.forecast_fusion import ForecastHVFusion


def _inputs(
    *,
    batch: int = 1,
    channels: int = 2,
    patches: int = 3,
    d_model: int = 4,
    patch_len: int = 6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    route_latents = torch.arange(
        batch * channels * patches * 8 * d_model,
        dtype=torch.float32,
    ).reshape(batch, channels, patches, 8, d_model) / 100.0
    route_available = torch.ones(
        batch, channels, patches, 8, dtype=torch.bool
    )
    teacher_route_logits = torch.linspace(
        -0.4,
        0.4,
        steps=batch * channels * patches * 8,
    ).reshape(batch, channels, patches, 8)
    missing_indicator = torch.zeros(
        batch, channels, patches, patch_len, dtype=torch.bool
    )
    missing_indicator[..., ::2] = True
    return route_latents, route_available, teacher_route_logits, missing_indicator


def test_forecast_fusion_returns_finite_normalized_hv_none_weights() -> None:
    torch.manual_seed(7)
    fusion = ForecastHVFusion(d_model=4, patch_len=6)
    route_latents, availability, teacher_logits, missing_indicator = _inputs()

    fused, weights = fusion(
        route_latents,
        availability,
        teacher_logits,
        missing_indicator,
    )

    assert fused.shape == (1, 2, 3, 4)
    assert weights.shape == (1, 2, 3, 4)
    assert fused.dtype == route_latents.dtype
    assert weights.dtype == route_latents.dtype
    assert torch.equal(weights[..., 2], torch.zeros_like(weights[..., 2]))
    assert torch.all(weights[..., (0, 1, 3)] > 0.0)
    torch.testing.assert_close(
        weights.sum(dim=-1), torch.ones_like(weights[..., 0])
    )
    assert torch.isfinite(fused).all()
    assert torch.isfinite(weights).all()


def test_forecast_fusion_masks_unavailable_hv_and_exactly_falls_back_to_none() -> None:
    torch.manual_seed(11)
    fusion = ForecastHVFusion(d_model=4, patch_len=6)
    route_latents, availability, teacher_logits, missing_indicator = _inputs(
        channels=1
    )
    availability[0, 0, 0, 1] = False
    availability[0, 0, 1, 3] = False
    availability[0, 0, 2] = False
    teacher_logits[0, 0, 0, 1] = 1_000.0
    teacher_logits[0, 0, 1, 3] = 1_000.0

    fused, weights = fusion(
        route_latents,
        availability,
        teacher_logits,
        missing_indicator,
    )

    assert weights[0, 0, 0, 1].item() == 0.0
    assert weights[0, 0, 1, 3].item() == 0.0
    torch.testing.assert_close(
        weights[0, 0, 2],
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        fused[0, 0, 2],
        route_latents[0, 0, 2, 0],
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    ("input_name", "replacement", "message"),
    [
        (
            "route_latents",
            lambda value: value[..., :3],
            "route_latents must have shape",
        ),
        (
            "route_available",
            lambda value: value[..., :7],
            "route_available must have shape",
        ),
        (
            "teacher_route_logits",
            lambda value: value[..., :7],
            "teacher_route_logits must have shape",
        ),
        (
            "missing_indicator",
            lambda value: value[..., :5],
            "missing_indicator must have shape",
        ),
        (
            "route_available",
            lambda value: value.to(torch.float32),
            "route_available must be bool",
        ),
        (
            "teacher_route_logits",
            lambda value: value.to(torch.int64),
            "teacher_route_logits must be floating-point",
        ),
        (
            "missing_indicator",
            lambda value: value.to(torch.float32),
            "missing_indicator must be bool",
        ),
        (
            "route_latents",
            lambda value: value.masked_fill(
                torch.arange(value.numel()).reshape(value.shape).eq(0), torch.nan
            ),
            "route_latents must contain only finite values",
        ),
        (
            "teacher_route_logits",
            lambda value: value.masked_fill(
                torch.arange(value.numel()).reshape(value.shape).eq(0), torch.inf
            ),
            "teacher_route_logits must contain only finite values",
        ),
    ],
)
def test_forecast_fusion_rejects_invalid_inputs(
    input_name: str,
    replacement: object,
    message: str,
) -> None:
    fusion = ForecastHVFusion(d_model=4, patch_len=6)
    names = (
        "route_latents",
        "route_available",
        "teacher_route_logits",
        "missing_indicator",
    )
    values = list(_inputs())
    index = names.index(input_name)
    values[index] = replacement(values[index])  # type: ignore[operator]

    with pytest.raises((TypeError, ValueError), match=message):
        fusion(*values)


def test_forecast_fusion_detaches_teacher_prior_but_trains_from_forecast_mse() -> None:
    torch.manual_seed(23)
    fusion = ForecastHVFusion(d_model=4, patch_len=6, hidden_dim=9)
    route_latents, availability, teacher_logits, missing_indicator = _inputs()
    teacher_logits.requires_grad_(True)

    fused, weights = fusion(
        route_latents,
        availability,
        teacher_logits,
        missing_indicator,
    )
    loss = torch.nn.functional.mse_loss(fused, torch.zeros_like(fused))
    loss.backward()

    assert teacher_logits.grad is None
    assert torch.equal(weights[..., 2], torch.zeros_like(weights[..., 2]))
    parameter_gradients = [
        parameter.grad for parameter in fusion.parameters() if parameter.requires_grad
    ]
    assert parameter_gradients
    assert all(gradient is not None for gradient in parameter_gradients)
    assert all(
        torch.isfinite(gradient).all()
        for gradient in parameter_gradients
        if gradient is not None
    )
    assert any(
        torch.count_nonzero(gradient).item() > 0
        for gradient in parameter_gradients
        if gradient is not None
    )
