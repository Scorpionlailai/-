import pytest
import torch

from malr.models.reconstruction import (
    LatentReconstructionHead,
    aggregate_patch_predictions,
)


def test_reconstruction_head_maps_each_latent_to_one_patch() -> None:
    head = LatentReconstructionHead(d_model=4, patch_len=16)
    latents = torch.randn(2, 3, 12, 4)

    patches = head(latents)

    assert patches.shape == (2, 3, 12, 16)
    assert patches.requires_grad


def test_overlapping_patch_predictions_are_averaged_once_per_time_point() -> None:
    patches = torch.zeros(1, 1, 2, 4)
    patches[:, :, 0] = 2.0
    patches[:, :, 1] = 6.0

    window, coverage = aggregate_patch_predictions(
        patches,
        window_len=6,
        patch_len=4,
        stride=2,
    )

    expected = torch.tensor([[[2.0], [2.0], [4.0], [4.0], [6.0], [6.0]]])
    torch.testing.assert_close(window, expected)
    torch.testing.assert_close(coverage, torch.tensor([[[1], [1], [2], [2], [1], [1]]]))


def test_reconstruction_excludes_replicated_padding_from_average() -> None:
    patches = torch.zeros(1, 1, 4, 4)
    patches[:, :, 0] = 1.0
    patches[:, :, 1] = 9.0
    patches[:, :, 2] = 7.0
    patches[:, :, 3] = torch.tensor([11.0, 11.0, 1000.0, 1000.0])
    validity = torch.ones_like(patches, dtype=torch.bool)
    validity[:, :, -1, 2:] = False

    window, coverage = aggregate_patch_predictions(
        patches,
        window_len=8,
        patch_len=4,
        stride=2,
        validity=validity,
    )

    assert window.shape == (1, 8, 1)
    assert coverage.shape == (1, 8, 1)
    torch.testing.assert_close(window[:, 6:], torch.full((1, 2, 1), 9.0))
    torch.testing.assert_close(
        coverage,
        torch.tensor([[[1], [1], [2], [2], [2], [2], [2], [2]]]),
    )


def test_reconstruction_requires_positive_coverage_and_finite_output() -> None:
    patches = torch.randn(1, 1, 1, 4)
    invalid = torch.zeros_like(patches, dtype=torch.bool)

    with pytest.raises(ValueError, match="no valid coverage"):
        aggregate_patch_predictions(
            patches,
            window_len=4,
            patch_len=4,
            stride=2,
            validity=invalid,
        )
