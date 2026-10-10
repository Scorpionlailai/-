from __future__ import annotations

import pytest
import torch

from malr.models.explicit_repair import build_explicit_repair


def _repair_inputs() -> tuple[torch.Tensor, ...]:
    # Two overlapping patches cover a length-6 sequence.  Coordinates 2 and 3
    # are missing in both patches, while the candidate predictions disagree
    # across the overlap but average to four at each missing coordinate.
    candidate = torch.tensor(
        [[[[10.0, 20.0, 2.0, 6.0], [6.0, 2.0, 30.0, 40.0]]]],
        requires_grad=True,
    )
    observed_values = torch.tensor(
        [[[[1.0, 2.0, 0.0, 0.0], [0.0, 0.0, 5.0, 6.0]]]]
    )
    observed_mask = torch.tensor(
        [[[[True, True, False, False], [False, False, True, True]]]]
    )
    validity = torch.ones_like(observed_mask)
    mean = torch.tensor([[2.0]])
    std = torch.tensor([[3.0]])
    return candidate, observed_values, observed_mask, validity, mean, std


def test_explicit_repair_aggregates_once_clamps_and_repatches() -> None:
    candidate, observed_values, observed_mask, validity, mean, std = _repair_inputs()
    repaired = build_explicit_repair(
        candidate,
        observed_values,
        observed_mask,
        validity,
        mean,
        std,
        window_len=6,
        patch_len=4,
        stride=2,
    )

    torch.testing.assert_close(
        repaired.normalized_sequence[:, 2:4],
        torch.full((1, 2, 1), 4.0),
    )
    assert repaired.sequence_missing.sum().item() == 2
    assert torch.equal(
        repaired.normalized_sequence[repaired.sequence_observed],
        repaired.observed_sequence[repaired.sequence_observed],
    )
    assert repaired.repaired_patches.requires_grad
    assert repaired.observation_patches.shape == repaired.repaired_patches.shape
    assert repaired.validity_patches.shape == repaired.repaired_patches.shape

    clean = torch.zeros_like(repaired.standardized_sequence)
    loss = torch.nn.functional.smooth_l1_loss(
        repaired.standardized_sequence[repaired.sequence_missing],
        clean[repaired.sequence_missing],
    )
    loss.backward()
    assert candidate.grad is not None
    assert torch.isfinite(candidate.grad).all()
    assert torch.equal(candidate.grad[observed_mask], torch.zeros_like(candidate.grad[observed_mask]))


def test_explicit_repair_rejects_inconsistent_overlapping_provenance() -> None:
    candidate, observed_values, observed_mask, validity, mean, std = _repair_inputs()
    inconsistent = observed_mask.clone()
    inconsistent[:, :, 1, 0] = True

    with pytest.raises(ValueError, match="overlapping provenance"):
        build_explicit_repair(
            candidate,
            observed_values,
            inconsistent,
            validity,
            mean,
            std,
            window_len=6,
            patch_len=4,
            stride=2,
        )
