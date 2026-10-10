import torch

from malr.models.e23.nucleus import nucleus_prefix


def test_nucleus_uses_smallest_stable_probability_prefix():
    logits = torch.zeros(1, 4)
    selected = nucleus_prefix(logits, torch.ones_like(logits, dtype=torch.bool), p=0.5)
    assert selected.tolist() == [[True, True, False, False]]
    assert nucleus_prefix(logits, torch.ones_like(logits, dtype=torch.bool), p=0.1).sum() == 1


def test_nucleus_ignores_invalid_slots_and_empty_rows():
    logits = torch.tensor([[0., 100., 0., 0.], [1., 2., 3., 4.]])
    eligible = torch.tensor([[True, False, True, False], [False, False, False, False]])
    selected = nucleus_prefix(logits, eligible, p=0.9)
    assert selected.tolist() == [[True, False, True, False], [False, False, False, False]]


def test_nucleus_rejects_nonfinite_valid_score():
    logits = torch.tensor([[float("nan"), 0.]])
    with torch.no_grad():
        try:
            nucleus_prefix(logits, torch.ones_like(logits, dtype=torch.bool), p=0.9)
        except ValueError as error:
            assert "finite" in str(error)
        else:
            raise AssertionError("nonfinite eligible score was accepted")


def test_nucleus_exact_boundary_does_not_gain_extra_candidate_from_subtraction_rounding():
    logits = torch.zeros(1, 6, dtype=torch.float32)
    selected = nucleus_prefix(logits, torch.ones_like(logits, dtype=torch.bool), p=1 / 3)
    assert selected.tolist() == [[True, True, False, False, False, False]]
