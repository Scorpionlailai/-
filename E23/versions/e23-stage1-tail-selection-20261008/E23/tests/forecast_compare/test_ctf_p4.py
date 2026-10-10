from pathlib import Path

import torch

from integrations.forecast_compare.ctf_p4 import (
    ctf_inputs, group_by_full_patch_visibility, load_official_ctf,
)
from integrations.forecast_compare.protocol import P4Batch


def test_ctf_groups_samples_by_their_own_empty_patches():
    mask = torch.ones(3, 4, 1, dtype=torch.bool)
    mask[1, :2] = False
    mask[2, 2:] = False
    groups = group_by_full_patch_visibility(mask, patch_len=2)
    assert [indices.tolist() for indices in groups] == [[0], [1], [2]]


def test_ctf_inputs_hide_missing_values():
    batch = P4Batch(
        normalized_observed=torch.tensor([[[1.0], [0.0], [-1.0], [0.0]]]),
        normalized_target=torch.tensor([[[1.0], [999.0], [-1.0], [999.0]]]),
        observed_mask=torch.tensor([[[True], [False], [True], [False]]]),
        query_mean=torch.tensor([[10.0]]), query_std=torch.tensor([[2.0]]),
        forecast_target=torch.full((1, 1, 4), 999.0),
    )
    values, mask = ctf_inputs(batch)
    torch.testing.assert_close(values.flatten(), torch.tensor([12.0, 0.0, 8.0, 0.0]))
    torch.testing.assert_close(mask.flatten(), torch.tensor([1.0, 0.0, 1.0, 0.0]))


def test_ctf_missing_model_predicts_with_its_own_mask():
    model = load_official_ctf(Path('artifacts/upstream/ChannelTokenFormer'), device=torch.device('cpu'))
    model.eval()
    values = torch.zeros(2, 96, 7)
    mask = torch.ones_like(values)
    mask[:, :24, 0] = 0
    with torch.no_grad():
        prediction = model(values, None, None, None, missing_flag=mask)
    assert prediction.shape == (2, 96, 7)
    assert torch.isfinite(prediction).all()
