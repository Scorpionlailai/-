from pathlib import Path

import torch

from integrations.forecast_compare.crib_p4 import (
    crib_input, crib_training_loss, load_official_crib, verify_crib_selection,
)
from integrations.forecast_compare.protocol import P4Batch


def test_crib_input_uses_only_visible_standardized_history():
    batch = P4Batch(
        normalized_observed=torch.tensor([[[1.0], [0.0], [-1.0], [0.0]]]),
        normalized_target=torch.tensor([[[1.0], [999.0], [-1.0], [999.0]]]),
        observed_mask=torch.tensor([[[True], [False], [True], [False]]]),
        query_mean=torch.tensor([[10.0]]),
        query_std=torch.tensor([[2.0]]),
        forecast_target=torch.full((1, 1, 4), 999.0),
    )
    actual = crib_input(batch, patch_len=2)
    assert actual.shape == (1, 2, 1, 2)
    torch.testing.assert_close(actual.flatten(), torch.tensor([12.0, 0.0, 8.0, 0.0]))


def test_crib_official_forward_has_p4_forecast_shape():
    root = Path('artifacts/upstream/CRIB')
    model = load_official_crib(root, device=torch.device('cpu'), batch_size=2)
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(2, 12, 7, 8), test_flag=True)
    assert output[4].shape == (2, 96, 7)
    assert torch.isfinite(output[4]).all()


def test_crib_training_loss_keeps_official_three_terms():
    output = (torch.zeros(2, 3), None, torch.ones(2, 3), None,
              torch.zeros(2, 4, 1), None, torch.tensor(10.0))
    loss = crib_training_loss(output, torch.ones(2, 4, 1))
    torch.testing.assert_close(loss, torch.tensor(2.00001))


def test_crib_selection_rejects_changed_validation_mask():
    selection = {
        'source_csv_sha256': 'source',
        'validation_mask_sha256': 'old-mask',
        'test_mask_sha256': 'test-mask',
        'epochs_completed': 100,
        'best_epoch': 42,
    }
    try:
        verify_crib_selection(selection, source_sha256='source',
                              validation_mask_sha256='new-mask',
                              test_mask_sha256='test-mask')
    except ValueError as error:
        assert 'validation mask' in str(error)
    else:
        raise AssertionError('changed validation mask was accepted')
