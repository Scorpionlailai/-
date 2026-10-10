from pathlib import Path

import torch

from integrations.forecast_compare.coifnet_p4 import (
    coifnet_inputs, coifnet_training_loss, load_official_coifnet,
    load_timestamp_features,
)
from integrations.forecast_compare.protocol import P4Batch
from malr.data.windows import WindowRef


def test_coifnet_inputs_hide_missing_history_and_align_time_features():
    batch = P4Batch(
        normalized_observed=torch.tensor([[[1.0], [0.0], [-1.0], [0.0]]]),
        normalized_target=torch.tensor([[[1.0], [999.0], [-1.0], [999.0]]]),
        observed_mask=torch.tensor([[[True], [False], [True], [False]]]),
        query_mean=torch.tensor([[10.0]]), query_std=torch.tensor([[2.0]]),
        forecast_target=torch.full((1, 1, 4), 999.0),
        refs=(WindowRef(1, 5, 5, 9, 'train'),),
    )
    stamps = torch.arange(20).reshape(10, 2)
    values, mask, features = coifnet_inputs(batch, stamps)
    torch.testing.assert_close(values.flatten(), torch.tensor([12.0, 0.0, 8.0, 0.0]))
    torch.testing.assert_close(mask.flatten(), torch.tensor([1.0, 0.0, 1.0, 0.0]))
    torch.testing.assert_close(features, stamps[1:5].unsqueeze(0))


def test_official_coifnet_predicts_history_and_future_without_future_input():
    model = load_official_coifnet(Path('artifacts/upstream/CoIFNet'), device=torch.device('cpu'))
    model.eval()
    with torch.no_grad():
        prediction = model.impute(torch.zeros(2, 96, 7), torch.ones(2, 96, 7),
                                  feat=torch.zeros(2, 96, 2))
    assert prediction.shape == (2, 192, 7)
    assert torch.isfinite(prediction).all()


def test_coifnet_loss_uses_official_forecast_reconstruction_weights():
    prediction = torch.zeros(1, 4, 1)
    observed_values = torch.tensor([[[2.0], [0.0]]])
    observed_mask = torch.tensor([[[1.0], [0.0]]])
    forecast = torch.ones(1, 2, 1)
    loss = coifnet_training_loss(prediction, observed_values, observed_mask, forecast)
    torch.testing.assert_close(loss, torch.tensor(1.9))


def test_coifnet_timestamp_features_follow_official_simple_encoding(tmp_path):
    path = tmp_path / 'timestamps.csv'
    path.write_text('date,x\n2016-07-01 00:00:00,1\n2016-07-02 03:00:00,2\n')
    actual = load_timestamp_features(path)
    torch.testing.assert_close(actual, torch.tensor([[1, 0], [2, 3]]))
