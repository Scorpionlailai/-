from pathlib import Path

import pytest
import torch

from integrations.forecast_compare.bitgraph_p4 import bitgraph_inputs, load_official_bitgraph
from integrations.forecast_compare.protocol import P4Batch


def test_bitgraph_inputs_zero_missing_standardized_values():
    batch = P4Batch(
        normalized_observed=torch.tensor([[[1.0], [0.0], [-1.0], [0.0]]]),
        normalized_target=torch.tensor([[[1.0], [999.0], [-1.0], [999.0]]]),
        observed_mask=torch.tensor([[[True], [False], [True], [False]]]),
        query_mean=torch.tensor([[10.0]]), query_std=torch.tensor([[2.0]]),
        forecast_target=torch.full((1, 1, 4), 999.0),
    )
    values, mask = bitgraph_inputs(batch)
    assert values.shape == (1, 4, 1, 1)
    torch.testing.assert_close(values.flatten(), torch.tensor([12.0, 0.0, 8.0, 0.0]))
    torch.testing.assert_close(mask.flatten(), torch.tensor([1.0, 0.0, 1.0, 0.0]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='official BiTGraph hardcodes CUDA')
def test_official_bitgraph_forward_has_p4_shape():
    model = load_official_bitgraph(Path('artifacts/upstream/BiTGraph'), device=torch.device('cuda:0'))
    model.eval()
    with torch.no_grad():
        prediction = model(torch.zeros(2, 96, 7, 1, device='cuda:0'),
                           torch.ones(2, 96, 7, 1, device='cuda:0'), 0)
    assert prediction.shape == (2, 96, 7, 1)
    assert torch.isfinite(prediction).all()
