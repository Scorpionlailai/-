import json
from hashlib import sha256
from pathlib import Path

import pytest
import torch
from torch import nn

from integrations.forecast_compare.protocol import P4Batch
from integrations.forecast_compare.protocol import load_p4_protocol
from integrations.forecast_compare.saits_model import SaitsForecastModel
from integrations.forecast_compare.saits_run import evaluate_selected


class TinyImputer(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(0.5))

    def impute(self, inputs):
        return inputs["X"] + self.value, []


class TinyForecast(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.seen = None

    def forward(self, values, mask, mean, std, *, patch_validity):
        self.seen = (values.detach(), mask.detach(), patch_validity.detach())
        return (values.mean(dim=(2, 3)) * self.weight + mean).unsqueeze(-1).expand(-1, -1, 96)


def example_batch():
    observed = torch.ones(2, 96, 7, dtype=torch.bool)
    observed[:, 24:48, :] = False
    return P4Batch(torch.randn(2, 96, 7).masked_fill(~observed, 0),
                   torch.randn(2, 96, 7), observed, torch.zeros(2, 7),
                   torch.ones(2, 7), torch.randn(2, 7, 96))


def test_forecast_mask_and_shape():
    predictor = TinyForecast()
    model = SaitsForecastModel(TinyImputer(), predictor)
    item = example_batch()
    assert model(item).shape == (2, 7, 96)
    values, mask, validity = predictor.seen
    assert values.shape == mask.shape == validity.shape == (2, 7, 12, 16)
    assert not mask[:, :, 3, :].all()
    assert not validity[:, :, -1, -8:].any()


def test_forecast_loss_updates_saits_and_predictor():
    imputer, predictor = TinyImputer(), TinyForecast()
    model = SaitsForecastModel(imputer, predictor)
    item = example_batch()
    loss = (model(item) - item.forecast_target).square().mean()
    loss.backward()
    assert imputer.value.grad is not None and imputer.value.grad.abs() > 0
    assert predictor.weight.grad is not None and predictor.weight.grad.abs() > 0


def test_future_target_not_used_at_inference():
    model = SaitsForecastModel(TinyImputer(), TinyForecast())
    item = example_batch()
    changed = P4Batch(item.normalized_observed, item.normalized_target,
                      item.observed_mask, item.query_mean, item.query_std,
                      item.forecast_target + 1000)
    torch.testing.assert_close(model(item), model(changed))


def test_forecast_selection_rejects_wrong_checkpoint(tmp_path):
    checkpoint = tmp_path / "selected.pt"
    checkpoint.write_bytes(b"not actual weights")
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"stage": "forecast", "checkpoint": str(checkpoint),
                                     "checkpoint_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="SHA-256"):
        evaluate_selected(object(), selection=selection, output=tmp_path / "evaluation.json",
                          device=torch.device("cpu"))


def test_evaluate_rejects_changed_architecture_before_model_load(tmp_path):
    protocol = load_p4_protocol(
        Path("configs/stage1_seq96_dynamic_temporal_repair.yaml"),
        Path("../E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context"),
    )
    checkpoint = tmp_path / "selected.pt"
    checkpoint.write_bytes(b"placeholder")
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({
        "stage": "forecast", "best_epoch": 1,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint.read_bytes()).hexdigest(),
        "source_csv_sha256": protocol.source_sha256,
        "validation_mask_sha256": protocol.mask_sha256["validation"],
        "test_mask_sha256": protocol.mask_sha256["test"],
        "model_config_sha256": "0" * 64,
    }))
    with pytest.raises(ValueError, match="config"):
        evaluate_selected(protocol, selection=selection,
                          output=tmp_path / "evaluation.json", device=torch.device("cpu"))
