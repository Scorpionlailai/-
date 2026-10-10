import torch
from torch import nn
import pytest

from integrations.forecast_compare.protocol import P4Batch
from integrations.forecast_compare.saits_model import (
    build_official_saits, impute_with_observed_clamp, saits_repair_loss,
)


class FakeOfficial(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_inputs = None
        self.last_stage = None

    def forward(self, inputs, stage):
        self.last_inputs, self.last_stage = inputs, stage
        return {
            "reconstruction_loss": torch.tensor(2.0),
            "imputation_loss": torch.tensor(3.0),
            "imputed_data": torch.full_like(inputs["X"], 100),
        }


def batch():
    observed = torch.tensor([[[True, False], [False, True]]])
    return P4Batch(
        torch.tensor([[[1., 0.], [0., 4.]]]),
        torch.tensor([[[1., 2.], [3., 4.]]]),
        observed,
        torch.zeros(1, 2), torch.ones(1, 2), torch.zeros(1, 2, 2),
    )


def test_saits_mask_mapping():
    model = FakeOfficial()
    item = batch()
    saits_repair_loss(model, item)
    inputs = model.last_inputs
    assert torch.equal(inputs["X"], item.normalized_observed)
    assert torch.equal(inputs["missing_mask"], item.observed_mask.float())
    assert torch.equal(inputs["X_holdout"], item.normalized_target)
    assert torch.equal(inputs["indicating_mask"], (~item.observed_mask).float())
    assert (inputs["missing_mask"] * inputs["indicating_mask"]).sum() == 0


def test_loss_is_ort_plus_mit():
    assert saits_repair_loss(FakeOfficial(), batch()).item() == 5.0


def test_imputer_preserves_observed_values():
    item = batch()
    result = impute_with_observed_clamp(FakeOfficial(), item)
    assert torch.equal(result[item.observed_mask], item.normalized_observed[item.observed_mask])
    assert torch.all(result[~item.observed_mask] == 100)


def test_no_extra_random_holdout():
    model = FakeOfficial()
    item = batch()
    saits_repair_loss(model, item)
    assert model.last_stage == "train"
    assert torch.equal(model.last_inputs["indicating_mask"].bool(), ~item.observed_mask)


def test_rejects_unpinned_source_before_import(tmp_path):
    source = tmp_path / "modeling"
    source.mkdir()
    (source / "saits.py").write_text("# changed upstream source\n")
    with pytest.raises(ValueError, match="source SHA-256"):
        build_official_saits(tmp_path, device=torch.device("cpu"))
