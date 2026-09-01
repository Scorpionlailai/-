from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from malr.training.checkpoints import load_checkpoint
from malr.training.trainer import BestCheckpointTracker, PreparedBatch, Trainer


def test_checkpoint_selection_requires_strictly_lower_validation_mse(tmp_path: Path):
    path = tmp_path / "best.pt"
    tracker = BestCheckpointTracker(path, {"dataset": "fixture"})
    state = {"weight": torch.tensor([1.0])}

    assert tracker.consider(1.0, val_mae=0.8, epoch=0, model_state=state)
    assert not tracker.consider(1.0, val_mae=0.7, epoch=1, model_state=state)
    assert tracker.consider(0.5, val_mae=0.6, epoch=2, model_state=state)

    payload = load_checkpoint(path, expected_metadata={"dataset": "fixture"})
    assert payload["epoch"] == 2
    assert payload["val_mse"] == 0.5
    assert payload["val_mae"] == 0.6
    assert tracker.best_mae == pytest.approx(0.6)


class _FixedForecastModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, *args: object, **kwargs: object) -> SimpleNamespace:
        forecast = torch.tensor([[[0.0, 4.0]]], device=self.anchor.device)
        return SimpleNamespace(forecast=forecast + self.anchor * 0.0)


def test_validation_metrics_compute_mse_and_mae_in_one_pass(tmp_path: Path) -> None:
    model = _FixedForecastModel()
    trainer = Trainer(
        model,  # type: ignore[arg-type]
        checkpoint_path=tmp_path / "fixed.pt",
        checkpoint_metadata={"dataset": "fixture"},
    )
    batch = PreparedBatch(
        patch_values=torch.zeros(1, 1, 1, 1),
        patch_mask=torch.ones(1, 1, 1, 1, dtype=torch.bool),
        rho=torch.ones(1, 1, 1),
        state=torch.zeros(1, 1, 1, dtype=torch.long),
        query_mean=torch.zeros(1, 1),
        query_std=torch.ones(1, 1),
        forecast_target=torch.tensor([[[0.0, 2.0]]]),
        clean_raw_patches=torch.zeros(1, 1, 1, 1),
        patch_valid=torch.ones(1, 1, 1, dtype=torch.bool),
        raw_context=torch.zeros(1, 1, 1),
        context_mask=torch.ones(1, 1, 1, dtype=torch.bool),
        train_std=torch.ones(1),
    )

    metrics = trainer.validation_metrics([batch])

    assert metrics.mse == pytest.approx(2.0)
    assert metrics.mae == pytest.approx(1.0)


def test_incompatible_dependency_fails_before_model_factory_is_called():
    from malr.training.trainer import validate_dependency_metadata

    called = False

    def allocate_model():
        nonlocal called
        called = True

    with pytest.raises(ValueError, match="dependency metadata mismatch"):
        validate_dependency_metadata(
            {"dataset": "ETTh1", "config_hash": "old"},
            {"dataset": "Weather", "config_hash": "new"},
        )

    assert called is False
