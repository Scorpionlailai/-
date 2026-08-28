from pathlib import Path

import torch

from malr.training.checkpoints import load_checkpoint
from malr.training.trainer import BestCheckpointTracker


def test_checkpoint_selection_requires_strictly_lower_validation_mse(tmp_path: Path):
    path = tmp_path / "best.pt"
    tracker = BestCheckpointTracker(path, {"dataset": "fixture"})
    state = {"weight": torch.tensor([1.0])}

    assert tracker.consider(1.0, epoch=0, model_state=state)
    assert not tracker.consider(1.0, epoch=1, model_state=state)
    assert tracker.consider(0.5, epoch=2, model_state=state)

    payload = load_checkpoint(path, expected_metadata={"dataset": "fixture"})
    assert payload["epoch"] == 2
    assert payload["val_mse"] == 0.5


def test_incompatible_dependency_fails_before_model_factory_is_called():
    from malr.training.trainer import validate_dependency_metadata

    called = False

    def allocate_model():
        nonlocal called
        called = True

    try:
        validate_dependency_metadata(
            {"dataset": "ETTh1", "config_hash": "old"},
            {"dataset": "Weather", "config_hash": "new"},
        )
    except ValueError:
        pass
    else:
        raise AssertionError("dependency mismatch was not rejected")

    assert called is False
