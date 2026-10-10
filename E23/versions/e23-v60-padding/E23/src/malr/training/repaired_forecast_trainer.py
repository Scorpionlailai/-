from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F

from malr.data.repaired import RepairedForecastBatch
from malr.evaluation.metrics import ForecastMetrics
from malr.models.repaired_forecast import MaskAwareForecastModel
from malr.training.checkpoints import load_checkpoint
from malr.training.trainer import BestCheckpointTracker, TrainingResult


@torch.no_grad()
def evaluate_repaired_forecast(
    model: MaskAwareForecastModel,
    batches: Iterable[RepairedForecastBatch],
    *,
    device: str | torch.device = "cpu",
) -> ForecastMetrics:
    evaluation_device = torch.device(device)
    model.to(evaluation_device).eval()
    squared = absolute = 0.0
    elements = 0
    for batch in batches:
        current = batch.to(evaluation_device)
        forecast = model(
            current.patch_values,
            current.patch_mask,
            current.query_mean,
            current.query_std,
            patch_validity=current.patch_validity,
        )
        error = forecast - current.forecast_target
        squared += float(error.square().sum().cpu())
        absolute += float(error.abs().sum().cpu())
        elements += error.numel()
    if elements == 0:
        raise ValueError("repaired forecast iterator produced no batches")
    return ForecastMetrics(mse=squared / elements, mae=absolute / elements)


class RepairedForecastTrainer:
    """Forecast-only Stage-2 training; no Stage-1 module is owned or updated."""

    def __init__(
        self,
        model: MaskAwareForecastModel,
        *,
        checkpoint_path: Path,
        checkpoint_metadata: Mapping[str, object],
        max_epochs: int,
        patience: int,
        learning_rate: float,
        weight_decay: float = 0.0,
        device: str | torch.device = "cpu",
        progress_every: int = 10,
    ) -> None:
        if max_epochs <= 0 or patience <= 0 or learning_rate <= 0.0:
            raise ValueError("forecast trainer settings are invalid")
        if weight_decay < 0.0 or progress_every <= 0:
            raise ValueError("forecast trainer regularization settings are invalid")
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.progress_every = int(progress_every)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.tracker = BestCheckpointTracker(checkpoint_path, checkpoint_metadata)

    def _train_epoch(self, batches: Iterable[RepairedForecastBatch], epoch: int) -> float:
        self.model.train()
        total = 0.0
        count = 0
        started = perf_counter()
        print(f"epoch={epoch}/{self.max_epochs} phase=forecast start", flush=True)
        for batch_index, batch in enumerate(batches, start=1):
            current = batch.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            forecast = self.model(
                current.patch_values,
                current.patch_mask,
                current.query_mean,
                current.query_std,
                patch_validity=current.patch_validity,
            )
            loss = F.mse_loss(forecast, current.forecast_target)
            loss.backward()
            self.optimizer.step()
            value = float(loss.detach().cpu())
            total += value
            count += 1
            if batch_index == 1 or batch_index % self.progress_every == 0:
                print(
                    f"epoch={epoch}/{self.max_epochs} phase=forecast "
                    f"batch={batch_index} loss={value:.6f} "
                    f"elapsed={perf_counter() - started:.1f}s",
                    flush=True,
                )
        if count == 0:
            raise ValueError("repaired training iterator produced no batches")
        return total / count

    def fit(
        self,
        train_batches_for_epoch: Callable[[int], Iterable[RepairedForecastBatch]],
        validation_batches: Callable[[], Iterable[RepairedForecastBatch]],
    ) -> TrainingResult:
        started = perf_counter()
        stale = 0
        completed = 0
        for epoch in range(1, self.max_epochs + 1):
            epoch_started = perf_counter()
            train_loss = self._train_epoch(train_batches_for_epoch(epoch - 1), epoch)
            validation = evaluate_repaired_forecast(
                self.model, validation_batches(), device=self.device
            )
            improved = self.tracker.consider(
                validation.mse,
                val_mae=validation.mae,
                val_diagnostics={},
                epoch=epoch,
                model_state=self.model.state_dict(),
            )
            stale = 0 if improved else stale + 1
            completed = epoch
            print(
                f"epoch={epoch}/{self.max_epochs} phase=forecast-validation "
                f"train_loss={train_loss:.6f} val_mse={validation.mse:.7f} "
                f"val_mae={validation.mae:.7f} best_epoch={self.tracker.best_epoch} "
                f"stale={stale} epoch_seconds={perf_counter() - epoch_started:.1f}",
                flush=True,
            )
            if stale >= self.patience:
                break
        payload = load_checkpoint(self.tracker.path)
        self.model.load_state_dict(payload["model_state"], strict=True)
        return TrainingResult(
            best_checkpoint=self.tracker.path,
            best_val_mse=self.tracker.best_metric,
            best_val_mae=self.tracker.best_mae,
            best_val_diagnostics={},
            best_epoch=self.tracker.best_epoch,
            epochs_completed=completed,
            training_seconds=perf_counter() - started,
            trainable_parameters=sum(
                parameter.numel()
                for parameter in self.model.parameters()
                if parameter.requires_grad
            ),
        )


__all__ = ["RepairedForecastTrainer", "evaluate_repaired_forecast"]
