from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch import Tensor

from malr.data.patching import PATCH_OBSERVED
from malr.models.stage1 import Stage1Model

from .checkpoints import load_checkpoint, save_checkpoint


def patchtst_type3_learning_rate(
    base_learning_rate: float, completed_epoch: int
) -> float:
    """Match PatchTST's one-based ``type3`` epoch schedule."""

    if base_learning_rate <= 0:
        raise ValueError("base_learning_rate must be positive")
    if completed_epoch <= 0:
        raise ValueError("completed_epoch must be positive")
    if completed_epoch < 3:
        return float(base_learning_rate)
    return float(base_learning_rate) * 0.9 ** (completed_epoch - 3)


@dataclass(frozen=True)
class PreparedBatch:
    patch_values: Tensor
    patch_mask: Tensor
    rho: Tensor
    state: Tensor
    query_mean: Tensor
    query_std: Tensor
    forecast_target: Tensor
    clean_raw_patches: Tensor
    patch_valid: Tensor
    raw_context: Tensor
    context_mask: Tensor
    train_std: Tensor

    def to(self, device: torch.device | str) -> "PreparedBatch":
        return PreparedBatch(
            **{
                field.name: getattr(self, field.name).to(device)
                for field in fields(self)
            }
        )


@dataclass(frozen=True)
class TrainingResult:
    best_checkpoint: Path
    best_val_mse: float
    best_val_mae: float
    best_val_diagnostics: dict[str, float]
    best_epoch: int
    epochs_completed: int
    training_seconds: float
    trainable_parameters: int
    test_evaluated: bool = False


@dataclass(frozen=True)
class ValidationMetrics:
    mse: float
    mae: float
    diagnostics: dict[str, float]


def validate_dependency_metadata(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    mismatches = {
        key: (actual.get(key), value)
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatches:
        raise ValueError(f"dependency metadata mismatch: {mismatches}")


class BestCheckpointTracker:
    def __init__(self, path: Path, metadata: Mapping[str, Any]) -> None:
        self.path = path
        self.metadata = dict(metadata)
        self.best_metric = float("inf")
        self.best_mae = float("inf")
        self.best_diagnostics: dict[str, float] = {}
        self.best_epoch = -1

    def consider(
        self,
        metric: float,
        *,
        val_mae: float,
        val_diagnostics: Mapping[str, float] | None = None,
        epoch: int,
        model_state: Mapping[str, Any],
    ) -> bool:
        if not torch.isfinite(torch.tensor([metric, val_mae])).all():
            raise ValueError("validation metrics must be finite")
        if metric >= self.best_metric:
            return False
        diagnostics = dict(val_diagnostics or {})
        save_checkpoint(
            self.path,
            model_state,
            self.metadata,
            epoch=epoch,
            val_mse=float(metric),
            val_mae=float(val_mae),
            validation_diagnostics=diagnostics,
        )
        self.best_metric = float(metric)
        self.best_mae = float(val_mae)
        self.best_diagnostics = diagnostics
        self.best_epoch = epoch
        return True


class Trainer:
    def __init__(
        self,
        model: Stage1Model,
        *,
        checkpoint_path: Path,
        checkpoint_metadata: Mapping[str, Any],
        max_epochs: int = 50,
        patience: int = 5,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        lr_schedule: str = "constant",
        lambda_rec: float = 1.0,
        lambda_keep: float = 0.0,
        use_recovery_loss: bool = True,
        device: str | torch.device = "cpu",
    ) -> None:
        if max_epochs <= 0 or patience <= 0:
            raise ValueError("max_epochs and patience must be positive")
        if lr_schedule not in {"constant", "type3"}:
            raise ValueError("lr_schedule must be 'constant' or 'type3'")
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.checkpoint_path = checkpoint_path
        self.max_epochs = max_epochs
        self.patience = patience
        self.base_learning_rate = float(learning_rate)
        self.lr_schedule = lr_schedule
        self.lambda_rec = lambda_rec
        self.lambda_keep = lambda_keep
        self.use_recovery_loss = use_recovery_loss
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not trainable:
            raise ValueError("model has no trainable parameters")
        self.optimizer = torch.optim.Adam(
            trainable, lr=learning_rate, weight_decay=weight_decay
        )
        self.tracker = BestCheckpointTracker(checkpoint_path, checkpoint_metadata)

    def _train_epoch(self, batches: Iterable[PreparedBatch]) -> None:
        self.model.train()
        saw_batch = False
        for batch in batches:
            saw_batch = True
            current = batch.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            result = self.model.training_objective(
                current.patch_values,
                current.patch_mask,
                current.rho,
                current.state,
                current.query_mean,
                current.query_std,
                current.forecast_target,
                current.clean_raw_patches,
                current.patch_valid,
                raw_context=current.raw_context,
                context_mask=current.context_mask,
                train_std=current.train_std,
                lambda_rec=self.lambda_rec,
                lambda_keep=self.lambda_keep,
                use_recovery_loss=self.use_recovery_loss,
            )
            result.loss.total.backward()
            self.optimizer.step()
        if not saw_batch:
            raise ValueError("training batch factory produced no batches")

    @torch.no_grad()
    def validation_metrics(self, batches: Iterable[PreparedBatch]) -> ValidationMetrics:
        self.model.eval()
        squared_error = 0.0
        absolute_error = 0.0
        element_count = 0
        filter_totals: dict[str, dict[str, float]] = {}
        relation_names = ("temporal", "variable", "historical")
        rho_bands = {
            "rho_eq_0": lambda rho: rho == 0,
            "rho_0_025": lambda rho: (rho > 0) & (rho <= 0.25),
            "rho_025_050": lambda rho: (rho > 0.25) & (rho <= 0.50),
            "rho_050_100": lambda rho: (rho > 0.50) & (rho < 1.0),
        }
        for batch in batches:
            current = batch.to(self.device)
            output = self.model(
                current.patch_values,
                current.patch_mask,
                current.rho,
                current.state,
                current.query_mean,
                current.query_std,
                raw_context=current.raw_context,
                context_mask=current.context_mask,
                train_std=current.train_std,
            )
            error = output.forecast - current.forecast_target
            squared_error += float(error.square().sum().cpu())
            absolute_error += float(error.abs().sum().cpu())
            element_count += current.forecast_target.numel()
            recovery = getattr(output, "recovery", None)
            if recovery is None:
                continue
            raw_diagnostics = recovery.diagnostics
            damaged = (
                (current.state != PATCH_OBSERVED)
                & current.patch_valid.to(dtype=torch.bool)
            )
            for relation in relation_names:
                candidate_key = f"{relation}_filter_candidate_count"
                if candidate_key not in raw_diagnostics:
                    continue
                retained = raw_diagnostics[
                    f"{relation}_filter_retained_count"
                ].to(dtype=torch.float64)
                candidate = raw_diagnostics[candidate_key].to(dtype=torch.float64)
                pool = raw_diagnostics[
                    f"{relation}_filter_pool_count"
                ].to(dtype=torch.float64)
                totals = filter_totals.setdefault(
                    relation,
                    {
                        "pool": 0.0,
                        "candidate": 0.0,
                        "retained": 0.0,
                        "candidate_confidence": 0.0,
                        "retained_confidence": 0.0,
                        "damaged": 0.0,
                        "zero_edge": 0.0,
                        **{
                            f"{band}_retained": 0.0
                            for band in rho_bands
                        },
                        **{f"{band}_count": 0.0 for band in rho_bands},
                        "threshold": float("nan"),
                        "max_edges": float("nan"),
                    },
                )
                totals["pool"] += float(pool.sum().cpu())
                totals["candidate"] += float(candidate.sum().cpu())
                totals["retained"] += float(retained.sum().cpu())
                totals["candidate_confidence"] += float(
                    raw_diagnostics[
                        f"{relation}_filter_candidate_confidence_sum"
                    ].sum().cpu()
                )
                totals["retained_confidence"] += float(
                    raw_diagnostics[
                        f"{relation}_filter_retained_confidence_sum"
                    ].sum().cpu()
                )
                totals["damaged"] += float(damaged.sum().cpu())
                totals["zero_edge"] += float(
                    ((retained == 0) & damaged).sum().cpu()
                )
                for band, selector in rho_bands.items():
                    selected_targets = damaged & selector(current.rho)
                    totals[f"{band}_retained"] += float(
                        retained[selected_targets].sum().cpu()
                    )
                    totals[f"{band}_count"] += float(
                        selected_targets.sum().cpu()
                    )
                for scalar_name in ("threshold", "max_edges"):
                    value = float(
                        raw_diagnostics[
                            f"{relation}_filter_{scalar_name}"
                        ].cpu()
                    )
                    previous = totals[scalar_name]
                    if torch.isfinite(torch.tensor(previous)) and previous != value:
                        raise ValueError(
                            f"{relation} filter {scalar_name} changed during validation"
                        )
                    totals[scalar_name] = value
        if element_count == 0:
            raise ValueError("validation batch factory produced no batches")
        diagnostics: dict[str, float] = {}
        for relation, totals in filter_totals.items():
            candidate = totals["candidate"]
            retained = totals["retained"]
            diagnostics.update(
                {
                    f"{relation}_candidate_availability": candidate
                    / max(totals["pool"], 1.0),
                    f"{relation}_keep_rate": retained / max(candidate, 1.0),
                    f"{relation}_zero_edge_rate": totals["zero_edge"]
                    / max(totals["damaged"], 1.0),
                    f"{relation}_mean_confidence": totals[
                        "candidate_confidence"
                    ]
                    / max(candidate, 1.0),
                    f"{relation}_mean_retained_confidence": totals[
                        "retained_confidence"
                    ]
                    / max(retained, 1.0),
                    f"{relation}_threshold": totals["threshold"],
                    f"{relation}_max_edges": totals["max_edges"],
                }
            )
            for band in rho_bands:
                diagnostics[f"{relation}_mean_retained_{band}"] = totals[
                    f"{band}_retained"
                ] / max(totals[f"{band}_count"], 1.0)
        return ValidationMetrics(
            mse=squared_error / element_count,
            mae=absolute_error / element_count,
            diagnostics=diagnostics,
        )

    @torch.no_grad()
    def validation_mse(self, batches: Iterable[PreparedBatch]) -> float:
        return self.validation_metrics(batches).mse

    def fit(
        self,
        train_batches_for_epoch: Callable[[int], Iterable[PreparedBatch]],
        validation_batches: Callable[[], Iterable[PreparedBatch]],
    ) -> TrainingResult:
        start_time = perf_counter()
        epochs_without_improvement = 0
        epochs_completed = 0
        for epoch in range(self.max_epochs):
            # This is the only call per epoch, so a timeline mask manager is also
            # invoked exactly once when owned by this factory.
            epoch_batches = train_batches_for_epoch(epoch)
            self._train_epoch(epoch_batches)
            validation = self.validation_metrics(validation_batches())
            improved = self.tracker.consider(
                validation.mse,
                val_mae=validation.mae,
                val_diagnostics=validation.diagnostics,
                epoch=epoch,
                model_state=self.model.state_dict(),
            )
            epochs_completed = epoch + 1
            epochs_without_improvement = 0 if improved else epochs_without_improvement + 1
            current_lr = float(self.optimizer.param_groups[0]["lr"])
            filter_summary = ""
            if validation.diagnostics:
                filter_summary = " " + " ".join(
                    f"{relation}_keep={validation.diagnostics[f'{relation}_keep_rate']:.3f} "
                    f"{relation}_zero={validation.diagnostics[f'{relation}_zero_edge_rate']:.3f}"
                    for relation in ("temporal", "variable", "historical")
                    if f"{relation}_keep_rate" in validation.diagnostics
                )
            print(
                f"epoch={epoch + 1}/{self.max_epochs} "
                f"val_mse={validation.mse:.7f} val_mae={validation.mae:.7f} "
                f"best_val_mse={self.tracker.best_metric:.7f} "
                f"best_val_mae={self.tracker.best_mae:.7f} "
                f"lr={current_lr:.8g}{filter_summary}"
            )
            if epochs_without_improvement >= self.patience:
                break
            if self.lr_schedule == "type3":
                next_lr = patchtst_type3_learning_rate(
                    self.base_learning_rate, epoch + 1
                )
                for parameter_group in self.optimizer.param_groups:
                    parameter_group["lr"] = next_lr

        elapsed = perf_counter() - start_time
        best = load_checkpoint(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(best["model_state"], strict=True)
        trainable_parameters = sum(
            parameter.numel()
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )
        return TrainingResult(
            best_checkpoint=self.checkpoint_path,
            best_val_mse=self.tracker.best_metric,
            best_val_mae=self.tracker.best_mae,
            best_val_diagnostics=dict(self.tracker.best_diagnostics),
            best_epoch=self.tracker.best_epoch,
            epochs_completed=epochs_completed,
            training_seconds=elapsed,
            trainable_parameters=trainable_parameters,
        )
