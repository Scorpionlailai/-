from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch import Tensor

from malr.data.patching import PATCH_MISSING, PATCH_OBSERVED, PATCH_PARTIAL
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
    imputation_target: Tensor | None = None
    patch_starts: Tensor | None = None
    patch_validity: Tensor | None = None

    def to(self, device: torch.device | str) -> "PreparedBatch":
        return PreparedBatch(
            **{
                field.name: (
                    None
                    if getattr(self, field.name) is None
                    else getattr(self, field.name).to(device)
                )
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
        filter_learning_rate: float | None = None,
        weight_decay: float = 0.0,
        lr_schedule: str = "constant",
        lambda_rec: float = 1.0,
        lambda_keep: float = 0.0,
        use_recovery_loss: bool = True,
        lambda_imputation_mse: float = 0.0,
        lambda_imputation_mae: float = 0.0,
        lambda_imputation_huber: float = 0.0,
        imputation_huber_delta: float = 1.0,
        lambda_edge_utility: float = 0.0,
        device: str | torch.device = "cpu",
    ) -> None:
        if max_epochs <= 0 or patience <= 0:
            raise ValueError("max_epochs and patience must be positive")
        if lr_schedule not in {"constant", "type3"}:
            raise ValueError("lr_schedule must be 'constant' or 'type3'")
        if filter_learning_rate is not None and filter_learning_rate <= 0.0:
            raise ValueError("filter_learning_rate must be positive")
        if (
            lambda_imputation_mse < 0.0
            or lambda_imputation_mae < 0.0
            or lambda_imputation_huber < 0.0
            or lambda_edge_utility < 0.0
        ):
            raise ValueError("imputation loss weights cannot be negative")
        if imputation_huber_delta <= 0.0:
            raise ValueError("imputation Huber delta must be positive")
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
        self.lambda_imputation_mse = float(lambda_imputation_mse)
        self.lambda_imputation_mae = float(lambda_imputation_mae)
        self.lambda_imputation_huber = float(lambda_imputation_huber)
        self.imputation_huber_delta = float(imputation_huber_delta)
        self.lambda_edge_utility = float(lambda_edge_utility)
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not trainable:
            raise ValueError("model has no trainable parameters")
        parameter_groups: list[dict[str, Any]] = [
            {"params": trainable, "lr": learning_rate, "name": "base"}
        ]
        if filter_learning_rate is not None:
            recovery = model.recovery
            edge_filter = None if recovery is None else recovery.edge_filter
            if edge_filter is None:
                raise ValueError(
                    "filter_learning_rate requires a missing-aware edge filter"
                )
            filter_parameters = [
                parameter
                for parameter in edge_filter.parameters()
                if parameter.requires_grad
            ]
            node_gate_parameters = (
                []
                if recovery is None or recovery.node_gate is None
                else [
                    parameter
                    for parameter in recovery.node_gate.parameters()
                    if parameter.requires_grad
                ]
            )
            auxiliary_parameter_ids = {
                id(parameter)
                for parameter in (*filter_parameters, *node_gate_parameters)
            }
            base_parameters = [
                parameter
                for parameter in trainable
                if id(parameter) not in auxiliary_parameter_ids
            ]
            parameter_groups = [
                {"params": base_parameters, "lr": learning_rate, "name": "base"},
                {
                    "params": filter_parameters,
                    "lr": float(filter_learning_rate),
                    "name": "filter",
                },
            ]
            if node_gate_parameters:
                parameter_groups.append(
                    {
                        "params": node_gate_parameters,
                        "lr": float(filter_learning_rate),
                        "name": "node_gate",
                    }
                )
        self.optimizer = torch.optim.Adam(
            parameter_groups, lr=learning_rate, weight_decay=weight_decay
        )
        self.tracker = BestCheckpointTracker(checkpoint_path, checkpoint_metadata)

    def _train_epoch(self, batches: Iterable[PreparedBatch]) -> dict[str, float]:
        self.model.train()
        saw_batch = False
        batch_count = 0
        loss_totals: dict[str, Tensor] = {}
        for batch in batches:
            saw_batch = True
            batch_count += 1
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
                imputation_target=current.imputation_target,
                lambda_imputation_mse=self.lambda_imputation_mse,
                lambda_imputation_mae=self.lambda_imputation_mae,
                lambda_imputation_huber=self.lambda_imputation_huber,
                imputation_huber_delta=self.imputation_huber_delta,
                lambda_edge_utility=self.lambda_edge_utility,
                patch_starts=current.patch_starts,
            )
            result.loss.total.backward()
            self.optimizer.step()
            current_losses = {
                "total": result.loss.total,
                "forecast_mse": result.loss.forecast_mse,
                "recovery_mse": result.loss.recovery_mse,
                "filter_keep": result.loss.filter_keep,
                "weighted_filter_keep": self.lambda_keep
                * result.loss.filter_keep,
                "imputation_mse": result.loss.imputation_mse,
                "imputation_mae": result.loss.imputation_mae,
                "weighted_imputation_mse": result.loss.weighted_imputation_mse,
                "weighted_imputation_mae": result.loss.weighted_imputation_mae,
                "imputation_huber": result.loss.imputation_huber,
                "weighted_imputation_huber": result.loss.weighted_imputation_huber,
                "edge_utility": result.loss.edge_utility,
                "weighted_edge_utility": result.loss.weighted_edge_utility,
            }
            for name, value in current_losses.items():
                detached = value.detach()
                loss_totals[name] = loss_totals.get(name, torch.zeros_like(detached)) + detached
        if not saw_batch:
            raise ValueError("training batch factory produced no batches")
        return {
            name: float((total / batch_count).cpu())
            for name, total in loss_totals.items()
        }

    @torch.no_grad()
    def validation_metrics(self, batches: Iterable[PreparedBatch]) -> ValidationMetrics:
        self.model.eval()
        squared_error = 0.0
        absolute_error = 0.0
        element_count = 0
        filter_totals: dict[str, dict[str, float]] = {}
        historical_totals = {
            "group_similarity": 0.0,
            "group_count": 0.0,
            "visible_skip": 0.0,
            "damaged": 0.0,
            "selected": 0.0,
        }
        node_gate_totals = {
            "sum": 0.0,
            "active": 0.0,
            "damaged": 0.0,
            "partial_sum": 0.0,
            "partial_count": 0.0,
            "missing_sum": 0.0,
            "missing_count": 0.0,
            "applied_norm": 0.0,
        }
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
                patch_starts=current.patch_starts,
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
            if "node_gate" in raw_diagnostics:
                gate = raw_diagnostics["node_gate"].to(dtype=torch.float64)
                partial = damaged & (current.state == PATCH_PARTIAL)
                missing = damaged & (current.state == PATCH_MISSING)
                node_gate_totals["sum"] += float(gate[damaged].sum().cpu())
                node_gate_totals["active"] += float(
                    (gate[damaged] > 0.0).sum().cpu()
                )
                node_gate_totals["damaged"] += float(damaged.sum().cpu())
                node_gate_totals["partial_sum"] += float(
                    gate[partial].sum().cpu()
                )
                node_gate_totals["partial_count"] += float(partial.sum().cpu())
                node_gate_totals["missing_sum"] += float(
                    gate[missing].sum().cpu()
                )
                node_gate_totals["missing_count"] += float(missing.sum().cpu())
                node_gate_totals["applied_norm"] += float(
                    raw_diagnostics["node_gate_applied_norm"][damaged].sum().cpu()
                )
            if (
                "historical_group_similarity" in raw_diagnostics
                and "historical_visible_count" in raw_diagnostics
            ):
                historical_similarity = raw_diagnostics[
                    "historical_group_similarity"
                ]
                historical_visible = raw_diagnostics["historical_visible_count"]
                historical_selected = raw_diagnostics["historical_selected"]
                historical_targets = damaged
                historical_totals["group_similarity"] += float(
                    historical_similarity[historical_targets].sum().cpu()
                )
                historical_totals["group_count"] += float(
                    historical_targets.sum().cpu()
                )
                historical_totals["visible_skip"] += float(
                    (historical_targets & (historical_visible < 4)).sum().cpu()
                )
                historical_totals["damaged"] += float(historical_targets.sum().cpu())
                historical_totals["selected"] += float(
                    ((historical_selected >= 0).sum(dim=-1) > 0)[historical_targets]
                    .sum()
                    .cpu()
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
        if historical_totals["group_count"] > 0.0:
            diagnostics.update(
                {
                    "historical_group_mean_similarity": historical_totals[
                        "group_similarity"
                    ]
                    / historical_totals["group_count"],
                    "historical_visible_skip_rate": historical_totals[
                        "visible_skip"
                    ]
                    / max(historical_totals["damaged"], 1.0),
                    "historical_selected_rate": historical_totals["selected"]
                    / max(historical_totals["damaged"], 1.0),
                }
            )
        if node_gate_totals["damaged"] > 0.0:
            diagnostics.update(
                {
                    "node_gate_mean": node_gate_totals["sum"]
                    / node_gate_totals["damaged"],
                    "node_gate_active_rate": node_gate_totals["active"]
                    / node_gate_totals["damaged"],
                    "node_gate_partial_mean": node_gate_totals["partial_sum"]
                    / max(node_gate_totals["partial_count"], 1.0),
                    "node_gate_missing_mean": node_gate_totals["missing_sum"]
                    / max(node_gate_totals["missing_count"], 1.0),
                    "node_gate_mean_applied_norm": node_gate_totals["applied_norm"]
                    / node_gate_totals["damaged"],
                }
            )
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
            train_metrics = self._train_epoch(epoch_batches)
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
            learning_rates = {
                str(group["name"]): float(group["lr"])
                for group in self.optimizer.param_groups
            }
            base_lr = learning_rates["base"]
            filter_lr = learning_rates.get("filter")
            node_gate_lr = learning_rates.get("node_gate")
            filter_summary = ""
            if validation.diagnostics:
                filter_summary = " " + " ".join(
                    f"{relation}_keep={validation.diagnostics[f'{relation}_keep_rate']:.3f} "
                    f"{relation}_zero={validation.diagnostics[f'{relation}_zero_edge_rate']:.3f}"
                    for relation in ("temporal", "variable", "historical")
                    if f"{relation}_keep_rate" in validation.diagnostics
                )
                if "node_gate_mean" in validation.diagnostics:
                    filter_summary += (
                        f" node_gate_mean={validation.diagnostics['node_gate_mean']:.3f}"
                        f" node_gate_active={validation.diagnostics['node_gate_active_rate']:.3f}"
                    )
            print(
                f"epoch={epoch + 1}/{self.max_epochs} "
                f"val_mse={validation.mse:.7f} val_mae={validation.mae:.7f} "
                f"best_val_mse={self.tracker.best_metric:.7f} "
                f"best_val_mae={self.tracker.best_mae:.7f} "
                f"train_forecast_mse={train_metrics['forecast_mse']:.7f} "
                f"train_recovery_mse={train_metrics['recovery_mse']:.7f} "
                f"train_filter_keep={train_metrics['filter_keep']:.7f} "
                f"train_weighted_filter_keep={train_metrics['weighted_filter_keep']:.7g} "
                f"train_imputation_mse={train_metrics['imputation_mse']:.7f} "
                f"train_imputation_mae={train_metrics['imputation_mae']:.7f} "
                f"train_weighted_imputation_mse={train_metrics['weighted_imputation_mse']:.7g} "
                f"train_weighted_imputation_mae={train_metrics['weighted_imputation_mae']:.7g} "
                f"train_imputation_huber={train_metrics['imputation_huber']:.7f} "
                f"train_weighted_imputation_huber={train_metrics['weighted_imputation_huber']:.7g} "
                f"train_edge_utility={train_metrics['edge_utility']:.7f} "
                f"train_weighted_edge_utility={train_metrics['weighted_edge_utility']:.7g} "
                f"base_lr={base_lr:.8g} "
                f"filter_lr={('none' if filter_lr is None else format(filter_lr, '.8g'))}"
                f" node_gate_lr={('none' if node_gate_lr is None else format(node_gate_lr, '.8g'))}"
                f"{filter_summary}"
            )
            if epochs_without_improvement >= self.patience:
                break
            if self.lr_schedule == "type3":
                next_lr = patchtst_type3_learning_rate(
                    self.base_learning_rate, epoch + 1
                )
                for parameter_group in self.optimizer.param_groups:
                    if parameter_group["name"] == "base":
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
