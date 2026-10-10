"""Small, explicit training loop for the online Stage-1 composition root."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch
from torch import Tensor

from malr.models.online_stage1 import OnlineStage1Model
from malr.models.online_stage1_v2 import (
    RelationV2OnlineStage1Model,
    RelationV2TrainingResult,
)
from malr.models.dynamic_context_model import DynamicContextOnlineModel
from malr.models.reconstruction import aggregate_patch_predictions

from .online_checkpoint import load_online_checkpoint, save_online_checkpoint


@dataclass(frozen=True)
class OnlineTrainingResult:
    best_checkpoint: Path
    best_val_mse: float
    best_val_mae: float
    best_epoch: int
    epochs_completed: int
    training_seconds: float
    trainable_parameters: int


OnlineModel = OnlineStage1Model | RelationV2OnlineStage1Model


class OnlineTrainer:
    """Train one online model while keeping memory and model snapshots paired."""

    def __init__(
        self,
        model: OnlineModel,
        *,
        checkpoint_path: Path,
        checkpoint_metadata: Mapping[str, object],
        max_epochs: int = 100,
        patience: int = 10,
        learning_rate: float = 1e-4,
        calibration_relation_lr_scale: float = 0.1,
        weight_decay: float = 0.0,
        device: str | torch.device = "cpu",
        progress_every: int = 10,
    ) -> None:
        if (
            max_epochs <= 0
            or patience <= 0
            or learning_rate <= 0.0
            or not 0.0 < calibration_relation_lr_scale <= 1.0
            or weight_decay < 0.0
            or progress_every <= 0
        ):
            raise ValueError("online trainer optimizer and epoch settings are invalid")
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.checkpoint_path = checkpoint_path
        self.metadata = dict(checkpoint_metadata)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.progress_every = int(progress_every)
        self.learning_rate = float(learning_rate)
        self.calibration_relation_lr_scale = float(calibration_relation_lr_scale)
        self.weight_decay = float(weight_decay)
        if isinstance(self.model, DynamicContextOnlineModel) and self.model._training_stage == "forecast":
            assert self.model.forecast_core is not None
            assert self.model.forecast_mask_projection is not None
            forecast_ids = {
                id(parameter)
                for module in (self.model.forecast_core, self.model.forecast_mask_projection)
                for parameter in module.parameters()
            }
            active = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
            repair = [parameter for parameter in active if id(parameter) not in forecast_ids]
            forecast = [parameter for parameter in active if id(parameter) in forecast_ids]
            groups = [
                {
                    "name": "repair",
                    "params": repair,
                    "lr": self.learning_rate * self.model.stage2_repair_lr_scale,
                },
                {"name": "forecast", "params": forecast, "lr": self.learning_rate},
            ]
        elif isinstance(self.model, RelationV2OnlineStage1Model):
            groups = [
                {"name": "backbone", "params": list(self.model.backbone_parameters()), "lr": self.learning_rate},
                {"name": "relations", "params": list(self.model.relation_parameters()), "lr": self.learning_rate},
            ]
            parameter_ids = [id(parameter) for group in groups for parameter in group["params"]]
            expected_ids = {
                id(parameter) for parameter in self.model.parameters() if parameter.requires_grad
            }
            if len(parameter_ids) != len(set(parameter_ids)) or set(parameter_ids) != expected_ids:
                raise ValueError("online optimizer parameter groups must be disjoint and complete")
        else:
            groups = [
                {
                    "name": "all",
                    "params": [parameter for parameter in self.model.parameters() if parameter.requires_grad],
                    "lr": self.learning_rate,
                }
            ]
        self.optimizer = torch.optim.Adam(groups, lr=self.learning_rate, weight_decay=self.weight_decay)

    @staticmethod
    def _batch_to_device(batch: object, device: torch.device) -> object:
        to = getattr(batch, "to", None)
        return to(device) if callable(to) else batch

    def _configure_epoch(self, epoch: int, *, objective: str = "joint") -> None:
        """Apply the V2 calibration schedule without replacing optimizer state."""

        for group in self.optimizer.param_groups:
            if isinstance(self.model, DynamicContextOnlineModel) and self.model._training_stage == "forecast":
                group["lr"] = (
                    self.learning_rate
                    if group["name"] == "forecast"
                    else self.learning_rate * self.model.stage2_repair_lr_scale
                )
            else:
                group["lr"] = self.learning_rate
        if (
            objective != "forecast"
            and isinstance(self.model, RelationV2OnlineStage1Model)
            and self.model.training_phase(epoch) == "calibration"
        ):
            for group in self.optimizer.param_groups:
                if group["name"] == "relations":
                    group["lr"] = self.learning_rate * self.calibration_relation_lr_scale

    def _train_epoch(
        self,
        batches: Iterable[object],
        epoch: int,
        *,
        objective: str = "joint",
        phase_label: str = "train",
        display_total: int | None = None,
        model_epoch_index: int | None = None,
    ) -> float:
        self.model.train()
        total = 0.0
        count = 0
        epoch_start = perf_counter()
        print(
            f"epoch={epoch}/{display_total or self.max_epochs} phase={phase_label} start",
            flush=True,
        )
        for batch_index, batch in enumerate(batches, start=1):
            current = self._batch_to_device(batch, self.device)
            self.optimizer.zero_grad(set_to_none=True)
            objective_kwargs = {
                "patch_starts": current.patch_starts,
                "epoch_index": epoch if model_epoch_index is None else model_epoch_index,
                "patch_valid": current.patch_valid,
            }
            if isinstance(self.model, RelationV2OnlineStage1Model):
                objective_kwargs["objective"] = objective
                objective_kwargs["patch_validity"] = current.patch_validity
            elif objective != "joint":
                raise ValueError("repair and forecast objectives require relation-v2")
            result = self.model.training_objective(
                current.patch_values,
                current.patch_mask,
                current.rho,
                current.state,
                current.query_mean,
                current.query_std,
                current.forecast_target,
                current.clean_raw_patches,
                **objective_kwargs,
            )
            result.loss.total.backward()
            self.optimizer.step()
            self.model.update_training_statistics(result)
            if objective != "forecast":
                self.model.stage_observed_batch(
                    current.clean_raw_patches,
                    current.patch_mask,
                    current.rho,
                    current.state,
                    current.patch_starts,
                )
            batch_loss = float(result.loss.total.detach().cpu())
            total += batch_loss
            count += 1
            if batch_index == 1 or batch_index % self.progress_every == 0:
                component_summary = ""
                if isinstance(result, RelationV2TrainingResult):
                    branches = result.branch_repair_losses.detach().cpu().tolist()
                    component_summary = (
                        f" repair={float(result.loss.imputation_huber.detach().cpu()):.6f}"
                        f" gain={float(result.loss.edge_gain.detach().cpu()):.6f}"
                        f" memory={float(result.loss.memory.detach().cpu()):.6f}"
                        f" H={branches[0]:.6f} T={branches[1]:.6f} V={branches[2]:.6f}"
                    )
                print(
                    f"epoch={epoch}/{display_total or self.max_epochs} phase={phase_label} "
                    f"batch={batch_index} loss={batch_loss:.6f} "
                    f"elapsed={perf_counter() - epoch_start:.1f}s"
                    f"{component_summary}",
                    flush=True,
                )
        if count == 0:
            raise ValueError("online training batch factory produced no batches")
        return total / count

    @torch.no_grad()
    def _repair_validation(self, batches: Iterable[object]) -> tuple[float, float]:
        if not isinstance(self.model, RelationV2OnlineStage1Model):
            raise ValueError("repair validation requires relation-v2")
        self.model.eval()
        before = self.model.online_state()
        squared = 0.0
        absolute = 0.0
        elements = 0
        for batch in batches:
            current = self._batch_to_device(batch, self.device)
            forward_kwargs = {
                "patch_starts": current.patch_starts,
                "epoch_index": self.model.memory.snapshot().epoch,
                "patch_valid": current.patch_valid,
            }
            if isinstance(self.model, RelationV2OnlineStage1Model):
                forward_kwargs["patch_validity"] = current.patch_validity
            output = self.model(
                current.patch_values,
                current.patch_mask,
                current.rho,
                current.state,
                current.query_mean,
                current.query_std,
                **forward_kwargs,
            )
            if hasattr(output, "explicit"):
                coordinate_validity = (
                    current.patch_valid.unsqueeze(-1).expand_as(current.patch_mask)
                    if current.patch_validity is None
                    else current.patch_validity.to(
                        device=current.patch_mask.device, dtype=torch.bool
                    )
                )
                clean_sequence, _ = aggregate_patch_predictions(
                    current.clean_raw_patches,
                    window_len=self.model.core.seq_len,  # type: ignore[attr-defined]
                    patch_len=self.model.core.patch_len,  # type: ignore[attr-defined]
                    stride=self.model.core.stride,  # type: ignore[attr-defined]
                    validity=coordinate_validity,
                )
                error = (
                    output.explicit.standardized_sequence[output.explicit.sequence_missing]
                    - clean_sequence[output.explicit.sequence_missing]
                )
                squared += float(error.square().sum().cpu())
                absolute += float(error.abs().sum().cpu())
                elements += error.numel()
                continue
            coordinate_validity = (
                current.patch_valid.unsqueeze(-1).expand_as(current.patch_mask)
                if current.patch_validity is None
                else current.patch_validity.to(
                    device=current.patch_mask.device, dtype=torch.bool
                )
            )
            missing = (~current.patch_mask) & coordinate_validity
            recovered_raw = self.model._decode_raw_patches(
                output.latents, current.query_mean, current.query_std
            )
            observed_raw = (
                current.patch_values
                * current.query_std.unsqueeze(-1).unsqueeze(-1)
                + current.query_mean.unsqueeze(-1).unsqueeze(-1)
            )
            reconstructed = torch.where(
                current.patch_mask, observed_raw, recovered_raw
            )
            error = (reconstructed - current.clean_raw_patches)[missing]
            squared += float(error.square().sum().cpu())
            absolute += float(error.abs().sum().cpu())
            elements += error.numel()
        if elements == 0:
            raise ValueError("online repair validation batch factory produced no batches")
        after = self.model.online_state()
        if (
            before["memory_hash"] != after["memory_hash"]
            or before["ema_hash"] != after["ema_hash"]
            or before["relation_prior_hash"] != after["relation_prior_hash"]
        ):
            raise RuntimeError("online repair validation changed read-only state")
        return squared / elements, absolute / elements

    @torch.no_grad()
    def _validation(self, batches: Iterable[object]) -> tuple[float, float]:
        self.model.eval()
        before = self.model.online_state()
        squared = 0.0
        absolute = 0.0
        elements = 0
        for batch in batches:
            current = self._batch_to_device(batch, self.device)
            forward_kwargs = {
                "patch_starts": current.patch_starts,
                "epoch_index": self.model.memory.snapshot().epoch,
                "patch_valid": current.patch_valid,
            }
            if isinstance(self.model, RelationV2OnlineStage1Model):
                forward_kwargs["patch_validity"] = current.patch_validity
            output = self.model(
                current.patch_values,
                current.patch_mask,
                current.rho,
                current.state,
                current.query_mean,
                current.query_std,
                **forward_kwargs,
            )
            error = output.forecast - current.forecast_target
            squared += float(error.square().sum().cpu())
            absolute += float(error.abs().sum().cpu())
            elements += error.numel()
        if elements == 0:
            raise ValueError("online validation batch factory produced no batches")
        after = self.model.online_state()
        if (
            before["memory_hash"] != after["memory_hash"]
            or before["ema_hash"] != after["ema_hash"]
            or before["relation_prior_hash"] != after["relation_prior_hash"]
        ):
            raise RuntimeError("online validation changed read-only state")
        return squared / elements, absolute / elements

    def _save_best(
        self, *, epoch: int, val_mse: float, val_mae: float,
        hashes: Mapping[str, str], training_epoch: int,
    ) -> None:
        state = self.model.online_state()
        save_online_checkpoint(
            self.checkpoint_path,
            self.model.state_dict(),
            state["memory_state"],  # type: ignore[arg-type]
            state["ema_state"],  # type: ignore[arg-type]
            state["relation_prior_state"],  # type: ignore[arg-type]
            {
                **self.metadata, "epoch": epoch, "training_epoch": training_epoch,
                "val_mse": val_mse, "val_mae": val_mae,
            },
            epoch=epoch,
            config_hash=hashes["config_hash"],
            data_hash=hashes["data_hash"],
            mask_hash=hashes["mask_hash"],
            memory_hash=str(state["memory_hash"]),
        )

    def _restore_best(self, hashes: Mapping[str, str]) -> None:
        payload = load_online_checkpoint(self.checkpoint_path, expected_hashes=hashes)
        self.model.load_state_dict(payload["model_state"], strict=True)
        self.model.load_online_state(payload["online_state"])

    def fit(
        self,
        train_batches_for_epoch: Callable[[int], Iterable[object]],
        validation_batches: Callable[[], Iterable[object]],
        *,
        config_hash: str,
        data_hash: str,
        mask_hash: str,
        objective: str = "joint",
        validation_objective: str = "forecast",
        epoch_offset: int = 0,
        model_epoch_offset: int | None = None,
        stage_label: str | None = None,
        freeze_online_state: bool = False,
        preserve_initial_best: bool = False,
    ) -> OnlineTrainingResult:
        valid_objectives = {"joint", "repair", "forecast"}
        if objective not in valid_objectives or validation_objective not in {
            "repair",
            "forecast",
        }:
            raise ValueError("online trainer objectives are invalid")
        if type(epoch_offset) is not int or epoch_offset < 0:
            raise ValueError("epoch_offset must be a non-negative integer")
        if model_epoch_offset is not None and (
            type(model_epoch_offset) is not int or model_epoch_offset < 0
        ):
            raise ValueError("model_epoch_offset must be a non-negative integer")
        if type(freeze_online_state) is not bool:
            raise TypeError("freeze_online_state must be bool")
        if preserve_initial_best and (
            objective != "forecast" or validation_objective != "forecast"
            or not freeze_online_state or epoch_offset <= 0
        ):
            raise ValueError("preserve_initial_best requires frozen forecast warm-start")
        if objective != "joint" and not isinstance(self.model, RelationV2OnlineStage1Model):
            raise ValueError("repair and forecast objectives require relation-v2")
        hashes = {"config_hash": config_hash, "data_hash": data_hash, "mask_hash": mask_hash}
        if any(not isinstance(value, str) or not value for value in hashes.values()):
            raise ValueError("online trainer hashes must be non-empty strings")
        start_time = perf_counter()
        best_mse = float("inf")
        best_mae = float("inf")
        best_epoch = -1
        stale = 0
        completed = 0
        display_total = epoch_offset + self.max_epochs
        effective_model_offset = (
            epoch_offset if model_epoch_offset is None else model_epoch_offset
        )
        progress_phase = stage_label or "train"
        if preserve_initial_best:
            print(f"epoch={epoch_offset}/{display_total} phase=warm-start-validation start", flush=True)
            best_mse, best_mae = self._validation(validation_batches())
            if not torch.isfinite(torch.tensor([best_mse, best_mae])).all():
                raise ValueError("warm-start validation metrics must be finite")
            best_epoch = epoch_offset
            self._save_best(
                epoch=int(self.model.online_state()["epoch"]),
                training_epoch=epoch_offset, val_mse=best_mse, val_mae=best_mae, hashes=hashes,
            )
            print(
                f"epoch={epoch_offset}/{display_total} phase=warm-start-validation "
                f"val_mse={best_mse:.6f} val_mae={best_mae:.6f} best_epoch={best_epoch}",
                flush=True,
            )
        for local_epoch in range(1, self.max_epochs + 1):
            epoch = epoch_offset + local_epoch
            model_epoch = effective_model_offset + local_epoch
            epoch_start = perf_counter()
            self._configure_epoch(model_epoch, objective=objective)
            # Forecast fine-tuning reuses the Stage-1 memory/EMA/statistics as
            # a read-only feature source.  It still receives the continued
            # model epoch for route scheduling, but must not publish a new
            # online snapshot or advance the retrieval EMA.
            if not freeze_online_state:
                self.model.begin_epoch(model_epoch)
            train_loss = self._train_epoch(
                train_batches_for_epoch(epoch - 1),
                epoch,
                objective=objective,
                phase_label=progress_phase,
                display_total=display_total,
                model_epoch_index=model_epoch,
            )
            if not freeze_online_state:
                self.model.end_epoch(model_epoch)
            if validation_objective == "repair":
                val_mse, val_mae = self._repair_validation(validation_batches())
                validation_phase = f"{progress_phase}-validation"
            else:
                val_mse, val_mae = self._validation(validation_batches())
                validation_phase = "validation"
            if val_mse < best_mse:
                best_mse, best_mae, best_epoch = val_mse, val_mae, epoch
                stale = 0
                checkpoint_epoch = (
                    int(self.model.online_state()["epoch"])
                    if freeze_online_state
                    else model_epoch
                )
                self._save_best(
                    epoch=checkpoint_epoch,
                    training_epoch=epoch,
                    val_mse=val_mse,
                    val_mae=val_mae,
                    hashes=hashes,
                )
            else:
                stale += 1
            completed = epoch
            print(
                f"epoch={epoch}/{display_total} phase={validation_phase} "
                f"train_loss={train_loss:.6f} val_mse={val_mse:.6f} "
                f"val_mae={val_mae:.6f} best_epoch={best_epoch} stale={stale} "
                f"epoch_seconds={perf_counter() - epoch_start:.1f}",
                flush=True,
            )
            if stale >= self.patience:
                break
        if best_epoch < 0:
            raise RuntimeError("online trainer did not produce a finite validation checkpoint")
        self._restore_best(hashes)
        elapsed = perf_counter() - start_time
        trainable = sum(parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad)
        return OnlineTrainingResult(self.checkpoint_path, best_mse, best_mae, best_epoch, completed, elapsed, trainable)


__all__ = ["OnlineTrainer", "OnlineTrainingResult"]
