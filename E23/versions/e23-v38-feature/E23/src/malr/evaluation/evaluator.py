from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch

from malr.models.stage1 import Stage1Model
from malr.models.online_stage1 import OnlineStage1Model
from malr.models.online_stage1_v2 import RelationV2OnlineStage1Model
from malr.models.dynamic_context_model import DynamicContextOnlineModel
from malr.models.routing_v2 import oracle_route_labels
from malr.models.edge_gain import counterfactual_relative_gain, masked_patch_huber_error
from malr.models.reconstruction import aggregate_patch_predictions
from malr.training.trainer import PreparedBatch

from .metrics import ForecastMetrics
from .online_diagnostics import OnlineDiagnosticAccumulator


@dataclass(frozen=True)
class OnlineEvaluation:
    metrics: ForecastMetrics
    imputation_mse: float
    imputation_mae: float
    diagnostics: dict[str, float]
    inference_ablations: dict[str, ForecastMetrics]


def _forward_v2(model: RelationV2OnlineStage1Model, batch: PreparedBatch, disabled: frozenset[str]):
    return model(
        batch.patch_values, batch.patch_mask, batch.rho, batch.state,
        batch.query_mean, batch.query_std, patch_starts=batch.patch_starts,
        epoch_index=model.memory.snapshot().epoch, patch_valid=batch.patch_valid,
        patch_validity=batch.patch_validity,
        disabled_relations=disabled,
    )


@torch.no_grad()
def _evaluate_dynamic_context_model(
    model: DynamicContextOnlineModel,
    batches: Iterable[PreparedBatch],
    *,
    device: str | torch.device,
) -> OnlineEvaluation:
    """Evaluate E22 while keeping H/V route diagnostics compatible with V2."""

    evaluation_device = torch.device(device)
    model.to(evaluation_device).eval()
    totals = {name: [0.0, 0.0] for name in (
        "full", "without_historical", "without_temporal", "without_variable", "historical_only"
    )}
    disabled = {
        "full": frozenset(),
        "without_historical": frozenset({"historical"}),
        "without_temporal": frozenset({"temporal"}),
        "without_variable": frozenset({"variable"}),
        "historical_only": frozenset({"temporal", "variable"}),
    }
    elements = 0
    imputation_squared = imputation_absolute = imputation_elements = 0.0
    diagnostics = OnlineDiagnosticAccumulator()
    before = model.online_state()
    for batch in batches:
        current = batch.to(evaluation_device)
        outputs = {
            name: model(
                current.patch_values,
                current.patch_mask,
                current.rho,
                current.state,
                current.query_mean,
                current.query_std,
                patch_starts=current.patch_starts,
                epoch_index=model.memory.snapshot().epoch,
                patch_valid=current.patch_valid,
                patch_validity=current.patch_validity,
                disabled_relations=value,
            )
            for name, value in disabled.items()
        }
        for name, output in outputs.items():
            if output.forecast is None:
                raise RuntimeError("E22 evaluation requires an installed forecast predictor")
            error = output.forecast - current.forecast_target
            totals[name][0] += float(error.square().sum().cpu())
            totals[name][1] += float(error.abs().sum().cpu())
        elements += current.forecast_target.numel()
        output = outputs["full"]
        coordinate_validity = (
            current.patch_valid.unsqueeze(-1).expand_as(current.patch_mask)
            if current.patch_validity is None
            else current.patch_validity.to(device=current.patch_mask.device, dtype=torch.bool)
        )
        clean_sequence, _ = aggregate_patch_predictions(
            current.clean_raw_patches,
            window_len=model.core.seq_len,
            patch_len=model.core.patch_len,
            stride=model.core.stride,
            validity=coordinate_validity,
        )
        sequence_error = (
            output.explicit.standardized_sequence[output.explicit.sequence_missing]
            - clean_sequence[output.explicit.sequence_missing]
        )
        imputation_squared += float(sequence_error.square().sum().cpu())
        imputation_absolute += float(sequence_error.abs().sum().cpu())
        imputation_elements += sequence_error.numel()
        damaged = current.state.ne(0) & current.patch_valid
        h_candidate = output.historical.candidate_eligible
        v_candidate = output.variable.candidate_eligible
        t_candidate = output.temporal.local_selected | output.temporal.global_selected
        zeros_h = torch.zeros_like(h_candidate, dtype=current.patch_values.dtype)
        zeros_v = torch.zeros_like(v_candidate, dtype=current.patch_values.dtype)
        zeros_t = torch.zeros_like(t_candidate, dtype=current.patch_values.dtype)
        diagnostics.update(
            relation_candidate={"historical": h_candidate, "temporal": t_candidate, "variable": v_candidate},
            relation_selected={
                "historical": output.historical.candidate_selected,
                "temporal": t_candidate,
                "variable": output.variable.candidate_selected,
            },
            relation_predicted_gain={"historical": output.historical.candidate_predicted_gain, "temporal": zeros_t, "variable": output.variable.candidate_predicted_gain},
            relation_actual_gain={"historical": zeros_h, "temporal": zeros_t, "variable": zeros_v},
            temporal_local_selected=output.temporal.local_selected,
            temporal_global_selected=output.temporal.global_selected,
            route_index=output.route_index,
            route_actual_gain=torch.zeros_like(output.route_index, dtype=current.patch_values.dtype),
            oracle_route_index=output.route_index,
            route_repair_regret=torch.zeros_like(output.route_index, dtype=current.patch_values.dtype),
            damaged=damaged,
        )
    if elements == 0:
        raise ValueError("test batch iterator produced no batches")
    after = model.online_state()
    if before["memory_hash"] != after["memory_hash"] or before["ema_hash"] != after["ema_hash"] or before["relation_prior_hash"] != after["relation_prior_hash"]:
        raise RuntimeError("E22 evaluation changed read-only online state")
    metric_values = {
        name: ForecastMetrics(mse=values[0] / elements, mae=values[1] / elements)
        for name, values in totals.items()
    }
    return OnlineEvaluation(
        metrics=metric_values.pop("full"),
        imputation_mse=imputation_squared / imputation_elements if imputation_elements else 0.0,
        imputation_mae=imputation_absolute / imputation_elements if imputation_elements else 0.0,
        diagnostics=diagnostics.compute(),
        inference_ablations=metric_values,
    )


@torch.no_grad()
def evaluate_online_model(
    model: RelationV2OnlineStage1Model,
    batches: Iterable[PreparedBatch],
    *,
    device: str | torch.device = "cpu",
) -> OnlineEvaluation:
    """Evaluate V2 once per inference ablation without mutating online state."""

    if isinstance(model, DynamicContextOnlineModel):
        return _evaluate_dynamic_context_model(model, batches, device=device)

    evaluation_device = torch.device(device)
    model.to(evaluation_device).eval()
    totals = {
        "full": [0.0, 0.0],
        "without_historical": [0.0, 0.0],
        "without_temporal": [0.0, 0.0],
        "without_variable": [0.0, 0.0],
        "historical_only": [0.0, 0.0],
    }
    disabled = {
        "full": frozenset(),
        "without_historical": frozenset({"historical"}),
        "without_temporal": frozenset({"temporal"}),
        "without_variable": frozenset({"variable"}),
        "historical_only": frozenset({"temporal", "variable"}),
    }
    elements = 0
    imputation_squared = imputation_absolute = imputation_elements = 0.0
    diagnostics = OnlineDiagnosticAccumulator()
    for batch in batches:
        current = batch.to(evaluation_device)
        outputs = {name: _forward_v2(model, current, value) for name, value in disabled.items()}
        for name, output in outputs.items():
            error = output.forecast - current.forecast_target
            totals[name][0] += float(error.square().sum().cpu())
            totals[name][1] += float(error.abs().sum().cpu())
        elements += current.forecast_target.numel()
        output = outputs["full"]
        damaged = current.state.ne(0) & current.patch_valid
        coordinate_validity = (
            current.patch_valid.unsqueeze(-1).expand_as(current.patch_mask)
            if current.patch_validity is None
            else current.patch_validity.to(
                device=current.patch_mask.device, dtype=torch.bool
            )
        )
        if coordinate_validity.shape != current.patch_mask.shape:
            raise ValueError("patch_validity must match patch_mask")
        missing_mask = (
            (~current.patch_mask)
            & coordinate_validity
            & damaged.unsqueeze(-1)
        )
        # ``reconstructed_patches`` is normalized because it is fed back to
        # PatchTST.  Evaluation labels are raw patches, so decode latents and
        # restore observed normalized values before measuring imputation.
        recovered_raw = model._decode_raw_patches(
            output.latents, current.query_mean, current.query_std
        )
        observed_raw = current.patch_values * current.query_std.unsqueeze(-1).unsqueeze(-1) + current.query_mean.unsqueeze(-1).unsqueeze(-1)
        reconstructed_raw = torch.where(current.patch_mask, observed_raw, recovered_raw)
        imputation_error = (reconstructed_raw - current.clean_raw_patches)[missing_mask]
        imputation_squared += float(imputation_error.square().sum().cpu())
        imputation_absolute += float(imputation_error.abs().sum().cpu())
        imputation_elements += float(imputation_error.numel())
        target_eligible = damaged & missing_mask.any(dim=-1)
        branches = {"historical": output.historical, "temporal": output.temporal, "variable": output.variable}
        actual_gain = {}
        for name, branch in branches.items():
            candidate_latents = model.graph.recover_candidates(output.base_latents, branch.candidate_messages, target_eligible)
            baseline = model._decode_raw_patches(output.base_latents, current.query_mean, current.query_std)
            candidate = model._decode_raw_patches(candidate_latents, current.query_mean, current.query_std)
            actual_gain[name] = counterfactual_relative_gain(baseline, candidate, current.clean_raw_patches, missing_mask)
        baseline_prediction = model._decode_raw_patches(output.base_latents, current.query_mean, current.query_std)
        baseline_error = masked_patch_huber_error(baseline_prediction, current.clean_raw_patches, missing_mask)
        route_prediction = model._decode_raw_patches(output.graph.route_latents, current.query_mean, current.query_std)
        route_errors = masked_patch_huber_error(route_prediction, current.clean_raw_patches, missing_mask).clone()
        route_errors[..., 0] = baseline_error
        relation_available = (
            output.route_available[..., 1:4] & target_eligible.unsqueeze(-1)
        )
        oracle = oracle_route_labels(
            route_errors,
            baseline_error,
            relation_available,
            complexity_margin=torch.zeros(
                3, device=route_errors.device, dtype=route_errors.dtype
            ),
            allow_combinations=False,
        )
        selected_error = route_errors.gather(-1, output.route_index.unsqueeze(-1)).squeeze(-1)
        oracle_error = route_errors.gather(
            -1, oracle.index.unsqueeze(-1)
        ).squeeze(-1)
        route_repair_regret = (
            (selected_error - oracle_error) / (baseline_error + 1e-8)
        ).clamp_min(0.0)
        route_gain = (
            (baseline_error - selected_error) / (baseline_error + 1e-8)
        ).nan_to_num(nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
        diagnostics.update(
            relation_candidate={name: branch.candidate_eligible for name, branch in branches.items()},
            relation_selected={name: branch.candidate_selected for name, branch in branches.items()},
            relation_predicted_gain={name: branch.candidate_predicted_gain for name, branch in branches.items()},
            relation_actual_gain=actual_gain,
            temporal_local_selected=output.temporal.diagnostics.get(
                "local_selected", torch.zeros_like(output.temporal.candidate_selected)
            ),
            temporal_global_selected=output.temporal.diagnostics.get(
                "global_selected", torch.zeros_like(output.temporal.candidate_selected)
            ),
            route_index=output.route_index,
            route_actual_gain=route_gain,
            oracle_route_index=oracle.index,
            route_repair_regret=route_repair_regret,
            damaged=target_eligible,
        )
    if elements == 0:
        raise ValueError("test batch iterator produced no batches")
    metric_values = {name: ForecastMetrics(mse=values[0] / elements, mae=values[1] / elements) for name, values in totals.items()}
    return OnlineEvaluation(
        metrics=metric_values.pop("full"),
        imputation_mse=imputation_squared / imputation_elements if imputation_elements else 0.0,
        imputation_mae=imputation_absolute / imputation_elements if imputation_elements else 0.0,
        diagnostics=diagnostics.compute(),
        inference_ablations=metric_values,
    )


@torch.no_grad()
def evaluate_model(
    model: Stage1Model | OnlineStage1Model,
    batches: Iterable[PreparedBatch],
    *,
    device: str | torch.device = "cpu",
) -> ForecastMetrics:
    evaluation_device = torch.device(device)
    model.to(evaluation_device).eval()
    squared_error = 0.0
    absolute_error = 0.0
    element_count = 0
    for batch in batches:
        current = batch.to(evaluation_device)
        if isinstance(model, OnlineStage1Model):
            output = model(
                current.patch_values,
                current.patch_mask,
                current.rho,
                current.state,
                current.query_mean,
                current.query_std,
                patch_starts=current.patch_starts,
                epoch_index=model.memory.snapshot().epoch,
                patch_valid=current.patch_valid,
            )
        else:
            output = model(
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
        element_count += error.numel()
    if element_count == 0:
        raise ValueError("test batch iterator produced no batches")
    return ForecastMetrics(
        mse=squared_error / element_count,
        mae=absolute_error / element_count,
    )
