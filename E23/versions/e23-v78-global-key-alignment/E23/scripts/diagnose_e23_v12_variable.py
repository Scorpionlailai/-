"""Read-only validation audit of E23 v12's cross-variable source choice."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from integrations.e23.config import load_e23_p4_protocol, load_e23_settings
from integrations.e23.train import _new_memory, _starts, build_forecast_model, load_stage2_selection
from integrations.forecast_compare.protocol import iter_p4_batches
from malr.data.patching import patchify_preserving_values, patchify_with_state
from malr.migration import MigrationCatalog


def _train_pairwise_affine(protocol) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit source-to-target maps using only jointly observed training coordinates."""
    count = torch.zeros((7, 7), dtype=torch.float64)
    target_sum = torch.zeros_like(count)
    source_sum = torch.zeros_like(count)
    source_square = torch.zeros_like(count)
    cross = torch.zeros_like(count)
    for batch in iter_p4_batches(protocol, "train", batch_size=128, epoch=0):
        values = batch.normalized_target.reshape(-1, 7).double()
        observed = batch.observed_mask.reshape(-1, 7).double()
        visible_values = values * observed
        count += observed.T @ observed
        target_sum += visible_values.T @ observed
        source_sum += observed.T @ visible_values
        source_square += observed.T @ (visible_values.square())
        cross += visible_values.T @ visible_values
    denominator = source_square - source_sum.square() / count.clamp_min(1)
    slope = (cross - target_sum * source_sum / count.clamp_min(1)) / denominator.clamp_min(1e-8)
    intercept = (target_sum - slope * source_sum) / count.clamp_min(1)
    return slope.float(), intercept.float(), count


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    trial = root / "artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/e23_v3_window_v12"
    protocol = load_e23_p4_protocol(
        catalog=MigrationCatalog.load(root.parent / "E22/migration/paper_catalog.jsonl", project="E23"),
        protocol_config=root / "configs/stage1_seq96_dynamic_temporal_repair.yaml",
        mask_artifacts=root.parent / "E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context",
    )
    settings = load_e23_settings(root / "configs/e23_multi_expert_p4_width64_window_v12.yaml", protocol=protocol)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    manifest = trial / "stage2/runs/e23-stage2.json"
    selected = load_stage2_selection(manifest, protocol, settings, device=device)
    model = build_forecast_model(protocol, settings, device=device)
    model.load_state_dict(selected["model_state"], strict=True)
    model.repair.historical.retrieval.load_state_dict(selected["retrieval_state"])
    memory = _new_memory(protocol)
    memory.load_state_dict(selected["memory_state"])
    model.repair.historical.refresh_snapshot(memory.snapshot())
    model.eval()
    slope, intercept, fit_count = _train_pairwise_affine(protocol)
    slope, intercept = slope.to(device), intercept.to(device)

    totals = torch.zeros(9, dtype=torch.float64, device=device)
    # dynamic minus global-topk: negative means v12's window-conditioned choice helps
    delta_mae = torch.zeros(4, dtype=torch.float64, device=device)
    delta_mse = torch.zeros_like(delta_mae)
    bucket_count = torch.zeros_like(delta_mae)
    horizon_mae = torch.zeros_like(delta_mae)
    horizon_mse = torch.zeros_like(delta_mae)
    hidden_count = torch.zeros(7, dtype=torch.float64, device=device)
    raw_mae = torch.zeros_like(hidden_count)
    mapped_mae = torch.zeros_like(hidden_count)
    zero_mae = torch.zeros_like(hidden_count)
    raw_mse = torch.zeros_like(hidden_count)
    mapped_mse = torch.zeros_like(hidden_count)
    zero_mse = torch.zeros_like(hidden_count)
    windows = 0
    with torch.inference_mode():
        for cpu in iter_p4_batches(protocol, "validation", batch_size=settings.batch_size):
            batch = cpu.to(device)
            starts = _starts(batch, stride=model.repair.stride,
                             patches=model.repair.patch_count, device=device)
            prediction, output = model(batch.normalized_observed, batch.observed_mask,
                                       batch.query_mean, batch.query_std, starts)
            model.repair.variable.selection = "global_topk"
            try:
                global_prediction, global_output = model(
                    batch.normalized_observed, batch.observed_mask,
                    batch.query_mean, batch.query_std, starts,
                )
            finally:
                model.repair.variable.selection = "window_context_topk_v1"
            windows += prediction.shape[0]
            difference_mae = (prediction - batch.forecast_target).abs() - (
                global_prediction - batch.forecast_target).abs()
            difference_mse = (prediction - batch.forecast_target).square() - (
                global_prediction - batch.forecast_target).square()
            totals[5] += (prediction - batch.forecast_target).abs().sum()
            totals[6] += (prediction - batch.forecast_target).square().sum()
            totals[7] += (global_prediction - batch.forecast_target).abs().sum()
            totals[8] += (global_prediction - batch.forecast_target).square().sum()
            totals[0] += difference_mae.sum()
            totals[1] += difference_mse.sum()
            totals[2] += difference_mae.numel()
            dynamic = output.evidence["variable"].provenance["source_variables"]
            global_source = global_output.evidence["variable"].provenance["source_variables"]
            same_set = (dynamic.sort(-1).values == global_source[None, :, None, :].sort(-1).values).all(-1)
            damaged = output.evidence["variable"].available
            totals[3] += (same_set & damaged).sum()
            totals[4] += damaged.sum()
            fraction = batch.observed_mask.float().mean(1)
            bucket = torch.clamp((fraction * 4).long(), max=3)
            for index in range(4):
                selected_bucket = bucket == index
                delta_mae[index] += difference_mae.sum(-1)[selected_bucket].sum()
                delta_mse[index] += difference_mse.sum(-1)[selected_bucket].sum()
                bucket_count[index] += selected_bucket.sum() * 96
                horizon_mae[index] += difference_mae[..., index * 24:(index + 1) * 24].sum()
                horizon_mse[index] += difference_mse[..., index * 24:(index + 1) * 24].sum()

            observed_patch = patchify_with_state(batch.normalized_observed,
                                                 batch.observed_mask, 16, 8)
            clean_patch = patchify_preserving_values(batch.normalized_target,
                                                     torch.ones_like(batch.observed_mask), 16, 8)
            patch_values = observed_patch.values
            patch_mask = observed_patch.effective_mask
            target = clean_patch.values
            size, channels, patches, candidates = dynamic.shape
            batch_index = torch.arange(size, device=device)[:, None, None, None]
            patch_index = torch.arange(patches, device=device)[None, None, :, None]
            source_value = patch_values[batch_index, dynamic, patch_index]
            source_observed = patch_mask[batch_index, dynamic, patch_index]
            target_observed = patch_mask[:, :, :, None, :]
            hidden_with_source = ~target_observed & source_observed
            target_value = target[:, :, :, None, :]
            target_index = torch.arange(channels, device=device)[None, :, None, None]
            mapped = (slope[target_index, dynamic][..., None] * source_value
                      + intercept[target_index, dynamic][..., None])
            raw_error = (source_value - target_value) * hidden_with_source
            mapped_error = (mapped - target_value) * hidden_with_source
            zero_error = -target_value * hidden_with_source
            hidden_count += hidden_with_source.sum(dim=(0, 2, 3, 4))
            raw_mae += raw_error.abs().sum(dim=(0, 2, 3, 4))
            mapped_mae += mapped_error.abs().sum(dim=(0, 2, 3, 4))
            zero_mae += zero_error.abs().sum(dim=(0, 2, 3, 4))
            raw_mse += raw_error.square().sum(dim=(0, 2, 3, 4))
            mapped_mse += mapped_error.square().sum(dim=(0, 2, 3, 4))
            zero_mse += zero_error.square().sum(dim=(0, 2, 3, 4))

    output_path = trial / "benchmarks/v12_variable_source_diagnostic_v3.json"
    report = {
        "split": "validation", "windows": windows,
        "note": "Inference source-swap and train-only affine source proxy; neither retrains nor evaluates a new model.",
        "dynamic_minus_global_mae": (totals[0] / totals[2]).item(),
        "dynamic_minus_global_mse": (totals[1] / totals[2]).item(),
        "dynamic_validation_mae": (totals[5] / totals[2]).item(),
        "dynamic_validation_mse": (totals[6] / totals[2]).item(),
        "global_swap_validation_mae": (totals[7] / totals[2]).item(),
        "global_swap_validation_mse": (totals[8] / totals[2]).item(),
        "damaged_same_source_set_fraction": (totals[3] / totals[4]).item(),
        "observation_buckets": ["[0,.25)", "[.25,.5)", "[.5,.75)", "[.75,1]"],
        "bucket_coordinate_count": bucket_count.cpu().tolist(),
        "bucket_dynamic_minus_global_mae": (delta_mae / bucket_count.clamp_min(1)).cpu().tolist(),
        "bucket_dynamic_minus_global_mse": (delta_mse / bucket_count.clamp_min(1)).cpu().tolist(),
        "horizon_dynamic_minus_global_mae": (horizon_mae / (windows * 7 * 24)).cpu().tolist(),
        "horizon_dynamic_minus_global_mse": (horizon_mse / (windows * 7 * 24)).cpu().tolist(),
        "selected_source_hidden_coordinate_count_by_target": hidden_count.cpu().tolist(),
        "selected_source_raw_mae_by_target": (raw_mae / hidden_count.clamp_min(1)).cpu().tolist(),
        "selected_source_affine_mae_by_target": (mapped_mae / hidden_count.clamp_min(1)).cpu().tolist(),
        "zero_baseline_mae_by_target": (zero_mae / hidden_count.clamp_min(1)).cpu().tolist(),
        "selected_source_raw_mse_by_target": (raw_mse / hidden_count.clamp_min(1)).cpu().tolist(),
        "selected_source_affine_mse_by_target": (mapped_mse / hidden_count.clamp_min(1)).cpu().tolist(),
        "zero_baseline_mse_by_target": (zero_mse / hidden_count.clamp_min(1)).cpu().tolist(),
        "train_pairwise_min_observed_count": fit_count[~torch.eye(7, dtype=torch.bool)].min().item(),
        "train_pairwise_negative_slope_count": int((slope.cpu()[~torch.eye(7, dtype=torch.bool)] < 0).sum()),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
