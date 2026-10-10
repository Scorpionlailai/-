"""Read-only v12 validation ablation of each expert before router softmax."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import torch

from integrations.e23.config import load_e23_p4_protocol, load_e23_settings
from integrations.e23.train import (
    _key_sha256, _new_memory, _starts, build_forecast_model, load_stage2_selection,
)
from integrations.forecast_compare.protocol import iter_p4_batches
from malr.migration import MigrationCatalog


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = root / "configs/e23_multi_expert_p4_width64_window_v12.yaml"
    manifest = (root / "artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments"
                / "e23_v3_window_v12/stage2/runs/e23-stage2.json")
    protocol = load_e23_p4_protocol(
        catalog=MigrationCatalog.load(root.parent / "E22/migration/paper_catalog.jsonl", project="E23"),
        protocol_config=root / "configs/stage1_seq96_dynamic_temporal_repair.yaml",
        mask_artifacts=root.parent / "E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context",
    )
    settings = load_e23_settings(config, protocol=protocol)
    device = torch.device(args.device)
    selected = load_stage2_selection(manifest, protocol, settings, device=device)
    record = json.loads(manifest.read_text(encoding="utf-8"))
    model = build_forecast_model(protocol, settings, device=device)
    model.load_state_dict(selected["model_state"], strict=True)
    model.repair.historical.retrieval.load_state_dict(selected["retrieval_state"])
    memory = _new_memory(protocol)
    memory.load_state_dict(selected["memory_state"])
    model.repair.historical.refresh_snapshot(memory.snapshot())
    before = _key_sha256(model.repair.historical.retrieval.state_dict())
    if before != record["key_sha256"]:
        raise ValueError("H key state mismatches v12 manifest")
    model.eval()

    variants = {"full": None, "without_historical": "historical",
                "without_variable": "variable", "without_temporal": "temporal",
                "without_all_experts": "all"}
    totals = {name: {"squared": 0.0, "absolute": 0.0, "count": 0,
                     "band_squared": [0.0] * 4, "band_absolute": [0.0] * 4,
                     "band_count": [0] * 4} for name in variants}
    original_router = model.repair.router.forward
    windows = 0
    with torch.inference_mode():
        for cpu in iter_p4_batches(protocol, "validation", batch_size=settings.batch_size):
            batch = cpu.to(device)
            starts = _starts(batch, stride=model.repair.stride,
                             patches=model.repair.patch_count, device=device)
            windows += batch.normalized_observed.shape[0]
            for variant, disabled in variants.items():
                if disabled is None:
                    model.repair.router.forward = original_router
                else:
                    def ablated_router(inputs, evidence, *, disabled=disabled):
                        changed = {
                            name: replace(item, available=torch.zeros_like(item.available))
                            if disabled == "all" or name == disabled else item
                            for name, item in evidence.items()
                        }
                        return original_router(inputs, changed)
                    model.repair.router.forward = ablated_router
                forecast, _ = model(batch.normalized_observed, batch.observed_mask,
                                    batch.query_mean, batch.query_std, starts)
                error = forecast - batch.forecast_target
                stats = totals[variant]
                stats["squared"] += error.square().sum().item()
                stats["absolute"] += error.abs().sum().item()
                stats["count"] += error.numel()
                for band in range(4):
                    part = error[..., 24 * band:24 * (band + 1)]
                    stats["band_squared"][band] += part.square().sum().item()
                    stats["band_absolute"][band] += part.abs().sum().item()
                    stats["band_count"][band] += part.numel()
    model.repair.router.forward = original_router
    after = _key_sha256(model.repair.historical.retrieval.state_dict())
    if before != after:
        raise ValueError("H key state changed during ablation")
    results = {}
    for variant, stats in totals.items():
        results[variant] = {
            "mse": stats["squared"] / stats["count"],
            "mae": stats["absolute"] / stats["count"],
            "horizon_bands": [{
                "steps": [24 * band + 1, 24 * (band + 1)],
                "mse": stats["band_squared"][band] / stats["band_count"][band],
                "mae": stats["band_absolute"][band] / stats["band_count"][band],
            } for band in range(4)],
        }
    if (abs(results["full"]["mse"] - record["best_val_mse"]) > 1e-6 or
            abs(results["full"]["mae"] - record["best_val_mae"]) > 1e-6):
        raise ValueError("full v12 replay differs from selected manifest")
    for variant in variants:
        if variant == "full":
            continue
        results[variant]["delta_mse"] = results[variant]["mse"] - results["full"]["mse"]
        results[variant]["delta_mae"] = results[variant]["mae"] - results["full"]["mae"]
    output = {
        "split": "validation", "method": "mask expert availability before router softmax",
        "windows": windows, "coordinates_per_variant": totals["full"]["count"],
        "stage2_manifest": str(manifest), "stage2_checkpoint_sha256": record["checkpoint_sha256"],
        "h_key_sha256_before": before, "h_key_sha256_after": after,
        "results": results,
    }
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output))


if __name__ == "__main__":
    main()
