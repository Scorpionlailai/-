"""Replay a selected E23 Stage-2 checkpoint on locked P4 validation only."""

from __future__ import annotations

import argparse
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
    parser.add_argument("--e23-config", type=Path, required=True)
    parser.add_argument("--stage2-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    protocol = load_e23_p4_protocol(
        catalog=MigrationCatalog.load(root.parent / "E22/migration/paper_catalog.jsonl", project="E23"),
        protocol_config=root / "configs/stage1_seq96_dynamic_temporal_repair.yaml",
        mask_artifacts=root.parent / "E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context",
    )
    settings = load_e23_settings(args.e23_config, protocol=protocol)
    device = torch.device(args.device)
    selected = load_stage2_selection(args.stage2_manifest, protocol, settings, device=device)
    record = json.loads(args.stage2_manifest.read_text(encoding="utf-8"))
    model = build_forecast_model(
        protocol, settings, device=device,
        forecast_mixer=record.get("forecast_mixer", "none"),
        history_adapter=record.get("history_adapter", "none"),
        forecast_input=record.get("forecast_input", "explicit_v1"),
    )
    model.load_state_dict(selected["model_state"], strict=True)
    model.repair.historical.retrieval.load_state_dict(selected["retrieval_state"])
    memory = _new_memory(protocol)
    memory.load_state_dict(selected["memory_state"])
    model.repair.historical.refresh_snapshot(memory.snapshot())
    before = _key_sha256(model.repair.historical.retrieval.state_dict())
    if before != record["key_sha256"]:
        raise ValueError("H key state mismatches selected manifest before replay")
    model.eval()
    absolute = squared = 0.0
    coordinates = windows = 0
    with torch.inference_mode():
        for cpu in iter_p4_batches(protocol, "validation", batch_size=settings.batch_size):
            batch = cpu.to(device)
            starts = _starts(batch, stride=model.repair.stride,
                             patches=model.repair.patch_count, device=device)
            forecast, _ = model(batch.normalized_observed, batch.observed_mask,
                                batch.query_mean, batch.query_std, starts)
            error = forecast - batch.forecast_target
            absolute += error.abs().sum().item()
            squared += error.square().sum().item()
            coordinates += error.numel()
            windows += forecast.shape[0]
    after = _key_sha256(model.repair.historical.retrieval.state_dict())
    if after != before:
        raise ValueError("H key state changed during validation replay")
    mse, mae = squared / coordinates, absolute / coordinates
    if abs(mse - record["best_val_mse"]) > 1e-6 or abs(mae - record["best_val_mae"]) > 1e-6:
        raise ValueError("validation replay differs from selected manifest")
    result = {
        "split": "validation", "windows": windows, "coordinates": coordinates,
        "mse": mse, "mae": mae, "h_key_sha256_before": before,
        "h_key_sha256_after": after,
        "stage2_manifest": str(args.stage2_manifest.resolve()),
        "stage2_checkpoint_sha256": record["checkpoint_sha256"],
    }
    if args.output.exists():
        raise FileExistsError(f"replay output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
