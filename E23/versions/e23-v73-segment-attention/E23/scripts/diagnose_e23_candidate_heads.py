"""Read-only validation diagnostics for the eight candidate-reading E23 heads."""

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
    if settings.multihead_router not in (
        "eight_candidate_heads_v1", "eight_full_width_gated_v1",
    ):
        raise ValueError("head diagnostics require the eight-head config")
    device = torch.device(args.device)
    selected = load_stage2_selection(args.stage2_manifest, protocol, settings, device=device)
    record = json.loads(args.stage2_manifest.read_text())
    model = build_forecast_model(protocol, settings, device=device)
    model.load_state_dict(selected["model_state"], strict=True)
    model.repair.historical.retrieval.load_state_dict(selected["retrieval_state"])
    memory = _new_memory(protocol)
    memory.load_state_dict(selected["memory_state"])
    model.repair.historical.refresh_snapshot(memory.snapshot())
    before = _key_sha256(model.repair.historical.retrieval.state_dict())
    if before != record["key_sha256"]:
        raise ValueError("selected H key state mismatch")
    model.eval()
    router = model.repair.router
    original = router.forward
    variants = ("full", *(f"without_head_{index}" for index in range(8)))
    stats = {name: {"square": 0.0, "absolute": 0.0, "count": 0} for name in variants}
    route_sum = torch.zeros((8, 4), dtype=torch.float64)
    gate_sum = torch.zeros(9, dtype=torch.float64)
    damaged_count = 0
    attention_difference = {name: 0.0 for name in ("historical", "variable", "temporal")}
    attention_count = {name: 0 for name in attention_difference}
    with torch.inference_mode():
        for cpu in iter_p4_batches(protocol, "validation", batch_size=settings.batch_size):
            batch = cpu.to(device)
            starts = _starts(batch, stride=model.repair.stride,
                             patches=model.repair.patch_count, device=device)
            for variant in variants:
                if variant == "full":
                    router.forward = original
                else:
                    index = int(variant.rsplit("_", 1)[-1])

                    def ablated(inputs, evidence, *, index=index):
                        routed = original(inputs, evidence)
                        if routed.head_gate_weights is None:
                            result = routed.latents.clone()
                            width = result.shape[-1] // 8
                            result[..., index * width:(index + 1) * width] = (
                                inputs.base_latents[..., index * width:(index + 1) * width]
                            )
                        else:
                            result = routed.latents - (
                                routed.head_gate_weights[..., index + 1, None]
                                * routed.head_deltas[..., index, :]
                            )
                        return replace(routed, latents=result)

                    router.forward = ablated
                forecast, output = model(batch.normalized_observed, batch.observed_mask,
                                         batch.query_mean, batch.query_std, starts)
                error = forecast - batch.forecast_target
                state = stats[variant]
                state["square"] += error.square().sum().item()
                state["absolute"] += error.abs().sum().item()
                state["count"] += error.numel()
                if variant != "full":
                    continue
                damaged = ((~output.explicit.observation_patches.to(torch.bool))
                           & output.explicit.validity_patches.to(torch.bool)).any(-1)
                damaged_count += damaged.sum().item()
                route_sum += output.router.head_weights[damaged].sum(0).double().cpu()
                if output.router.head_gate_weights is not None:
                    gate_sum += output.router.head_gate_weights[damaged].sum(0).double().cpu()
                for name, item in output.evidence.items():
                    valid = item.candidates.valid.sum(-1).ge(2) & damaged
                    if not valid.any():
                        continue
                    attention = output.router.candidate_attention[name][valid]
                    # Average L1 distance over all distinct pairs, normalized by candidates.
                    diff = (attention[:, :, None, :] - attention[:, None, :, :]).abs().mean(-1)
                    pairs = torch.triu(torch.ones(8, 8, device=device, dtype=torch.bool), diagonal=1)
                    attention_difference[name] += diff[:, pairs].sum().item()
                    attention_count[name] += int(valid.sum().item()) * int(pairs.sum().item())
    router.forward = original
    after = _key_sha256(model.repair.historical.retrieval.state_dict())
    if before != after:
        raise ValueError("H key state changed during head ablation")
    metrics = {name: {"mse": value["square"] / value["count"],
                      "mae": value["absolute"] / value["count"]}
               for name, value in stats.items()}
    if abs(metrics["full"]["mae"] - record["best_val_mae"]) > 1e-6 or abs(
        metrics["full"]["mse"] - record["best_val_mse"]
    ) > 1e-6:
        raise ValueError("full validation replay differs from selected checkpoint")
    for name in variants[1:]:
        metrics[name]["delta_mae"] = metrics[name]["mae"] - metrics["full"]["mae"]
        metrics[name]["delta_mse"] = metrics[name]["mse"] - metrics["full"]["mse"]
    result = {
        "split": "validation", "stage2_manifest": str(args.stage2_manifest.resolve()),
        "checkpoint_sha256": record["checkpoint_sha256"],
        "h_key_sha256_before": before, "h_key_sha256_after": after,
        "head_ablation": metrics,
        "mean_head_route_weights_on_damaged": (route_sum / damaged_count).tolist(),
        **({"mean_head_gate_weights_on_damaged": (gate_sum / damaged_count).tolist()}
           if settings.multihead_router == "eight_full_width_gated_v1" else {}),
        "mean_pairwise_candidate_attention_l1": {
            name: attention_difference[name] / max(attention_count[name], 1)
            for name in attention_difference
        },
    }
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
