"""Read-only fixed-validation sensitivity for the selected grouped E23 checkpoint."""

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import torch

from integrations.e23.config import load_e23_p4_protocol, load_e23_settings
from integrations.e23.train import (_key_sha256, _new_memory, _starts,
                                    build_forecast_model, load_stage2_selection)
from integrations.forecast_compare.protocol import iter_p4_batches
from malr.data.patching import patchify_with_state
from malr.migration import MigrationCatalog


ROOT = Path(__file__).resolve().parents[7]
SELECTION = ROOT / "artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/e23_grouped_history_current_width64_v5/stage2_p4/runs/e23-stage2.json"
OUT = Path(__file__).with_name("selected_validation_group_ablations.json")
MODES = {
    "full": (),
    "without_historical_group": ("historical", "historical_variable"),
    "without_current_group": ("temporal", "variable"),
    "without_historical": ("historical",),
    "without_historical_variable": ("historical_variable",),
    "without_temporal": ("temporal",),
    "without_current_variable": ("variable",),
}


def main() -> None:
    device = torch.device("cuda")
    catalog = MigrationCatalog.load(ROOT.parent / "E22/migration/paper_catalog.jsonl", project="E23")
    protocol = load_e23_p4_protocol(
        catalog=catalog,
        protocol_config=ROOT / "configs/stage1_seq96_dynamic_temporal_repair.yaml",
        mask_artifacts=ROOT.parent / "E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context",
    )
    settings = load_e23_settings(ROOT / "configs/e23_grouped_history_current_p4_width64.yaml",
                                 protocol=protocol)
    payload = load_stage2_selection(SELECTION, protocol, settings, device=device)
    model = build_forecast_model(protocol, settings, device=device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.repair.historical.retrieval.load_state_dict(payload["retrieval_state"])
    memory = _new_memory(protocol)
    memory.load_state_dict(payload["memory_state"])
    model.repair.historical.refresh_snapshot(memory.snapshot())
    model.eval()
    memory_before = memory.snapshot_hash()
    key_before = _key_sha256(model.repair.historical.retrieval.state_dict())
    original = model.repair.router.forward
    metrics = {mode: {"squared": 0.0, "absolute": 0.0, "count": 0}
               for mode in MODES}
    group_weight = torch.zeros(3, dtype=torch.float64)
    group_available = torch.zeros(3, dtype=torch.float64)
    leaf_weight = torch.zeros(4, dtype=torch.float64)
    leaf_available = torch.zeros(4, dtype=torch.float64)
    damaged_count = windows = 0
    leaves = ("historical", "historical_variable", "temporal", "variable")
    with torch.inference_mode():
        for batch_cpu in iter_p4_batches(protocol, "validation", batch_size=settings.batch_size):
            batch = batch_cpu.to(device)
            starts = _starts(batch, stride=model.repair.stride,
                             patches=model.repair.patch_count, device=device)
            for mode, disabled in MODES.items():
                def route(inputs, evidence, *, disabled=disabled):
                    if disabled:
                        evidence = {name: replace(item, available=torch.zeros_like(item.available))
                                    if name in disabled else item for name, item in evidence.items()}
                    return original(inputs, evidence)
                model.repair.router.forward = route
                prediction, output = model(
                    batch.normalized_observed, batch.observed_mask,
                    batch.query_mean, batch.query_std, starts,
                )
                error = prediction - batch.forecast_target
                metrics[mode]["squared"] += error.square().sum().item()
                metrics[mode]["absolute"] += error.abs().sum().item()
                metrics[mode]["count"] += error.numel()
                if mode == "full":
                    damaged = patchify_with_state(
                        batch.normalized_observed, batch.observed_mask,
                        model.repair.patch_length, model.repair.stride,
                    ).rho.lt(1.0)
                    damaged_count += damaged.sum().item()
                    group_weight += output.router.weights[damaged].sum(0).double().cpu()
                    group_available += output.router.available[damaged].sum(0).double().cpu()
                    leaf_weight += output.router.leaf_weights[damaged].sum(0).double().cpu()
                    leaf_available += output.router.leaf_available[damaged].sum(0).double().cpu()
            windows += batch.normalized_observed.shape[0]
    model.repair.router.forward = original
    if windows != 2688 or not damaged_count:
        raise ValueError("fixed validation window count mismatch")
    if memory.snapshot_hash() != memory_before or _key_sha256(model.repair.historical.retrieval.state_dict()) != key_before:
        raise RuntimeError("read-only H state changed")
    result = {
        "split": "validation", "windows": windows,
        "selection_manifest_sha256": sha256(SELECTION.read_bytes()).hexdigest(),
        "selected_epoch": payload["epoch"],
        "metrics": {name: {"mse": item["squared"] / item["count"],
                           "mae": item["absolute"] / item["count"]}
                    for name, item in metrics.items()},
        "damaged_patch_targets": damaged_count,
        "group_weight_mean": dict(zip(("none", "historical_group", "current_group"),
                                      (group_weight / damaged_count).tolist())),
        "group_availability_rate": dict(zip(("none", "historical_group", "current_group"),
                                            (group_available / damaged_count).tolist())),
        "leaf_weight_mean": dict(zip(leaves, (leaf_weight / damaged_count).tolist())),
        "leaf_availability_rate": dict(zip(leaves, (leaf_available / damaged_count).tolist())),
        "h_state_unchanged": True,
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
