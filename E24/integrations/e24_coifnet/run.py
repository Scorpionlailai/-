"""One isolated Weather experiment; parent repair and official baselines stay sealed."""
from time import perf_counter
PROCESS_STARTED = perf_counter()

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch

from integrations.e24.data import read_manifest, load_dataset_protocol, with_test_mask
from integrations.e23.config import load_e23_settings
from integrations.e23.forecast_sensitive import parameter_hash
from integrations.e23.train import (load_stage1_selection, build_repair_model, _new_memory,
                                    _key_sha256, _atomic_json, _atomic_torch_save)
from integrations.forecast_compare.author_profiles import load_author_profiles, author_coifnet_loss
from integrations.forecast_compare.coifnet_p4 import coifnet_inputs, load_timestamp_features
from integrations.forecast_compare.protocol import iter_p4_batches
from integrations.forecast_compare.resources import TrainingResources, selected_training_resources
from integrations.weather_baselines.models import build_author
from integrations.weather_baselines.run import seed
from malr.evaluation.report import file_sha256
from .model import FrozenRepairCoIFNet, repaired_coifnet_output

ROOT = Path(__file__).resolve().parents[2]
MODEL = "E24-v79+CoIFNet-v1"
PROTOCOL = "E24-weather-frozen-repair-coifnet-sr010-bl8-v1"
DEFAULT_MANIFEST = ROOT / "artifacts/Weather/seq96_pred96/sr010_bl8/shared_protocol_v1/dataset_manifest.json"
DEFAULT_PARENT = ROOT / "artifacts/Weather/seq96_pred96/sr010_bl8/e24_v79_weather_20261007/seed_2023/E24-v79/stage1/runs/e23-stage1.json"
DEFAULT_PROFILES = ROOT / "configs/baselines/weather_sr010_bl8_author_20261008.json"


def state_identity(model):
    return dict(repair=parameter_hash(model.repair), memory=model.memory.snapshot_hash(),
                retrieval=_key_sha256(model.repair.historical.retrieval.state_dict()))


def construct(args, protocol, profile, device):
    settings = load_e23_settings(args.e24_config, protocol=protocol)
    payload = load_stage1_selection(args.parent, protocol, settings, device=device)
    repair = build_repair_model(protocol, settings, device=device)
    repair.load_state_dict(payload["model_state"], strict=True)
    repair.historical.retrieval.load_state_dict(payload["retrieval_state"])
    memory = _new_memory(protocol)
    memory.load_state_dict(payload["memory_state"])
    repair.historical.refresh_snapshot(memory.snapshot())
    # Repair construction must not change the CoIFNet initialization sequence.
    seed(profile["seed"])
    core = build_author("CoIFNet", profile, protocol, device)
    model = FrozenRepairCoIFNet(repair, core, memory)
    return model.to(device)


def identity(args, protocol, profile):
    parent = json.loads(args.parent.read_text())
    return dict(model=MODEL, protocol_id=PROTOCOL, data_protocol_id=read_manifest(args.dataset_manifest)["protocol_id"],
        dataset_manifest_sha256=file_sha256(args.dataset_manifest), source_csv_sha256=protocol.source_sha256,
        repair_selection_manifest=str(args.parent), repair_selection_sha256=file_sha256(args.parent),
        stage1_checkpoint_sha256=parent["checkpoint_sha256"], profiles_sha256=file_sha256(args.profiles),
        e24_config_sha256=file_sha256(args.e24_config), training_seed=profile["seed"],
        model_adapter_sha256=file_sha256(Path(__file__).with_name("model.py")),
        runner_sha256=file_sha256(Path(__file__)), upstream_revision=profile["revision"],
        repair_update_policy="fully_frozen", normalization="observed_only_stats_preserve_repaired_values_v1",
        mask_semantics="original_true_visibility; repaired coordinates remain unobserved",
        reconstruction_supervision="observed_history_only; repaired pseudo-values are not targets",
        exploratory=True, public_patchtst_protocol_compliant=False)


@torch.no_grad()
def validation_cache(model, protocol, device):
    # Only this fixed validation mask is cached. Dynamic training masks are
    # repaired afresh every epoch, and each test mask is evaluated separately.
    model.eval()
    return [(batch, model.repaired_history(batch.to(device)).cpu())
            for batch in iter_p4_batches(protocol, "validation", batch_size=128)]


@torch.no_grad()
def score(model, protocol, split, device, timestamps, *, cache=None):
    model.eval()
    sq = absolute = count = windows = 0
    repair = {k: 0 for k in ("sq96", "abs96", "n96", "sq32", "abs32", "n32")}
    batches = cache if cache is not None else ((batch, None)
        for batch in iter_p4_batches(protocol, split, batch_size=128))
    for cpu, cached in batches:
        batch = cpu.to(device)
        output, repaired = model(batch, timestamps, repaired=None if cached is None else cached.to(device))
        prediction = output[:, 96:].permute(0, 2, 1)
        error = (prediction - batch.forecast_target).double()
        sq += error.square().sum().item(); absolute += error.abs().sum().item(); count += error.numel()
        visible, _, _ = coifnet_inputs(batch, timestamps)
        if not torch.equal(repaired[batch.observed_mask], visible[batch.observed_mask]):
            raise RuntimeError("visible input values changed")
        clean = batch.normalized_target * batch.query_std[:, None] + batch.query_mean[:, None]
        difference = (repaired - clean).double()
        for length in (96, 32):
            part = difference[:, -length:][~batch.observed_mask[:, -length:]]
            repair["sq" + str(length)] += part.square().sum().item()
            repair["abs" + str(length)] += part.abs().sum().item()
            repair["n" + str(length)] += part.numel()
        windows += len(batch.refs)
    if windows != len(protocol.refs[split]):
        raise RuntimeError("evaluation dropped windows")
    result = dict(mse=sq / count, mae=absolute / count, windows=windows, target_elements=count)
    for length in (96, 32):
        n = repair["n" + str(length)]
        result.update({f"repair{length}_mse": repair["sq" + str(length)] / n,
                       f"repair{length}_mae": repair["abs" + str(length)] / n,
                       f"hidden{length}_coordinates": n})
    return result


def preflight(args, protocol, profile, device, ident):
    model = construct(args, protocol, profile, device).eval()
    timestamps = load_timestamp_features(Path(protocol.config.data.datasets["Weather"].path))
    batch = next(iter_p4_batches(protocol, "train", batch_size=16, epoch=0)).to(device)
    before = state_identity(model)
    with torch.no_grad():
        output, repaired = model(batch, timestamps)
        changed = replace(batch, normalized_target=torch.full_like(batch.normalized_target, 12345),
                          forecast_target=torch.full_like(batch.forecast_target, -12345))
        replay, _ = model(changed, timestamps)
        if not torch.equal(output, replay):
            raise RuntimeError("supervision labels leaked into inputs")
        visible, mask, time = coifnet_inputs(batch, timestamps)
        alternative = repaired + (~batch.observed_mask).float() * .5
        perturbed = repaired_coifnet_output(model.coifnet, alternative, visible, mask, time)
        if torch.allclose(output[:, 96:], perturbed[:, 96:]):
            raise RuntimeError("repair values do not reach the forecast")
        model.coifnet.revin_layer(visible, "norm", mask)
        means = model.coifnet.revin_layer.mean.clone()
        neutral = torch.where(batch.observed_mask, visible, means.expand_as(visible))
        bridged = repaired_coifnet_output(model.coifnet, neutral, visible, mask, time)
        original = model.coifnet.impute(visible, mask, feat=time)
        if not torch.allclose(bridged, original, atol=2e-6, rtol=2e-6):
            raise RuntimeError("adapter changed the original visible-input path")
    model.train()
    prediction, _ = model(batch, timestamps)
    loss = author_coifnet_loss(prediction, visible, mask, batch.forecast_target.permute(0, 2, 1), weight=.2)
    loss.backward()
    if not torch.isfinite(loss) or any(p.grad is not None for p in model.repair.parameters()):
        raise RuntimeError("invalid gradient or repair gradients appeared")
    gradients = [p.grad for p in model.coifnet.parameters() if p.grad is not None]
    if not gradients or any(not torch.isfinite(g).all() for g in gradients):
        raise RuntimeError("CoIFNet has invalid gradients")
    torch.optim.Adam(model.coifnet.parameters(), lr=.001).step()
    if before != state_identity(model) or model.repair.training:
        raise RuntimeError("frozen repair, retrieval or memory changed")
    args.output.mkdir(parents=True, exist_ok=True)
    _atomic_json(dict(status="passed", identity=ident, labels_excluded=True,
        repaired_values_reach_forecast=True, observed_only_normalization=True,
        original_visible_path_parity=True, repair_state_unchanged=True,
        finite_coifnet_backward=True, output_shape=list(output.shape),
        smoke_weights_discarded=True), args.output / "acceptance.json")
    print(json.dumps(dict(status="preflight_passed", output=str(args.output))), flush=True)


def train(args, protocol, profile, device, ident):
    output = args.output
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    model = construct(args, protocol, profile, device)
    frozen = state_identity(model)
    initial = parameter_hash(model.coifnet)
    timestamps = load_timestamp_features(Path(protocol.config.data.datasets["Weather"].path))
    training = profile["training"]
    optimizer = torch.optim.Adam(model.coifnet.parameters(), lr=training["learning_rate"],
                                 weight_decay=training["weight_decay"])
    cache = validation_cache(model, protocol, device)
    best = float("inf"); stale = updates = 0; history = []; best_epoch = 0; best_validation = None
    for epoch in range(training["epochs"]):
        model.train()
        if model.repair.training or any(p.requires_grad for p in model.repair.parameters()):
            raise RuntimeError("repair is not frozen")
        total = len(protocol.refs["train"]) // training["batch_size"]
        for index, cpu in enumerate(iter_p4_batches(protocol, "train", batch_size=training["batch_size"], epoch=epoch), 1):
            batch = cpu.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction, _ = model(batch, timestamps)
            visible, mask, _ = coifnet_inputs(batch, timestamps)
            loss = author_coifnet_loss(prediction, visible, mask, batch.forecast_target.permute(0, 2, 1),
                                       weight=profile["model"]["loss_lambda"])
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite training loss")
            loss.backward(); optimizer.step(); updates += 1
            if any(p.grad is not None for p in model.repair.parameters()):
                raise RuntimeError("repair gradients appeared")
            if index == 1 or index == total:
                print(json.dumps(dict(stage="coifnet", epoch=epoch + 1, batch=index,
                    batches=total, loss=loss.item(), optimizer_updates=updates)), flush=True)
        metrics = score(model, protocol, "validation", device, timestamps, cache=cache)
        if state_identity(model) != frozen:
            raise RuntimeError("frozen repair state changed")
        if metrics["mae"] < best:
            best = metrics["mae"]; best_epoch = epoch + 1; best_validation = metrics; stale = 0
            _atomic_torch_save(dict(coifnet_state=model.coifnet.state_dict(), identity=ident,
                repair_state_identity=frozen, best_epoch=best_epoch, validation=metrics), output / "selected.pt")
        else:
            stale += 1
        row = dict(epoch=epoch + 1, validation=metrics, best_epoch=best_epoch, stale=stale, optimizer_updates=updates)
        history.append(row); _atomic_json(history, output / "history.json")
        print(json.dumps(row), flush=True)
        if stale >= training["patience"]:
            break
    record = dict(**ident, status="validation_finalized", stage="coifnet", training=training,
        epochs_completed=epoch + 1, best_epoch=best_epoch, best_validation=best_validation,
        optimizer_updates=updates, actual_training_batches_per_epoch=total,
        actual_training_windows_per_epoch=total * training["batch_size"],
        checkpoint=str((output / "selected.pt").resolve()), checkpoint_sha256=file_sha256(output / "selected.pt"),
        predictor_initial_sha256=initial, frozen_repair_identity=frozen, test_metrics_seen=False,
        stage1_reused=True, stage1_original_training_seconds=json.loads(args.parent.read_text())["training_seconds"],
        validation_repairs_cached=True, training_repairs_cached=False, repair_forward_chunk_size=128)
    _atomic_json(record, output / "selection.json")
    return output / "selection.json"


def evaluate(args, protocol, profile, device, ident):
    if args.output.exists():
        raise FileExistsError(args.output)
    selected = json.loads(args.selection.read_text())
    if any(selected.get(k) != v for k, v in ident.items()):
        raise ValueError("selected experiment identity changed")
    checkpoint = Path(selected["checkpoint"])
    if file_sha256(checkpoint) != selected["checkpoint_sha256"]:
        raise ValueError("selected checkpoint changed")
    model = construct(args, protocol, profile, device).eval()
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.coifnet.load_state_dict(payload["coifnet_state"], strict=True)
    before = state_identity(model), parameter_hash(model.coifnet)
    if before[0] != selected["frozen_repair_identity"]:
        raise ValueError("parent repair state changed")
    timestamps = load_timestamp_features(Path(protocol.config.data.datasets["Weather"].path))
    validation = score(model, protocol, "validation", device, timestamps)
    if any(abs(validation[k] - selected["best_validation"][k]) > 1e-6 for k in ("mse", "mae")):
        raise ValueError("selected validation replay differs")
    manifest = read_manifest(args.dataset_manifest); rows = []
    for item in (x for x in manifest["masks"] if x["split"] == "test"):
        current = with_test_mask(protocol, manifest, item)
        metrics = score(model, current, "test", device, timestamps)
        rows.append(dict(seed=item["seed"], mask_sha256=item["sha256"], **metrics))
        print(json.dumps(dict(split="test", **rows[-1])), flush=True)
    if before != (state_identity(model), parameter_hash(model.coifnet)):
        raise RuntimeError("test changed model state")
    keys = ("mse", "mae", "repair96_mse", "repair96_mae", "repair32_mse", "repair32_mae")
    summary = {k: dict(mean=float(np.mean([r[k] for r in rows])),
        mask_std_ddof1=float(np.std([r[k] for r in rows], ddof=1))) for k in keys}
    resources = selected_training_resources([args.selection])
    parent_resources = selected_training_resources([args.parent])
    result = dict(**ident, dataset="Weather", start_rate=.1, block_len=8, selection=str(args.selection),
        checkpoint_sha256=selected["checkpoint_sha256"], validation=validation, draws=rows,
        summary=summary, training_resources=resources, reused_stage1_resources=parent_resources,
        cached_stage1_plus_new_predictor=True,
        historical_total_training_seconds=parent_resources["full_workflow_seconds"] + resources["full_workflow_seconds"],
        repair_frozen_verified=True, model_state_unchanged=True, visible_values_exact=True,
        repair_metric_note="frozen E24-v79 input repair; not CoIFNet output reconstruction",
        training_seed_std=None, drr=None)
    args.output.mkdir(parents=True)
    _atomic_json(result, args.output / "twenty-test-draws.json")
    with (args.output / "per_mask.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    baselines = list(csv.DictReader((ROOT / "docs/experiments/Weather_sr010_bl8_E24_baselines_20261008.csv").open()))
    comparison = [{k: r[k] for k in ("model", "forecast_mse", "forecast_mae", "repair96_mse", "repair96_mae", "full_training_workflow_seconds")}
        for r in baselines if r["model"] in ("E24-v79", "T1", "CoIFNet") and r["primary"] == "True"]
    comparison.append(dict(model=MODEL, forecast_mse=summary["mse"]["mean"], forecast_mae=summary["mae"]["mean"],
        repair96_mse=summary["repair96_mse"]["mean"], repair96_mae=summary["repair96_mae"]["mean"],
        full_training_workflow_seconds=resources["full_workflow_seconds"]))
    for row in comparison:
        row["timing_note"] = "new CoIFNet training with cached Stage1; original Stage1 cost separately recorded" if row["model"] == MODEL else "original recorded workflow"
    with (args.output / "comparison.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison[0])); writer.writeheader(); writer.writerows(comparison)
    print(json.dumps(dict(status="completed", summary=summary, output=str(args.output))), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["preflight", "train", "evaluate"])
    parser.add_argument("--dataset-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--e24-config", type=Path, default=ROOT / "configs/e24_v79.yaml")
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    for name in ("dataset_manifest", "parent", "profiles", "e24_config"):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    torch.set_num_threads(4); device = torch.device(args.device)
    manifest = read_manifest(args.dataset_manifest)
    if manifest["start_rate"] != .1 or manifest["block_len"] != 8:
        raise ValueError("this first version is locked to Weather 0.10/8")
    protocol = load_dataset_protocol(args.dataset_manifest, config_path=Path(manifest["config_path"]), artifacts=args.dataset_manifest.parent)
    profile = load_author_profiles(args.profiles)["models"]["CoIFNet"]
    protocol = replace(protocol, training_seed=profile["seed"])
    ident = identity(args, protocol, profile)
    if args.action == "preflight":
        preflight(args, protocol, profile, device, ident)
    elif args.action == "train":
        with TrainingResources(args.output, device=device, batch_size=512, model=MODEL,
                               mode="training", started_at=PROCESS_STARTED) as resources:
            with resources.stage("coifnet"):
                resources.selection_manifest = train(args, protocol, profile, device, ident)
    else:
        if args.selection is None:
            raise ValueError("evaluation needs --selection")
        with TrainingResources(args.output, device=device, batch_size=128, model=MODEL,
                               mode="evaluation", started_at=PROCESS_STARTED):
            evaluate(args, protocol, profile, device, ident)


if __name__ == "__main__":
    main()
