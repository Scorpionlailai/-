"""Independent SAITS repair and forecast stages for the locked P4 comparison."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from time import monotonic
from typing import Sequence

import torch

from malr.config import config_sha256
from malr.models.patchtst import PatchTSTCore
from malr.models.repaired_forecast import MaskAwareForecastModel

from .protocol import P4Protocol, iter_p4_batches, load_p4_protocol
from .saits_model import SAITS_COMMIT, SAITS_SETTINGS, SaitsForecastModel, build_official_saits, impute_with_observed_clamp, saits_repair_loss


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _adapter_code_sha256() -> str:
    digest = sha256()
    for name in ("protocol.py", "saits_model.py"):
        path = Path(__file__).with_name(name)
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _predictor_settings(protocol: P4Protocol) -> dict:
    return {
        "seq_len": protocol.config.data.seq_len,
        "pred_len": protocol.config.data.pred_len,
        "patch_length": protocol.config.patch.length,
        "patch_stride": protocol.config.patch.stride,
        "profile": asdict(protocol.config.model.for_dataset("ETTh1")),
    }


def _atomic_torch_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@torch.no_grad()
def _repair_validation_mae(model, protocol: P4Protocol, *, device: torch.device, epoch: int) -> float:
    del epoch
    model.eval()
    absolute = 0.0
    count = 0
    batches = iter_p4_batches(protocol, "validation", batch_size=128)
    for index, cpu_batch in enumerate(batches, 1):
        batch = cpu_batch.to(device)
        repaired = impute_with_observed_clamp(model, batch)
        hidden = ~batch.observed_mask
        absolute += (repaired - batch.normalized_target).abs()[hidden].sum().item()
        count += hidden.sum().item()
        if index % 5 == 0:
            print(f"stage=repair validation batch={index}/21", flush=True)
    if count == 0:
        raise ValueError("validation has no hidden coordinates")
    return absolute / count


def train_repair(
    protocol: P4Protocol,
    *,
    output: Path,
    repo_root: Path,
    device: torch.device,
    epochs: int = 100,
) -> Path:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    output = Path(output)
    torch.manual_seed(2021)
    model = build_official_saits(repo_root, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
    checkpoint = output / "checkpoints" / "saits-stage1.pt"
    manifest_path = output / "runs" / "saits-stage1.json"
    best_mae = float("inf")
    best_epoch = 0
    started = monotonic()
    train_total = len(protocol.refs["train"]) // 128 if hasattr(protocol, "refs") else 1
    for epoch in range(epochs):
        model.train()
        for index, cpu_batch in enumerate(iter_p4_batches(protocol, "train", batch_size=128, epoch=epoch), 1):
            batch = cpu_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = saits_repair_loss(model, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite SAITS repair loss")
            loss.backward()
            optimizer.step()
            if index == 1 or index % 5 == 0 or index == train_total:
                print(f"stage=repair epoch={epoch + 1}/{epochs} batch={index}/{train_total} loss={loss.item():.6f}", flush=True)
        val_mae = _repair_validation_mae(model, protocol, device=device, epoch=epoch)
        if val_mae < best_mae:
            best_mae = val_mae
            best_epoch = epoch + 1
            _atomic_torch_save({"model_state": model.state_dict(), "epoch": best_epoch,
                                "validation_imputation_mae": best_mae}, checkpoint)
        print(f"stage=repair epoch={epoch + 1}/{epochs} val_imputation_mae={val_mae:.8f} best_epoch={best_epoch}", flush=True)
    if best_epoch == 0:
        raise RuntimeError("no valid Stage 1 checkpoint was selected")
    manifest = {
        "stage": "repair", "best_epoch": best_epoch,
        "epochs_completed": epochs, "best_val_imputation_mae": best_mae,
        "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": _sha256(checkpoint),
        "source_csv_sha256": protocol.source_sha256,
        "validation_mask_sha256": protocol.mask_sha256["validation"],
        "test_mask_sha256": protocol.mask_sha256["test"],
        "config_path": str(protocol.config_path), "upstream_commit": SAITS_COMMIT,
        "model_config_sha256": config_sha256(protocol.config),
        "adapter_code_sha256": _adapter_code_sha256(),
        "settings": SAITS_SETTINGS, "batch_size": 128, "learning_rate": 1e-3,
        "weight_decay": 0, "seed": 2021, "training_seconds": monotonic() - started,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "selection": "lowest fixed-validation hidden-coordinate imputation MAE",
    }
    _atomic_json(manifest, manifest_path)
    return manifest_path


def _read_selected_manifest(path: Path, *, stage: str) -> dict:
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if record.get("stage") != stage:
        raise ValueError(f"expected selected {stage} manifest")
    checkpoint = Path(record["checkpoint"])
    if not checkpoint.is_file() or _sha256(checkpoint) != record.get("checkpoint_sha256"):
        raise ValueError("selected checkpoint SHA-256 mismatch")
    if int(record.get("best_epoch", 0)) <= 0:
        raise ValueError("selected manifest has no validation-selected epoch")
    return record


def _verify_protocol_identity(record: dict, protocol: P4Protocol) -> None:
    if record.get("model_config_sha256") != config_sha256(protocol.config):
        raise ValueError("selected model config SHA-256 mismatch")
    if record.get("adapter_code_sha256") != _adapter_code_sha256():
        raise ValueError("selected adapter code SHA-256 mismatch")
    if record.get("settings") != SAITS_SETTINGS:
        raise ValueError("selected SAITS architecture settings mismatch")
    if record.get("stage") == "forecast" and record.get("predictor_settings") != _predictor_settings(protocol):
        raise ValueError("selected predictor architecture settings mismatch")
    if record.get("source_csv_sha256") != protocol.source_sha256:
        raise ValueError("selected source CSV SHA-256 mismatch")
    for split in ("validation", "test"):
        if record.get(f"{split}_mask_sha256") != protocol.mask_sha256[split]:
            raise ValueError(f"selected {split} mask SHA-256 mismatch")


def _build_forecast_model(protocol: P4Protocol, repo_root: Path, device: torch.device) -> SaitsForecastModel:
    profile = protocol.config.model.for_dataset("ETTh1")
    core = PatchTSTCore(
        c_in=7, seq_len=96, pred_len=96, patch_len=16, stride=8,
        d_model=profile.d_model, n_heads=profile.n_heads, d_ff=profile.d_ff,
        n_layers=profile.encoder_layers, dropout=profile.dropout,
        fc_dropout=profile.fc_dropout, head_dropout=profile.head_dropout,
        attn_dropout=profile.attn_dropout,
    )
    predictor = MaskAwareForecastModel(core)
    imputer = build_official_saits(repo_root, device=device)
    return SaitsForecastModel(imputer, predictor).to(device)


@torch.no_grad()
def _forecast_metrics(model: SaitsForecastModel, protocol: P4Protocol, *, split: str,
                      device: torch.device) -> dict[str, float | int]:
    model.eval()
    squared = absolute = 0.0
    count = windows = 0
    total = len(protocol.refs[split]) // 128
    for index, cpu_batch in enumerate(iter_p4_batches(protocol, split, batch_size=128), 1):
        batch = cpu_batch.to(device)
        pred = model(batch)
        if pred.shape != batch.forecast_target.shape:
            raise ValueError("forecast output shape mismatch")
        error = pred - batch.forecast_target
        squared += error.square().sum().item()
        absolute += error.abs().sum().item()
        count += error.numel()
        windows += error.shape[0]
        if index == 1 or index % 5 == 0 or index == total:
            print(f"stage=forecast split={split} batch={index}/{total}", flush=True)
    if count == 0:
        raise ValueError(f"{split} has no evaluation targets")
    return {"mse": squared / count, "mae": absolute / count,
            "windows": windows, "target_elements": count}


def train_forecast(
    protocol: P4Protocol,
    *,
    stage1_manifest: Path,
    output: Path,
    repo_root: Path,
    device: torch.device,
    epochs: int = 100,
) -> Path:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    stage1 = _read_selected_manifest(stage1_manifest, stage="repair")
    _verify_protocol_identity(stage1, protocol)
    torch.manual_seed(2021)
    model = _build_forecast_model(protocol, repo_root, device)
    stage1_payload = torch.load(stage1["checkpoint"], map_location=device, weights_only=False)
    model.imputer.load_state_dict(stage1_payload["model_state"], strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)
    output = Path(output)
    checkpoint = output / "checkpoints" / "saits-stage2.pt"
    manifest_path = output / "runs" / "saits-stage2.json"
    best_mse, best_mae, best_epoch = float("inf"), float("inf"), 0
    train_total = len(protocol.refs["train"]) // 128
    started = monotonic()
    for epoch in range(epochs):
        model.train()
        for index, cpu_batch in enumerate(iter_p4_batches(protocol, "train", batch_size=128, epoch=epoch), 1):
            batch = cpu_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            forecast = model(batch)
            loss = (forecast - batch.forecast_target).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite SAITS forecast loss")
            loss.backward()
            optimizer.step()
            if index == 1 or index % 5 == 0 or index == train_total:
                print(f"stage=forecast epoch={epoch + 1}/{epochs} batch={index}/{train_total} mse={loss.item():.6f}", flush=True)
        validation = _forecast_metrics(model, protocol, split="validation", device=device)
        if validation["mse"] < best_mse:
            best_mse, best_mae, best_epoch = float(validation["mse"]), float(validation["mae"]), epoch + 1
            _atomic_torch_save({"model_state": model.state_dict(), "epoch": best_epoch,
                                "validation_mse": best_mse}, checkpoint)
        print(f"stage=forecast epoch={epoch + 1}/{epochs} val_mse={validation['mse']:.8f} val_mae={validation['mae']:.8f} best_epoch={best_epoch}", flush=True)
    if best_epoch == 0:
        raise RuntimeError("no valid Stage 2 checkpoint was selected")
    manifest = {
        "stage": "forecast", "best_epoch": best_epoch, "epochs_completed": epochs,
        "best_val_mse": best_mse, "best_val_mae": best_mae,
        "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": _sha256(checkpoint),
        "stage1_manifest": str(Path(stage1_manifest).resolve()),
        "stage1_manifest_sha256": _sha256(Path(stage1_manifest)),
        "stage1_checkpoint_sha256": stage1["checkpoint_sha256"],
        "source_csv_sha256": protocol.source_sha256,
        "validation_mask_sha256": protocol.mask_sha256["validation"],
        "test_mask_sha256": protocol.mask_sha256["test"],
        "config_path": str(protocol.config_path), "saits_root": str(Path(repo_root).resolve()),
        "upstream_commit": SAITS_COMMIT, "settings": SAITS_SETTINGS,
        "model_config_sha256": config_sha256(protocol.config),
        "adapter_code_sha256": _adapter_code_sha256(),
        "predictor_settings": _predictor_settings(protocol),
        "batch_size": 128, "learning_rate": 1e-4, "weight_decay": 0, "seed": 2021,
        "loss": "forecast_mse_only", "optimizer_state_restored": False,
        "training_seconds": monotonic() - started,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "selection": "lowest fixed-validation forecast MSE",
    }
    _atomic_json(manifest, manifest_path)
    return manifest_path


def evaluate_selected(protocol: P4Protocol, *, selection: Path, output: Path,
                      device: torch.device) -> dict[str, object]:
    selection = Path(selection)
    record = _read_selected_manifest(selection, stage="forecast")
    _verify_protocol_identity(record, protocol)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"evaluation already exists: {output}")
    model = _build_forecast_model(protocol, Path(record["saits_root"]), device)
    payload = torch.load(record["checkpoint"], map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"], strict=True)
    result = {
        "model": "SAITS-adapted-to-P4 + same-structure mask-aware predictor",
        "stage": "forecast", "split": "test", "metric_space": "train_standardized",
        "selection_manifest": str(selection.resolve()),
        "selection_manifest_sha256": _sha256(selection),
        "checkpoint_sha256": record["checkpoint_sha256"],
        "model_config_sha256": record["model_config_sha256"],
        "adapter_code_sha256": record["adapter_code_sha256"],
        "upstream_commit": record["upstream_commit"],
        "source_csv_sha256": protocol.source_sha256,
        "validation_mask_sha256": protocol.mask_sha256["validation"],
        "test_mask_sha256": protocol.mask_sha256["test"],
        "seed": 2021, "best_epoch": record["best_epoch"],
        "best_val_mse": record["best_val_mse"],
        "trainable_parameters": record["trainable_parameters"],
        "training_seconds": record["training_seconds"],
        **_forecast_metrics(model, protocol, split="test", device=device),
    }
    if result["windows"] != 2688:
        raise ValueError(f"expected exactly 2688 test windows, got {result['windows']}")
    _atomic_json(result, output)
    print(json.dumps(result, indent=2), flush=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("train-repair", "train-forecast", "evaluate"):
        command = sub.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--mask-artifacts", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--device", default="cuda")
        if name != "evaluate":
            command.add_argument("--saits-root", type=Path, required=True)
        if name == "train-forecast":
            command.add_argument("--stage1-manifest", type=Path, required=True)
        if name == "evaluate":
            command.add_argument("--selection", type=Path, required=True)
    args = parser.parse_args(argv)
    protocol = load_p4_protocol(args.config, args.mask_artifacts)
    if args.command == "train-repair":
        manifest = train_repair(protocol, output=args.output, repo_root=args.saits_root,
                                device=torch.device(args.device))
        print(json.dumps({"manifest": str(manifest)}, indent=2), flush=True)
    elif args.command == "train-forecast":
        manifest = train_forecast(protocol, stage1_manifest=args.stage1_manifest,
                                  output=args.output, repo_root=args.saits_root,
                                  device=torch.device(args.device))
        print(json.dumps({"manifest": str(manifest)}, indent=2), flush=True)
    else:
        evaluate_selected(protocol, selection=args.selection, output=args.output,
                          device=torch.device(args.device))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
