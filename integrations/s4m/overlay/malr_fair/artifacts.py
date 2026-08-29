from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import tempfile
from typing import Any

import torch
from torch import Tensor


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    ) + "\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(content)
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


class MetricAccumulator:
    def __init__(self) -> None:
        self.sum_squared_error = 0.0
        self.sum_absolute_error = 0.0
        self.element_count = 0

    def update(self, prediction: Tensor, target: Tensor) -> None:
        predicted = prediction.detach()
        actual = target.detach()
        if predicted.shape != actual.shape or predicted.numel() == 0:
            raise ValueError("prediction and target must have the same non-empty shape")
        if not torch.isfinite(predicted).all() or not torch.isfinite(actual).all():
            raise ValueError("prediction and target must be finite")
        error = predicted.to(dtype=torch.float64) - actual.to(dtype=torch.float64)
        self.sum_squared_error += float(error.square().sum().cpu())
        self.sum_absolute_error += float(error.abs().sum().cpu())
        self.element_count += int(error.numel())

    def compute(self) -> dict[str, float]:
        if self.element_count <= 0:
            raise ValueError("cannot compute metrics without elements")
        return {
            "mse": self.sum_squared_error / self.element_count,
            "mae": self.sum_absolute_error / self.element_count,
        }


def write_training_manifest(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise ValueError(f"training manifest already exists: {path}")
    if payload.get("status") != "validation_finalized":
        raise ValueError("training manifest status must be validation_finalized")
    if not payload.get("run_id"):
        raise ValueError("training manifest requires run_id")
    validation_mse = float(payload.get("validation_mse", -1.0))
    if validation_mse < 0.0:
        raise ValueError("training manifest requires non-negative validation_mse")
    _write_json_atomic(path, payload)


def authorize_test(
    selection_path: Path,
    run_id: str,
    checkpoint: Path,
) -> dict[str, Any]:
    try:
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid S4M selection file: {selection_path}") from exc
    selected = str(selection.get("selected_run_id", ""))
    if run_id != selected:
        raise ValueError(f"run {run_id!r} is not the selected run {selected!r}")
    if bool(selection.get("test_metrics_seen", False)):
        raise ValueError("selection already contains test metrics")
    expected_path = Path(str(selection.get("checkpoint", ""))).resolve()
    actual_path = checkpoint.resolve()
    if actual_path != expected_path:
        raise ValueError(
            f"selected checkpoint mismatch: expected {expected_path}, got {actual_path}"
        )
    if not actual_path.is_file():
        raise ValueError(f"selected checkpoint is missing: {actual_path}")
    actual_digest = file_sha256(actual_path)
    expected_digest = str(selection.get("checkpoint_sha256", ""))
    if actual_digest != expected_digest:
        raise ValueError(
            f"selected checkpoint SHA-256 mismatch: expected {expected_digest}, "
            f"got {actual_digest}"
        )
    return selection


def write_evaluation_result(
    path: Path,
    payload: dict[str, Any],
    *,
    authorized: bool,
) -> None:
    if not authorized:
        raise ValueError("test result was not authorized by selection.json")
    if path.exists():
        raise ValueError(f"evaluation result already exists: {path}")
    if payload.get("status") != "test_finalized":
        raise ValueError("evaluation result status must be test_finalized")
    _write_json_atomic(path, payload)


def environment_identity() -> dict[str, object]:
    gpu_name = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": gpu_name,
    }
