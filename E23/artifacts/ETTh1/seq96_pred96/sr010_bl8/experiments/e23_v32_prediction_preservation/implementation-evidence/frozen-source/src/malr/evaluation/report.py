from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path
import tempfile
from typing import Any


@dataclass(frozen=True)
class ResultRecord:
    run_id: str
    dataset: str
    variant: str
    ablations: dict[str, bool]
    mse: float
    mae: float
    drr_mse: float | None
    drr_mae: float | None
    final_rate: float
    trainable_parameters: int
    training_seconds: float
    metric_space: str
    config_hash: str
    checkpoint_sha256: str
    mask_sha256: str | None
    bank_sha256: str | None
    source_csv_sha256: str
    seed: int
    timestamp_utc: str
    degradation_mse_absolute: float | None = None
    degradation_mae_absolute: float | None = None
    degradation_mse_relative_percent: float | None = None
    degradation_mae_relative_percent: float | None = None
    clean_baseline_fingerprint: str | None = None
    missing_baseline_fingerprint: str | None = None
    online_memory_sha256: str | None = None
    online_ema_sha256: str | None = None
    online_relation_prior_sha256: str | None = None
    online_diagnostics: dict[str, float] = field(default_factory=dict)
    inference_ablations: dict[str, dict[str, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.run_id or not self.dataset or not self.variant:
            raise ValueError("result identity fields cannot be empty")
        if self.metric_space != "train_standardized":
            raise ValueError("result metric_space must equal train_standardized")
        if self.mse < 0 or self.mae < 0 or not 0.0 <= self.final_rate <= 1.0:
            raise ValueError("result metrics or final_rate are invalid")
        if self.trainable_parameters <= 0 or self.training_seconds < 0:
            raise ValueError("training provenance is invalid")
        for key, value in self.online_diagnostics.items():
            if not isinstance(key, str) or not key or not isinstance(value, (int, float)) or not isfinite(value):
                raise ValueError("online diagnostics must have non-empty keys and finite values")
        for name, metric in self.inference_ablations.items():
            if not isinstance(name, str) or not name or set(metric) != {"mse", "mae"}:
                raise ValueError("inference ablations must contain named mse/mae pairs")
            if any(not isinstance(value, (int, float)) or not isfinite(value) or value < 0.0 for value in metric.values()):
                raise ValueError("inference ablation metrics must be finite and non-negative")


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    )


def _display(value: Any) -> Any:
    return "N/A" if value is None else value


def _write_summaries(directory: Path) -> None:
    records = []
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if "run_id" in payload and "mse" in payload and "mae" in payload:
            records.append(payload)
    columns = [
        "run_id",
        "dataset",
        "variant",
        "mse",
        "mae",
        "degradation_mse_absolute",
        "degradation_mae_absolute",
        "degradation_mse_relative_percent",
        "degradation_mae_relative_percent",
        "drr_mse",
        "drr_mae",
        "final_rate",
        "trainable_parameters",
        "training_seconds",
    ]
    lines: list[str] = []
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for record in records:
        writer.writerow({key: _display(record.get(key)) for key in columns})
    _atomic_text(directory / "summary.csv", buffer.getvalue())

    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for record in records:
        lines.append(
            "| "
            + " | ".join(str(_display(record.get(key))) for key in columns)
            + " |"
        )
    _atomic_text(directory / "summary.md", "\n".join(lines) + "\n")


def write_result_record(record: ResultRecord, path: Path) -> None:
    """Commit a final test result exactly once for a finalized run id."""

    if path.exists():
        raise ValueError(f"run {record.run_id!r} has already been evaluated")
    write_json_atomic(path, asdict(record))
    _write_summaries(path.parent)
