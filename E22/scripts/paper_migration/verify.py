"""Read-only acceptance checks for the curated paper workspace migration."""

from dataclasses import asdict, dataclass
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess

from malr.migration import MigrationCatalog
from .baseline import GIT, UPSTREAM, verify_baseline
from .inventory import build_inventory, file_sha256, inventory_digest
from .selection import SelectedRun, load_selection


OLD_PROJECT = Path("/home/xyw/missing_aware_latent_recovery")
OLD_CHECKOUT = OLD_PROJECT / ".superpowers/sdd/2026-08-31-residual-mlp-recovery/checkout"
SOURCE_OVERLAYS = frozenset({
    "E22/src/malr/cli.py", "E22/integrations/forecast_compare/protocol.py",
    "E23/src/malr/cli.py", "E23/integrations/forecast_compare/protocol.py",
    "S4M/bridge/integrations/forecast_compare/s4m_p4.py",
    "S4M/bridge/integrations/s4m/verify_upstream.py",
})


@dataclass(frozen=True)
class AcceptanceReport:
    accepted: bool
    expected_files: int
    expected_bytes: int
    verified_files: int
    verified_bytes: int
    inventory_digest: str
    overlays: tuple[str, ...]
    metrics: tuple[dict, ...]
    git: dict[str, dict]
    baselines: dict[str, dict]
    errors: tuple[str, ...]


def _source(reference: str, checkout: Path, project: Path) -> Path:
    prefix, separator, relative = reference.partition(":")
    if separator != ":" or prefix not in {"checkout", "project"}:
        raise ValueError(f"invalid selection reference: {reference}")
    return (checkout if prefix == "checkout" else project) / relative


def _ledger_metrics(paper_root: Path) -> dict[str, dict]:
    values = {}
    for csv_path in (paper_root / "E22/docs/experiments").glob("*missing_scenarios.csv"):
        with csv_path.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                source = row.get("Source JSON", "")
                if source:
                    values[source] = row
    return values


def _git_state(project: Path) -> dict:
    def run(*args: str) -> str:
        result = subprocess.run([GIT, "-C", str(project), *args], capture_output=True, text=True)
        if result.returncode:
            raise ValueError(result.stderr.strip())
        return result.stdout.strip()
    tracked = run("ls-files", "-z").split("\0")
    bad = [name for name in tracked if name and (name.startswith("artifacts/") or name.startswith("data/"))]
    if bad:
        raise ValueError(f"binary artifact/data tracked: {bad[:3]}")
    return {"head": run("rev-parse", "HEAD"), "branch": run("branch", "--show-current"),
            "tracked_paths": len([name for name in tracked if name]), "status": run("status", "--short")}


def _checked_source_overlay(project: Path, relative: str, original_sha256: str) -> None:
    """Require the original seed and current overlay to match committed blobs."""
    revision_range = (
        ["rev-list", "--reverse", f"{UPSTREAM['S4M']}..HEAD"]
        if project.name == "S4M" else ["rev-list", "--max-parents=0", "HEAD"]
    )
    first = subprocess.run([GIT, "-C", str(project), *revision_range],
                           capture_output=True, text=True, check=True).stdout.splitlines()[0]
    for revision, expected in ((first, original_sha256), ("HEAD", file_sha256(project / relative))):
        result = subprocess.run([GIT, "-C", str(project), "show", f"{revision}:{relative}"],
                                capture_output=True)
        if result.returncode:
            raise ValueError(f"source overlay is not tracked: {relative}")
        if hashlib.sha256(result.stdout).hexdigest() != expected:
            if revision == "HEAD":
                raise ValueError(f"uncommitted source overlay: {relative}")
            raise ValueError(f"original source seed mismatch: {relative}")


def _check_selected_lineage(
    run: SelectedRun, manifest_path: Path, evaluation: dict, checkpoint: Path,
    catalog: MigrationCatalog,
) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("run identity: manifest is not an object")
    manifest_run_id = manifest.get("run_id", manifest_path.stem if run.kind == "baseline_saits" else None)
    evaluation_run_id = evaluation.get(
        "source_run_id" if run.kind == "zero_shot" else "run_id",
        run.run_id if run.kind == "baseline_saits" else None,
    )
    if manifest_run_id != run.run_id or evaluation_run_id != run.run_id:
        raise ValueError("run identity mismatch between selection, manifest and evaluation")
    if manifest.get("dataset", run.dataset) != run.dataset or evaluation.get("dataset", run.dataset) != run.dataset:
        raise ValueError("run identity dataset mismatch")
    if run.kind == "zero_shot" and (evaluation.get("kind") != "zero_shot_missing_stress" or evaluation.get("split") != "test"):
        raise ValueError("run identity zero-shot record mismatch")
    if catalog.resolve(manifest.get("checkpoint", ""), expected_sha256=manifest.get("checkpoint_sha256")) != checkpoint:
        raise ValueError("checkpoint lineage differs from selected checkpoint")
    checkpoint_hash = file_sha256(checkpoint)
    if manifest.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("checkpoint lineage manifest SHA-256 mismatch")
    evaluation_hash_key = "source_checkpoint_sha256" if run.kind == "zero_shot" else "checkpoint_sha256"
    if evaluation.get(evaluation_hash_key) != checkpoint_hash:
        raise ValueError("checkpoint lineage evaluation SHA-256 mismatch")
    if evaluation.get("checkpoint") and catalog.resolve(evaluation["checkpoint"]) != checkpoint:
        raise ValueError("checkpoint lineage evaluation path mismatch")
    if evaluation.get("selection_manifest"):
        if catalog.resolve(evaluation["selection_manifest"], expected_sha256=evaluation.get("selection_manifest_sha256")) != manifest_path:
            raise ValueError("checkpoint lineage selection manifest mismatch")
    if run.kind in {"trained", "retrained", "zero_shot"}:
        stage1 = manifest.get("stage1_checkpoint")
        if not stage1:
            raise ValueError("checkpoint lineage lacks Stage 1")
        catalog.resolve(stage1, expected_sha256=manifest.get("stage1_checkpoint_sha256"))
        metadata = manifest.get("metadata", {})
        if metadata.get("stage1_checkpoint") and metadata["stage1_checkpoint"] != stage1:
            raise ValueError("checkpoint lineage Stage 1 metadata mismatch")
        if metadata.get("stage1_checkpoint_sha256") and metadata["stage1_checkpoint_sha256"] != manifest.get("stage1_checkpoint_sha256"):
            raise ValueError("checkpoint lineage Stage 1 hash mismatch")
        for warm_start in (metadata.get("stage2_warm_start"), manifest.get("training", {}).get("stage2_warm_start")):
            if warm_start:
                catalog.resolve(warm_start["checkpoint"], expected_sha256=warm_start["checkpoint_sha256"])
    if run.kind == "baseline_saits":
        stage1_manifest = catalog.resolve(
            manifest["stage1_manifest"], expected_sha256=manifest.get("stage1_manifest_sha256"),
        )
        stage1_record = json.loads(stage1_manifest.read_text(encoding="utf-8"))
        if stage1_record.get("checkpoint_sha256") != manifest.get("stage1_checkpoint_sha256"):
            raise ValueError("checkpoint lineage SAITS Stage 1 hash mismatch")
        catalog.resolve(stage1_record["checkpoint"], expected_sha256=stage1_record["checkpoint_sha256"])


def verify_migration(
    *, paper_root: Path, inventory: Path, selection: Path,
    source_checkout: Path = OLD_CHECKOUT, old_project: Path = OLD_PROJECT,
    require_git: bool = True,
) -> AcceptanceReport:
    paper_root = Path(paper_root).resolve()
    selected = load_selection(selection)
    expected = build_inventory(selected, source_checkout=source_checkout, old_project=old_project)
    rows = {row["destination_relative"]: row for row in (
        json.loads(line) for line in Path(inventory).read_text(encoding="utf-8").splitlines()
    )}
    errors: list[str] = []
    overlays: list[str] = []
    verified_files = verified_bytes = 0
    for entry in expected:
        relative = entry.destination_relative
        row = rows.get(relative)
        if row is None:
            errors.append(f"catalog missing: {relative}")
            continue
        if row["source"] != str(entry.source) or row["size"] != entry.size or row["sha256"] != entry.sha256:
            errors.append(f"catalog inventory mismatch: {relative}")
            continue
        destination = paper_root / relative
        if not destination.is_file() or destination.is_symlink():
            errors.append(f"destination missing or unsafe: {relative}")
            continue
        if destination.stat().st_size != entry.size or file_sha256(destination) != entry.sha256:
            if relative in SOURCE_OVERLAYS and require_git:
                try:
                    project_name, source_relative = relative.split("/", 1)
                    _checked_source_overlay(paper_root / project_name, source_relative, entry.sha256)
                    overlays.append(relative)
                except (ValueError, OSError, subprocess.CalledProcessError) as exc:
                    errors.append(f"unverified source overlay {relative}: {exc}")
                continue
            errors.append(f"destination SHA-256 mismatch: {relative}")
            continue
        verified_files += 1
        verified_bytes += entry.size
    unexpected = set(rows) - {entry.destination_relative for entry in expected}
    if unexpected:
        errors.append(f"catalog contains {len(unexpected)} unselected entries")
    catalog = MigrationCatalog.load(inventory)
    ledgers = _ledger_metrics(paper_root)
    metrics: list[dict] = []
    for run in selected.runs:
        try:
            manifest = catalog.resolve(_source(run.manifest, source_checkout, old_project))
            evaluation = catalog.resolve(_source(run.evaluation, source_checkout, old_project))
            checkpoint = catalog.resolve(_source(run.checkpoint, source_checkout, old_project))
            for dependency in run.dependencies:
                catalog.resolve(_source(dependency, source_checkout, old_project))
            if run.kind not in {"baseline_saits", "baseline_s4m"}:
                experiment = evaluation.parent.parent
                for split in ("validation", "test"):
                    catalog.resolve(experiment / "masks" / f"{run.dataset}_{split}.npz")
            record = json.loads(evaluation.read_text(encoding="utf-8"))
            if not isinstance(record, dict):
                raise ValueError("evaluation is not an object")
            _check_selected_lineage(run, manifest, record, checkpoint, catalog)
            if record.get("metric_space") != "train_standardized":
                raise ValueError("evaluation metric space mismatch")
            item = {"dataset": run.dataset, "scenario": run.scenario, "kind": run.kind,
                    "run_id": run.run_id, "manifest": str(manifest.relative_to(paper_root)),
                    "checkpoint": str(checkpoint.relative_to(paper_root)),
                    "evaluation": str(evaluation.relative_to(paper_root)),
                    "metric_space": record.get("metric_space", "absent")}
            for metric in ("mse", "mae", "drr_mse", "drr_mae"):
                value = record.get(metric)
                if value is None:
                    item[metric] = "absent"
                elif isinstance(value, (int, float)) and math.isfinite(value):
                    item[metric] = value
                else:
                    raise ValueError(f"invalid {metric}")
            if item["mse"] == "absent" or item["mae"] == "absent":
                raise ValueError("evaluation lacks MSE/MAE")
            source_key = run.evaluation.removeprefix("checkout:")
            if source_key in ledgers:
                ledger = ledgers[source_key]
                for metric in ("mse", "mae"):
                    if abs(float(ledger[metric.upper()]) - item[metric]) > 1e-6:
                        raise ValueError(f"ledger {metric} mismatch")
                for metric, column in (("drr_mse", "DRR-MSE (%)"), ("drr_mae", "DRR-MAE (%)")):
                    item[f"ledger_{metric}"] = float(ledger[column]) if ledger.get(column) else "absent"
                    if item[metric] != "absent" and item[f"ledger_{metric}"] != "absent":
                        if abs(item[metric] - item[f"ledger_{metric}"]) > 1e-6:
                            raise ValueError(f"ledger {metric} mismatch")
            metrics.append(item)
        except (ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
            errors.append(f"selected run {run.dataset}/{run.scenario}/{run.kind}: {exc}")
    git: dict[str, dict] = {}
    baselines: dict[str, dict] = {}
    if require_git:
        for name in ("E22", "E23", "SAITS", "S4M"):
            try:
                git[name] = _git_state(paper_root / name)
            except ValueError as exc:
                errors.append(f"Git {name}: {exc}")
        for name in ("SAITS", "S4M"):
            try:
                baselines[name] = asdict(verify_baseline(name, project=paper_root / name, catalog=catalog))
            except ValueError as exc:
                errors.append(f"baseline {name}: {exc}")
    accepted = not errors and verified_files + len(overlays) == len(expected) and len(metrics) == len(selected.runs)
    return AcceptanceReport(accepted, len(expected), sum(entry.size for entry in expected),
                            verified_files, verified_bytes, inventory_digest(expected),
                            tuple(overlays), tuple(metrics), git, baselines, tuple(errors))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-root", type=Path, default=Path("/home/xyw/paper"))
    parser.add_argument("--inventory", type=Path, default=Path("migration/paper_catalog.jsonl"))
    parser.add_argument("--selection", type=Path, default=Path("migration/paper_selection.json"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify_migration(paper_root=args.paper_root, inventory=args.inventory, selection=args.selection)
    payload = json.dumps(asdict(report), ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload)
    return 0 if report.accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
