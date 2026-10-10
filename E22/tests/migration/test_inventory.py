import hashlib
import json
from pathlib import Path

import pytest

from scripts.paper_migration.inventory import build_inventory
from scripts.paper_migration.selection import load_selection


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_inventory_rejects_ambiguous_or_missing_lineage(tmp_path: Path) -> None:
    checkout = tmp_path / "old" / "checkout"
    run = checkout / "artifacts/ETTh1/main/runs/ours.json"
    _write_json(run, {"run_id": "ours", "checkpoint": "missing.pt"})
    selection_path = tmp_path / "selection.json"
    selected = {
        "schema_version": 1,
        "copy_roots": [{"source": "checkout:artifacts/ETTh1/main", "destination": "E22/artifacts/ETTh1/main", "owner": "E22"}],
        "runs": [{"dataset": "ETTh1", "scenario": "sr010_bl8", "kind": "trained", "run_id": "ours", "manifest": "checkout:artifacts/ETTh1/main/runs/ours.json", "evaluation": "checkout:artifacts/ETTh1/main/evaluations/ours.json", "checkpoint": "checkout:artifacts/ETTh1/main/checkpoints/missing.pt"}],
    }
    _write_json(selection_path, selected)
    selection = load_selection(selection_path)
    with pytest.raises(ValueError, match="missing selected"):
        build_inventory(selection, source_checkout=checkout, old_project=checkout.parent)

    selected["runs"].append(dict(selected["runs"][0]))
    _write_json(selection_path, selected)
    with pytest.raises(ValueError, match="duplicate selected run"):
        load_selection(selection_path)


def test_inventory_excludes_old_trials_and_source_mutation(tmp_path: Path) -> None:
    checkout = tmp_path / "old" / "checkout"
    source = checkout / "src/malr/model.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"E22 model\n")
    old_trial = checkout / "artifacts/E01/checkpoint.pt"
    old_trial.parent.mkdir(parents=True, exist_ok=True)
    old_trial.write_bytes(b"old model")
    failed_job = checkout / "src/jobs/failed/console.log"
    failed_job.parent.mkdir(parents=True, exist_ok=True)
    failed_job.write_bytes(b"failed")
    selected = {
        "schema_version": 1,
        "copy_roots": [{"source": "checkout:src", "destination": "E22/src", "owner": "E22", "dataset": "ETTh1", "scenario": "sr010_bl8", "run_id": "ours"}],
        "runs": [],
    }
    selection_path = tmp_path / "selection.json"
    _write_json(selection_path, selected)
    before = source.stat().st_mtime_ns
    entries = build_inventory(load_selection(selection_path), source_checkout=checkout, old_project=checkout.parent)
    assert [(entry.destination_relative, entry.sha256) for entry in entries] == [
        ("E22/src/malr/model.py", hashlib.sha256(b"E22 model\n").hexdigest())
    ]
    assert source.stat().st_mtime_ns == before
    assert old_trial.read_bytes() == b"old model"
    assert all("jobs" not in entry.destination_relative for entry in entries)
    assert (entries[0].dataset, entries[0].scenario, entries[0].run_id) == ("ETTh1", "sr010_bl8", "ours")


def test_e23_selection_uses_e22_seed_and_only_current_design_documents() -> None:
    selection = load_selection(Path(__file__).parents[2] / "migration/paper_selection.json")
    e23_roots = [item for item in selection.copy_roots if item.owner == "E23"]
    destinations = {item.destination for item in e23_roots}
    assert "E23/src" in destinations
    assert "E23/configs" in destinations
    assert "E23/docs/superpowers/specs/2026-09-29-e23-multi-expert-explicit-recovery-design.md" in destinations
    assert "E23/docs/superpowers/plans/2026-09-29-e23-multi-expert-explicit-recovery.md" in destinations
    assert all("E23/docs/experiments/" not in path for path in destinations)
