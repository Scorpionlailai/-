import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from scripts.paper_migration.verify import _checked_source_overlay, verify_migration


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fixture(tmp_path: Path):
    source = tmp_path / "old"
    experiment = source / "artifacts" / "ETTh1" / "exp"
    for sub in ("runs", "evaluations", "checkpoints", "masks"):
        (experiment / sub).mkdir(parents=True)
    checkpoint = experiment / "checkpoints" / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    stage1 = experiment / "checkpoints" / "stage1.pt"
    stage1.write_bytes(b"stage1")
    manifest = experiment / "runs" / "run.json"
    manifest.write_text(json.dumps({"run_id": "run", "dataset": "ETTh1", "checkpoint": str(checkpoint), "checkpoint_sha256": _sha(b"checkpoint"), "stage1_checkpoint": str(stage1), "stage1_checkpoint_sha256": _sha(b"stage1")}))
    evaluation = experiment / "evaluations" / "run.json"
    evaluation.write_text(json.dumps({"run_id": "run", "dataset": "ETTh1", "checkpoint_sha256": _sha(b"checkpoint"), "mse": 0.4, "mae": 0.5, "metric_space": "train_standardized"}))
    for split in ("validation", "test"):
        (experiment / "masks" / f"ETTh1_{split}.npz").write_bytes(split.encode())
    paper = tmp_path / "paper"
    catalog = paper / "E22/migration/paper_catalog.jsonl"
    catalog.parent.mkdir(parents=True)
    rows = []
    for path in experiment.rglob("*"):
        if not path.is_file():
            continue
        relative = "E22/" + path.relative_to(source).as_posix()
        destination = paper / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
        rows.append({"source": str(path), "destination_relative": relative,
                     "size": path.stat().st_size, "sha256": _sha(path.read_bytes()),
                     "owner": "E22", "dataset": "ETTh1", "scenario": "sr010_bl8",
                     "run_id": "run", "role": "asset"})
    catalog.write_text("".join(json.dumps(row) + "\n" for row in rows))
    selection = paper / "E22/migration/paper_selection.json"
    selection.write_text(json.dumps({"schema_version": 1, "copy_roots": [{
        "source": "checkout:artifacts/ETTh1/exp", "destination": "E22/artifacts/ETTh1/exp",
        "owner": "E22", "dataset": "ETTh1", "scenario": "sr010_bl8", "run_id": "run",
    }], "runs": [{"dataset": "ETTh1", "scenario": "sr010_bl8", "kind": "trained", "run_id": "run",
                    "manifest": "checkout:artifacts/ETTh1/exp/runs/run.json",
                    "evaluation": "checkout:artifacts/ETTh1/exp/evaluations/run.json",
                    "checkpoint": "checkout:artifacts/ETTh1/exp/checkpoints/model.pt"}]}))
    return paper, source, catalog, selection


def _verify(paths):
    paper, source, catalog, selection = paths
    return verify_migration(paper_root=paper, inventory=catalog, selection=selection,
                            source_checkout=source, old_project=source, require_git=False)


def test_verify_fails_on_missing_checkpoint_or_mask(tmp_path: Path):
    paths = _fixture(tmp_path)
    paper = paths[0]
    (paper / "E22/artifacts/ETTh1/exp/checkpoints/model.pt").unlink()
    report = _verify(paths)
    assert not report.accepted
    assert any("checkpoint" in error for error in report.errors)
    paths = _fixture(tmp_path / "second")
    (paths[0] / "E22/artifacts/ETTh1/exp/masks/ETTh1_test.npz").unlink()
    report = _verify(paths)
    assert not report.accepted
    assert any("mask" in error for error in report.errors)


def test_verify_labels_unreported_drr_absent(tmp_path: Path):
    report = _verify(_fixture(tmp_path))
    assert report.accepted
    assert report.metrics[0]["drr_mse"] == "absent"
    assert report.metrics[0]["drr_mae"] == "absent"


def test_verify_blocks_partial_copy(tmp_path: Path):
    paths = _fixture(tmp_path)
    lines = paths[2].read_text().splitlines()
    paths[2].write_text("\n".join(lines[:-1]) + "\n")
    report = _verify(paths)
    assert not report.accepted
    assert any("catalog" in error for error in report.errors)


def test_verify_rejects_uncommitted_source_overlay(tmp_path: Path):
    root = tmp_path / "E22"
    root.mkdir()
    git = "/home/xyw/miniforge3/envs/malr/bin/git"
    subprocess.run([git, "init", "-q", "-b", "paper", str(root)], check=True)
    path = root / "src/malr/cli.py"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"seed")
    subprocess.run([git, "-C", str(root), "add", "src/malr/cli.py"], check=True)
    subprocess.run([git, "-C", str(root), "-c", "user.name=Test", "-c", "user.email=test@local", "commit", "-q", "-m", "seed"], check=True)
    path.write_bytes(b"approved")
    subprocess.run([git, "-C", str(root), "add", "src/malr/cli.py"], check=True)
    subprocess.run([git, "-C", str(root), "-c", "user.name=Test", "-c", "user.email=test@local", "commit", "-q", "-m", "overlay"], check=True)
    _checked_source_overlay(root, "src/malr/cli.py", _sha(b"seed"))
    path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="uncommitted"):
        _checked_source_overlay(root, "src/malr/cli.py", _sha(b"seed"))


def test_verify_rejects_selected_run_identity_or_wrong_checkpoint(tmp_path: Path):
    paths = _fixture(tmp_path / "identity")
    selection = json.loads(paths[3].read_text())
    selection["runs"][0]["run_id"] = "another-run"
    paths[3].write_text(json.dumps(selection))
    report = _verify(paths)
    assert not report.accepted
    assert any("run identity" in error for error in report.errors)

    paths = _fixture(tmp_path / "checkpoint")
    selection = json.loads(paths[3].read_text())
    selection["runs"][0]["checkpoint"] = "checkout:artifacts/ETTh1/exp/masks/ETTh1_test.npz"
    paths[3].write_text(json.dumps(selection))
    report = _verify(paths)
    assert not report.accepted
    assert any("checkpoint lineage" in error for error in report.errors)


def test_verify_rejects_reported_drr_disagreement(tmp_path: Path):
    paths = _fixture(tmp_path)
    source_evaluation = paths[1] / "artifacts/ETTh1/exp/evaluations/run.json"
    destination_evaluation = paths[0] / "E22/artifacts/ETTh1/exp/evaluations/run.json"
    payload = json.loads(source_evaluation.read_text())
    payload["drr_mse"] = 20.0
    payload["drr_mae"] = 30.0
    source_evaluation.write_text(json.dumps(payload))
    destination_evaluation.write_bytes(source_evaluation.read_bytes())
    rows = [json.loads(line) for line in paths[2].read_text().splitlines()]
    for row in rows:
        if row["destination_relative"].endswith("evaluations/run.json"):
            row["size"] = source_evaluation.stat().st_size
            row["sha256"] = _sha(source_evaluation.read_bytes())
    paths[2].write_text("".join(json.dumps(row) + "\n" for row in rows))
    ledger = paths[0] / "E22/docs/experiments/E22_ETTh1_missing_scenarios.csv"
    ledger.parent.mkdir(parents=True)
    ledger.write_text("MSE,MAE,DRR-MSE (%),DRR-MAE (%),Source JSON\n0.4,0.5,21.0,30.0,artifacts/ETTh1/exp/evaluations/run.json\n")
    report = _verify(paths)
    assert not report.accepted
    assert any("ledger drr_mse mismatch" in error for error in report.errors)
