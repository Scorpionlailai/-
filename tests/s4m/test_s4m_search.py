from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from integrations.s4m.search import (
    SearchCandidate,
    build_train_command,
    dropout_refinement_candidates,
    main,
    official_candidates,
    select_best,
    tuned_stage1_candidates,
)


def test_official_profile_searches_paper_layer_choices() -> None:
    candidates = official_candidates()

    assert {
        (candidate.e_layers, candidate.learning_rate, candidate.dropout)
        for candidate in candidates
    } == {(2, 0.005, 0.1), (4, 0.005, 0.1), (8, 0.005, 0.1)}


def test_tuned_stage1_searches_layer_and_learning_rate_grid() -> None:
    candidates = tuned_stage1_candidates()

    assert len(candidates) == 9
    assert {candidate.e_layers for candidate in candidates} == {2, 4, 8}
    assert {candidate.learning_rate for candidate in candidates} == {
        0.001,
        0.003,
        0.005,
    }
    assert {candidate.dropout for candidate in candidates} == {0.1}


def test_dropout_refinement_uses_two_best_stage1_candidates() -> None:
    candidates = tuned_stage1_candidates()
    scores = {
        candidate.run_id: float(index)
        for index, candidate in enumerate(reversed(candidates))
    }

    refined = dropout_refinement_candidates(candidates, scores)

    assert len(refined) == 4
    assert {candidate.dropout for candidate in refined} == {0.0, 0.2}
    assert len({(candidate.e_layers, candidate.learning_rate) for candidate in refined}) == 2


def test_selection_uses_validation_mse_then_mae(tmp_path: Path) -> None:
    checkpoint_a = tmp_path / "a.pth"
    checkpoint_b = tmp_path / "b.pth"
    checkpoint_a.write_bytes(b"a")
    checkpoint_b.write_bytes(b"b")
    manifests = []
    for run_id, mse, mae, checkpoint in (
        ("a", 0.7, 0.6, checkpoint_a),
        ("b", 0.7, 0.5, checkpoint_b),
    ):
        path = tmp_path / f"{run_id}.json"
        path.write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "status": "validation_finalized",
                    "validation_mse": mse,
                    "validation_mae": mae,
                    "checkpoint": str(checkpoint.resolve()),
                    "checkpoint_sha256": __import__("hashlib").sha256(
                        checkpoint.read_bytes()
                    ).hexdigest(),
                    "test_evaluated": False,
                }
            ),
            encoding="utf-8",
        )
        manifests.append(path)

    output = tmp_path / "selection.json"
    selected = select_best(manifests, output)

    assert selected["selected_run_id"] == "b"
    assert selected["test_metrics_seen"] is False
    assert json.loads(output.read_text(encoding="utf-8")) == selected


def test_selection_rejects_manifest_that_accessed_test(tmp_path: Path) -> None:
    path = tmp_path / "leaked.json"
    path.write_text(
        json.dumps(
            {
                "run_id": "leaked",
                "status": "validation_finalized",
                "validation_mse": 0.1,
                "validation_mae": 0.1,
                "checkpoint": "unused",
                "checkpoint_sha256": "unused",
                "test_evaluated": True,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="test"):
        select_best([path], tmp_path / "selection.json")


def test_train_command_uses_official_entry_and_fair_train_mode(tmp_path: Path) -> None:
    candidate = SearchCandidate(
        e_layers=4,
        learning_rate=0.003,
        dropout=0.2,
        profile="tuned",
    )
    command = build_train_command(
        candidate,
        python="python",
        s4m_root=tmp_path / "S4M",
        config=tmp_path / "stage1.yaml",
        artifacts=tmp_path / "malr-artifacts",
        output=tmp_path / "s4m-artifacts",
        device="cuda",
        variant="missing",
    )

    joined = " ".join(command)
    assert command[1].endswith("S4M/s4m/run.py")
    assert "--fair_mode train" in joined
    assert "--fair_device cuda" in joined
    assert "--malr_config" in command
    assert "evaluate" not in command
    assert "--seed 2021" in joined


def test_s4m_subprocess_environment_makes_bundled_s4_importable(
    tmp_path: Path,
) -> None:
    from integrations.s4m import search

    s4m_root = tmp_path / "S4M"
    package = s4m_root / "s4m" / "model" / "S4"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("VALUE = 17\n", encoding="utf-8")

    build_environment = getattr(search, "build_s4m_subprocess_environment", None)
    assert build_environment is not None, (
        "the S4M launcher must provide the bundled model directory on PYTHONPATH"
    )
    environment = build_environment(
        s4m_root,
        base_environment={"PYTHONPATH": "existing-entry", "KEEP_ME": "yes"},
    )
    completed = subprocess.run(
        [sys.executable, "-c", "import S4; print(S4.VALUE)"],
        env={**os.environ, **environment},
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "17"
    assert environment["KEEP_ME"] == "yes"
    assert environment["PYTHONPATH"].endswith(
        os.pathsep + "existing-entry"
    )


def test_smoke_keeps_enough_samples_for_official_memory_clustering(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = main(
        [
            "--config",
            str(tmp_path / "config.yaml"),
            "--artifacts",
            str(tmp_path / "malr-artifacts"),
            "--s4m-root",
            str(tmp_path / "S4M"),
            "--output",
            str(tmp_path / "s4m-artifacts"),
            "--smoke",
            "--device",
            "cuda",
            "--dry-run",
        ]
    )

    command = capsys.readouterr().out
    assert result == 0
    assert "--batch_size 16" in command
