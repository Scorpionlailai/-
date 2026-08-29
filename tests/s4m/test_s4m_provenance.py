from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from integrations.s4m.verify_upstream import verify_checkout


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _repository(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "S4M"
    protected = repo / "s4m" / "model" / "S4M.py"
    protected.parent.mkdir(parents=True)
    protected.write_text("official core\n", encoding="utf-8")
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "fixture")
    metadata = tmp_path / "upstream.json"
    metadata.write_text(
        json.dumps(
            {
                "commit": _git(repo, "rev-parse", "HEAD"),
                "protected_blobs": {
                    "s4m/model/S4M.py": _git(
                        repo, "hash-object", "s4m/model/S4M.py"
                    )
                },
                "patch_targets": ["s4m/run.py"],
                "overlay_prefix": "s4m/malr_fair/",
            }
        ),
        encoding="utf-8",
    )
    return repo, metadata


def test_verify_checkout_rejects_wrong_commit(tmp_path: Path) -> None:
    repo, metadata = _repository(tmp_path)
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    payload["commit"] = "0" * 40
    metadata.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="commit"):
        verify_checkout(repo, metadata, allow_patched=False)


def test_verify_checkout_rejects_modified_protected_core(tmp_path: Path) -> None:
    repo, metadata = _repository(tmp_path)
    (repo / "s4m" / "model" / "S4M.py").write_text(
        "modified core\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="protected"):
        verify_checkout(repo, metadata, allow_patched=True)


def test_verify_checkout_allows_only_declared_patch_paths(tmp_path: Path) -> None:
    repo, metadata = _repository(tmp_path)
    run_path = repo / "s4m" / "run.py"
    run_path.write_text("patched\n", encoding="utf-8")
    overlay_path = repo / "s4m" / "malr_fair" / "data.py"
    overlay_path.parent.mkdir(parents=True)
    overlay_path.write_text("bridge\n", encoding="utf-8")

    report = verify_checkout(repo, metadata, allow_patched=True)

    assert report["commit"] == _git(repo, "rev-parse", "HEAD")
    assert report["dirty_paths"] == ["s4m/malr_fair/data.py", "s4m/run.py"]


def test_verify_checkout_rejects_unrelated_dirty_path(tmp_path: Path) -> None:
    repo, metadata = _repository(tmp_path)
    (repo / "unexpected.txt").write_text("not allowed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected.txt"):
        verify_checkout(repo, metadata, allow_patched=True)
