import hashlib
from pathlib import Path

import pytest

from scripts.paper_migration.copy_verified import copy_entries
from scripts.paper_migration.inventory import MigrationEntry


def _entry(source: Path, destination: str = "E22/artifacts/result.pt") -> MigrationEntry:
    data = source.read_bytes()
    return MigrationEntry(source, destination, len(data), hashlib.sha256(data).hexdigest(), "E22")


def test_copy_refuses_conflicting_destination(tmp_path: Path) -> None:
    source = tmp_path / "old.pt"
    source.write_bytes(b"old selected result")
    destination = tmp_path / "paper/E22/artifacts/result.pt"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"different user content")
    with pytest.raises(ValueError, match="conflicting destination"):
        copy_entries([_entry(source)], target_root=tmp_path / "paper", owners=frozenset({"E22"}), apply=True)
    assert destination.read_bytes() == b"different user content"
    assert source.read_bytes() == b"old selected result"


def test_copy_refuses_changed_source(tmp_path: Path) -> None:
    source = tmp_path / "old.pt"
    source.write_bytes(b"inventoried")
    entry = _entry(source)
    source.write_bytes(b"changed later")
    with pytest.raises(ValueError, match="source changed"):
        copy_entries([entry], target_root=tmp_path / "paper", owners=frozenset({"E22"}), apply=True)
    assert not (tmp_path / "paper/E22/artifacts/result.pt").exists()


def test_copy_resumes_matching_files(tmp_path: Path) -> None:
    source = tmp_path / "old.pt"
    source.write_bytes(b"same result")
    entry = _entry(source)
    target = tmp_path / "paper"
    first = copy_entries([entry], target_root=target, owners=frozenset({"E22"}), apply=True)
    second = copy_entries([entry], target_root=target, owners=frozenset({"E22"}), apply=True)
    assert (first.copied, first.skipped, second.copied, second.skipped) == (1, 0, 0, 1)
    assert (target / entry.destination_relative).read_bytes() == b"same result"
    assert (target / "E22/migration/paper_catalog.jsonl").is_file()


def test_copy_rejects_path_escape(tmp_path: Path) -> None:
    source = tmp_path / "old.pt"
    source.write_bytes(b"content")
    entry = _entry(source, "../escape.pt")
    with pytest.raises(ValueError, match="destination escapes"):
        copy_entries([entry], target_root=tmp_path / "paper", owners=frozenset({"E22"}), apply=True)
    assert not (tmp_path / "escape.pt").exists()


def test_copy_dry_run_writes_nothing(tmp_path: Path) -> None:
    source = tmp_path / "old.pt"
    source.write_bytes(b"content")
    target = tmp_path / "paper"
    report = copy_entries([_entry(source)], target_root=target, owners=frozenset({"E22"}))
    assert (report.copied, report.skipped) == (0, 0)
    assert not target.exists()
