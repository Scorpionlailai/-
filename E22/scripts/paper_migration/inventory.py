"""Read-only content inventory of explicitly selected migration inputs."""

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterable

from .selection import Selection


_SKIP_PARTS = frozenset({".git", ".pytest_cache", "__pycache__", ".codex_tmp", "jobs"})


@dataclass(frozen=True)
class MigrationEntry:
    source: Path
    destination_relative: str
    size: int
    sha256: str
    owner: str
    dataset: str = ""
    scenario: str = ""
    run_id: str = ""
    role: str = "asset"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_path(reference: str, *, source_checkout: Path, old_project: Path) -> Path:
    prefix, sep, relative = reference.partition(":")
    if not sep or prefix not in {"checkout", "project"}:
        raise ValueError(f"invalid source reference: {reference}")
    part = Path(relative)
    if part.is_absolute() or ".." in part.parts or not part.parts:
        raise ValueError(f"source escapes reviewed root: {reference}")
    base = source_checkout if prefix == "checkout" else old_project
    path = base / part
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents if parent != base.parent):
        raise ValueError(f"symlink source is not allowed: {reference}")
    if not path.resolve().is_relative_to(base.resolve()):
        raise ValueError(f"source escapes reviewed root: {reference}")
    return path


def _files(root: Path) -> Iterable[Path]:
    if root.is_file():
        yield root
    elif root.is_dir():
        for path in root.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"symlink source is not allowed: {path}")
            if path.is_file() and not (_SKIP_PARTS & set(path.relative_to(root).parts)) and path.suffix not in {".pyc", ".pyo"}:
                yield path
    else:
        raise ValueError(f"missing copy source: {root}")


def build_inventory(selection: Selection, *, source_checkout: Path, old_project: Path) -> list[MigrationEntry]:
    """Hash selected files and check selected run dependency closure."""
    entries: dict[str, MigrationEntry] = {}
    selected_sources: set[Path] = set()
    for root in selection.copy_roots:
        source = _source_path(root.source, source_checkout=source_checkout, old_project=old_project)
        if not source.exists():
            raise ValueError(f"missing copy source: {source}")
        for path in _files(source):
            relative = path.relative_to(source) if source.is_dir() else Path()
            destination = (Path(root.destination) / relative).as_posix()
            entry = MigrationEntry(path, destination, path.stat().st_size, file_sha256(path), root.owner, root.dataset, root.scenario, root.run_id, root.role)
            if destination in entries and entries[destination].source != path:
                raise ValueError(f"conflicting destination: {destination}")
            entries[destination] = entry
            selected_sources.add(path)
    for run in selection.runs:
        references = (run.manifest, run.evaluation, run.checkpoint, *run.dependencies)
        for reference in references:
            path = _source_path(reference, source_checkout=source_checkout, old_project=old_project)
            if not path.is_file() or path not in selected_sources:
                raise ValueError(f"missing selected run dependency: {run.run_id}: {reference}")
    return [entries[key] for key in sorted(entries)]


def inventory_digest(entries: Iterable[MigrationEntry]) -> str:
    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda item: item.destination_relative):
        digest.update(f"{entry.destination_relative}\0{entry.size}\0{entry.sha256}\n".encode())
    return digest.hexdigest()
