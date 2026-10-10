"""Strict access to the immutable files selected for the paper workspace."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class _Entry:
    source: Path
    destination: Path
    size: int
    sha256: str
    owner: str


class MigrationCatalog:
    """Map an old absolute path to one checked copy under the paper root."""

    def __init__(self, entries: dict[str, _Entry], paper_root: Path):
        self._entries = entries
        self.paper_root = paper_root

    @classmethod
    def load(cls, path: Path, *, project: str | None = None) -> "MigrationCatalog":
        path = Path(path).resolve(strict=True)
        if path.name != "paper_catalog.jsonl" or path.parent.name != "migration":
            raise ValueError(f"invalid migration catalog location: {path}")
        paper_root = path.parent.parent.parent
        project = project or path.parent.parent.name
        entries: dict[str, _Entry] = {}
        by_source: dict[str, list[_Entry]] = {}
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            row = json.loads(line)
            source = Path(row["source"])
            relative = Path(row["destination_relative"])
            digest = row["sha256"]
            if not source.is_absolute() or not relative.parts or relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"invalid catalog destination/source on line {line_number}")
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError(f"invalid catalog SHA-256 on line {line_number}")
            destination = paper_root / relative
            entry = _Entry(source, destination, row["size"], digest, row["owner"])
            key = str(destination)
            if key in entries and entries[key] != entry:
                raise ValueError(f"ambiguous catalog destination: {key}")
            entries[key] = entry
            by_source.setdefault(str(source), []).append(entry)
        for source, candidates in by_source.items():
            preferred = [entry for entry in candidates if entry.owner == project]
            chosen = preferred or candidates
            if len(chosen) != 1:
                raise ValueError(f"ambiguous catalog source: {source}")
            entries[source] = chosen[0]
        return cls(entries, paper_root)

    def resolve(self, original: str | Path, *, expected_sha256: str | None = None) -> Path:
        reference = Path(original)
        key = str(reference if reference.is_absolute() else reference.resolve(strict=False))
        entry = self._entries.get(key)
        if entry is None:
            raise ValueError(f"unmapped migration artifact: {key}")
        path = entry.destination
        if not path.is_relative_to(self.paper_root) or path.is_symlink() or not path.is_file():
            raise ValueError(f"missing or unsafe catalog destination: {path}")
        if any(parent.is_symlink() for parent in path.parents if parent != self.paper_root.parent):
            raise ValueError(f"unsafe catalog destination: {path}")
        if path.stat().st_size != entry.size:
            raise ValueError(f"catalog size mismatch: {path}")
        if expected_sha256 is not None and expected_sha256 != entry.sha256:
            raise ValueError(f"expected SHA-256 mismatch: {path}")
        if _sha256(path) != entry.sha256:
            raise ValueError(f"catalog SHA-256 mismatch: {path}")
        return path

    def owner_entries(self, owner: str) -> tuple[_Entry, ...]:
        """Return each cataloged destination for one project exactly once."""
        by_destination = {
            str(entry.destination): entry for entry in self._entries.values()
            if entry.owner == owner
        }
        return tuple(by_destination[key] for key in sorted(by_destination))


def load_migrated_run(manifest_path: Path, catalog: MigrationCatalog) -> tuple[dict, dict[str, Path]]:
    """Read a cataloged run manifest and validate its checkpoint lineage."""
    path = catalog.resolve(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"invalid run manifest: {path}")
    paths: dict[str, Path] = {}
    for name in ("checkpoint", "stage1_checkpoint", "clean_checkpoint"):
        value = manifest.get(name)
        if value:
            paths[name] = catalog.resolve(value, expected_sha256=manifest.get(f"{name}_sha256"))
    if "checkpoint" not in paths:
        raise ValueError(f"run manifest lacks checkpoint: {path}")
    return manifest, paths
