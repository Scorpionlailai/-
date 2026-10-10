"""Copy a pinned inventory without overwriting any destination file."""

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Callable, Sequence

from .inventory import MigrationEntry, build_inventory, file_sha256, inventory_digest
from .selection import load_selection


@dataclass(frozen=True)
class CopyReport:
    copied: int
    skipped: int
    total: int
    bytes_verified: int


def _destination(target_root: Path, relative: str) -> Path:
    part = Path(relative)
    if part.is_absolute() or ".." in part.parts or not part.parts:
        raise ValueError(f"destination escapes target root: {relative}")
    if target_root.is_symlink():
        raise ValueError(f"destination escapes target root: {target_root}")
    path = target_root / part
    current = path.parent
    while current != target_root.parent:
        if current.is_symlink():
            raise ValueError(f"destination escapes target root through symlink: {current}")
        if current == target_root:
            break
        current = current.parent
    return path


def _verify_source(entry: MigrationEntry) -> None:
    if entry.source.is_symlink() or not entry.source.is_file():
        raise ValueError(f"source changed: {entry.source}")
    if entry.source.stat().st_size != entry.size or file_sha256(entry.source) != entry.sha256:
        raise ValueError(f"source changed since inventory: {entry.source}")


def _write_catalog(target_root: Path, entries: Sequence[MigrationEntry]) -> None:
    catalog = target_root / "E22/migration/paper_catalog.jsonl"
    catalog.parent.mkdir(parents=True, exist_ok=True)
    by_destination: dict[str, dict] = {}
    if catalog.exists():
        for line in catalog.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            by_destination[row["destination_relative"]] = row
    for entry in entries:
        row = {**asdict(entry), "source": str(entry.source)}
        previous = by_destination.get(entry.destination_relative)
        if previous and (previous["sha256"] != entry.sha256 or previous["source"] != str(entry.source)):
            raise ValueError(f"conflicting catalog entry: {entry.destination_relative}")
        by_destination[entry.destination_relative] = row
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=catalog.parent, prefix=".catalog-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            for key in sorted(by_destination):
                stream.write(json.dumps(by_destination[key], ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, catalog)


def copy_entries(
    entries: Sequence[MigrationEntry], *, target_root: Path, owners: frozenset[str],
    apply: bool = False, progress: Callable[[int, int, str], None] | None = None,
) -> CopyReport:
    """Verify every source and safely copy the selected owner subset."""
    selected = [entry for entry in entries if entry.owner in owners]
    if not selected:
        raise ValueError(f"no inventory entries for owners: {sorted(owners)}")
    destinations = [_destination(target_root, entry.destination_relative) for entry in selected]
    copied = skipped = bytes_verified = 0
    for index, (entry, destination) in enumerate(zip(selected, destinations), 1):
        _verify_source(entry)
        if destination.exists() or destination.is_symlink():
            if destination.is_symlink() or not destination.is_file() or destination.stat().st_size != entry.size or file_sha256(destination) != entry.sha256:
                raise ValueError(f"conflicting destination: {destination}")
            action = "SKIP"
            if apply:
                skipped += 1
        elif not apply:
            action = "PLAN"
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            _destination(target_root, entry.destination_relative)
            with tempfile.NamedTemporaryFile("wb", dir=destination.parent, prefix=".copy-", delete=False) as stream:
                temporary = Path(stream.name)
                try:
                    with entry.source.open("rb") as source_stream:
                        shutil.copyfileobj(source_stream, stream, length=1024 * 1024)
                    stream.flush()
                    os.fsync(stream.fileno())
                except BaseException:
                    temporary.unlink(missing_ok=True)
                    raise
            try:
                if file_sha256(temporary) != entry.sha256:
                    raise ValueError(f"source changed during copy: {entry.source}")
                _verify_source(entry)
                try:
                    os.link(temporary, destination)
                    copied += 1
                    action = "COPY"
                except FileExistsError:
                    if destination.is_symlink() or file_sha256(destination) != entry.sha256:
                        raise ValueError(f"conflicting destination: {destination}")
                    skipped += 1
                    action = "SKIP"
            finally:
                temporary.unlink(missing_ok=True)
        bytes_verified += entry.size
        if progress:
            progress(index, len(selected), f"{action} {entry.destination_relative}")
    if apply:
        _write_catalog(target_root, selected)
    return CopyReport(copied, skipped, len(selected), bytes_verified)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--source-checkout", type=Path, required=True)
    parser.add_argument("--old-project", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--owners", required=True)
    parser.add_argument("--inventory-digest", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    entries = build_inventory(load_selection(args.selection), source_checkout=args.source_checkout, old_project=args.old_project)
    actual_digest = inventory_digest(entries)
    if actual_digest != args.inventory_digest:
        raise ValueError(f"source inventory changed: expected {args.inventory_digest}, got {actual_digest}")
    report = copy_entries(
        entries, target_root=args.target_root, owners=frozenset(args.owners.split(",")), apply=args.apply,
        progress=lambda i, n, message: print(f"[{i}/{n}] {message}", flush=True),
    )
    print(json.dumps(asdict(report), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
