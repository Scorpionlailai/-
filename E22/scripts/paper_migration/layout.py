"""Check the two model projects' portable directory contract."""

from dataclasses import dataclass
import json
from pathlib import Path

from .inventory import file_sha256


@dataclass(frozen=True)
class LayoutReport:
    projects: tuple[str, str]
    data_root: Path
    checked_catalog_files: int


def verify_project_layout(paper_root: Path) -> LayoutReport:
    data_root = paper_root / "data"
    if not data_root.is_dir():
        raise ValueError(f"missing shared data: {data_root}")
    for name in ("E22", "E23"):
        project = paper_root / name
        git_path = project / ".git"
        if git_path.is_file() and "C:/" in git_path.read_text(encoding="utf-8"):
            raise ValueError(f"Windows gitdir remains: {git_path}")
        if git_path.is_symlink():
            raise ValueError(f"non-independent Git directory: {git_path}")
        link = project / "data"
        if not link.is_symlink() or link.readlink() != Path("../data") or link.resolve() != data_root.resolve():
            raise ValueError(f"expected relative data link: {link}")
        if not (project / "src/malr/__init__.py").is_file() or not (project / "configs").is_dir():
            raise ValueError(f"incomplete model source: {project}")
    e23_artifacts = paper_root / "E23/artifacts"
    if e23_artifacts.exists() and any(e23_artifacts.iterdir()):
        raise ValueError(f"E23 artifacts must start empty: {e23_artifacts}")
    catalog = paper_root / "E22/migration/paper_catalog.jsonl"
    checked = 0
    if catalog.is_file():
        for line in catalog.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            if entry["owner"] not in {"E22", "E23"} or entry["role"] != "source":
                continue
            destination = paper_root / entry["destination_relative"]
            if not destination.is_file() or destination.stat().st_size != entry["size"] or file_sha256(destination) != entry["sha256"]:
                raise ValueError(f"source hash mismatch: {destination}")
            checked += 1
    return LayoutReport(("E22", "E23"), data_root, checked)
