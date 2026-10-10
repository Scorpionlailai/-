"""Load the explicit, reviewable set of experiments to preserve."""

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class CopyRoot:
    source: str
    destination: str
    owner: str
    role: str = "asset"
    dataset: str = ""
    scenario: str = ""
    run_id: str = ""


@dataclass(frozen=True)
class SelectedRun:
    dataset: str
    scenario: str
    kind: str
    run_id: str
    manifest: str
    evaluation: str
    checkpoint: str
    dependencies: tuple[str, ...] = ()


@dataclass(frozen=True)
class Selection:
    copy_roots: tuple[CopyRoot, ...]
    runs: tuple[SelectedRun, ...]


def load_selection(path: Path) -> Selection:
    """Reject ambiguous selections before any file is inventoried or copied."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported migration selection schema")
    roots = tuple(CopyRoot(**item) for item in payload["copy_roots"])
    runs = tuple(
        SelectedRun(**{**item, "dependencies": tuple(item.get("dependencies", ()))})
        for item in payload["runs"]
    )
    seen: set[tuple[str, str, str]] = set()
    for run in runs:
        key = (run.dataset, run.scenario, run.kind)
        if key in seen:
            raise ValueError(f"duplicate selected run: {key}")
        seen.add(key)
        if not all((run.dataset, run.scenario, run.kind, run.run_id, run.manifest, run.evaluation, run.checkpoint)):
            raise ValueError(f"incomplete selected run: {key}")
    for root in roots:
        if root.owner not in {"data", "E22", "E23", "SAITS", "S4M"}:
            raise ValueError(f"invalid owner: {root.owner}")
        destination = Path(root.destination)
        if destination.is_absolute() or ".." in destination.parts or not destination.parts:
            raise ValueError(f"invalid destination: {root.destination}")
    return Selection(roots, runs)
