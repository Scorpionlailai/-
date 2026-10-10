"""Verify pinned local SAITS and S4M clones and their selected P4 assets."""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Literal

from malr.migration import MigrationCatalog
from .inventory import file_sha256


GIT = "/home/xyw/miniforge3/envs/malr/bin/git"
UPSTREAM = {
    "SAITS": "660b87f19c1277065f314f24134f646229e89ca9",
    "S4M": "a718823addd3606e763dfc261174e0135b2535f4",
}
S4M_BRIDGE_OVERLAYS = frozenset({
    "bridge/integrations/forecast_compare/s4m_p4.py",
    "bridge/integrations/s4m/verify_upstream.py",
})


@dataclass(frozen=True)
class BaselineReport:
    name: str
    upstream_commit: str
    project_head: str
    bridge_hash: str
    selected_evaluation_hash: str
    file_count: int
    portable: bool


def _git(project: Path, *args: str) -> str:
    result = subprocess.run([GIT, "-C", str(project), *args], capture_output=True, text=True)
    if result.returncode:
        raise ValueError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _combined_hash(entries: list[tuple[str, str]]) -> str:
    digest = hashlib.sha256()
    for path, sha256 in sorted(entries):
        digest.update(f"{path}\0{sha256}\n".encode())
    return digest.hexdigest()


def _verified_bridge_overlay(project: Path, relative: str, expected_seed_sha256: str) -> str:
    bridge_commits = _git(project, "rev-list", "--reverse", f"{UPSTREAM['S4M']}..HEAD").splitlines()
    if not bridge_commits:
        raise ValueError("reviewed S4M bridge commit is missing")
    path = project / relative
    current = file_sha256(path)
    for revision, expected in ((bridge_commits[0], expected_seed_sha256), ("HEAD", current)):
        result = subprocess.run(
            [GIT, "-C", str(project), "show", f"{revision}:{relative}"], capture_output=True,
        )
        if result.returncode or hashlib.sha256(result.stdout).hexdigest() != expected:
            raise ValueError(f"reviewed S4M bridge overlay mismatch: {relative} at {revision}")
    return current


def _check_import_preflight(name: str, project: Path) -> None:
    python = f"/home/xyw/miniforge3/envs/malr-{name.lower()}/bin/python"
    code = (
        "from pathlib import Path; "
        "from integrations.forecast_compare.s4m_p4 import _verify_s4m_checkout; "
        "print(_verify_s4m_checkout(Path('.'))['commit'])"
        if name == "S4M" else
        "from integrations.forecast_compare import saits_model, saits_run; print('SAITS bridge OK')"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = "bridge:../E22/src"
    result = subprocess.run(
        [python, "-c", code], cwd=project, env=environment, capture_output=True, text=True,
    )
    if result.returncode:
        raise ValueError(f"{name} import/preflight failed: {result.stderr.strip()}")
    if name == "S4M" and result.stdout.strip() != UPSTREAM[name]:
        raise ValueError("S4M preflight reported wrong upstream commit")


def verify_baseline(
    name: Literal["SAITS", "S4M"], *, project: Path, catalog: MigrationCatalog,
) -> BaselineReport:
    if name not in UPSTREAM:
        raise ValueError(f"unknown baseline: {name}")
    project = Path(project).resolve()
    head = _git(project, "rev-parse", "HEAD")
    commit = UPSTREAM[name]
    try:
        ancestor = _git(project, "merge-base", "HEAD", commit)
    except ValueError as exc:
        raise ValueError(f"{name} upstream commit is unavailable") from exc
    if ancestor != commit:
        raise ValueError(f"{name} upstream commit mismatch: {head}")
    metadata = (
        json.loads((catalog.paper_root / "E22/integrations/s4m/upstream.json").read_text())
        if name == "S4M" else {}
    )
    patch_targets = set(metadata.get("patch_targets", ()))
    allowed = ("bridge/", "migration/", "artifacts/", "s4m/malr_fair/")
    for relative in _git(project, "diff", "--name-only", f"{commit}..HEAD").splitlines():
        if relative not in {".gitignore", "data", "PAPER_HANDOFF.md", *patch_targets} and not relative.startswith(allowed):
            raise ValueError(f"unexpected committed upstream change: {relative}")
    if name == "S4M":
        if metadata["commit"] != commit:
            raise ValueError("S4M upstream metadata mismatch")
        for relative, expected in metadata["protected_blobs"].items():
            actual = _git(project, "hash-object", relative)
            if actual != expected:
                raise ValueError(f"protected S4M file changed: {relative}")
        patch = project / "bridge/integrations/s4m/patches/s4m-malr-fair.patch"
        installed = subprocess.run(
            [GIT, "-C", str(project), "apply", "--reverse", "--ignore-space-change", "--check", str(patch)],
            capture_output=True, text=True,
        )
        if installed.returncode:
            raise ValueError(f"reviewed S4M patch is not installed: {installed.stderr.strip()}")
    for line in _git(project, "status", "--porcelain=v1", "--untracked-files=all").splitlines():
        relative = line[3:]
        if relative not in {".gitignore", "PAPER_HANDOFF.md"} and not relative.startswith(allowed) and relative not in patch_targets:
            raise ValueError(f"unexpected dirty upstream file: {relative}")
    bridge: list[tuple[str, str]] = []
    evaluations: list[tuple[str, str]] = []
    entries = catalog.owner_entries(name)
    if not entries:
        raise ValueError(f"{name} is missing from migration catalog")
    for entry in entries:
        relative = entry.destination.relative_to(catalog.paper_root / name).as_posix()
        if name == "S4M" and relative in S4M_BRIDGE_OVERLAYS:
            digest = _verified_bridge_overlay(project, relative, entry.sha256)
            path = project / relative
        else:
            path = catalog.resolve(entry.destination)
            digest = entry.sha256
        if relative.startswith("bridge/") or relative.startswith("s4m/malr_fair/"):
            bridge.append((relative, digest))
        if "/evaluations/" in relative and relative.endswith(".json"):
            evaluations.append((relative, digest))
    if not bridge or not evaluations:
        raise ValueError(f"{name} lacks checked bridge or selected evaluations")
    _check_import_preflight(name, project)
    return BaselineReport(name, commit, head, _combined_hash(bridge), _combined_hash(evaluations), len(entries), True)
