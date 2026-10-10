from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any


def patch_paths(path: Path) -> set[str]:
    if not path.is_file():
        raise ValueError(f"S4M patch is missing: {path}")
    touched: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("+++ b/"):
            continue
        touched.add(line[len("+++ b/") :])
    if not touched:
        raise ValueError(f"S4M patch does not contain file changes: {path}")
    return touched


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ValueError(f"git {' '.join(arguments)} failed: {detail}")
    return completed.stdout.rstrip()


def _dirty_paths(root: Path) -> list[str]:
    output = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    paths: list[str] = []
    for line in output.splitlines():
        if len(line) < 4:
            continue
        value = line[3:].replace("\\", "/")
        if " -> " in value:
            value = value.split(" -> ", 1)[1]
        paths.append(value.strip('"'))
    return sorted(paths)


def verify_checkout(
    root: Path,
    metadata_path: Path,
    *,
    allow_patched: bool,
) -> dict[str, object]:
    checkout = root.resolve()
    if not (checkout / ".git").exists():
        raise ValueError(f"S4M checkout is not a git repository: {checkout}")
    metadata: dict[str, Any] = json.loads(
        metadata_path.resolve().read_text(encoding="utf-8")
    )
    expected_commit = str(metadata["commit"])
    actual_commit = _git(checkout, "rev-parse", "HEAD")
    try:
        ancestor = _git(checkout, "merge-base", "HEAD", expected_commit)
    except ValueError as exc:
        raise ValueError(f"S4M upstream commit is unavailable: {expected_commit}") from exc
    if ancestor != expected_commit:
        raise ValueError(
            f"S4M commit mismatch: expected {expected_commit}, got {actual_commit}"
        )
    committed = _git(checkout, "diff", "--name-only", f"{expected_commit}..HEAD").splitlines()
    patch_targets = {str(path) for path in metadata.get("patch_targets", [])}
    overlay_prefix = str(metadata.get("overlay_prefix", "s4m/malr_fair/"))
    if committed and not allow_patched:
        raise ValueError(f"S4M checkout has local commits: {', '.join(committed)}")
    unexpected_committed = [
        path for path in committed
        if path not in patch_targets
        and path not in {".gitignore", "data"}
        and not path.startswith(("bridge/", "migration/", overlay_prefix))
    ]
    if unexpected_committed:
        raise ValueError("S4M checkout has unapproved commits: " + ", ".join(unexpected_committed))

    protected = dict(metadata.get("protected_blobs", {}))
    protected_report: dict[str, str] = {}
    for relative, expected_blob in protected.items():
        path = checkout / relative
        if not path.is_file():
            raise ValueError(f"protected S4M file is missing: {relative}")
        actual_blob = _git(checkout, "hash-object", relative)
        if actual_blob != expected_blob:
            raise ValueError(
                f"protected S4M file changed: {relative}; "
                f"expected {expected_blob}, got {actual_blob}"
            )
        protected_report[relative] = actual_blob

    dirty = _dirty_paths(checkout)
    if dirty and not allow_patched:
        raise ValueError(f"S4M checkout is dirty: {', '.join(dirty)}")
    if allow_patched:
        unexpected = [
            path
            for path in dirty
            if path not in patch_targets and not path.startswith(overlay_prefix)
        ]
        if unexpected:
            raise ValueError(
                "S4M checkout contains unexpected dirty paths: "
                + ", ".join(unexpected)
            )
        patch = metadata_path.resolve().parent / "patches/s4m-malr-fair.patch"
        _git(checkout, "apply", "--reverse", "--ignore-space-change", "--check", str(patch))

    return {
        "root": str(checkout),
        "commit": expected_commit,
        "project_head": actual_commit,
        "dirty_paths": dirty,
        "protected_blobs": protected_report,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify the pinned S4M checkout")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path(__file__).with_name("upstream.json"),
    )
    parser.add_argument("--allow-patched", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = verify_checkout(
        args.root,
        args.metadata,
        allow_patched=bool(args.allow_patched),
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
