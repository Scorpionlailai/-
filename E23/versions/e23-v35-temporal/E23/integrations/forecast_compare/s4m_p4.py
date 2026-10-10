"""P4-safe wrapper for the existing official-profile S4M bridge."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha1, sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence

from integrations.s4m.overlay.malr_fair.data import FairS4MDataset

from .protocol import P4Protocol, load_p4_protocol

UPSTREAM_COMMIT = "a718823addd3606e763dfc261174e0135b2535f4"
GIT_BIN = Path("/home/xyw/miniforge3/pkgs/git-2.55.0-pl5321h5685339_1/bin/git")


def _ref_hash(refs: tuple) -> str:
    payload = json.dumps([asdict(ref) for ref in refs], sort_keys=True,
                         separators=(",", ":"), ensure_ascii=True)
    return sha256(payload.encode("utf-8")).hexdigest()


def _verify_s4m_checkout(s4m_root: Path) -> dict:
    if not GIT_BIN.is_file():
        raise ValueError(f"pinned git binary missing: {GIT_BIN}")
    script = Path(__file__).resolve().parents[1] / "s4m" / "verify_upstream.py"
    environment = dict(os.environ)
    environment["PATH"] = f"{GIT_BIN.parent}:{environment.get('PATH', '')}"
    environment["LD_LIBRARY_PATH"] = "/home/xyw/miniforge3/lib"
    completed = subprocess.run(
        [sys.executable, str(script), "--root", str(s4m_root.resolve()), "--allow-patched"],
        capture_output=True, text=True, check=False, env=environment,
    )
    if completed.returncode:
        # This Windows-origin checkout has CRLF on many tracked text files.
        # Permit only byte-level line-ending drift; real source changes still fail.
        metadata = json.loads(script.with_name("upstream.json").read_text(encoding="utf-8"))

        def git(*args: str) -> str:
            call = subprocess.run([str(GIT_BIN), "-C", str(s4m_root.resolve()), *args],
                                  capture_output=True, text=True, check=False, env=environment)
            if call.returncode:
                raise ValueError(f"S4M git check failed: {call.stderr.strip()}")
            return call.stdout.strip()

        if git("rev-parse", "HEAD") != UPSTREAM_COMMIT:
            raise ValueError("S4M upstream commit mismatch")
        for relative, expected_blob in metadata["protected_blobs"].items():
            raw = (s4m_root / relative).read_bytes().replace(b"\r\n", b"\n")
            blob = sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
            if blob != expected_blob:
                raise ValueError(f"protected S4M core changed beyond line endings: {relative}")
        allowed = set(metadata["patch_targets"])
        changed = git("diff", "--ignore-space-at-eol", "--name-only").splitlines()
        unexpected = [path for path in changed
                      if path not in allowed and "__pycache__" not in path]
        untracked = git("ls-files", "--others", "--exclude-standard").splitlines()
        unexpected += [path for path in untracked
                       if not path.startswith(metadata["overlay_prefix"])
                       and "__pycache__" not in path]
        if unexpected:
            raise ValueError("unexpected semantic S4M changes: " + ", ".join(unexpected))
        result = {"commit": UPSTREAM_COMMIT, "line_endings_normalized": True,
                  "protected_blobs": metadata["protected_blobs"]}
    else:
        result = json.loads(completed.stdout)
    if result["commit"] != UPSTREAM_COMMIT:
        raise ValueError("S4M upstream commit mismatch")
    return result


def preflight_s4m_p4(*, config: Path, mask_artifacts: Path, s4m_root: Path) -> dict[str, object]:
    common: P4Protocol = load_p4_protocol(config, mask_artifacts)
    upstream = _verify_s4m_checkout(s4m_root)
    args = argparse.Namespace(
        malr_config=str(Path(config).resolve()), malr_dataset="ETTh1",
        malr_artifacts=str(Path(mask_artifacts).resolve()), fair_variant="missing",
        fair_eval_reference_batch=128, batch_size=16, num_workers=0, seed=2021,
    )
    identities: dict[str, dict] = {}
    for split, flag in (("validation", "val"), ("test", "test")):
        dataset = FairS4MDataset(args, flag)
        identity = dataset.identity()
        common_refs = common.refs[split][:2688]
        if len(dataset.refs) != 2688 or tuple(dataset.refs) != common_refs:
            raise ValueError(f"S4M {split} window refs diverge from common P4 protocol")
        if identity["window_indices_sha256"] != _ref_hash(common_refs):
            raise ValueError(f"S4M {split} ref SHA-256 mismatch")
        if identity["mask_sha256"] != common.mask_sha256[split]:
            raise ValueError(f"S4M {split} mask SHA-256 mismatch")
        if identity["source_csv_sha256"] != common.source_sha256:
            raise ValueError("S4M source CSV SHA-256 mismatch")
        if identity["metric_space"] != "train_standardized":
            raise ValueError("S4M metric space mismatch")
        identities[split] = identity
    return {
        "upstream_commit": upstream["commit"],
        "line_endings_normalized_for_verification": bool(upstream.get("line_endings_normalized", False)),
        "source_csv_sha256": common.source_sha256,
        "validation_mask_sha256": common.mask_sha256["validation"],
        "test_mask_sha256": common.mask_sha256["test"],
        "validation_windows": identities["validation"]["sample_count"],
        "test_windows": identities["test"]["sample_count"],
        "validation_refs_sha256": identities["validation"]["window_indices_sha256"],
        "test_refs_sha256": identities["test"]["window_indices_sha256"],
        "common_test_refs_sha256": _ref_hash(common.refs["test"][:2688]),
        "train_mask_protocol": "epoch_timeline_dynamic; differs from E22/SAITS per-sample masks",
    }


def require_s4m_selection(output: Path, *, variant: str) -> Path:
    if variant not in ("missing", "clean"):
        raise ValueError("S4M variant must be missing or clean")
    name = "selection-official.json" if variant == "missing" else "selection-official-clean.json"
    path = Path(output) / name
    if not path.is_file():
        raise ValueError(f"S4M {variant} validation selection is missing: {path}")
    record = json.loads(path.read_text(encoding="utf-8"))
    if not record.get("selected_run_id") or not record.get("selected_manifest") or record.get("test_metrics_seen") is not False:
        raise ValueError(f"S4M {variant} selection is not validation-finalized")
    if not isinstance(record.get("checkpoint_sha256"), str) or len(record["checkpoint_sha256"]) != 64:
        raise ValueError(f"S4M {variant} selection checkpoint SHA-256 missing")
    return path


def s4m_subprocess_environment(base: dict[str, str] | None = None) -> dict[str, str]:
    environment = dict(os.environ if base is None else base)
    environment["PYTHONUNBUFFERED"] = "1"
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    return environment


def build_s4m_p4_commands(
    *, config: Path, mask_artifacts: Path, s4m_root: Path,
    output: Path, python: str,
) -> tuple[list[str], list[str], list[str], list[str]]:
    base = [python, "-m", "integrations.s4m.search",
            "--config", str(Path(config).resolve()),
            "--artifacts", str(Path(mask_artifacts).resolve()),
            "--s4m-root", str(Path(s4m_root).resolve()),
            "--output", str(Path(output).resolve()), "--device", "cuda"]
    official = ["--train-epochs", "100", "--patience", "100"]
    smoke = [*base, "--smoke"]
    search = [*base, "--profile", "official", *official]
    missing = [*base, "--evaluate-selected", "--selection",
               str(Path(output).resolve() / "selection-official.json"), *official]
    clean = [*base, "--evaluate-selected", "--selection",
             str(Path(output).resolve() / "selection-official-clean.json"), *official]
    return smoke, search, missing, clean


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mask-artifacts", type=Path, required=True)
    parser.add_argument("--s4m-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--phase", choices=("smoke", "search", "evaluate-missing", "evaluate-clean"), required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(preflight_s4m_p4(config=args.config, mask_artifacts=args.mask_artifacts,
                                      s4m_root=args.s4m_root), indent=2), flush=True)
    commands = build_s4m_p4_commands(config=args.config, mask_artifacts=args.mask_artifacts,
                                      s4m_root=args.s4m_root, output=args.output, python=args.python)
    index = ("smoke", "search", "evaluate-missing", "evaluate-clean").index(args.phase)
    if index >= 2 and not args.dry_run:
        require_s4m_selection(args.output, variant="missing" if index == 2 else "clean")
    command = commands[index]
    if args.dry_run:
        command = [*command, "--dry-run"]
    print("command:", " ".join(command), flush=True)
    completed = subprocess.run(command, check=False, env=s4m_subprocess_environment())
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
