#!/usr/bin/env python3
"""Finish the ETTh1 P4 comparison after its detached S4M search succeeds.

This is a one-off experiment continuation, not a training entry point. Every
evaluation and the final report gets its own run_background.py job and log.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/forecast_comparison_v1"
MASKS = ROOT / "artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context"
CONFIG = ROOT / "configs/stage1_seq96_dynamic_temporal_repair.yaml"
S4M_ROOT = Path("/home/xyw/missing_aware_latent_recovery/external/S4M")
MALR_PYTHON = Path("/home/xyw/miniforge3/envs/malr/bin/python")
S4M_PYTHON = Path("/home/xyw/miniforge3/envs/malr-s4m/bin/python")
SEARCH_JOB = BASE / "jobs/s4m_search_unbuffered_20260929"


def _status(job: Path) -> dict:
    path = job / "status.json"
    if not path.is_file():
        raise RuntimeError(f"missing background-job status: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def wait_for_success(job: Path, poll_seconds: float) -> None:
    while True:
        status = _status(job)
        if status.get("state") == "finished":
            if status.get("exit_code") != 0:
                raise RuntimeError(f"background job failed: {job} (exit {status.get('exit_code')})")
            print(f"finished successfully: {job}", flush=True)
            return
        if status.get("state") == "running" and not _alive(status.get("supervisor_pid")):
            raise RuntimeError(f"background supervisor disappeared with stale status: {job}")
        print(f"waiting: {job}; state={status.get('state')}; log={job / 'console.log'}", flush=True)
        time.sleep(poll_seconds)


def launch_and_wait(job: Path, command: list[str], poll_seconds: float) -> None:
    if job.exists():
        raise RuntimeError(f"job directory already exists; refusing to overwrite: {job}")
    subprocess.run(
        [str(MALR_PYTHON), str(ROOT / "scripts/run_background.py"),
         "--job-dir", str(job), "--", *command],
        cwd=ROOT, check=True,
    )
    wait_for_success(job, poll_seconds)


def s4m_evaluation_command(phase: str) -> list[str]:
    if phase not in ("evaluate-missing", "evaluate-clean"):
        raise ValueError(f"invalid S4M evaluation phase: {phase}")
    return [
        "/usr/bin/env", "CUDA_DEVICE_ORDER=PCI_BUS_ID", "CUDA_VISIBLE_DEVICES=1",
        "PYTHONUNBUFFERED=1", f"PYTHONPATH={ROOT}:{ROOT / 'src'}",
        str(S4M_PYTHON), "-u", "-m", "integrations.forecast_compare.s4m_p4",
        "--config", str(CONFIG), "--mask-artifacts", str(MASKS),
        "--s4m-root", str(S4M_ROOT), "--output", str(BASE / "s4m/official_p4_v2"),
        "--python", str(S4M_PYTHON), "--phase", phase,
    ]


def report_command() -> list[str]:
    return [
        "/usr/bin/env", f"PYTHONPATH={ROOT}:{ROOT / 'src'}", str(MALR_PYTHON), "-u",
        "-m", "integrations.forecast_compare.report",
        "--clean-cache", str(ROOT / "artifacts/ETTh1/seq96_pred96/clean/b5932e46c76ee3e89ccbc51bec219b521bb1c8b4ed085d063a8c1b89102148fe/baseline.json"),
        "--missing-cache", str(ROOT / "artifacts/ETTh1/seq96_pred96/sr010_bl8/missing/292570de0560129a4bd96568f13fc19875e06e80fd7041a7e7f8d06a89710f10/baseline.json"),
        "--e22-evaluation", str(MASKS / "evaluations/ETTh1-ours-dea387986927.json"),
        "--saits-evaluation", str(BASE / "saits/evaluations/saits-test-v2.json"),
        "--s4m-output", str(BASE / "s4m/official_p4_v2"),
        "--output", str(BASE / "reports/comparison_p4_v2"),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    jobs = (
        (BASE / "jobs/s4m_missing_test_20260929", s4m_evaluation_command("evaluate-missing")),
        (BASE / "jobs/s4m_clean_test_20260929", s4m_evaluation_command("evaluate-clean")),
        (BASE / "jobs/report_20260929", report_command()),
    )
    if args.dry_run:
        print(json.dumps({"search_job": str(SEARCH_JOB),
                          "steps": [{"job": str(job), "command": command} for job, command in jobs]},
                         indent=2), flush=True)
        return 0
    for job, _ in jobs:
        if job.exists():
            raise RuntimeError(f"job directory already exists; refusing to overwrite: {job}")
    wait_for_success(SEARCH_JOB, args.poll_seconds)
    for filename in ("selection-official.json", "selection-official-clean.json"):
        selection = BASE / "s4m/official_p4_v2" / filename
        if not selection.is_file():
            raise RuntimeError(f"S4M validation selection is missing: {selection}")
    for job, command in jobs:
        launch_and_wait(job, command, args.poll_seconds)
    report = BASE / "reports/comparison_p4_v2/comparison.json"
    rows = json.loads(report.read_text(encoding="utf-8"))["rows"]
    print(json.dumps({"report": str(report), "rows": rows}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
