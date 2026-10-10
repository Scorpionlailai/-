#!/usr/bin/env python3
"""Run a command in a detached session, preserving logs, PIDs and exit status."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _run(job):
    # Session detachment handles terminal loss; ignore explicit shell HUP too.
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    spec = json.loads((job / "launch.json").read_text(encoding="utf-8"))
    status = {"state": "running", "supervisor_pid": os.getpid(), "started_at": _now()}
    _write(job / "status.json", status)
    code = 127
    try:
        child = subprocess.Popen(spec["command"], cwd=spec["cwd"], stdin=subprocess.DEVNULL)
        status["child_pid"] = child.pid
        _write(job / "status.json", status)
        code = child.wait()
    except Exception as exc:
        status["error"] = str(exc)
        print(f"background launch failed: {exc}", flush=True)
    finally:
        status.update(state="finished", exit_code=code, finished_at=_now())
        _write(job / "status.json", status)
        print(f"background job finished: exit_code={code}", flush=True)
    return code if code >= 0 else 128 - code


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--_run":
        return _run(Path(sys.argv[2]))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("provide a command after --")
    cwd = args.cwd.resolve(strict=True)
    job = args.job_dir.resolve()
    try:
        job.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f"job directory already exists: {job}; choose a new name")
    _write(job / "launch.json", {"command": command, "cwd": str(cwd), "created_at": _now()})
    _write(job / "status.json", {"state": "launching"})
    with (job / "console.log").open("xb") as log:
        child = subprocess.Popen(
            [sys.executable, "-u", str(Path(__file__).resolve()), "--_run", str(job)],
            cwd=cwd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, close_fds=True,
        )
    (job / "pid").write_text(str(child.pid) + "\n", encoding="utf-8")
    print(json.dumps({"supervisor_pid": child.pid, "job_dir": str(job)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
