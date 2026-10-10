import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest


LAUNCHER = Path(__file__).resolve().parents[2] / "scripts" / "run_background.py"


@pytest.mark.parametrize("code", [0, 7])
def test_background_job_detaches_and_records_exit_status(tmp_path, code):
    job = tmp_path / "job with spaces"
    launched = subprocess.run([
        sys.executable, str(LAUNCHER), "--job-dir", str(job), "--cwd", str(tmp_path), "--",
        sys.executable, "-u", "-c",
        "import os,time,sys; time.sleep(.3); "
        "print('progress', os.getcwd(), os.getsid(0), flush=True); "
        f"sys.exit({code})",
    ], capture_output=True, text=True, timeout=5)
    assert launched.returncode == 0, launched.stderr
    info = json.loads(launched.stdout)
    assert int((job / "pid").read_text()) == info["supervisor_pid"]
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        status = json.loads((job / "status.json").read_text())
        if status["state"] == "finished":
            break
        time.sleep(.05)
    assert status["state"] == "finished"
    assert status["exit_code"] == code
    log = (job / "console.log").read_text()
    assert f"progress {tmp_path} {info['supervisor_pid']}" in log
    assert info["supervisor_pid"] != os.getsid(0)
    assert (job / "launch.json").is_file()


def test_background_job_does_not_overwrite_existing_job(tmp_path):
    job = tmp_path / "existing"
    job.mkdir()
    marker = job / "console.log"
    marker.write_text("previous result")
    launched = subprocess.run([
        sys.executable, str(LAUNCHER), "--job-dir", str(job), "--",
        sys.executable, "-c", "print('unwanted')",
    ], capture_output=True, text=True, timeout=5)
    assert launched.returncode != 0
    assert "already exists" in launched.stderr
    assert marker.read_text() == "previous result"
