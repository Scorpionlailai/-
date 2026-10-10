"""Move a still-unstarted experiment to an idle GPU without changing its model."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
from zoneinfo import ZoneInfo

from . import workflow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["launch", "supervise"])
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--replaces", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    workflow.GPU = args.gpu_uuid
    if args.action == "supervise":
        workflow.supervise(args.output.resolve())
        return
    if workflow.gpu_jobs():
        raise RuntimeError("destination GPU is occupied")
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    output = (args.output or workflow.ROOT / f"artifacts/Weather/seq96_pred96/sr010_bl8/e24_v79_coifnet_v1_{stamp}_gpu0").resolve()
    output.mkdir(parents=True, exist_ok=False)
    previous = args.replaces.resolve(strict=True)
    state = json.loads((previous / "status.json").read_text())
    if state["status"] != "waiting_for_gpu" or state.get("child_pid") is not None:
        raise RuntimeError("only a waiting experiment with no training child may be moved")
    pid = int(state["controller_pid"])
    command_line = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    if "integrations.e24_coifnet.workflow supervise" not in command_line or str(previous) not in command_line:
        raise RuntimeError("waiting controller process identity differs")
    os.kill(pid, signal.SIGTERM)
    state.update(status="superseded", active_stage=None, child_pid=None,
                 replacement_output=str(output), reason="GPU0 became idle; original training never started",
                 finished_utc=workflow.utc(), controller_termination_signal="SIGTERM")
    workflow.write(previous / "status.json", state)
    workflow.write(output / "replacement.json", dict(previous_output=str(previous), previous_pid=pid,
        original_training_started=False, gpu_uuid=args.gpu_uuid, numerical_configuration_unchanged=True))
    command = [workflow.PYTHON, "-u", "-m", "integrations.e24_coifnet.relocate", "supervise",
               "--gpu-uuid", args.gpu_uuid, "--output", str(output)]
    env = dict(os.environ, PYTHONPATH=".:src", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONUNBUFFERED="1")
    with (output / "console.log").open("x") as stream:
        child = subprocess.Popen(command, cwd=workflow.ROOT, env=env, stdin=subprocess.DEVNULL,
                                 stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    (output / "pid").write_text(str(child.pid) + "\n")
    workflow.write(output / "launch.json", dict(pid=child.pid, started_utc=workflow.utc(),
        cwd=str(workflow.ROOT), command=command, detached=True, gpu_uuid=args.gpu_uuid,
        previous_output=str(previous), python=workflow.PYTHON,
        environment={k: env[k] for k in ("PYTHONPATH", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}))
    print(json.dumps(dict(pid=child.pid, output=str(output), replaced=str(previous), detached=True)), flush=True)


if __name__ == "__main__":
    main()
