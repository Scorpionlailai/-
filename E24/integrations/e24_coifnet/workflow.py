"""Detached supervisor: CPU acceptance, wait for GPU1, train, then 20-mask test."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
GPU = "GPU-1315e15c-d657-e61f-bde5-6e0b38e6e54b"
PYTHON = "/home/xyw/miniforge3/envs/malr/bin/python"


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def utc():
    return datetime.now(timezone.utc).isoformat()


def gpu_jobs():
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader"],
                            capture_output=True, text=True, check=True)
    return [line.strip() for line in result.stdout.splitlines() if GPU in line]


def supervise(output):
    from malr.evaluation.report import file_sha256
    from .run import DEFAULT_MANIFEST, DEFAULT_PARENT, DEFAULT_PROFILES, MODEL, PROTOCOL
    lock = (output / "supervisor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (output / "plan.json").exists():
        raise FileExistsError("experiment already prepared")
    paths = {p for directory in ("src", "integrations") for p in (ROOT / directory).rglob("*.py")}
    paths.update((DEFAULT_MANIFEST, DEFAULT_PARENT, DEFAULT_PROFILES, ROOT / "configs/e24_v79.yaml"))
    paths.update((ROOT.parent / "CoIFNet" / "model").glob("*.py"))
    paths.update((ROOT.parent / "CoIFNet" / "conf").rglob("*.yaml"))
    hashes = {str(p.resolve()): file_sha256(p) for p in sorted(paths)}
    with tarfile.open(output / "source.tar.gz", "w:gz") as archive:
        for path in hashes:
            archive.add(path, arcname=str(Path(path).relative_to(ROOT.parent)), recursive=False)
    plan = dict(model=MODEL, protocol_id=PROTOCOL, dataset="Weather", start_rate=.1, block_len=8,
        created_utc=utc(), stage1_reused=str(DEFAULT_PARENT), stage1_reused_sha256=file_sha256(DEFAULT_PARENT),
        trainable="CoIFNet only", repair="entire repair/retrieval/memory frozen", original_mask_retained=True,
        normalization="observed-only stats; completed history is not masked back to zero",
        coifnet=dict(batch=512, epochs=100, patience=10, learning_rate=.001, precision="float32",
                    loss="0.2 future MAE + 0.8 observed-history reconstruction MAE", seed=2023),
        repair_forward_batch=128, validation_repair_cache=True, dynamic_training_masks=True,
        primary_selection="fixed validation MAE", test_masks=20, exploratory=True,
        public_patchtst_protocol_compliant=False, gpu_uuid=GPU, wait_for_gpu_idle=True,
        dataset_manifest=str(DEFAULT_MANIFEST), source_sha256=hashes,
        source_archive_sha256=file_sha256(output / "source.tar.gz"), monitor_interval_seconds=300)
    write(output / "plan.json", plan)
    state = dict(status="running", controller_pid=os.getpid(), started_utc=utc(), stages=[])
    write(output / "status.json", state)
    def verify():
        for path, expected in hashes.items():
            if file_sha256(Path(path)) != expected:
                raise RuntimeError("sealed source changed: " + path)
    env = dict(os.environ, PYTHONPATH=".:src", CUDA_VISIBLE_DEVICES=GPU,
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONUNBUFFERED="1")
    for name in ("MALR_MISSING_SCENARIO", "E24_DATASET_MANIFEST", "MALR_E23_CODE_FREEZE"):
        env.pop(name, None)
    def execute(stage, args):
        verify()
        command = [PYTHON, "-u", "-m", "integrations.e24_coifnet.run", *args]
        log = output / (stage + ".log")
        started = time.monotonic()
        with log.open("x") as stream:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT)
            state.update(status="running", active_stage=stage, child_pid=child.pid, log=str(log))
            write(output / "status.json", state)
            print(json.dumps(dict(event="started", stage=stage, pid=child.pid, log=str(log))), flush=True)
            while True:
                try:
                    code = child.wait(timeout=300)
                    break
                except subprocess.TimeoutExpired:
                    progress = dict(time=utc(), stage=stage, pid=child.pid,
                        elapsed_seconds=time.monotonic()-started, tail=log.read_text(errors="replace").splitlines()[-3:])
                    with (output / "monitor.jsonl").open("a") as handle:
                        handle.write(json.dumps(progress) + "\n")
                    print(json.dumps(progress), flush=True)
        state["stages"].append(dict(stage=stage, exit_code=code, wall_seconds=time.monotonic()-started))
        write(output / "status.json", state)
        if code:
            raise RuntimeError(stage + " failed; see " + str(log))
    try:
        execute("preflight", ["preflight", "--device", "cpu", "--output", str(output / "preflight")])
        state.update(status="waiting_for_gpu", active_stage="waiting_for_gpu", child_pid=None)
        write(output / "status.json", state)
        idle_checks = 0; last_report = 0.
        while idle_checks < 3:
            jobs = gpu_jobs()
            idle_checks = 0 if jobs else idle_checks + 1
            state.update(gpu_jobs=jobs, last_check_utc=utc())
            write(output / "status.json", state)
            if time.monotonic() - last_report >= 300:
                print(json.dumps(dict(event="waiting_for_gpu", jobs=jobs, time=utc())), flush=True)
                last_report = time.monotonic()
            time.sleep(15)
        verify()
        state["gpu_jobs_at_training_start"] = gpu_jobs()
        execute("training", ["train", "--device", "cuda:0", "--output", str(output / "training")])
        execute("evaluation", ["evaluate", "--device", "cuda:0",
            "--selection", str(output / "training/selection.json"), "--output", str(output / "evaluation")])
        verify()
        state.update(status="completed", active_stage=None, child_pid=None, exit_code=0, finished_utc=utc())
        write(output / "status.json", state)
        print(json.dumps(dict(event="completed", output=str(output))), flush=True)
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}", exit_code=1, finished_utc=utc())
        write(output / "status.json", state)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["launch", "supervise"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "supervise":
        supervise(args.output.resolve())
        return
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    output = args.output or ROOT / f"artifacts/Weather/seq96_pred96/sr010_bl8/e24_v79_coifnet_v1_{stamp}"
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    command = [PYTHON, "-u", "-m", "integrations.e24_coifnet.workflow", "supervise", "--output", str(output)]
    env = dict(os.environ, PYTHONPATH=".:src", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONUNBUFFERED="1")
    with (output / "console.log").open("x") as stream:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    (output / "pid").write_text(str(child.pid) + "\n")
    write(output / "launch.json", dict(pid=child.pid, started_utc=utc(), cwd=str(ROOT),
        command=command, python=PYTHON, detached=True, environment={k: env[k] for k in ("PYTHONPATH", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}))
    print(json.dumps(dict(pid=child.pid, output=str(output), detached=True)), flush=True)


if __name__ == "__main__":
    main()
