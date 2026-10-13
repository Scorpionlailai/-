"""Sequential GPU queue; complete all repair comparisons before forecasting."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from .data import ROOT, SCENES, write_json
from .report import MODELS, render, read


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--gpu', default='1')
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (root / 'queue.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    matrix = [(model, dataset, scene) for model in MODELS
              for dataset in ('Weather', 'ETTh1') for scene in SCENES]
    envs = dict(TimesNet='paper-timesnet', ImputeFormer='paper-imputeformer', BRITS='paper-brits', CSDI='paper-csdi')
    write_json(root / 'queue_plan.json', dict(expected_repair_runs=32, expected_forecast_runs=32,
        matrix=[dict(model=m, dataset=d, scene=s) for m,d,s in matrix], gpu=args.gpu,
        phase_barrier='all32_repair_evaluations_before_any_forecast_training',
        github_policy='local_only_until_user_review'))
    (root / 'logs').mkdir(exist_ok=True)
    render(root)
    for phase in ('stage1', 'stage2'):
        if phase == 'stage2' and not all((root/d/s/m/'seed_2023/evaluation/repair_summary.json').exists() for m,d,s in matrix):
            write_json(root / 'queue_status.json', dict(state='repair_failures_require_fix', phase=phase))
            return
        for model, dataset, scene in matrix:
            if dataset == 'ETTh1' and not (root / 'etth1_rebuilt_protocol_accepted.json').exists():
                write_json(root / 'queue_status.json', dict(state='waiting_etth1_protocol_decision', phase=phase))
                return
            if (root / 'PAUSE').exists():
                write_json(root / 'queue_status.json', dict(state='paused_before_next_run', phase=phase))
                return
            run = root / dataset / scene / model / 'seed_2023'
            status = read(run / 'status.json') or {}
            if status.get('phase') == phase+'_complete' or (phase=='stage1' and (run/'evaluation/repair_summary.json').exists()):
                continue
            acceptance = read(root / f'preflight_{model}.json')
            if not acceptance or not acceptance.get('passed'):
                raise RuntimeError(f'preflight missing/failed: {model}')
            interpreter = Path('/home/xyw/.local/share/mamba/envs') / envs[model] / 'bin/python'
            environments = root / 'environments'
            environments.mkdir(exist_ok=True)
            snapshot = environments / f'{model}.txt'
            if not snapshot.exists():
                snapshot.write_text(subprocess.check_output([str(interpreter), '-m', 'pip', 'freeze'], text=True))
            command = [str(interpreter), '-u', '-m', 'integrations.repair_baselines.run',
                       '--model', model, '--dataset', dataset, '--scene', scene,
                       '--phase', phase, '--root', str(root)]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                       PYTHONPATH=f'{ROOT}/E24/src:{ROOT}/E24')
            logfile = root / 'logs' / f'{phase}_{model}_{dataset}_{scene}.log'
            started = time.time()
            with logfile.open('a') as stream:
                child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
                launch = dict(state='running', phase=phase, model=model, dataset=dataset, scene=scene,
                              child_pid=child.pid, queue_pid=os.getpid(), command=command, log=str(logfile), started_unix=started)
                write_json(root / 'queue_status.json', launch)
                print(json.dumps(launch), flush=True)
                next_report = time.monotonic()+300
                while child.poll() is None:
                    time.sleep(5)
                    if time.monotonic() >= next_report:
                        current = dict(launch, elapsed_seconds=time.time()-started, run_status=read(run/'status.json'))
                        with (root / 'monitor.jsonl').open('a') as monitor:
                            monitor.write(json.dumps(current)+'\n')
                        write_json(root / 'queue_status.json', current)
                        render(root)
                        next_report = time.monotonic()+300
            finished = dict(launch, state='completed' if child.returncode==0 else 'failed',
                            returncode=child.returncode, elapsed_seconds=time.time()-started)
            with (root / 'queue_journal.jsonl').open('a') as journal:
                journal.write(json.dumps(finished)+'\n')
            if child.returncode != 0:
                run.mkdir(parents=True, exist_ok=True)
                write_json(run / 'status.json', dict(phase=phase+'_failed', log=str(logfile), returncode=child.returncode))
            render(root)
    write_json(root / 'queue_status.json', dict(state='complete', github_uploaded=False))
    render(root)


if __name__ == '__main__':
    main()
