"""One independent experiment per GPU, with a global repair/forecast barrier."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from .data import ROOT, SCENES, write_json
from .report import MODELS, read, render


def process_token(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except FileNotFoundError:
        return None


def complete(root, job, phase):
    model, dataset, scene = job
    run = root / dataset / scene / model / 'seed_2023'
    if phase == 'stage1':
        return (run / 'evaluation/repair_summary.json').exists()
    return (read(run / 'status.json') or {}).get('phase') == 'stage2_complete'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--gpus', nargs='+', default=['0', '1'])
    parser.add_argument('--adopt', type=Path)
    args = parser.parse_args()
    if len(set(args.gpus)) != len(args.gpus):
        raise ValueError('GPU identifiers must be distinct')
    root = args.root.resolve()
    lock = (root / 'queue.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    matrix = [(m, d, s) for m in MODELS for d in ('Weather', 'ETTh1') for s in SCENES]
    envs = dict(TimesNet='paper-timesnet', ImputeFormer='paper-imputeformer',
                BRITS='paper-brits', CSDI='paper-csdi')
    write_json(root / 'queue_plan.json', dict(expected_repair_runs=32, expected_forecast_runs=32,
        matrix=[dict(model=m, dataset=d, scene=s) for m,d,s in matrix], gpus=args.gpus,
        concurrency=len(args.gpus), phase_barrier='all32_repair_evaluations_before_any_forecast_training',
        github_policy='local_only_until_user_review'))
    active = {}
    if args.adopt:
        payload = read(args.adopt)
        records = payload.get('active', [payload])
        for record in records:
            gpu = str(record['gpu'])
            if gpu not in args.gpus or record['phase'] != 'stage1' or gpu in active:
                raise ValueError('invalid adopted GPU or phase')
            if process_token(record['child_pid']) == record['process_token']:
                active[gpu] = dict(record, queue_pid=os.getpid(),
                                   job=(record['model'], record['dataset'], record['scene']),
                                   child=None, stream=None)
            elif not complete(root, (record['model'], record['dataset'], record['scene']), 'stage1'):
                raise RuntimeError('adopted experiment exited without completion')
    for phase in ('stage1', 'stage2'):
        if phase == 'stage2' and not all(complete(root, j, 'stage1') for j in matrix):
            write_json(root / 'queue_status.json', dict(state='repair_failures_require_fix', phase=phase))
            render(root)
            return
        adopted = {r['job'] for r in active.values()}
        pending = [j for j in matrix if not complete(root, j, phase) and j not in adopted]
        failed = []
        next_report = 0.
        while pending or active:
            for gpu, record in list(active.items()):
                child = record['child']
                alive = child.poll() is None if child is not None else process_token(record['child_pid']) == record['process_token']
                if alive:
                    continue
                success = complete(root, record['job'], phase) and (child is None or child.returncode == 0)
                if record['stream'] is not None:
                    record['stream'].close()
                finished = {k:v for k,v in record.items() if k not in ('child', 'stream', 'job')}
                finished.update(state='completed' if success else 'failed', elapsed_seconds=time.time()-record['started_unix'])
                with (root / 'queue_journal.jsonl').open('a') as stream:
                    stream.write(json.dumps(finished)+'\n')
                if not success:
                    failed.append(record['job'])
                del active[gpu]
                render(root)
            paused = (root / 'PAUSE').exists()
            for gpu in args.gpus:
                if paused or gpu in active or not pending:
                    continue
                model, dataset, scene = pending.pop(0)
                if dataset == 'ETTh1' and not (root / 'etth1_rebuilt_protocol_accepted.json').exists():
                    raise RuntimeError('ETTh1 protocol authorization missing')
                if not (read(root / f'preflight_{model}.json') or {}).get('passed'):
                    raise RuntimeError(f'preflight missing: {model}')
                interpreter = Path('/home/xyw/.local/share/mamba/envs') / envs[model] / 'bin/python'
                environments = root / 'environments'
                environments.mkdir(exist_ok=True)
                snapshot = environments / f'{model}.txt'
                if not snapshot.exists():
                    snapshot.write_text(subprocess.check_output([str(interpreter), '-m', 'pip', 'freeze'], text=True))
                command = [str(interpreter), '-u', '-m', 'integrations.repair_baselines.run',
                           '--model', model, '--dataset', dataset, '--scene', scene,
                           '--phase', phase, '--root', str(root)]
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                           PYTHONPATH=f'{ROOT}/E24/src:{ROOT}/E24')
                logfile = root / 'logs' / f'{phase}_{model}_{dataset}_{scene}.log'
                stream = logfile.open('a')
                child = subprocess.Popen(command, env=env, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                record = dict(gpu=gpu, model=model, dataset=dataset, scene=scene, phase=phase,
                              child_pid=child.pid, queue_pid=os.getpid(), command=command,
                              log=str(logfile), started_unix=time.time(), process_token=process_token(child.pid))
                print(json.dumps(record), flush=True)
                active[gpu] = dict(record, job=(model,dataset,scene), child=child, stream=stream)
            if time.monotonic() >= next_report or not active:
                running = [{k:v for k,v in r.items() if k not in ('child','stream','job')} for r in active.values()]
                write_json(root / 'queue_status.json', dict(state='running' if active else 'paused_before_next_run' if paused else 'phase_finished',
                    phase=phase, queue_pid=os.getpid(), gpus=args.gpus, active=running,
                    pending_runs=len(pending), failed_runs=failed))
                render(root)
                next_report = time.monotonic()+30
            if paused and not active:
                return
            if pending or active:
                time.sleep(5)
        if failed:
            write_json(root / 'queue_status.json', dict(state='failures_require_fix', phase=phase, failed_runs=failed))
            return
    write_json(root / 'queue_status.json', dict(state='complete', gpus=args.gpus, github_uploaded=False))
    render(root)


if __name__ == '__main__':
    main()
