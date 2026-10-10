"""Run the predeclared v125 scene transfer; retain every artifact and failure."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, subprocess, sys, time, traceback

ROOT = Path(__file__).resolve().parent


def status(state, **extra):
    record = dict(status=state, updated_at=datetime.now(timezone.utc).isoformat(), **extra)
    temporary = ROOT/'status.tmp'
    temporary.write_text(json.dumps(record, indent=2)+'\n')
    temporary.replace(ROOT/'status.json')
    print(json.dumps(record), flush=True)


def frozen_inputs():
    for path, digest in json.loads((ROOT/'source_freeze.json').read_text())['files'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path


def environment(gpu_uuid):
    env = dict(os.environ)
    env['CUDA_VISIBLE_DEVICES'] = gpu_uuid
    env['PYTHONPATH'] = workflow['source']+':'+workflow['source']+'/src'
    env['OMP_NUM_THREADS'] = '4'
    return env


def tail(path):
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size-12000))
        lines = stream.read().decode(errors='replace').splitlines()
    return lines[-1] if lines else ''


def sequential(script, args, directory, gpu_uuid):
    frozen_inputs()
    log = directory/(Path(script).stem+'.log')
    with log.open('xb') as stream:
        process = subprocess.Popen([sys.executable, '-u', str(directory/script), *args],
                                   cwd=directory, env=environment(gpu_uuid),
                                   stdout=stream, stderr=subprocess.STDOUT)
        while process.poll() is None:
            status('running', phase=script, pid=process.pid, last_line=tail(log))
            time.sleep(15)
        assert process.returncode == 0, f'{script} exited {process.returncode}; inspect {log}'


def main():
    frozen_inputs()
    processes = []
    streams = []
    for branch in workflow['branches']:
        directory = Path(branch['root'])
        stream = (directory/'train_refinement.log').open('xb')
        streams.append(stream)
        process = subprocess.Popen([sys.executable, '-u', str(directory/'train_refinement.py'), 'sr015_bl8'],
                                   cwd=directory, env=environment(branch['gpu_uuid']),
                                   stdout=stream, stderr=subprocess.STDOUT)
        (directory/'launch.json').write_text(json.dumps(dict(pid=process.pid, width=branch['width'],
            gpu_uuid=branch['gpu_uuid'], source=workflow['source'],
            started_at=datetime.now(timezone.utc).isoformat()), indent=2)+'\n')
        processes.append((branch, process))
    while any(process.poll() is None for _, process in processes):
        progress = []
        for branch, process in processes:
            progress.append(dict(width=branch['width'], pid=process.pid, exit_code=process.poll(),
                                 last_line=tail(Path(branch['root'])/'train_refinement.log')))
        status('running', phase='parallel_calibration_training', progress=progress)
        time.sleep(30)
    for stream in streams:
        stream.close()
    assert all(process.returncode == 0 for _, process in processes), 'A calibration failed; inspect branch logs'
    for branch, _ in processes:
        record = json.loads(Path(branch['selection']).read_text())
        assert record['epochs_completed'] == 100 and record['optimizer_updates'] == 28600
        assert record['selection_metric'] == 'mae' and record['seed'] == 2023
    with (ROOT/'progress.md').open('a') as stream:
        stream.write('\nTask2: complete — both full100-epoch/28600-update calibrations finished, MAE/MSE paired records saved.\n')
    directory = Path(workflow['ensemble_root'])
    gpu_uuid = workflow['branches'][0]['gpu_uuid']
    sequential('materialize.py', [], directory, gpu_uuid)
    preflight = json.loads((directory/'preflight.json').read_text())
    assert preflight['validation_alpha_locked'] == .5 and preflight['strict_restore_passed']
    sequential('evaluate_repair.py', ['--selection', 'mae'], directory, gpu_uuid)
    sequential('summarize_transfer.py', [], ROOT, gpu_uuid)
    frozen_inputs()
    result = json.loads((directory/'E23-v125/sr015_bl8/repair_evaluation_mae/twenty-test-draws.json').read_text())
    status('completed', exit_code=0, forecast_deferred=True,
           metrics={k: result['summary'][k]['mean'] for k in result['summary'] if k.startswith('repair')},
           comparison_csv=str(ROOT/'repair_comparison.csv'), report=str(ROOT/'result.md'))


workflow = json.loads((ROOT/'workflow.json').read_text())
if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status('failed', error=repr(error), traceback=traceback.format_exc())
        raise
