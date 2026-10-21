"""User-authorized scheduling preemption; numerical acceptance code is unchanged."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

L=Path('/home/xyw/v125-shared-prefix-acceptance-20261021')
A=Path('/home/xyw/paper/E24/artifacts/v125_a4_seed_recheck_20261021')
PIDS={2024:647981,2025:647982}
SUPERVISOR=647976
WAITER=656334


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def state(pid):
    try:return Path(f'/proc/{pid}/status').read_text().split('State:')[1].split()[0]
    except (OSError,IndexError):return None


def main():
    assert not (L/'executed_once.json').exists()
    record=dict(authorized_by='User: 你先把要上传的任务优先处理了',started_unix=time.time(),
                numerical_protocol_changed=False,resume_original_A4_after_upload=True,workers={})
    write(L/'priority_status.json',dict(phase='waiting_for_committed_A4_epoch_boundary',**record))
    env=os.environ.copy()
    original=dict(x.split(b'=',1) for x in Path(f'/proc/{SUPERVISOR}/environ').read_bytes().split(b'\0') if b'=' in x)
    for name in ('PYTHONPATH','CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','CUBLAS_WORKSPACE_CONFIG','PYTHONUNBUFFERED','PATH'):
        if name.encode() in original:env[name]=original[name.encode()].decode()
    cwd=os.readlink(f'/proc/{SUPERVISOR}/cwd')
    starting={seed:json.loads((A/'A4'/f'seed_{seed}/status.json').read_text())['epoch'] for seed in PIDS}
    pending=set(PIDS);deadline=time.time()+180
    while pending:
        for seed in list(pending):
            p=A/'A4'/f'seed_{seed}/status.json'
            try:row=json.loads(p.read_text())
            except (OSError,json.JSONDecodeError):continue
            if row.get('epoch',0)<=starting[seed]:continue
            os.kill(PIDS[seed],signal.SIGSTOP)
            while state(PIDS[seed]) not in ('T','t'):time.sleep(.005)
            target=L/'priority_local_assets'/f'seed_{seed}';target.mkdir(parents=True,exist_ok=False)
            for name in ('resume.pt','selected.pt'):
                source=p.parent/'coarse'/name;shutil.copyfile(source,target/name)
                assert sha(source)==sha(target/name)
            shutil.copyfile(p,target/'status.json')
            record['workers'][str(seed)]=dict(pid=PIDS[seed],committed_epoch=row['epoch'],optimizer_steps=row['optimizer_steps'],
                resume_path=str(target/'resume.pt'),resume_sha256=sha(target/'resume.pt'),
                selected_path=str(target/'selected.pt'),selected_sha256=sha(target/'selected.pt'),
                scope='Original epoch checkpoint contains model/Adam/Python-NumPy-CPU-CUDA RNG/retrieval EMA/best/selected/cursors; original worker resumes it. Any uncommitted work after checkpoint is discarded and not selected.')
            pending.remove(seed)
        if time.time()>deadline:raise RuntimeError('no committed epoch boundary within180seconds; acceptance not launched')
        time.sleep(.01)
    # These CPU-only reads verify completeness before releasing either GPU context.
    import torch
    for seed in PIDS:
        info=record['workers'][str(seed)];checkpoint=torch.load(info['resume_path'],map_location='cpu',weights_only=False)
        assert checkpoint['epoch']==info['committed_epoch'] and checkpoint['steps']==info['optimizer_steps']==85*checkpoint['epoch']
        assert all(key in checkpoint for key in ('model','optimizer','rng','retrieval_ema','best','selected','model_epoch','batch_index'))
        assert checkpoint['optimizer']['state'] and checkpoint['rng']
        del checkpoint
    for name in ('supervisor.json','queue_status.json','report_status.json'):
        if (A/name).exists():shutil.copyfile(A/name,L/'priority_local_assets'/('before_'+name))
    write(L/'priority_pause_record.json',record)
    # Terminating stopped processes releases contexts; restarting the original supervisor
    # later reuses both committed checkpoints without changing its sealed sources.
    for pid in (SUPERVISOR,*PIDS.values()):os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
    deadline=time.time()+60
    while any(state(pid) not in (None,'Z','X') for pid in (SUPERVISOR,*PIDS.values())):
        if time.time()>deadline:raise RuntimeError('original A4 processes did not exit; no acceptance launch')
        time.sleep(.1)
    write(L/'priority_status.json',dict(phase='A4_checkpointed_GPU_released_acceptance_prioritized',updated_unix=time.time(),**record))
    os.kill(WAITER,signal.SIGCONT)
    # The original one-shot waiter owns execution and result publication. Never rerun it.
    while state(WAITER) not in (None,'Z','X'):time.sleep(5)
    active=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:args=(p/'cmdline').read_bytes().split(b'\0')
        except OSError:continue
        if str(L/'src/run_acceptance.py').encode() in args and state(int(p.name)) not in ('Z','X'):active.append(p.name)
    if active:raise RuntimeError('acceptance child still active after waiter exit; A4 remains paused to preserve timing')
    # Resume A4 even when the finite acceptance failed; neither experiment is retried.
    log=(L/'A4_resumed_supervisor.log').open('a')
    command=[sys.executable,'-u',str(A/'tools/supervisor3.py')]
    p=subprocess.Popen(command,cwd=cwd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close()
    result=dict(phase='acceptance_finished_A4_resumed',A4_supervisor_pid=p.pid,A4_command=command,A4_cwd=cwd,resumed_unix=time.time(),
        numerical_protocol_changed=False,acceptance_publication=json.loads((L/'publication.json').read_text()) if (L/'publication.json').exists() else None,
        acceptance_waiter_failure=json.loads((L/'waiter_failure.json').read_text()) if (L/'waiter_failure.json').exists() else None)
    write(L/'priority_status.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException:
        write(L/'priority_failure.json',dict(time_unix=time.time(),traceback=traceback.format_exc(),A4_states={str(p):state(p) for p in (SUPERVISOR,*PIDS.values())}))
        raise
