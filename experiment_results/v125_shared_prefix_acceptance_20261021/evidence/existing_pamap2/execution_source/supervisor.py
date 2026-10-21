"""Restricted two-GPU queue; verify, publish results, then stop without Stage2."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from integrations.v125_ablation.common import write_json
from calibration_worker import OUT,check_lock
from export_results import build
TASKS=[('A0',2024),('A1',2024),('A0',2025),('A1',2025),('A0',2023),('A1',2023)]

def main():
    check_lock()
    with (OUT/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pending=list(TASKS);running={};finished=[];failed=[];last_export=0.
        write_json(OUT/'supervisor.json',dict(pid=os.getpid(),tasks=TASKS,started_unix=time.time()))
        while pending or running:
            for gpu,(process,arm,seed,log) in list(running.items()):
                code=process.poll()
                if code is None:continue
                log.close();record=dict(gpu=gpu,arm=arm,seed=seed,exit_code=code);finished.append(record)
                if code:failed.append(record)
                del running[gpu]
            if failed:pending=[]
            for gpu in (0,1):
                if gpu in running or not pending:continue
                arm,seed=pending.pop(0);folder=OUT/arm/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
                if (folder/'evaluation_complete.json').exists():finished.append(dict(arm=arm,seed=seed,exit_code=0,reused_completed=True));continue
                log=(folder/'worker.log').open('a');env=os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
                process=subprocess.Popen([sys.executable,'-u',str(OUT/'tools/calibration_worker.py'),'--arm',arm,'--seed',str(seed)],
                    env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                running[gpu]=(process,arm,seed,log)
            state=dict(supervisor_pid=os.getpid(),running=[dict(gpu=g,pid=x[0].pid,arm=x[1],seed=x[2]) for g,x in running.items()],
                pending=[dict(arm=a,seed=s,kind='read_only_verified_reuse' if s==2023 else 'original64and256_100epochs_each') for a,s in pending],
                finished=finished,failed=failed,phase='failed' if failed else 'running' if pending or running else 'trained_and_evaluated',updated_unix=time.time())
            write_json(OUT/'queue_status.json',state)
            if time.time()-last_export>120 or not pending and not running:build(False);last_export=time.time()
            if pending or running:time.sleep(20)
        if failed:
            write_json(OUT/'completion.json',dict(complete=False,failed=failed,no_other_stages=True));return
        write_json(OUT/'queue_status.json',dict(**{**state,'phase':'final_verification'}))
        from verify_results import verify
        verification=verify();assert verification['passed']
        write_json(OUT/'queue_status.json',dict(**{**state,'phase':'publishing'}))
        build(True)
        write_json(OUT/'completion.json',dict(complete=True,finished=finished,failed=[],no_calibration_search_or_stage2=True,
            verification='final_verification.json',publication=json.loads((OUT/'publication.json').read_text()),completed_unix=time.time()))
        write_json(OUT/'queue_status.json',dict(**{**state,'phase':'complete','updated_unix':time.time()}))

if __name__=='__main__':
    try:main()
    except BaseException:
        write_json(OUT/'supervisor_failure.json',dict(pid=os.getpid(),traceback=traceback.format_exc(),time=time.time()));raise
