"""Two GPUs, mandatory coarse/calibration/global-test barriers, publish and stop."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from policy6 import OUT,REPO
from run6 import check_lock,bind_calibration_sources,lock_all_checkpoints
from export6 import build
from integrations.v125_ablation.common import write_json


def run_phase(phase,seeds):
    pending=list(seeds);running={};finished=[];last_export=0.
    while pending or running:
        for gpu,item in list(running.items()):
            process,seed,log=item;code=process.poll()
            if code is None:continue
            log.close();finished.append(dict(phase=phase,seed=seed,gpu=gpu,exit_code=code));del running[gpu]
            if code:
                write_json(OUT/'queue_failure.json',dict(phase=phase,failed_seed=seed,exit_code=code,remaining_jobs_stopped=True))
                # Let already-running paired job finish its phase; no dependent phase is launched.
                for other,_,handle in running.values():other.wait();handle.close()
                raise RuntimeError(f'{phase} seed{seed} failed; no subsequent stage launched')
        for gpu in (0,1):
            if gpu in running or not pending:continue
            seed=pending.pop(0);folder=OUT/'A5'/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
            marker=folder/({'coarse':'coarse_complete.json','calibration':'calibration_complete.json','evaluate':'evaluation_complete.json'}[phase])
            if marker.exists():finished.append(dict(phase=phase,seed=seed,reused_completed=True,exit_code=0));continue
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
            log=(folder/f'{phase}_worker.log').open('a')
            p=subprocess.Popen([sys.executable,'-u',str(OUT/'tools/run6.py'),'--phase',phase,'--seed',str(seed)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            running[gpu]=(p,seed,log)
        write_json(OUT/'queue_status.json',dict(phase=phase,supervisor_pid=os.getpid(),running=[dict(gpu=g,pid=p.pid,variant='A5',seed=s) for g,(p,s,l) in running.items()],pending=pending,finished=finished,updated_unix=time.time()))
        if time.time()-last_export>180 or not pending and not running:build();last_export=time.time()
        if pending or running:time.sleep(20)
    return finished


def main():
    check_lock()
    with (OUT/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write_json(OUT/'supervisor.json',dict(pid=os.getpid(),started_unix=time.time(),scope='A5seeds2024/2025coarse+calibration;reuse2023;barredtest;noStage2'))
        coarse=run_phase('coarse',[2024,2025])
        # Deliver validation/resource milestone before opening calibration training.
        if not (OUT/'coarse_publication.json').exists():build(coarse_milestone=True)
        bind_calibration_sources()
        calibration=run_phase('calibration',[2024,2025])
        lock_all_checkpoints()
        evaluation=run_phase('evaluate',[2023,2024,2025])
        write_json(OUT/'queue_status.json',dict(phase='capacity_same_GPU_no_training',running=[],pending=[],supervisor_pid=os.getpid(),updated_unix=time.time()))
        if not (OUT/'capacity_complete.json').exists():
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
            with (OUT/'capacity.log').open('a') as log:subprocess.run([sys.executable,'-u',str(OUT/'tools/run6.py'),'--phase','capacity'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        write_json(OUT/'queue_status.json',dict(phase='final_verification',running=[],pending=[],updated_unix=time.time()))
        from verify6 import verify
        assert verify()['passed'];build(final=True)
        write_json(OUT/'completion.json',dict(complete=True,coarse=coarse,calibration=calibration,evaluation=evaluation,
            no_stage2_or_other_variants=True,publication=json.loads((OUT/'publication.json').read_text()),completed_unix=time.time()))
        write_json(OUT/'queue_status.json',dict(phase='complete',running=[],pending=[],updated_unix=time.time()))
if __name__=='__main__':
    try:main()
    except BaseException:
        write_json(OUT/'supervisor_failure.json',dict(pid=os.getpid(),traceback=traceback.format_exc(),time=time.time()));raise
