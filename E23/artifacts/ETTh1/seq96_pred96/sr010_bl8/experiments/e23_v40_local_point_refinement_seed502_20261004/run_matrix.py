"""Run both predeclared scopes sequentially with a shared frozen source."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
WORK=Path('/home/xyw/paper/.e23-v40-point/E23')
PARENT=ROOT.parent/'e23_t1_five_seed_standard_20261004/seed_502/E23-v31-B/evaluation/validation-lock.json'


def main():
    assert json.loads((ROOT/'preflight.json').read_text())['status']=='passed'
    env={**os.environ,'PYTHONPATH':'src:.','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}
    commands=[]
    for label in ['tail32','full96']:
        command=[sys.executable,'scripts/run_e23_learning_trial.py','--root',str(ROOT/label),
            '--config',str(ROOT/label/'e23_seed502.yaml'),'--label','E23-v40-'+label,
            '--repair-batch-size','128','--parent',str(PARENT)]
        commands.append(command)
    with (ROOT/'matrix-commands.json').open('x') as stream:json.dump(commands,stream,indent=2)
    for label,command in zip(['tail32','full96'],commands):
        print(json.dumps({'event':'start_trial','trial':label}),flush=True)
        with (ROOT/label/'coordinator.log').open('x') as stream:
            subprocess.run(command,cwd=WORK,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
        print(json.dumps({'event':'completed_trial','trial':label}),flush=True)
    print(json.dumps({'status':'matrix_complete'}),flush=True)


if __name__=='__main__':main()
