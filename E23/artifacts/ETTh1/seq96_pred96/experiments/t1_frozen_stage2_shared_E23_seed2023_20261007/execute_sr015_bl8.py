"""Add the user-requested 0.15/8 scene without altering the original run plan."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert json.loads((ROOT/'training-complete.json').read_text())['status']=='completed'
    plan=json.loads((ROOT/'plan_sr015_bl8.json').read_text())
    frozen=json.loads((ROOT/'frozen-addendum.json').read_text())
    work=Path(plan['work'])
    runner=work/'scripts/run_t1_frozen_stage2.py'
    env={**os.environ,'PYTHONPATH':f'{work}/src:{work}','PYTHONUNBUFFERED':'1'}
    env.pop('MALR_MISSING_SCENARIO',None)
    for mode in ['preflight','training']:
        assert sha(ROOT/'plan_sr015_bl8.json')==frozen['plan_sha256']
        assert sha(runner)==frozen['runner_sha256']
        assert all(sha(p)==h for p,h in frozen['protected'].items())
        command=[sys.executable,str(runner),'--plan',str(ROOT/'plan_sr015_bl8.json'),'--scene','sr015_bl8']
        if mode=='preflight':command.append('--preflight')
        log=ROOT/'sr015_bl8'/f'{mode}.log'
        log.parent.mkdir(parents=True,exist_ok=True)
        print(json.dumps(dict(event='started',scene='sr015_bl8',mode=mode,log=str(log))),flush=True)
        with log.open('x') as stream:
            subprocess.run(command,cwd=work,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
        print(json.dumps(dict(event='completed',scene='sr015_bl8',mode=mode)),flush=True)
    assert all(sha(p)==h for p,h in frozen['protected'].items())
    with (ROOT/'addendum-complete.json').open('x') as stream:
        json.dump(dict(status='completed',scene='sr015_bl8',epochs=100,protected_records_verified=True),stream,indent=2)


if __name__=='__main__':
    main()
