"""Resume ONLY evaluations after the same-byte migration path correction."""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

R = Path(__file__).resolve().parent
p = json.loads((R/'plan.json').read_text())
w = Path(p['work'])
s = p['scenarios'][0]
cell = R/'E23-v99'/s['name']
dependency = R.parent/'e23_v100_stage1_validation_cache_20261008/status.json'
while json.loads(dependency.read_text())['status'] not in ['completed','equivalence_failed','failed']:
    time.sleep(60)
env = {**os.environ,'PYTHONPATH':f'{w}/src:{w}','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','PYTHONUNBUFFERED':'1','MALR_MISSING_SCENARIO':s['scenario_manifest']}

def verify():
    sources = json.loads((R/'frozen-definition.json').read_text())['sources']
    assert all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==sha for f,sha in sources.items())

try:
    verify()
    for metric in ['mae','mse']:
        selection=cell/'stage2/runs'/('e23-stage2.json' if metric=='mae' else 'e23-stage2-mse.json')
        args=[sys.executable,str(w/'scripts/evaluate_e23_t1_replicate.py'),'--model','E23-v31-B','--model-label','E23-v99-selective-bf16-'+metric,'--training-seed','2023','--e23-config',p['models'][0]['config'],'--selection',str(selection),'--test-mask-bank',s['test_bank'],'--output',str(cell/('evaluation_'+metric)),'--device','cuda:0']
        (cell/('evaluation_'+metric+'.resume.command.json')).write_text(json.dumps(args,indent=2))
        (R/'status.json').write_text(json.dumps(dict(status='running',stage='recovered_forecast_evaluation_'+metric)))
        with (cell/('evaluation_'+metric+'.resume.log')).open('x') as log:
            subprocess.run(args,cwd=w,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        verify()
    (R/'status.json').write_text(json.dumps(dict(status='pilot_completed',scene=s['name'],evaluation_path_recovery=True)))
    print('v99 pilot and both forecast evaluations completed',flush=True)
except Exception as error:
    (R/'status.json').write_text(json.dumps(dict(status='failed',error=repr(error))))
    raise
