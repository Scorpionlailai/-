import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

R = Path(__file__).resolve().parent
p = json.loads((R/'plan.json').read_text())
w = Path(p['work'])
sources = {str(f): hashlib.sha256(f.read_bytes()).hexdigest()
           for folder in ['src','integrations','scripts','configs']
           for f in (w/folder).rglob('*') if f.is_file() and f.suffix in ['.py','.yaml','.json']}
for name in ['plan.json','controller.py','train_refinement.py','evaluate_repair.py','refinement_config.yaml']:
    f = R/name
    sources[str(f)] = hashlib.sha256(f.read_bytes()).hexdigest()
(R/'frozen-definition.json').write_text(json.dumps(dict(sources=sources),indent=2))
with tarfile.open(R/'frozen-source-and-runner.tar.gz','w:gz') as archive:
    for f in sources:
        archive.add(f,arcname=str(Path(f)).lstrip('/'),recursive=False)

dependency = R.parent/'e23_v99_selective_bf16_20261008/status.json'
(R/'status.json').write_text(json.dumps(dict(status='waiting_for_v99_gpu0')))
while True:
    if dependency.exists():
        state = json.loads(dependency.read_text())['status']
        if state in ['pilot_completed','failed']:
            break
    time.sleep(60)

cell = R/'E23-v100/sr015_bl8'
cell.mkdir(parents=True)
args = [sys.executable,str(R/'train_refinement.py'),'sr015_bl8']
env = {**os.environ,'PYTHONPATH':f'{w}/src:{w}','PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}
(cell/'stage1.command.json').write_text(json.dumps(args,indent=2))

def verify():
    assert all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==sha for f,sha in sources.items())

try:
    verify()
    (R/'status.json').write_text(json.dumps(dict(status='running',stage='cached_refinement_100epochs')))
    with (cell/'stage1.log').open('x') as log:
        child = subprocess.Popen(args,cwd=w,env=env,stdout=log,stderr=subprocess.STDOUT)
        while True:
            try:
                rc = child.wait(timeout=300)
                break
            except subprocess.TimeoutExpired:
                verify()
                print(json.dumps(dict(event='five_minute_check',stage='v100_cached_refinement',tail=(cell/'stage1.log').read_text().splitlines()[-2:])),flush=True)
    if rc:
        raise RuntimeError(f'Cached refinement failed exit {rc}; inspect {cell}/stage1.log')
    verify()
    import torch
    old = R.parent/'e23_stage1_mean_and_dense_20261008/E23-v97/sr015_bl8/stage1/runs/e23-stage1.json'
    new = cell/'stage1/runs/e23-stage1.json'
    base, candidate = json.loads(old.read_text()), json.loads(new.read_text())
    before = torch.load(base['checkpoint'],map_location='cpu',weights_only=False)
    after = torch.load(candidate['checkpoint'],map_location='cpu',weights_only=False)
    state_equal = before['model_state'].keys()==after['model_state'].keys() and all(torch.equal(before['model_state'][k],after['model_state'][k]) for k in before['model_state'])
    checks = dict(entire_repair_state_exact_equal=state_equal,
                  selected_epoch_equal=base['best_epoch']==candidate['best_epoch'],
                  selected_refinement_epoch_equal=base['best_refinement_epoch']==candidate['best_refinement_epoch'],
                  validation_metrics_exact_equal=base['validation_repair_metrics']==candidate['validation_repair_metrics'],
                  memory_sha_equal=base['memory_sha256']==candidate['memory_sha256'],
                  key_sha_equal=base['key_sha256']==candidate['key_sha256'])
    checks['acceptance_passed'] = all(checks.values())
    checks['old_selection'] = str(old)
    checks['new_selection'] = str(new)
    checks['cache'] = candidate['fixed_stage1_validation_cache']
    checks['test_rerun'] = False
    checks['prediction_and_repair_results_reused_from_v97'] = checks['acceptance_passed']
    (R/'acceptance.json').write_text(json.dumps(checks,indent=2))
    (R/'status.json').write_text(json.dumps(dict(status='completed' if checks['acceptance_passed'] else 'equivalence_failed')))
    print(json.dumps(checks,indent=2),flush=True)
except Exception as error:
    (R/'status.json').write_text(json.dumps(dict(status='failed',error=repr(error)),indent=2))
    raise
