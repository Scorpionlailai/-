"""Preflight and run the three predeclared T1 frozen Stage2 experiments."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def main():
    plan = json.loads((ROOT/'plan.json').read_text())
    work = Path(plan['work'])
    env = {**os.environ, 'PYTHONPATH': f'{work}/src:{work}', 'PYTHONUNBUFFERED': '1'}
    env.pop('MALR_MISSING_SCENARIO', None)
    sources = {str(p.relative_to(work)): sha(p) for name in ['src','integrations','scripts','configs','tests']
               for p in (work/name).rglob('*') if p.is_file() and p.suffix in ['.py','.json','.yaml','.toml']}
    protected = {}
    for scene in plan['scenarios']:
        paths = [scene[k] for k in ['t1_repair_selection','t1_selection','t1_evaluation','test_bank',
                                  'e23_selection','e23_evaluation']]
        if scene['scenario_manifest']:
            paths.append(scene['scenario_manifest'])
        parent = json.loads(Path(scene['t1_repair_selection']).read_text())
        paths.append(parent['checkpoint'])
        for path in paths:
            protected[path] = sha(path)
    frozen = dict(sources=sources, protected=protected, plan_sha256=sha(ROOT/'plan.json'),
                  runner_sha256=sha(work/'scripts/run_t1_frozen_stage2.py'), controller_sha256=sha(__file__))
    write(ROOT/'frozen-definition.json', frozen)
    with tarfile.open(ROOT/'frozen-source.tar.gz','w:gz') as archive:
        for path in sources:
            archive.add(work/path, arcname=path, recursive=False)
        archive.add(ROOT/'plan.json',arcname='experiment/plan.json')
        archive.add(__file__,arcname='experiment/execute.py')

    def verify():
        assert sha(ROOT/'plan.json') == frozen['plan_sha256']
        assert sha(__file__) == frozen['controller_sha256']
        assert all(sha(work/p) == h for p,h in sources.items())
        assert all(sha(p) == h for p,h in protected.items())

    def run(scene, preflight):
        verify()
        mode = 'preflight' if preflight else 'training'
        command = [sys.executable, str(work/'scripts/run_t1_frozen_stage2.py'),
                   '--plan', str(ROOT/'plan.json'), '--scene', scene['name']]
        if preflight:
            command.append('--preflight')
        log = ROOT/scene['name']/f'{mode}.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        print(json.dumps(dict(event='started',scene=scene['name'],mode=mode,log=str(log))),flush=True)
        with log.open('x') as stream:
            subprocess.run(command,cwd=work,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
        verify()
        with (ROOT/'progress.md').open('a') as stream:
            stream.write(datetime.now(timezone.utc).isoformat()+f' {scene["name"]} {mode} complete.\n')
        print(json.dumps(dict(event='completed',scene=scene['name'],mode=mode)),flush=True)

    for scene in plan['scenarios']:
        path = ROOT/scene['name']/'preflight.json'
        if path.exists():
            assert json.loads(path.read_text())['status'] == 'passed'
        else:
            run(scene,True)
    write(ROOT/'preflight.json',dict(status='passed',scenes=3,unit_tests=3,no_test_scoring=True))
    for scene in plan['scenarios']:
        run(scene,False)
    verify()
    write(ROOT/'training-complete.json',dict(status='completed',scenes=3,epochs_per_scene=100,
                                           frozen_repair_state_verified=True,protected_sources_verified=True))
    print(json.dumps(dict(event='all_training_and_tests_completed',scenes=3)),flush=True)


if __name__ == '__main__':
    main()
