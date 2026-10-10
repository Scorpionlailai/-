"""Stage2-only trials using cached Stage1 selections and locked twenty test masks."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from datetime import datetime, timezone

WORK = Path(__file__).resolve().parents[1]
ROOT = Path('/home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/e23_v41_repair_drift_seed502_20261004')
FORMAL = ROOT.parent/'e23_t1_five_seed_standard_20261004'
MASKS = WORK.parent/'E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context'


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def write(path, value):
    with Path(path).open('x') as stream: json.dump(value, stream, indent=2); stream.write('\n')


def sources():
    paths = [WORK/'pyproject.toml']
    for folder in ['src','integrations','scripts','configs','tests']:
        paths.extend(p for p in (WORK/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc')
    return {str(p.relative_to(WORK)):digest(p) for p in sorted(paths)}


def main():
    assert read(ROOT/'preflight.json')['status'] == 'passed'
    plan = read(ROOT/'plan.json')
    parent_path = FORMAL/'seed_502/E23-v31-B/evaluation/validation-lock.json'
    parent = read(parent_path)
    files = {str(ROOT/'plan.json'):digest(ROOT/'plan.json'),str(parent_path):digest(parent_path),
             str(FORMAL/'test_masks/manifest.json'):digest(FORMAL/'test_masks/manifest.json')}
    for trial in plan['trials']:
        for key in ['config','stage1']: files[trial[key]] = digest(trial[key])
        checkpoint = read(trial['stage1'])['checkpoint']
        files[checkpoint] = digest(checkpoint)
    definition = dict(created_utc=datetime.now(timezone.utc).isoformat(),source_files=sources(),parent_files=files,
                      protected_files=read(ROOT/'protected-existing-files.json'),new_stage1_training=False)
    write(ROOT/'matrix-definition.json', definition)
    with tarfile.open(ROOT/'frozen-source.tar.gz','w:gz',dereference=True) as archive:
        for relative in definition['source_files']: archive.add(WORK/relative,arcname=relative,recursive=False)
    def verify():
        assert sources() == definition['source_files'], 'Frozen source changed'
        for path,h in {**definition['parent_files'],**definition['protected_files']}.items():
            assert digest(path) == h, f'Protected file changed: {path}'
    env = {**os.environ,'PYTHONPATH':'src:.','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}
    for trial in plan['trials']:
        cell = ROOT/trial['trial']
        stage1 = read(trial['stage1'])
        write(cell/'trial-definition.json',dict(**trial,matrix_definition_sha256=digest(ROOT/'matrix-definition.json'),
             stage1_checkpoint_sha256=stage1['checkpoint_sha256'],new_stage1_training=False,
             predictor_initial_sha256=parent['predictor_initial_sha256'],exploratory=True))
        common = ['--protocol-config',str(WORK/'configs/stage1_seq96_dynamic_temporal_repair.yaml'),
            '--migration-catalog',str(WORK.parent/'E22/migration/paper_catalog.jsonl'),'--mask-artifacts',str(MASKS),
            '--e23-config',trial['config'],'--training-seed','502','--device','cuda:1']
        commands = [[sys.executable,'-m','integrations.e23.run','train-forecast',*common,
            '--stage1-manifest',trial['stage1'],'--selection-metric','mae','--forecast-loss','mse_plus_0p2_mae',
            '--repair-retention','uniform_v1','--repair-update-policy',trial['policy'],'--output',str(cell/'stage2')],
            [sys.executable,'scripts/evaluate_e23_t1_replicate.py','--model','E23-v31-B','--model-label',trial['label'],
            '--training-seed','502','--selection',str(cell/'stage2/runs/e23-stage2.json'),'--e23-config',trial['config'],
            '--test-mask-bank',str(FORMAL/'test_masks/manifest.json'),'--output',str(cell/'evaluation'),'--device','cuda:1']]
        write(cell/'commands.json',commands)
        for step,command in enumerate(commands):
            verify()
            if step == 1:
                selected = read(cell/'stage2/runs/e23-stage2.json')
                assert selected['predictor_initial_sha256'] == parent['predictor_initial_sha256']
                assert selected['repair_update_policy'] == trial['policy'] and selected['epochs_completed'] == 100
                validation = dict(mse=selected['best_val_mse'],mae=selected['best_val_mae'])
                gate = validation['mse'] < parent['validation_replay']['mse'] and validation['mae'] <= parent['validation_replay']['mae']
                write(cell/'validation-decision.json',dict(validation=validation,parent=parent['validation_replay'],
                      passed=gate,test_metrics_seen=False,checkpoint_sha256=selected['checkpoint_sha256']))
            print(json.dumps(dict(event='start',trial=trial['trial'],step=step,utc=datetime.now(timezone.utc).isoformat())),flush=True)
            with (cell/f'step_{step}.log').open('x') as stream:
                subprocess.run(command,cwd=WORK,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
            verify()
        evaluation = read(cell/'evaluation/twenty-test-draws.json')
        write(cell/'completed-audit.json',dict(status='complete',source_unchanged=True,stage1_reused=True,
            decision=read(cell/'validation-decision.json'),test_mean=evaluation['test_mean'],
            training_resources=evaluation['training_resources'],head_initialization_matched=True))
        print(json.dumps(dict(event='completed',trial=trial['trial'],test_mean=evaluation['test_mean'])),flush=True)
    write(ROOT/'matrix-completed.json',dict(status='complete',source_unchanged=True,protected_files_unchanged=True,
        new_stage1_training=False,trial_count=3,utc=datetime.now(timezone.utc).isoformat()))


if __name__ == '__main__': main()
