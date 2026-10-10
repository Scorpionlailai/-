"""Frozen three-cell nonoverlap study; one fixed predictor and 20 test masks."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from datetime import datetime, timezone

WORK=Path(__file__).resolve().parents[1]
ROOT=Path('/home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/e23_v43_nonoverlap_seed502_20261004')
FORMAL=ROOT.parent/'e23_t1_five_seed_standard_20261004'
MASKS=WORK.parent/'E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context'

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(data,f,indent=2);f.write('\n')
def sources():
    files=[WORK/'pyproject.toml']
    for folder in ['src','integrations','scripts','configs','tests']:
        files.extend(p for p in (WORK/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc')
    return {str(p.relative_to(WORK)):digest(p) for p in sorted(files)}
def selected(cell,metric):return read(cell/'stage2/runs'/('e23-stage2.json' if metric=='mae' else 'e23-stage2-mse.json'))

def main():
    assert read(ROOT/'preflight.json')['status']=='passed'
    plan=read(ROOT/'plan.json')
    protected=read(ROOT/'protected-existing-files.json')
    definition=dict(created_utc=datetime.now(timezone.utc).isoformat(),sources=sources(),
        plan_sha256=digest(ROOT/'plan.json'),test_bank_sha256=digest(FORMAL/'test_masks/manifest.json'),
        protected=protected,configs={x['config']:digest(x['config']) for x in plan['cells']})
    write(ROOT/'matrix-definition.json',definition)
    with tarfile.open(ROOT/'frozen-source.tar.gz','w:gz',dereference=True) as archive:
        for relative in definition['sources']:archive.add(WORK/relative,arcname=relative,recursive=False)
    env={**os.environ,'PYTHONPATH':'src:.','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}
    def verify():
        assert sources()==definition['sources'],'Source changed'
        assert digest(ROOT/'plan.json')==definition['plan_sha256']
        assert digest(FORMAL/'test_masks/manifest.json')==definition['test_bank_sha256']
        for p,h in {**definition['protected'],**definition['configs']}.items():assert digest(p)==h,p
    def run(command,log):
        verify();print(json.dumps(dict(event='start',log=str(log),utc=datetime.now(timezone.utc).isoformat())),flush=True)
        with log.open('x') as f:subprocess.run(command,cwd=WORK,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        verify()
    for spec in plan['cells']:
        cell=ROOT/spec['name'];cell.mkdir()
        write(cell/'trial-definition.json',spec)
        common=['--protocol-config',str(WORK/'configs/stage1_seq96_dynamic_temporal_repair.yaml'),
            '--migration-catalog',str(WORK.parent/'E22/migration/paper_catalog.jsonl'),
            '--mask-artifacts',str(MASKS),'--e23-config',spec['config'],'--training-seed','502','--device','cuda:1']
        stage1=Path(spec['cached_stage1']) if spec.get('cached_stage1') else cell/'stage1/runs/e23-stage1.json'
        commands=[]
        if not spec.get('cached_stage1'):
            commands.append([sys.executable,'-m','integrations.e23.run','train-repair',*common,'--output',str(cell/'stage1')])
        commands.append([sys.executable,'-m','integrations.e23.run','train-forecast',*common,
            '--stage1-manifest',str(stage1),'--common-stage2','--selection-metric','mae',
            '--forecast-loss','mse','--repair-retention','none','--repair-update-policy','slow_lr_0p1_v1',
            '--output',str(cell/'stage2')])
        write(cell/'training-commands.json',commands)
        for i,command in enumerate(commands):run(command,cell/f'train_{i}.log')
        choices={}
        for metric in ['mae','mse']:
            r=selected(cell,metric)
            assert r['epochs_completed']==100 and r['loss']=='forecast_mse_only'
            assert r['predictor_initial_sha256']==plan['predictor_initial_sha256']
            assert r['repair_update_policy']=='slow_lr_0p1_v1'
            choices[metric]={k:r[k] for k in ['best_val_mse','best_val_mae','best_epoch','checkpoint_sha256']}
        write(cell/'validation-lock-before-tests.json',dict(choices=choices,test_metrics_seen=False))
        print(json.dumps(dict(event='validation_locked',cell=spec['name'],choices=choices)),flush=True)
        for metric in ['mae','mse']:
            selection=cell/'stage2/runs'/('e23-stage2.json' if metric=='mae' else 'e23-stage2-mse.json')
            command=[sys.executable,'scripts/evaluate_e23_t1_replicate.py','--model','E23-v31-B',
                '--model-label',spec['label']+'-select-'+metric,'--training-seed','502','--e23-config',spec['config'],
                '--selection',str(selection),'--test-mask-bank',str(FORMAL/'test_masks/manifest.json'),
                '--output',str(cell/f'evaluation_{metric}'),'--device','cuda:1']
            write(cell/f'evaluation-command-{metric}.json',command);run(command,cell/f'test-{metric}.log')
        command=[sys.executable,'scripts/diagnose_e23_nonoverlap.py','--cell',str(cell)]
        write(cell/'diagnostic-command.json',command);run(command,cell/'diagnostics.log')
        write(cell/'completed-audit.json',dict(status='complete',choices=choices,
            test_means={m:read(cell/f'evaluation_{m}/twenty-test-draws.json')['test_mean'] for m in ['mae','mse']}))
        print(json.dumps(dict(event='scored',cell=spec['name'],**read(cell/'completed-audit.json'))),flush=True)
    verify()
    control=selected(ROOT/plan['cells'][0]['name'],'mae')
    rows={}
    for spec in plan['cells'][1:]:
        r=selected(ROOT/spec['name'],'mae')
        rows[spec['name']]=dict(matched_gate=r['best_val_mse']<control['best_val_mse'] and r['best_val_mae']<=control['best_val_mae'],
            v41_reference_gate=r['best_val_mse']<plan['v41_reference_validation']['mse'] and
                               r['best_val_mae']<=plan['v41_reference_validation']['mae'])
    write(ROOT/'matrix-completed.json',dict(status='complete',validation_gates=rows,
        test_used_for_selection=False,new_T1_training=False,source_unchanged=True,protected_files_unchanged=True))
    print(json.dumps(read(ROOT/'matrix-completed.json')),flush=True)

if __name__=='__main__':main()
