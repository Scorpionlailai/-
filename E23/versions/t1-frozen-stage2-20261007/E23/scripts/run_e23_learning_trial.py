"""One frozen, paired seed-502 exploration with immediate twenty-mask evaluation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
FORMAL = Path('/home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/e23_t1_five_seed_standard_20261004')
MASKS = ROOT.parent/'E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context'

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def sources():
    paths = [ROOT/'pyproject.toml']
    for folder in ['src','integrations','scripts','configs','tests']:
        paths.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc')
    return {str(p.relative_to(ROOT)):digest(p) for p in sorted(paths)}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--label',required=True)
    parser.add_argument('--repair-batch-size',type=int,choices=(32,128),required=True)
    parser.add_argument('--parent',type=Path,required=True)
    args=parser.parse_args(); cell=args.root.resolve();cell.mkdir(exist_ok=True)
    parent=json.loads(args.parent.read_text()); expected_head=parent['predictor_initial_sha256']
    definition=dict(label=args.label,seed=502,device='cuda:1',created_utc=datetime.now(timezone.utc).isoformat(),
        parent=str(args.parent.resolve()),parent_sha256=digest(args.parent),
        parent_validation=parent['validation_replay'],repair_training_batch=args.repair_batch_size,
        predictor_initial_sha256=expected_head,configuration=str(args.config.resolve()),config_sha256=digest(args.config),
        test_bank=str(FORMAL/'test_masks/manifest.json'),test_bank_sha256=digest(FORMAL/'test_masks/manifest.json'),
        source_files=sources(),exploratory_seed_selected_from_old_test=True)
    seal=cell/'trial-definition.json'
    if seal.exists():raise FileExistsError(f'Already frozen: {seal}')
    seal.write_text(json.dumps(definition,indent=2)+'\n')
    (cell/'trial-definition.sha256').write_text(digest(seal)+'\n')
    with tarfile.open(cell/'frozen-source.tar.gz','w:gz',dereference=True) as archive:
        for relative in definition['source_files']:archive.add(ROOT/relative,arcname=relative,recursive=False)
    env={**os.environ,'PYTHONPATH':'src:.','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}
    common=['--protocol-config',str(ROOT/'configs/stage1_seq96_dynamic_temporal_repair.yaml'),
        '--migration-catalog',str(ROOT.parent/'E22/migration/paper_catalog.jsonl'),'--mask-artifacts',str(MASKS),
        '--e23-config',str(args.config.resolve()),'--training-seed','502','--device','cuda:1']
    commands=[
        [sys.executable,'-m','integrations.e23.run','train-repair',*common,'--repair-batch-size',str(args.repair_batch_size),'--output',str(cell/'stage1')],
        [sys.executable,'-m','integrations.e23.run','train-forecast',*common,'--stage1-manifest',str(cell/'stage1/runs/e23-stage1.json'),
         '--selection-metric','mae','--forecast-loss','mse_plus_0p2_mae','--repair-retention','uniform_v1','--output',str(cell/'stage2')],
        [sys.executable,'scripts/evaluate_e23_t1_replicate.py','--model','E23-v31-B','--model-label',args.label,'--training-seed','502',
         '--selection',str(cell/'stage2/runs/e23-stage2.json'),'--e23-config',str(args.config.resolve()),
         '--test-mask-bank',str(FORMAL/'test_masks/manifest.json'),'--output',str(cell/'evaluation'),'--device','cuda:1']]
    (cell/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
    def verify():
        if sources()!=definition['source_files'] or digest(args.config)!=definition['config_sha256'] or digest(args.parent)!=definition['parent_sha256']:
            raise ValueError('Frozen source/configuration/parent changed')
        if digest(seal)!=(cell/'trial-definition.sha256').read_text().strip():raise ValueError('Frozen definition changed')
        if digest(FORMAL/'test_masks/manifest.json')!=definition['test_bank_sha256']:raise ValueError('Mask bank changed')
    for index,command in enumerate(commands):
        verify()
        if index==2:
            chosen=json.loads((cell/'stage2/runs/e23-stage2.json').read_text())
            if chosen['predictor_initial_sha256']!=expected_head:raise ValueError('Original predictor initialization changed')
            validation=dict(mse=chosen['best_val_mse'],mae=chosen['best_val_mae'])
            gate=validation['mse']<parent['validation_replay']['mse'] and validation['mae']<=parent['validation_replay']['mae']
            (cell/'validation-decision.json').write_text(json.dumps(dict(validation=validation,parent=parent['validation_replay'],passed=gate,test_metrics_seen=False),indent=2)+'\n')
        log=cell/f'step_{index}.log'
        print(json.dumps(dict(event='start',step=index,label=args.label,log=str(log),utc=datetime.now(timezone.utc).isoformat())),flush=True)
        with log.open('x') as stream:subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
        verify()
    evaluation=json.loads((cell/'evaluation/twenty-test-draws.json').read_text())
    decision=json.loads((cell/'validation-decision.json').read_text())
    repair=json.loads((cell/'stage1/runs/e23-stage1.json').read_text())
    audit=dict(status='complete',label=args.label,source_unchanged=True,original_head_initialization_matched=True,
        repair_epochs=repair['epochs_completed'],repair_batch=repair['stage1_training_batch_size'],optimizer_updates=repair['optimizer_updates'],
        repair_validation_mse=repair['best_val_repair_mse'],decision=decision,test_mean=evaluation['test_mean'],
        training_resources=evaluation['training_resources'],source_archive_sha256=digest(cell/'frozen-source.tar.gz'))
    (cell/'completed-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2),flush=True)

if __name__=='__main__':main()
