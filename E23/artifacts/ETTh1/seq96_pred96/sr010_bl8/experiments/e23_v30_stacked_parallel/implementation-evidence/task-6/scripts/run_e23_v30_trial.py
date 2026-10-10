"""Run the predeclared stack matrix, lock validation, then evaluate frozen test draws."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from datetime import datetime,timezone
from scripts.report_e23_v30_trial import MATRIX,_digest,load_candidate,write_selection,choose_v30_candidate,report

ROOT=Path(__file__).resolve().parents[1]
MASKS=ROOT.parent/'E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context'
BANK=ROOT/'artifacts/ETTh1/seq96_pred96/sr010_bl8/test_mask_bank_20261002/manifest.json'
BANK_SHA='95fe49c78c6b884477e70c05f18a6c6c06f009e75d3afa01946275207b70e3a9'


def build_training_commands(root:Path,device:str):
    rows=[]
    for l,g in MATRIX:
        name=f'l{l}_g{g}';cell=root/name;config=ROOT/f'configs/e23_multi_expert_p4_width64_stack_{name}_v30.yaml'
        common=['--protocol-config',str(ROOT/'configs/stage1_seq96_dynamic_temporal_repair.yaml'),
            '--e23-config',str(config),'--migration-catalog',str(ROOT.parent/'E22/migration/paper_catalog.jsonl'),
            '--mask-artifacts',str(MASKS),'--device',device]
        rows.append(dict(cell=name,config=str(config),stage1_epochs=100,stage2_epochs=100,
            stage1=[sys.executable,'-m','integrations.e23.run','train-repair',*common,'--output',str(cell/'stage1')],
            stage2=[sys.executable,'-m','integrations.e23.run','train-forecast',*common,'--output',str(cell/'stage2'),
                    '--stage1-manifest',str(cell/'stage1/runs/e23-stage1.json'),'--selection-metric','mae',
                    '--forecast-loss','mse_plus_0p2_mae','--forecast-input','explicit_v1','--forecast-mixer','none']))
    return rows


def verify_freeze(root):
    freeze=json.loads((root/'code-freeze.json').read_text())
    if not freeze.get('files'): raise ValueError('empty code freeze')
    for name,sha in freeze['files'].items():
        if _digest(ROOT/name)!=sha: raise ValueError('code freeze mismatch: '+name)
    if _digest(BANK)!=BANK_SHA: raise ValueError('frozen test mask bank changed')
    return freeze


def evaluation_preconditions(root):
    for l,g in MATRIX:
        if (root/f'l{l}_g{g}/three-test-draws.json').exists(): raise FileExistsError('test already finalized')
    decision=json.loads((root/'validation_selection.json').read_text())
    verify_freeze(root)
    saved=decision.get('candidates',[])
    if len(saved)!=3 or [(r['recovery_layers'],r['expert_groups']) for r in saved]!=list(MATRIX):
        raise ValueError('all three locked candidates are required')
    fresh=[load_candidate(root/f'l{l}_g{g}',ROOT/f'configs/e23_multi_expert_p4_width64_stack_l{l}_g{g}_v30.yaml') for l,g in MATRIX]
    if fresh!=saved: raise ValueError('locked candidates changed')
    winner=choose_v30_candidate(fresh)
    if decision['winner']!=(None if winner is None else winner['model']):raise ValueError('validation decision changed')
    return decision


def _execute(command,log):
    log.parent.mkdir(parents=True,exist_ok=True)
    print(json.dumps({'event':'command_start','command':command,'log':str(log)}),flush=True)
    with log.open('x') as stream: subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
    print(json.dumps({'event':'command_complete','log':str(log)}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--device',default='cuda:1');parser.add_argument('--mode',choices=['train','evaluate','report'],required=True)
    parser.add_argument('--dry-run',action='store_true');args=parser.parse_args();root=args.root.resolve()
    commands=build_training_commands(root,args.device)
    if args.dry_run: print(json.dumps(commands,indent=2));return
    if args.mode=='train':
        verify_freeze(root)
        if (root/'validation_selection.json').exists() or any((root/row['cell']/stage).exists() for row in commands for stage in ('stage1','stage2')):
            raise FileExistsError('training outputs already exist')
        records=[]
        for row in commands:
            cell=root/row['cell'];cell.mkdir(parents=True,exist_ok=True)
            started=perf_counter()
            try:
                gpu_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader'],text=True)
            except (OSError,subprocess.CalledProcessError):gpu_processes='unavailable'
            environment=dict(started_utc=datetime.now(timezone.utc).isoformat(),device=args.device,
                concurrent_gpu_processes_at_start=gpu_processes,code_freeze_sha256=_digest(root/'code-freeze.json'))
            (cell/'formal_training_environment.json').write_text(json.dumps(environment,indent=2)+'\n')
            for stage in ('stage1','stage2'):_execute(row[stage],cell/f'{stage}.log')
            (cell/'training_entrypoint_runtime.json').write_text(json.dumps(dict(seconds=perf_counter()-started,
                scope='both complete training subprocesses; excludes replay, diagnostics, tests'),indent=2)+'\n')
            verify_freeze(root)
            _execute([sys.executable,'scripts/replay_e23_p4_validation.py','--e23-config',row['config'],
                '--stage2-manifest',str(cell/'stage2/runs/e23-stage2.json'),'--output',str(cell/'validation_replay.json'),
                '--device',args.device],cell/'validation_replay.log')
            _execute([sys.executable,'scripts/diagnose_e23_v30_stack.py','--e23-config',row['config'],
                '--stage1-manifest',str(cell/'stage1/runs/e23-stage1.json'),'--stage2-manifest',str(cell/'stage2/runs/e23-stage2.json'),
                '--output',str(cell/'layer_diagnostics.json'),'--device',args.device],cell/'layer_diagnostics.log')
            records.append(load_candidate(cell,Path(row['config'])))
        write_selection(records,root/'validation_selection.json')
    elif args.mode=='evaluate':
        decision=evaluation_preconditions(root)
        for row,record in zip(commands,decision['candidates']):
            cell=root/row['cell'];started=perf_counter()
            _execute([sys.executable,'scripts/evaluate_e23_three_test_draws.py','--e23-config',row['config'],
                '--selection',str(cell/'stage2/runs/e23-stage2.json'),'--output',str(cell/'three-test-draws.json'),
                '--model-label',record['model'],'--test-mask-bank',str(BANK),'--device',args.device],cell/'three-test-draws.log')
            (cell/'test_runtime.json').write_text(json.dumps(dict(seconds=perf_counter()-started,training_time_included=False),indent=2)+'\n')
            verify_freeze(root)
    else: verify_freeze(root);report(root,ROOT/'docs/experiments')

if __name__=='__main__':main()
