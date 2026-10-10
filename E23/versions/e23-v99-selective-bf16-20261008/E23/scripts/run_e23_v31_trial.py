"""Shared fresh Stage1, three independently initialized Stage2 runs and immediate tests."""
from __future__ import annotations
import argparse,json,subprocess,sys,os
from pathlib import Path
from datetime import datetime,timezone
from time import perf_counter
from scripts.run_e23_v30_trial import verify_freeze,_execute,MASKS,BANK
from scripts.report_e23_v31_trial import MODES,CONFIG,load_candidate,digest,report
ROOT=Path(__file__).resolve().parents[1]


def build_training_commands(root,device):
    common=['--protocol-config',str(ROOT/'configs/stage1_seq96_dynamic_temporal_repair.yaml'),'--e23-config',str(CONFIG),'--migration-catalog',str(ROOT.parent/'E22/migration/paper_catalog.jsonl'),'--mask-artifacts',str(MASKS),'--device',device]
    stage1=[sys.executable,'-m','integrations.e23.run','train-repair',*common,'--output',str(root/'common_stage1')]
    rows=[dict(condition=condition,stage2=[sys.executable,'-m','integrations.e23.run','train-forecast',*common,'--output',str(root/condition/'stage2'),'--stage1-manifest',str(root/'common_stage1/runs/e23-stage1.json'),'--selection-metric','mae','--forecast-loss','mse_plus_0p2_mae','--repair-retention',mode]) for condition,mode in MODES.items()]
    return stage1,rows


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--device',default='cuda:1');parser.add_argument('--mode',choices=['train','report'],required=True);parser.add_argument('--dry-run',action='store_true');args=parser.parse_args();root=args.root.resolve()
    first,rows=build_training_commands(root,args.device)
    if args.dry_run:print(json.dumps(dict(stage1=first,stage2=rows),indent=2));return
    verify_freeze(root)
    if args.mode=='report':report(root,ROOT/'docs/experiments');return
    if (root/'common_stage1').exists() or any((root/r['condition']/'stage2').exists() for r in rows):raise FileExistsError('formal outputs already exist')
    environment=dict(started_utc=datetime.now(timezone.utc).isoformat(),device=args.device,concurrent_gpu_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader'],text=True),code_freeze_sha256=digest(root/'code-freeze.json'))
    (root/'formal_training_environment.json').write_text(json.dumps(environment,indent=2)+'\n')
    _execute(first,root/'common_stage1.log',root/'code-freeze.json');verify_freeze(root)
    records=[]
    for row in rows:
        cell=root/row['condition'];cell.mkdir(parents=True,exist_ok=True)
        _execute(row['stage2'],cell/'stage2.log',root/'code-freeze.json');verify_freeze(root)
        selection=cell/'stage2/runs/e23-stage2.json'
        _execute([sys.executable,'scripts/replay_e23_p4_validation.py','--e23-config',str(CONFIG),'--stage2-manifest',str(selection),'--output',str(cell/'validation_replay.json'),'--device',args.device],cell/'validation_replay.log',root/'code-freeze.json')
        _execute([sys.executable,'scripts/diagnose_e23_v31_repair.py','--selection',str(selection),'--output',str(cell/'repair_diagnostic.json'),'--device',args.device],cell/'repair_diagnostic.log',root/'code-freeze.json')
        record=load_candidate(root,row['condition']);records.append(record)
        if len({r['predictor_initial_sha256'] for r in records})!=1:raise ValueError('different initial predictor state')
        (cell/'locked-validation.json').write_text(json.dumps(record,indent=2)+'\n')
        started=perf_counter()
        _execute([sys.executable,'scripts/evaluate_e23_three_test_draws.py','--e23-config',str(CONFIG),'--selection',str(selection),'--output',str(cell/'three-test-draws.json'),'--model-label',record['model'],'--test-mask-bank',str(BANK),'--device',args.device],cell/'three-test-draws.log',root/'code-freeze.json')
        (cell/'test_runtime.json').write_text(json.dumps(dict(seconds=perf_counter()-started,training_time_included=False),indent=2)+'\n');verify_freeze(root)
    report(root,ROOT/'docs/experiments')

if __name__=='__main__':main()
