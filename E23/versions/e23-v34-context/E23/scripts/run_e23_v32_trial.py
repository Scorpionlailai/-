"""Run independently trained control and Stage1 prediction-preservation conditions."""
from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
from time import perf_counter
from scripts.run_e23_v30_trial import verify_freeze,_execute,MASKS,BANK
from scripts.report_e23_v32_trial import ROOT,CONFIG,MODES,load_candidate,report,digest


def build_training_commands(root,device):
    common=['--protocol-config',str(ROOT/'configs/stage1_seq96_dynamic_temporal_repair.yaml'),'--e23-config',str(CONFIG),'--migration-catalog',str(ROOT.parent/'E22/migration/paper_catalog.jsonl'),'--mask-artifacts',str(MASKS),'--device',device]
    return [dict(condition=c,stage1=[sys.executable,'-m','integrations.e23.run','train-repair',*common,'--output',str(root/c/'stage1'),'--stage1-preservation',mode],
        stage2=[sys.executable,'-m','integrations.e23.run','train-forecast',*common,'--output',str(root/c/'stage2'),'--stage1-manifest',str(root/c/'stage1/runs/e23-stage1.json'),'--selection-metric','mae','--forecast-loss','mse']) for c,mode in MODES.items()]


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--device',default='cuda:1');p.add_argument('--mode',choices=('train','report'),required=True);p.add_argument('--dry-run',action='store_true');a=p.parse_args();root=a.root.resolve()
    commands=build_training_commands(root,a.device)
    if a.dry_run:print(json.dumps(commands,indent=2));return
    verify_freeze(root)
    if a.mode=='report':report(root,ROOT/'docs/experiments');return
    if any((root/r['condition']/'stage1').exists() or (root/r['condition']/'stage2').exists() for r in commands):raise FileExistsError('v32 outputs already exist')
    environment=dict(started_utc=datetime.now(timezone.utc).isoformat(),device=a.device,workspace=str(ROOT),code_freeze_sha256=digest(root/'code-freeze.json'),gpu_processes_at_start=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader'],text=True))
    (root/'formal_training_environment.json').write_text(json.dumps(environment,indent=2)+'\n')
    records=[]
    for row in commands:
        cell=root/row['condition'];cell.mkdir(parents=True,exist_ok=True)
        for stage in ('stage1','stage2'):
            _execute(row[stage],cell/(stage+'.log'),root/'code-freeze.json');verify_freeze(root)
        selection=cell/'stage2/runs/e23-stage2.json'
        _execute([sys.executable,'scripts/replay_e23_p4_validation.py','--e23-config',str(CONFIG),'--stage2-manifest',str(selection),'--output',str(cell/'validation_replay.json'),'--device',a.device],cell/'validation_replay.log',root/'code-freeze.json')
        _execute([sys.executable,'scripts/diagnose_e23_v32.py','--cell',str(cell),'--device',a.device],cell/'diagnostic.log',root/'code-freeze.json')
        record=load_candidate(root,row['condition']);records.append(record)
        if len({r['predictor_initial_sha256'] for r in records})!=1:raise ValueError('different predictor initialization')
        (cell/'locked-validation.json').write_text(json.dumps(record,indent=2)+'\n')
        started=perf_counter()
        _execute([sys.executable,'scripts/evaluate_e23_three_test_draws.py','--e23-config',str(CONFIG),'--selection',str(selection),'--output',str(cell/'three-test-draws.json'),'--model-label',record['model'],'--test-mask-bank',str(BANK),'--device',a.device],cell/'three-test-draws.log',root/'code-freeze.json')
        (cell/'test_runtime.json').write_text(json.dumps(dict(seconds=perf_counter()-started,training_time_included=False),indent=2)+'\n');verify_freeze(root)
    report(root,ROOT/'docs/experiments')


if __name__=='__main__':main()
