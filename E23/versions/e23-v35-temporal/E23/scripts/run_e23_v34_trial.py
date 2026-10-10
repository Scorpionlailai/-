"""Train A then B; lock each validation checkpoint before its three test draws."""
import argparse
import json
import subprocess
import sys
from datetime import datetime,timezone
from pathlib import Path
from time import perf_counter
from scripts.run_e23_v30_trial import _execute,verify_freeze,ROOT,MASKS,BANK
from scripts.preflight_e23_v34 import CONFIGS
from scripts.report_e23_v34_trial import load_candidate,report,digest


def build_commands(root,device):
    rows=[]
    for condition,config in CONFIGS.items():
        cell=root/condition
        common=['--protocol-config',str(ROOT/'configs/stage1_seq96_dynamic_temporal_repair.yaml'),
            '--e23-config',str(config),'--migration-catalog',str(ROOT.parent/'E22/migration/paper_catalog.jsonl'),
            '--mask-artifacts',str(MASKS),'--device',device]
        rows.append(dict(condition=condition,config=str(config),
            stage1=[sys.executable,'-m','integrations.e23.run','train-repair',*common,'--output',str(cell/'stage1')],
            stage2=[sys.executable,'-m','integrations.e23.run','train-forecast',*common,'--output',str(cell/'stage2'),
                '--stage1-manifest',str(cell/'stage1/runs/e23-stage1.json'),'--selection-metric','mae',
                '--forecast-loss','mse','--forecast-input','explicit_v1','--forecast-mixer','none']))
    return rows


def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True)
    p.add_argument('--device',default='cuda:1'); p.add_argument('--dry-run',action='store_true')
    p.add_argument('--resume',action='store_true'); a=p.parse_args(); root=a.root.resolve()
    commands=build_commands(root,a.device)
    if a.dry_run: print(json.dumps(commands,indent=2)); return
    verify_freeze(root)
    if not a.resume and any((root/r['condition']/s).exists() for r in commands for s in ('stage1','stage2')):
        raise FileExistsError('v34 outputs already exist')
    environment=dict(started_utc=datetime.now(timezone.utc).isoformat(),workspace=str(ROOT),device=a.device,
        gpu_processes_at_start=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
            '--format=csv,noheader'],text=True))
    (root/('resume-environment.json' if a.resume else 'formal-environment.json')).write_text(json.dumps(environment,indent=2)+'\n')
    for row in commands:
        cell=root/row['condition']; cell.mkdir(exist_ok=True)
        (root/'status.json').write_text(json.dumps(dict(status='training',condition=row['condition'],retained='E23_v31_B'),indent=2)+'\n')
        for stage in ('stage1','stage2'):
            manifest=cell/stage/'runs'/f'e23-{stage}.json'
            if a.resume and manifest.exists():
                saved=json.loads(manifest.read_text())
                if saved['epochs_completed']!=100 or saved['e23_config_sha256']!=digest(row['config']) or digest(saved['checkpoint'])!=saved['checkpoint_sha256']:
                    raise ValueError('resume stage identity mismatch')
                continue
            _execute(row[stage],cell/(stage+'.log'),root/'code-freeze.json'); verify_freeze(root)
        selection=cell/'stage2/runs/e23-stage2.json'
        if not (cell/'validation_replay.json').exists():
            _execute([sys.executable,'scripts/replay_e23_p4_validation.py','--e23-config',row['config'],
                '--stage2-manifest',str(selection),'--output',str(cell/'validation_replay.json'),'--device',a.device],
                cell/'validation-replay.log',root/'code-freeze.json')
        if not (cell/'historical-diagnostics.json').exists():
            _execute([sys.executable,'scripts/diagnose_e23_v34.py','--config',row['config'],
                '--cell',str(cell),'--device',a.device],cell/'historical-diagnostics.log',root/'code-freeze.json')
        record=load_candidate(root,row['condition'])
        locked=cell/'locked-validation.json'
        if locked.exists():
            if json.loads(locked.read_text())!=record: raise ValueError('locked validation changed')
        else: locked.write_text(json.dumps(record,indent=2)+'\n')
        if not (cell/'three-test-draws.json').exists():
            started=perf_counter()
            _execute([sys.executable,'scripts/evaluate_e23_three_test_draws.py','--e23-config',row['config'],
                '--selection',str(selection),'--output',str(cell/'three-test-draws.json'),
                '--model-label',record['model'],'--test-mask-bank',str(BANK),'--device',a.device],
                cell/'three-test-draws.log',root/'code-freeze.json')
            (cell/'test-runtime.json').write_text(json.dumps(dict(seconds=perf_counter()-started,training_time_included=False),indent=2)+'\n')
        verify_freeze(root); report(root,ROOT/'docs/experiments',partial=True)
    _,decision=report(root,ROOT/'docs/experiments')
    (root/'status.json').write_text(json.dumps(dict(status='complete',**decision),indent=2)+'\n')
    print(json.dumps(decision),flush=True)


if __name__=='__main__': main()
