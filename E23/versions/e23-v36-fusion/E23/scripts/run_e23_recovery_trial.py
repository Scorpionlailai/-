"""Run one frozen recovery trial with original Stage2 and test after locking."""
import argparse,json,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
from time import perf_counter
from scripts.run_e23_v30_trial import _execute,verify_freeze,ROOT,MASKS,BANK
from scripts.report_e23_recovery_trial import load_candidate,report,digest


def build_commands(root,trial,device):
    cell=root/trial['condition']
    common=['--protocol-config',str(ROOT/'configs/stage1_seq96_dynamic_temporal_repair.yaml'),
        '--e23-config',trial['config'],'--migration-catalog',str(ROOT.parent/'E22/migration/paper_catalog.jsonl'),
        '--mask-artifacts',str(MASKS),'--device',device]
    return dict(stage1=[sys.executable,'-m','integrations.e23.run','train-repair',*common,'--output',str(cell/'stage1')],
        stage2=[sys.executable,'-m','integrations.e23.run','train-forecast',*common,'--output',str(cell/'stage2'),
            '--stage1-manifest',str(cell/'stage1/runs/e23-stage1.json'),'--selection-metric','mae',
            '--forecast-loss','mse','--forecast-input','explicit_v1','--forecast-mixer','none'])


def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True)
    p.add_argument('--device',default='cuda:1'); p.add_argument('--dry-run',action='store_true')
    p.add_argument('--resume',action='store_true'); a=p.parse_args(); root=a.root.resolve()
    trial=json.loads((root/'trial.json').read_text()); commands=build_commands(root,trial,a.device)
    if a.dry_run: print(json.dumps(commands,indent=2)); return
    verify_freeze(root); cell=root/trial['condition']
    if json.loads((root/'code-freeze.json').read_text())['trial_sha256']!=digest(root/'trial.json'):
        raise ValueError('predeclared trial definition changed')
    if not a.resume and any((cell/s).exists() for s in ('stage1','stage2')): raise FileExistsError('trial outputs exist')
    cell.mkdir(exist_ok=True)
    environment=dict(started_utc=datetime.now(timezone.utc).isoformat(),workspace=str(ROOT),device=a.device,
        trial_sha256=digest(root/'trial.json'),gpu_processes_at_start=subprocess.check_output(
            ['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader'],text=True))
    (root/('resume-environment.json' if a.resume else 'formal-environment.json')).write_text(json.dumps(environment,indent=2)+'\n')
    for stage in ('stage1','stage2'):
        (root/'status.json').write_text(json.dumps(dict(status='training',stage=stage,model=trial['model']),indent=2)+'\n')
        manifest=cell/stage/'runs'/f'e23-{stage}.json'
        if a.resume and manifest.exists():
            saved=json.loads(manifest.read_text())
            if saved['epochs_completed']!=100 or saved['e23_config_sha256']!=digest(trial['config']) or digest(saved['checkpoint'])!=saved['checkpoint_sha256']:
                raise ValueError('resume stage identity mismatch')
            continue
        _execute(commands[stage],cell/(stage+'.log'),root/'code-freeze.json'); verify_freeze(root)
    selection=cell/'stage2/runs/e23-stage2.json'
    if not (cell/'validation_replay.json').exists():
        _execute([sys.executable,'scripts/replay_e23_p4_validation.py','--e23-config',trial['config'],
            '--stage2-manifest',str(selection),'--output',str(cell/'validation_replay.json'),'--device',a.device],
            cell/'validation-replay.log',root/'code-freeze.json')
    if not (cell/'recovery-diagnostics.json').exists():
        _execute([sys.executable,'scripts/diagnose_e23_recovery_trial.py','--config',trial['config'],
            '--cell',str(cell),'--device',a.device],cell/'recovery-diagnostics.log',root/'code-freeze.json')
    record=load_candidate(root,trial); locked=cell/'locked-validation.json'
    if locked.exists():
        if json.loads(locked.read_text())!=record: raise ValueError('locked validation changed')
    else: locked.write_text(json.dumps(record,indent=2)+'\n')
    if not (cell/'three-test-draws.json').exists():
        started=perf_counter()
        _execute([sys.executable,'scripts/evaluate_e23_three_test_draws.py','--e23-config',trial['config'],
            '--selection',str(selection),'--output',str(cell/'three-test-draws.json'),
            '--model-label',trial['model'],'--test-mask-bank',str(BANK),'--device',a.device],
            cell/'three-test-draws.log',root/'code-freeze.json')
        (cell/'test-runtime.json').write_text(json.dumps(dict(seconds=perf_counter()-started,training_time_included=False),indent=2)+'\n')
    verify_freeze(root); _,decision=report(root)
    (root/'status.json').write_text(json.dumps(dict(status='complete',**decision),indent=2)+'\n')
    print(json.dumps(decision),flush=True)


if __name__=='__main__': main()
