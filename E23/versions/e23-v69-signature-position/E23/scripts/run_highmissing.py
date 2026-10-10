import json,os,sys,subprocess,tarfile
from pathlib import Path
from datetime import datetime,timezone
from malr.evaluation.report import file_sha256
from scripts.prepare_highmissing import ROOT,WORK,MASKS,write

def read(p):return json.loads(Path(p).read_text())
def sources():
    files=[WORK/'pyproject.toml']
    for folder in ('src','integrations','scripts','configs','tests'):
        files.extend(p for p in (WORK/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc')
    return {str(p.relative_to(WORK)):file_sha256(p) for p in sorted(files)}
def main():
    assert read(ROOT/'preflight.json')['status']=='passed'
    plan=read(ROOT/'plan.json');protected=read(ROOT/'protected-files.json')
    fixed={str(p):file_sha256(p) for p in ROOT.rglob('*') if p.is_file() and p.suffix in ('.json','.yaml','.npz','.sha256')}
    definition=dict(created_utc=datetime.now(timezone.utc).isoformat(),sources=sources(),fixed=fixed,protected=protected)
    write(ROOT/'matrix-definition.json',definition)
    with tarfile.open(ROOT/'frozen-source.tar.gz','w:gz',dereference=True) as a:
        for rel in definition['sources']:a.add(WORK/rel,arcname=rel,recursive=False)
    def verify():
        assert sources()==definition['sources']
        for p,h in {**fixed,**protected}.items():assert file_sha256(Path(p))==h,p
    for scene in plan['scenarios']:
        env={**os.environ,'PYTHONPATH':'src:.','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','MALR_MISSING_SCENARIO':scene['scenario_manifest']}
        for label in plan['models']:
            cell=ROOT/scene['name']/label;cell.mkdir()
            if label=='E23-v41-slow':
                common=['--protocol-config',str(WORK/'configs/stage1_seq96_dynamic_temporal_repair.yaml'),'--migration-catalog',str(WORK.parent/'E22/migration/paper_catalog.jsonl'),'--mask-artifacts',str(MASKS),'--e23-config',plan['e23_config'],'--training-seed','502','--device','cuda:1']
                commands=[[sys.executable,'-m','integrations.e23.run','train-repair',*common,'--output',str(cell/'stage1')],
                    [sys.executable,'-m','integrations.e23.run','train-forecast',*common,'--stage1-manifest',str(cell/'stage1/runs/e23-stage1.json'),'--selection-metric','mae','--forecast-loss','mse_plus_0p2_mae','--repair-retention','uniform_v1','--repair-update-policy','slow_lr_0p1_v1','--output',str(cell/'stage2')]]
                selection=cell/'stage2/runs/e23-stage2.json'
            else:
                commands=[[sys.executable,'scripts/run_author_aligned_p4.py','--model','T1','--profiles',plan['author_profiles'],'--training-seed','502','--output',str(cell/'training'),'--device','cuda:1','--train']]
                selection=cell/'training/forecast-selection.json'
            write(cell/'training-commands.json',dict(commands=commands,environment={'MALR_MISSING_SCENARIO':scene['scenario_manifest']}))
            for i,command in enumerate(commands):
                verify();print(json.dumps(dict(event='train',scenario=scene['name'],model=label,step=i,utc=datetime.now(timezone.utc).isoformat())),flush=True)
                with (cell/f'train_{i}.log').open('x') as f:subprocess.run(command,cwd=WORK,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
                verify()
            record=read(selection)
            assert record['predictor_initial_sha256']==plan['predictor_initial_sha256']
            val=record['best_validation'] if label=='T1' else dict(mse=record['best_val_mse'],mae=record['best_val_mae'])
            write(cell/'validation-lock-before-tests.json',dict(selection=str(selection),checkpoint_sha256=record['checkpoint_sha256'],validation=val,test_metrics_seen=False))
            evaluation=[sys.executable,'scripts/evaluate_e23_t1_replicate.py','--model','T1' if label=='T1' else 'E23-v31-B','--model-label',label,'--training-seed','502','--selection',str(selection),'--test-mask-bank',scene['test_bank'],'--output',str(cell/'evaluation'),'--device','cuda:1']
            evaluation.extend(['--profiles',plan['author_profiles']] if label=='T1' else ['--e23-config',plan['e23_config']])
            write(cell/'evaluation-command.json',evaluation)
            with (cell/'test.log').open('x') as f:subprocess.run(evaluation,cwd=WORK,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
            verify();e=read(cell/'evaluation/twenty-test-draws.json')
            write(cell/'completed-audit.json',dict(status='complete',test_mean=e['test_mean'],validation=e['validation_replay'],checkpoint_sha256=e['checkpoint_sha256'],source_unchanged=True))
            print(json.dumps(dict(event='scored',scenario=scene['name'],model=label,test_mean=e['test_mean'],validation=e['validation_replay'])),flush=True)
    verify();write(ROOT/'matrix-completed.json',dict(status='complete',independent_runs=6,test_draws=120,test_used_for_selection=False,source_unchanged=True))
if __name__=='__main__':main()
