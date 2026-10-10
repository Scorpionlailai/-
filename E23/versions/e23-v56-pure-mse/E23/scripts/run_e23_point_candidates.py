"""Three authorized changes, sequentially accepted using fixed validation only."""
import hashlib,json,os,sys,subprocess,tarfile
from pathlib import Path
from datetime import datetime,timezone
import yaml

WORK=Path(__file__).resolve().parents[1]
ROOT=Path('/home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v51_candidate_readout_sr015_bl8_seed502_20261005')
def read(p):return json.loads(Path(p).read_text())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(data,f,indent=2);f.write('\n')
def sources():
    paths=[WORK/'pyproject.toml']
    for folder in ['src','integrations','scripts','configs','tests']:
        paths.extend(p for p in (WORK/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc')
    return {str(p.relative_to(WORK)):digest(p) for p in sorted(paths)}
def main():
    assert read(ROOT/'preflight.json')['status']=='passed' and read(ROOT/'test-results.json')['status']=='passed'
    plan=read(ROOT/'plan.json');protected=read(ROOT/'protected-files.json')
    definition=dict(created_utc=datetime.now(timezone.utc).isoformat(),sources=sources(),protected=protected,
        plan_sha256=digest(ROOT/'plan.json'),scenario_sha256=digest(plan['scenario_manifest']),bank_sha256=digest(plan['test_bank']))
    write(ROOT/'frozen-definition.json',definition)
    with tarfile.open(ROOT/'frozen-source.tar.gz','w:gz',dereference=True) as ar:
        for relative in definition['sources']:ar.add(WORK/relative,arcname=relative,recursive=False)
    env={**os.environ,'PYTHONPATH':'src:.','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','MALR_MISSING_SCENARIO':plan['scenario_manifest']}
    def verify():
        assert sources()==definition['sources'],'Source changed'
        assert digest(ROOT/'plan.json')==definition['plan_sha256']
        assert digest(plan['scenario_manifest'])==definition['scenario_sha256'] and digest(plan['test_bank'])==definition['bank_sha256']
        for p,h in protected.items():assert digest(p)==h,p
    def run(command,cell,name):
        verify();write(cell/(name+'-command.json'),dict(command=command,environment={'MALR_MISSING_SCENARIO':plan['scenario_manifest']}))
        print(json.dumps(dict(event='start',cell=cell.name,stage=name,utc=datetime.now(timezone.utc).isoformat())),flush=True)
        with (cell/(name+'.log')).open('x') as log:subprocess.run(command,cwd=WORK,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        verify()
    parent=plan['initial_parent'].copy();results=[]
    for step in plan['sequence']:
        spec={**parent,**step};options={k:parent[k] for k in ['point_router','overlap_fusion','stage1_structure','point_candidate_readout']};options[step['change']]=step['value']
        key='p'+str(int(options['point_router']!='none'))+'o'+str(int(options['overlap_fusion']!='none'))+'s'+str(int(options['stage1_structure']!='none'))+'c'+str(int(options['point_candidate_readout']!='none'))
        config=plan['configs'][key];cell=ROOT/step['name'];cell.mkdir()
        trial=dict(label=step['label'],config=config,config_key=key,parent=parent,only_change=step['change'],**options)
        write(cell/'plan.json',trial)
        common=['--protocol-config',str(WORK/'configs/stage1_seq96_dynamic_temporal_repair.yaml'),
            '--migration-catalog',str(WORK.parent/'E22/migration/paper_catalog.jsonl'),
            '--mask-artifacts',str(WORK.parent/'E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context'),
            '--e23-config',config,'--training-seed','502','--device','cuda:1']
        s1=cell/'stage1/runs/e23-stage1.json';s2=cell/'stage2/runs/e23-stage2.json'
        run([sys.executable,'-m','integrations.e23.run','train-repair',*common,'--output',str(cell/'stage1')],cell,'stage1')
        assert read(s1)['epochs_completed']==100
        run([sys.executable,'-m','integrations.e23.run','train-forecast',*common,'--stage1-manifest',str(s1),
            '--selection-metric','mae','--forecast-loss','mse_plus_0p2_mae','--repair-retention','uniform_v1',
            '--repair-update-policy','freeze_front_v1','--output',str(cell/'stage2')],cell,'stage2')
        r=read(s2);assert r['epochs_completed']==100 and r['predictor_initial_sha256']==plan['predictor_initial_sha256']
        assert r['repair_update_policy']=='freeze_front_v1' and r['loss']=='forecast_mse_plus_0p2_mae' and r['selection_metric']=='mae'
        validation={'mse':r['best_val_mse'],'mae':r['best_val_mae']}
        accepted=validation['mse']<parent['validation']['mse'] and validation['mae']<=parent['validation']['mae']
        vs_v41=validation['mse']<plan['reference_v41_validation']['mse'] and validation['mae']<=plan['reference_v41_validation']['mae']
        lock=dict(parent=parent,validation=validation,selected_epoch=r['best_epoch'],checkpoint_sha256=r['checkpoint_sha256'],
            accepted_on_validation=accepted,matched_v41_validation_gate=vs_v41,test_metrics_seen=False)
        write(cell/'validation-lock-before-tests.json',lock);print(json.dumps(dict(event='validation_locked',cell=cell.name,**lock)),flush=True)
        # Make the parent choice before any test metrics from this condition exist.
        if accepted:parent=dict(label=step['label'],root=str(cell),validation=validation,**options)
        write(cell/'parent-choice-before-tests.json',dict(next_parent=parent,uses_test_metrics=False))
        run([sys.executable,'scripts/evaluate_e23_t1_replicate.py','--model','E23-v31-B','--model-label',step['label'],
            '--training-seed','502','--e23-config',config,'--selection',str(s2),'--test-mask-bank',plan['test_bank'],
            '--output',str(cell/'evaluation'),'--device','cuda:1'],cell,'test')
        run([sys.executable,'scripts/diagnose_e23_front_context.py','--cell',str(cell)],cell,'diagnostics')
        score=read(cell/'evaluation/twenty-test-draws.json');assert len(score['test_draws'])==20
        result=dict(label=step['label'],cell=str(cell),trial=trial,validation_lock=lock,test_mean=score['test_mean'],
            test_std=score['test_mask_std_ddof1'],next_parent=parent)
        write(cell/'completed-audit.json',result);results.append(result)
        print(json.dumps(dict(event='scored',**result)),flush=True)
    verify();write(ROOT/'completed-audit.json',dict(status='complete',conditions=results,retained_parent=parent,
        test_used_for_checkpoint_or_automatic_parent_selection=False,source_unchanged=True,protected_files_unchanged=True))
    print(json.dumps(dict(event='complete',retained_parent=parent)),flush=True)
if __name__=='__main__':main()
