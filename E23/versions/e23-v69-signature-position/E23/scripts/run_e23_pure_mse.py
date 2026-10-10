"""Three authorized changes, sequentially accepted using fixed validation only."""
import hashlib,json,os,sys,subprocess,tarfile
from pathlib import Path
from datetime import datetime,timezone
import yaml

WORK=Path(__file__).resolve().parents[1]
ROOT=Path('/home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v56_pure_mse_sr015_bl8_seed502_20261005')
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
    original_parent=plan['initial_parent'].copy()
    options={k:original_parent[k] for k in ['point_router','overlap_fusion','stage1_structure','point_candidate_readout','variable_calibration','front_variable_context','repair_coordinates','point_variable_readout']}
    config=plan['base_config'];key='parent_architecture';cells=[]
    primary=ROOT/'v56_pure_mse_mae';primary.mkdir()
    source_stage1=Path(original_parent['root'])/'stage1'
    (primary/'stage1').symlink_to(source_stage1.resolve(),target_is_directory=True)
    trial=dict(label='E23-v56-pure-mse-MAE',config=config,config_key=key,parent=original_parent,only_change='stage2_pure_mse_without_retention',cached_stage1=str(source_stage1/'runs/e23-stage1.json'),**options)
    write(primary/'plan.json',trial)
    common=['--protocol-config',str(WORK/'configs/stage1_seq96_dynamic_temporal_repair.yaml'),'--migration-catalog',str(WORK.parent/'E22/migration/paper_catalog.jsonl'),'--mask-artifacts',str(WORK.parent/'E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context'),'--e23-config',config,'--training-seed','502','--device','cuda:1']
    s1=primary/'stage1/runs/e23-stage1.json'
    assert read(s1)['epochs_completed']==100
    write(primary/'cached-stage1.json',dict(manifest=str(s1.resolve()),sha256=digest(s1),checkpoint_sha256=read(s1)['checkpoint_sha256'],retrained=False,include_original_measured_cost=True))
    run([sys.executable,'-m','integrations.e23.run','train-forecast',*common,'--stage1-manifest',str(s1),'--selection-metric','mae','--forecast-loss','mse','--repair-retention','none','--repair-update-policy','freeze_front_v1','--output',str(primary/'stage2')],primary,'stage2')
    secondary=ROOT/'v56_pure_mse_mse';secondary.mkdir();(secondary/'stage1').symlink_to(source_stage1.resolve(),target_is_directory=True);(secondary/'stage2/runs').mkdir(parents=True)
    mse_manifest=primary/'stage2/runs/e23-stage2-mse.json';assert mse_manifest.exists()
    (secondary/'stage2/runs/e23-stage2.json').write_bytes(mse_manifest.read_bytes())
    secondary_trial={**trial,'label':'E23-v56-pure-mse-MSE','only_change':'stage2_pure_mse_without_retention_and_mse_selection'};write(secondary/'plan.json',secondary_trial)
    # Lock BOTH selections and a validation-only fallback choice before either test.
    locks=[]
    for cell,spec,metric in [(primary,trial,'mae'),(secondary,secondary_trial,'mse')]:
        r=read(cell/'stage2/runs/e23-stage2.json');assert r['epochs_completed']==100 and r['loss']=='forecast_mse_only' and r.get('repair_retention','none')=='none' and r['selection_metric']==metric and r['predictor_initial_sha256']==plan['predictor_initial_sha256']
        validation={'mse':r['best_val_mse'],'mae':r['best_val_mae']}
        accepted=validation['mse']<original_parent['validation']['mse'] and validation['mae']<=original_parent['validation']['mae']
        lock=dict(parent=original_parent,validation=validation,selected_epoch=r['best_epoch'],selection_metric=metric,checkpoint_sha256=r['checkpoint_sha256'],accepted_on_validation=accepted,matched_v41_validation_gate=validation['mse']<plan['reference_v41_validation']['mse'] and validation['mae']<=plan['reference_v41_validation']['mae'],test_metrics_seen=False)
        write(cell/'validation-lock-before-tests.json',lock);locks.append(lock);cells.append((cell,spec,lock));print(json.dumps(dict(event='validation_locked',cell=cell.name,**lock)),flush=True)
    retained=original_parent
    # Predeclared hierarchy: primary MAE if it passes, otherwise MSE if it passes.
    for cell,spec,lock in cells:
        if lock['accepted_on_validation']:
            retained=dict(label=spec['label'],root=str(cell),validation=lock['validation'],selection_metric=lock['selection_metric'],forecast_recipe='pure_mse_without_retention',**options);break
    for cell,spec,lock in cells:write(cell/'parent-choice-before-tests.json',dict(next_parent=retained,uses_test_metrics=False,predeclared_priority=['mae','mse']))
    results=[]
    for cell,spec,lock in cells:
        run([sys.executable,'scripts/evaluate_e23_t1_replicate.py','--model','E23-v31-B','--model-label',spec['label'],'--training-seed','502','--e23-config',config,'--selection',str(cell/'stage2/runs/e23-stage2.json'),'--test-mask-bank',plan['test_bank'],'--output',str(cell/'evaluation'),'--device','cuda:1'],cell,'test')
        run([sys.executable,'scripts/diagnose_e23_front_context.py','--cell',str(cell)],cell,'diagnostics')
        score=read(cell/'evaluation/twenty-test-draws.json');assert len(score['test_draws'])==20
        result=dict(label=spec['label'],cell=str(cell),trial=spec,validation_lock=lock,test_mean=score['test_mean'],test_std=score['test_mask_std_ddof1'],next_parent=retained)
        write(cell/'completed-audit.json',result);results.append(result);print(json.dumps(dict(event='scored',**result)),flush=True)
    verify();write(ROOT/'completed-audit.json',dict(status='complete',conditions=results,retained_parent=retained,stage1_reused=True,new_training_runs=1,test_used_for_checkpoint_or_automatic_parent_selection=False,source_unchanged=True,protected_files_unchanged=True));print(json.dumps(dict(event='complete',retained_parent=retained)),flush=True)
if __name__=='__main__':main()
