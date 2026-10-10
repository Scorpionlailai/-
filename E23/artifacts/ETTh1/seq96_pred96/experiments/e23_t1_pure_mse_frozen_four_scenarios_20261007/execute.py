"""Sequential matched pure-MSE frozen Stage2 runs; original Stage1 cached."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parent


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def e23_command(plan,scene):
    return [sys.executable,'-m','integrations.e23.run','train-forecast',
        '--protocol-config',plan['protocol_config'],
        '--migration-catalog','/home/xyw/paper/E22/migration/paper_catalog.jsonl',
        '--mask-artifacts',plan['mask_artifacts'],'--e23-config',plan['e23_config'],
        '--training-seed','2023','--device',plan['device'],
        '--stage1-manifest',scene['e23_stage1'],'--selection-metric','mae',
        '--forecast-loss','mse','--repair-retention','none','--repair-update-policy','frozen_v1',
        '--forecast-padding','zero_invalid_v1','--extra-mse-checkpoint',
        '--output',str(ROOT/'E23-v69'/scene['name']/'stage2')]


def check_e23(selection,plan,scene,metric):
    d=read(selection)
    expected=dict(seed=2023,epochs_completed=100,selection_metric=metric,
        loss='forecast_mse_only',repair_update_policy='frozen_v1',repair_trainable_parameters=0,
        predictor_initial_sha256=plan['predictor_initial_sha256'],
        validation_mask_sha256=scene['validation_mask_sha256'])
    for key,value in expected.items():assert d.get(key)==value,(key,d.get(key),value)
    assert d.get('repair_retention','none')=='none'
    assert d.get('forecast_padding','zero_invalid_v1')=='zero_invalid_v1'
    assert d['stage1_manifest']==scene['e23_stage1']
    assert sha(d['checkpoint'])==d['checkpoint_sha256']
    return dict(selected_epoch=d['best_epoch'],selection_metric=metric,
        selection_sha256=sha(selection),checkpoint_sha256=d['checkpoint_sha256'],
        validation=dict(mse=d['best_val_mse'],mae=d['best_val_mae']),test_metrics_seen=False)


def main():
    plan=read(ROOT/'plan.json');work=Path(plan['work'])
    env={**os.environ,'PYTHONPATH':f'{work}/src:{work}','PYTHONUNBUFFERED':'1',
         'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}
    env.pop('MALR_MISSING_SCENARIO',None)
    sources={str(p.relative_to(work)):sha(p) for name in ['src','integrations','scripts','configs']
             for p in (work/name).rglob('*') if p.is_file() and p.suffix in ['.py','.yaml','.json']}
    protected={}
    for scene in plan['scenarios']:
        for key in ['e23_stage1','t1_repair_selection','test_bank','e23_selection','e23_evaluation','t1_mixed_evaluation']:
            path=scene[key];protected[path]=sha(path)
        for key in ['e23_stage1','t1_repair_selection']:
            path=read(scene[key])['checkpoint'];protected[path]=sha(path)
        if scene['scenario_manifest']:protected[scene['scenario_manifest']]=sha(scene['scenario_manifest'])
    frozen=dict(sources=sources,protected=protected,plan_sha256=sha(ROOT/'plan.json'),
                t1_plan_sha256=sha(ROOT/'t1-plan.json'),controller_sha256=sha(__file__),preflight_sha256=sha(ROOT/'preflight.py'))
    write(ROOT/'frozen-definition.json',frozen)
    with tarfile.open(ROOT/'frozen-source.tar.gz','w:gz',dereference=True) as f:
        for rel in sources:f.add(work/rel,arcname=rel,recursive=False)
        for name in ['plan.json','t1-plan.json','execute.py','preflight.py']:f.add(ROOT/name,arcname='experiment/'+name)

    def verify():
        assert sha(ROOT/'plan.json')==frozen['plan_sha256']
        assert sha(ROOT/'t1-plan.json')==frozen['t1_plan_sha256']
        assert all(sha(work/p)==h for p,h in sources.items())
        assert all(sha(p)==h for p,h in protected.items())

    def run(command,log,scene=None):
        verify();log=Path(log);log.parent.mkdir(parents=True,exist_ok=True)
        job_env=dict(env)
        if scene and scene['scenario_manifest']:job_env['MALR_MISSING_SCENARIO']=scene['scenario_manifest']
        write(log.with_suffix('.command.json'),dict(command=command,scenario=scene['name'] if scene else None))
        print(json.dumps(dict(event='started',log=str(log))),flush=True)
        with log.open('x') as stream:subprocess.run(command,cwd=work,env=job_env,stdout=stream,stderr=subprocess.STDOUT,check=True)
        verify()
        with (ROOT/'progress.md').open('a') as f:f.write(datetime.now(timezone.utc).isoformat()+' completed '+str(log)+'\n')
        print(json.dumps(dict(event='completed',log=str(log))),flush=True)

    run([sys.executable,str(ROOT/'preflight.py')],ROOT/'preflight.log')
    assert read(ROOT/'preflight.json')['status']=='passed'
    for scene in plan['scenarios']:
        run([sys.executable,str(work/'scripts/run_t1_frozen_stage2.py'),
            '--plan',str(ROOT/'t1-plan.json'),'--scene',scene['name']],ROOT/'T1'/scene['name']/'training.log',scene)
        t1=read(ROOT/'T1'/scene['name']/'training/forecast-mae-selection.json')
        assert t1['loss']=='forecast_mse_only' and t1['epochs_completed']==100
        assert t1['repair_update_policy']=='frozen_v1' and t1['frozen_state_verified_epochs']==100
        run(e23_command(plan,scene),ROOT/'E23-v69'/scene['name']/'stage2.log',scene)
        for metric in ['mae','mse']:
            selection=ROOT/'E23-v69'/scene['name']/'stage2/runs'/('e23-stage2.json' if metric=='mae' else 'e23-stage2-mse.json')
            lock=check_e23(selection,plan,scene,metric)
            write(ROOT/'E23-v69'/scene['name']/f'validation-{metric}-lock-before-tests.json',lock)
            run([sys.executable,str(work/'scripts/evaluate_e23_t1_replicate.py'),
                '--model','E23-v31-B','--model-label','E23-v69-pureMSE-frozen-'+metric.upper(),
                '--training-seed','2023','--e23-config',plan['e23_config'],
                '--selection',str(selection),'--test-mask-bank',scene['test_bank'],
                '--output',str(ROOT/'E23-v69'/scene['name']/f'evaluation_{metric}'),
                '--device',plan['device']],ROOT/'E23-v69'/scene['name']/f'evaluation_{metric}.log',scene)
        a=read(ROOT/'E23-v69'/scene['name']/'evaluation_mae/twenty-test-draws.json')
        b=read(ROOT/'T1'/scene['name']/'evaluation_mae/twenty-test-draws.json')
        for x,y in zip(a['test_draws'],b['test_draws']):
            assert all(x[k]==y[k] for k in ['group_id','mask_seed','mask_sha256','windows','target_elements'])
        print(json.dumps(dict(event='scene_complete',scene=scene['name'],e23=a['test_mean'],t1=b['test_mean'])),flush=True)
    verify()
    write(ROOT/'complete.json',dict(status='completed',scenarios=4,models=2,epochs_per_run=100,pure_mse=True,frozen=True))


if __name__=='__main__':main()
