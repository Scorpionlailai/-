"""Run the shared v80 recent-band recipe on three fixed missing scenes."""
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


def command(plan,scene,stage):
    cell=ROOT/scene['name']/'E23-v80'
    args=[sys.executable,'-m','integrations.e23.run','train-'+stage,
          '--protocol-config',plan['protocol_config'],
          '--migration-catalog','/home/xyw/paper/E22/migration/paper_catalog.jsonl',
          '--mask-artifacts',plan['mask_artifacts'],'--e23-config',plan['e23_config'],
          '--training-seed','2023','--device',plan['device']]
    if stage=='repair':
        return args+['--auxiliary-forecast','none','--repair-batch-size','128','--output',str(cell/'stage1')]
    if stage=='forecast':
        return args+['--stage1-manifest',str(cell/'stage1/runs/e23-stage1.json'),
            '--selection-metric','mae','--forecast-loss','mse','--repair-retention','none',
            '--repair-update-policy','frozen_v1','--forecast-padding','zero_invalid_v1',
            '--extra-mse-checkpoint','--output',str(cell/'stage2')]
    raise ValueError(stage)


def main():
    plan=read(ROOT/'plan.json');baseline=Path(plan['baseline_root'])
    assert read(baseline/'complete.json')['status']=='completed'
    assert read(baseline/'verification.json')['status']=='verified'
    work=Path(plan['work'])
    env={**os.environ,'PYTHONPATH':f'{work}/src:{work}','PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}
    env.pop('MALR_MISSING_SCENARIO',None)
    sources={str(p.relative_to(work)):sha(p) for name in ['src','integrations','scripts','configs']
             for p in (work/name).rglob('*') if p.is_file() and p.suffix in ['.py','.json','.yaml']}
    protected={path:sha(path) for path in [plan['clean_reference'],plan['repair_reference']]}
    for scene in plan['scenarios']:
        for key in ['t1_pure_evaluation','v69_pure_evaluation','v79_pure_evaluation','v79_stage1','test_bank']:
            protected[scene[key]]=sha(scene[key])
        if scene['scenario_manifest']:protected[scene['scenario_manifest']]=sha(scene['scenario_manifest'])
        for key in ['t1_pure_evaluation','v69_pure_evaluation','v79_pure_evaluation']:
            evaluation=read(scene[key]);selection=Path(evaluation['selection_manifest'])
            protected[str(selection)]=sha(selection)
            checkpoint=read(selection)['checkpoint'];protected[checkpoint]=sha(checkpoint)
    frozen=dict(sources=sources,protected=protected,plan_sha256=sha(ROOT/'plan.json'),
                controller_sha256=sha(__file__),preflight_sha256=sha(ROOT/'preflight.py'))
    write(ROOT/'frozen-definition.json',frozen)
    with tarfile.open(ROOT/'frozen-source.tar.gz','w:gz',dereference=True) as f:
        for rel in sources:f.add(work/rel,arcname=rel,recursive=False)
        for name in ['plan.json','execute.py','preflight.py']:f.add(ROOT/name,arcname='experiment/'+name)
    def verify():
        assert sha(ROOT/'plan.json')==frozen['plan_sha256']
        assert sha(__file__)==frozen['controller_sha256']
        assert all(sha(work/p)==h for p,h in sources.items())
        assert all(sha(p)==h for p,h in protected.items())
    def run(args,log,scene=None):
        verify();log=Path(log);log.parent.mkdir(parents=True,exist_ok=True)
        job_env=dict(env)
        if scene and scene['scenario_manifest']:job_env['MALR_MISSING_SCENARIO']=scene['scenario_manifest']
        write(log.with_suffix('.command.json'),dict(command=args,scene=scene['name'] if scene else None))
        print(json.dumps(dict(event='started',log=str(log))),flush=True)
        with log.open('x') as f:subprocess.run(args,cwd=work,env=job_env,stdout=f,stderr=subprocess.STDOUT,check=True)
        verify()
        with (ROOT/'progress.md').open('a') as f:f.write(datetime.now(timezone.utc).isoformat()+' complete '+str(log)+'\n')
        print(json.dumps(dict(event='completed',log=str(log))),flush=True)
    run([sys.executable,str(ROOT/'preflight.py')],ROOT/'preflight.log')
    for scene in plan['scenarios']:
        cell=ROOT/scene['name']/'E23-v80'
        run(command(plan,scene,'repair'),cell/'stage1.log',scene)
        stage1=read(cell/'stage1/runs/e23-stage1.json')
        assert stage1['epochs_completed']==100 and stage1['optimizer_updates']==6600
        assert stage1.get('stage1_aux_forecast','none')=='none'
        assert stage1.get('stage1_gap_structure','none')=='none'
        assert stage1['recent_mean_projection']=='global_recent_bands_v3'
        run(command(plan,scene,'forecast'),cell/'stage2.log',scene)
        for metric in ['mae','mse']:
            selection=cell/'stage2/runs'/('e23-stage2.json' if metric=='mae' else 'e23-stage2-mse.json')
            d=read(selection)
            assert d['loss']=='forecast_mse_only' and d['epochs_completed']==100
            assert d.get('repair_retention','none')=='none'
            assert d['repair_update_policy']=='frozen_v1' and d['repair_trainable_parameters']==0
            assert d['predictor_initial_sha256']==plan['predictor_initial_sha256']
            assert d['validation_mask_sha256']==scene['validation_mask_sha256']
            assert d['recent_mean_global_extra_parameters']==3265 and d['recent_band_extra_parameters']==6210
            assert sha(d['checkpoint'])==d['checkpoint_sha256']
            write(cell/f'validation-{metric}-lock-before-tests.json',dict(selection_sha256=sha(selection),
                checkpoint_sha256=d['checkpoint_sha256'],validation=dict(mse=d['best_val_mse'],mae=d['best_val_mae']),
                selection_metric=metric,test_metrics_seen=False))
            run([sys.executable,str(work/'scripts/evaluate_e23_t1_replicate.py'),
                '--model','E23-v31-B','--model-label','E23-v80-pureMSE-frozen-'+metric.upper(),
                '--training-seed','2023','--e23-config',plan['e23_config'],'--selection',str(selection),
                '--test-mask-bank',scene['test_bank'],'--output',str(cell/f'evaluation_{metric}'),
                '--device',plan['device']],cell/f'evaluation_{metric}.log',scene)
        result=read(cell/'evaluation_mae/twenty-test-draws.json');reference=read(scene['t1_pure_evaluation'])
        for x,y in zip(result['test_draws'],reference['test_draws']):
            assert all(x[k]==y[k] for k in ['group_id','mask_seed','mask_sha256','windows','target_elements'])
        print(json.dumps(dict(event='scene_complete',scene=scene['name'],candidate=result['test_mean'],
                             t1=reference['test_mean'])),flush=True)
    verify()
    write(ROOT/'complete.json',dict(status='completed',scenarios=3,epochs_per_stage=100,
                                   pure_mse_frozen=True,architecture_changes=1))


if __name__=='__main__':main()
