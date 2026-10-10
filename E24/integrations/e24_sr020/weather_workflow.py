"""One sealed Weather v79 run: repair -> frozen forecast -> fixed-bank tests."""
import argparse
from datetime import datetime,timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .data import read_manifest

ROOT=Path(__file__).resolve().parents[2]


def digest(path): return sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    temp.replace(path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset-manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:1')
    args=parser.parse_args()
    output=args.output.resolve()
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True)
    manifest=args.dataset_manifest.resolve()
    dataset=read_manifest(manifest)
    preflight=manifest.parent/'preflight_acceptance.json'
    accepted=json.loads(preflight.read_text())
    assert accepted['status']=='passed' and accepted['repair_state_unchanged']
    paths=sorted([*ROOT.glob('src/**/*.py'),*ROOT.glob('integrations/**/*.py'),ROOT/'configs/e24_v79.yaml',
                  ROOT/'scripts/check_weather_v79_sr015.py',manifest,manifest.with_suffix('.sha256'),Path(dataset['config_path'])])
    sources={str(p.resolve()):digest(p) for p in paths}
    plan=dict(experiment_id='e24_v79_weather_sr020_bl8_seed2023_20261008_sr015',parent='E23-v79',
              protocol_id=dataset['protocol_id'],created_utc=datetime.now(timezone.utc).isoformat(),
              dataset_manifest=str(manifest),dataset_manifest_sha256=digest(manifest),
              preflight_acceptance=str(preflight),preflight_sha256=digest(preflight),
              output=str(output),device=args.device,training_seed=2023,
              stage1=dict(epochs=100,batch=128,lr=.001,loss='global_hidden_mse_plus_0p1_retrieval_plus_recent_mean_v2',
                          selection='fixed_validation_global_hidden_mse',optimizer_updates=28600),
              stage2=dict(epochs=100,batch=128,lr=.0001,loss='pure_mse',repair_update_policy='frozen_v1',
                          optimizer_updates=28600,primary_selection='validation_mae',secondary_selection='validation_mse',
                          predictor='original_D16_heads4_FF128_layers3_P16_S8_drop0p3'),
              training_windows_per_epoch=36608,validation_windows=5175,test_windows_per_mask=10444,
              test_masks=20,test_seeds=dataset['test_seeds'],source_sha256=sources,
              dimension_adaptation='21 variable identities/positions/global-summary inputs; all v79 active modules retained.',
              evaluation_adaptation='Keep all valid Weather evaluation windows, including partial batches; ETTh1 unchanged.',
              stage1_from_scratch=True,stage2_from_scratch=True,baseline_training_authorized=True,
              monitor_interval_seconds=3600,old_E23_queues_remain_paused=True)
    write(output/'plan.json',plan)
    write(output/'source_freeze.json',dict(created_before_training=True,sha256=sources))
    env=dict(os.environ,PYTHONPATH='src:.',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    for name in ('MALR_MISSING_SCENARIO','E24_DATASET_MANIFEST','MALR_E23_CODE_FREEZE'):
        env.pop(name,None)
    base=[sys.executable,'-m','integrations.e24_sr020.run']
    shared=['--dataset-manifest',str(manifest),'--device',args.device]
    stage1=output/'stage1/runs/e23-stage1.json'
    stage2=output/'stage2/runs/e23-stage2.json'
    jobs=[('stage1',base+['train-repair',*shared,'--output',str(output/'stage1')]),
          ('stage2',base+['train-forecast',*shared,'--stage1-manifest',str(stage1),'--output',str(output/'stage2')]),
          ('evaluation_mae',[sys.executable,'-m','integrations.e24_sr020.evaluate',*shared,
                             '--selection',str(stage2),'--output',str(output/'evaluation_mae')]),
          ('evaluation_mse',[sys.executable,'-m','integrations.e24_sr020.evaluate',*shared,
                             '--selection',str(output/'stage2/runs/e23-stage2-mse.json'),
                             '--output',str(output/'evaluation_mse')])]
    write(output/'commands.json',dict(cwd=str(ROOT),environment={k:env[k] for k in ('PYTHONPATH','OMP_NUM_THREADS','MKL_NUM_THREADS')},jobs=jobs))
    state=dict(status='running',controller_pid=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat(),stages=[])
    write(output/'status.json',state)
    try:
        for name,command in jobs:
            for path,expected in sources.items():
                if digest(path)!=expected:raise RuntimeError('sealed source changed: '+path)
            log=output/(name+'.log')
            started=time.monotonic()
            with log.open('x') as handle:
                process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
                state.update(active_stage=name,child_pid=process.pid,log=str(log),stage_started_utc=datetime.now(timezone.utc).isoformat())
                write(output/'status.json',state)
                print(json.dumps(dict(event='start',stage=name,pid=process.pid,log=str(log))),flush=True)
                while True:
                    try:code=process.wait(timeout=3600)
                    except subprocess.TimeoutExpired:
                        tail=log.read_text(errors='replace').splitlines()[-3:]
                        snapshot=dict(time_utc=datetime.now(timezone.utc).isoformat(),stage=name,pid=process.pid,
                                      elapsed_seconds=time.monotonic()-started,tail=tail)
                        with (output/'monitor.jsonl').open('a') as f:f.write(json.dumps(snapshot)+'\n')
                        print(json.dumps(dict(event='hourly_check',**snapshot)),flush=True)
                        continue
                    break
            state['stages'].append(dict(stage=name,exit_code=code,wall_seconds=time.monotonic()-started,log=str(log)))
            write(output/'status.json',state)
            print(json.dumps(dict(event='finish',**state['stages'][-1])),flush=True)
            if code:raise RuntimeError(f'{name} failed: see {log}')
        for path,expected in sources.items():
            if digest(path)!=expected:raise RuntimeError('sealed source changed: '+path)
        state.update(status='completed',finished_utc=datetime.now(timezone.utc).isoformat())
        write(output/'status.json',state)
        print(json.dumps(dict(event='completed',output=str(output))),flush=True)
    except BaseException as exc:
        state.update(status='failed',error=f'{type(exc).__name__}: {exc}',finished_utc=datetime.now(timezone.utc).isoformat())
        write(output/'status.json',state)
        raise


if __name__=='__main__': main()
