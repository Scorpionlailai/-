"""Two independent GPU queues, using per-model frozen Exchange runners."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import sys,os,json,time,subprocess,hashlib,csv,traceback,fcntl
Q=Path('/home/xyw/paper/E23/artifacts/Exchange-Rate/seq96_pred96/experiments/csdi_budget_100ep_train1_eval5_20261010')
PARENT=Q.parent/'v125_fixed_four_scenes_20261010'
GPUS=['GPU-382ce2ea-81a9-82ea-073f-729eaefd0268','GPU-1315e15c-d657-e61f-bde5-6e0b38e6e54b']
ORDERS=[['CSDI'],['CSDI']]
OLD=Q.parent/'v125_baselines_20261010'
SCENES_BY_GPU=[['sr010_bl8','sr020_bl8'],['sr015_bl8','sr010_bl16']]
def now():return datetime.now(timezone.utc).isoformat()
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)
def freeze(runner):
    for p,h in json.loads((runner/'source_freeze.json').read_text())['files'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
def tail(log):
    with log.open('rb') as f:f.seek(max(0,log.stat().st_size-2000));lines=f.read().decode(errors='replace').splitlines()
    return lines[-1] if lines else ''
def run(model,scene,gpu,phase,extra):
    runner=Q/'runners'/model;root=Q/'runs'/model/scene;root.mkdir(parents=True,exist_ok=True);freeze(runner)
    python='/home/xyw/miniforge3/envs/'+('malr-saits' if model=='SAITS' else 'malr')+'/bin/python'
    manifest=PARENT/scene/'shared_protocol/dataset_manifest.json'
    cmd=[python,'-u',str(runner/'worker.py'),*extra,'--model',model,'--dataset-manifest',str(manifest),'--profiles',str(runner/'profiles.json'),'--device','cuda:0']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPUS[gpu],OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    for k in ('E24_DATASET_MANIFEST','MALR_MISSING_SCENARIO','MALR_E23_CODE_FREEZE'):env.pop(k,None)
    log=root/(phase+'.log')
    with log.open('xb') as f:
        child=subprocess.Popen(cmd,cwd=runner,env=env,stdout=f,stderr=subprocess.STDOUT)
        write(root/(phase+'_launch.json'),dict(pid=child.pid,command=cmd,started_utc=now(),gpu_uuid=GPUS[gpu]))
        while child.poll() is None:
            s=dict(status='running',model=model,scene=scene,phase=phase,pid=child.pid,gpu_index=gpu,last_line=tail(log),updated_utc=now())
            write(root/'status.json',s);write(Q/f'gpu{gpu}_status.json',s);time.sleep(30)
        assert child.returncode==0,f'{model}/{scene}/{phase} exited {child.returncode}: {log}'
    freeze(runner)
def gpu_queue(gpu):
    for model in ORDERS[gpu]:
        runner=Q/'runners'/model
        while not (runner/'ready.json').exists():
            write(Q/f'gpu{gpu}_status.json',dict(status='waiting_for_adapter',model=model,gpu_index=gpu,updated_utc=now()));time.sleep(30)
        assert json.loads((runner/'ready.json').read_text())['status']=='ready'
        for scene in SCENES_BY_GPU[gpu]:
            root=Q/'runs'/model/scene
            try:
                run(model,scene,gpu,'preflight',['preflight','--output',str(root/'preflight')])
                assert json.loads((root/'preflight/acceptance.json').read_text())['status']=='passed'
                previous=OLD/'runs/CSDI'/scene/'stage1/selection.json'
                if previous.exists():
                    run(model,scene,gpu,'stage1',['reuse','--parent',str(previous),'--output',str(root/'stage1')])
                else:
                    run(model,scene,gpu,'stage1',['train','--stage','repair','--output',str(root/'stage1')])
                parent=root/'stage1/selection.json'
                run(model,scene,gpu,'repair_test20',['evaluate','--selection',str(parent),'--output',str(root/'repair_evaluation')])
                run(model,scene,gpu,'stage2',['train','--stage','forecast','--parent',str(parent),'--output',str(root/'stage2')])
                for metric,filename in [('mae','selection.json'),('mse','selection-mse.json')]:
                    run(model,scene,gpu,'forecast_test20_'+metric,['evaluate','--selection',str(root/'stage2'/filename),'--output',str(root/('forecast_evaluation_'+metric))])
                write(root/'status.json',dict(status='completed',model=model,scene=scene,updated_utc=now(),github_published=False))
            except BaseException as error:
                write(root/'status.json',dict(status='failed',model=model,scene=scene,error=repr(error),traceback=traceback.format_exc(),updated_utc=now()));raise
        write(Q/f'gpu{gpu}_status.json',dict(status='completed',gpu_index=gpu,updated_utc=now()))
def summarize():
    for kind,pattern in [('repair','*/ */repair_evaluation/twenty-test-draws.json'),('forecast','*/ */forecast_evaluation_*/twenty-test-draws.json')]:
        rows=[]
        for p in sorted((Q/'runs').glob(pattern.replace(' ',''))):
            x=json.loads(p.read_text());scene=p.parents[1].name;label=p.parent.name
            r=dict(model=x['model'],scene=scene,selection='mae' if label.endswith('mae') else 'mse' if label.endswith('mse') else 'stage1_rule',test_masks=len(x['draws']))
            for k,v in x['summary'].items():
                if isinstance(v,dict):r[k]=v['mean'];r[k+'_mask_std_ddof1']=v['mask_std_ddof1']
            cost=x['training_resources'];r.update(full_workflow_seconds=cost['full_workflow_seconds'],peak_allocated_gib=cost['peak_allocated_gib'],peak_reserved_gib=cost['peak_reserved_gib'],source=str(p));rows.append(r)
        if rows:
            keys=list(dict.fromkeys(k for row in rows for k in row))
            with (Q/(kind+'_results.csv')).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def main():
    with (Q/'queue.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);write(Q/'status.json',dict(status='running',gpu_orders=ORDERS,started_utc=now(),github_upload_requires_review=True))
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(gpu_queue,i) for i in (0,1)]
            while not all(f.done() for f in futures):summarize();time.sleep(30)
            for f in futures:f.result()
        summarize();write(Q/'status.json',dict(status='completed_awaiting_user_review',models=1,scenes_per_model=4,github_published=False,updated_utc=now()))
try:main()
except BaseException as error:
    write(Q/'status.json',dict(status='failed',error=repr(error),traceback=traceback.format_exc(),updated_utc=now()));raise
