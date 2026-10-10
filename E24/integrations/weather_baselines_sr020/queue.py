"""Persistent, sequential Weather baseline queue; independent of the IDE session."""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_ROOT=ROOT/'artifacts/Weather/seq96_pred96/sr020_bl8/weather_author_comparison_sr020_20261008'
PROFILES=ROOT/'configs/baselines/weather_sr020_bl8_author_20261008.json'
MANIFEST=ROOT/'artifacts/Weather/seq96_pred96/sr020_bl8/shared_protocol_v1/dataset_manifest.json'
CSV=ROOT/'docs/experiments/Weather_sr020_bl8_E24_baselines_20261008.csv'
REFERENCE=ROOT/'docs/experiments/Weather_sr020_bl8_E24_v79_seed2023_results.csv'
ORDER=['T1','SAITS','S4M','CRIB','CoIFNet','BiTGraph','ChannelTokenFormer']
GPU='GPU-1315e15c-d657-e61f-bde5-6e0b38e6e54b'
HEARTBEAT_SECONDS=3600


def utc():return datetime.now(timezone.utc).isoformat()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');tmp.replace(path)
def alive(pid):
    try:os.kill(int(pid),0);return True
    except (ProcessLookupError,ValueError,TypeError):return False


def gpu_processes():
    result=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],capture_output=True,text=True)
    return [line.strip() for line in result.stdout.splitlines() if GPU in line]


def python(name):
    environment='malr-s4m' if name=='S4M' else 'malr-saits' if name=='SAITS' else 'malr'
    return f'/home/xyw/miniforge3/envs/{environment}/bin/python'


def seal_sources(output):
    paths=set()
    for directory in ('src','integrations','scripts','configs'):
        paths.update(p for p in (ROOT/directory).rglob('*') if p.is_file() and p.suffix in ('.py','.json','.yaml','.yml','.ini','.toml','.sh'))
    profile=read(PROFILES)
    git='/home/xyw/miniforge3/envs/malr/bin/git'
    for cfg in profile['models'].values():
        repo=Path(cfg['root'])
        names=subprocess.check_output([git,'ls-files','-z'],cwd=repo).decode().split('\0')
        paths.update(repo/name for name in names if name and (repo/name).is_file()
                     and (Path(name).suffix in ('.py','.json','.yaml','.yml','.ini','.toml','.sh') or Path(name).name.startswith('README'))
                     and (repo/name).stat().st_size<5*2**20)
    dataset=read(MANIFEST)
    paths.update([PROFILES,MANIFEST,MANIFEST.with_suffix('.sha256'),Path(dataset['config_path']),REFERENCE])
    # The sealed dataset loader verifies all data/mask hashes at every entrypoint.
    records=[dict(path=str(p.resolve()),sha256=sha(p)) for p in sorted(paths)]
    archive=output/'experiment-source.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for row in records:
            path=Path(row['path']);tar.add(path,arcname=str(path.relative_to(ROOT.parent)),recursive=False)
    seal=dict(created_utc=utc(),files=records,dataset_manifest_sha256=sha(MANIFEST),archive=str(archive),archive_sha256=sha(archive))
    write(output/'source_seal.json',seal)
    Path(str(archive)+'.sha256').write_text(sha(archive)+'  '+archive.name+'\n')
    return seal


def verify_seal(output):
    seal=read(output/'source_seal.json')
    for row in seal['files']:
        if sha(row['path'])!=row['sha256']:raise ValueError('sealed experiment source changed: '+row['path'])
    if sha(MANIFEST)!=seal['dataset_manifest_sha256']:raise ValueError('dataset manifest changed')


def prepare(output):
    profile=read(PROFILES);output.mkdir(parents=True,exist_ok=True)
    for name in ORDER:
        acceptance=read(output/'preflight'/name/'acceptance.json')
        ident=acceptance['identity']
        if acceptance['status']!='passed' or not acceptance['finite_backward'] or not acceptance['labels_excluded']:raise ValueError('failed preflight: '+name)
        if profile['models'][name]['kind']=='imputer' and not acceptance['stage2_freeze_checked']:raise ValueError('freeze not checked: '+name)
        if ident['author_profiles_sha256']!=sha(PROFILES):raise ValueError('preflight profile changed: '+name)
        if ident['weather_adapter_sha256']!=sha(ROOT/'integrations/weather_baselines_sr020/run.py'):raise ValueError('preflight worker changed: '+name)
        if ident['weather_model_adapter_sha256']!=sha(ROOT/'integrations/weather_baselines_sr020/models.py'):raise ValueError('preflight builder changed: '+name)
    seal=seal_sources(output)
    plan=dict(created_utc=utc(),dataset='Weather',start_rate=0.2,block_len=8,test_draws=20,
        order=ORDER,profiles=str(PROFILES),profiles_sha256=sha(PROFILES),dataset_manifest=str(MANIFEST),
        dataset_manifest_sha256=sha(MANIFEST),source_seal_sha256=sha(output/'source_seal.json'),
        gpu_uuid=GPU,heartbeat_seconds=HEARTBEAT_SECONDS,max_attempts_per_stage=2,
        repair_update='fully_frozen',shared_stage2_loss='pure_mse',head_seed=2023,
        stage1_seeds={n:profile['models'][n]['seed'] for n in ORDER},
        python_environments={n:python(n) for n in ORDER},
        scope='one training run per model, fixed validation selection, same 20 test masks; no test tuning',
        concurrent_gpu_jobs_at_launch=gpu_processes(),archive_sha256=seal['archive_sha256'])
    write(output/'plan.json',plan)
    return plan


FIELDS=['dataset','model','model_type','status','selection_metric','primary','start_rate','block_len','seq_len','pred_len','n_vars',
 'stage1_seed','stage2_seed','test_masks','test_windows_per_mask','forecast_mse','forecast_mse_mask_std_ddof1',
 'forecast_mae','forecast_mae_mask_std_ddof1','repair96_mse','repair96_mse_mask_std_ddof1','repair96_mae','repair96_mae_mask_std_ddof1',
 'repair32_mse','repair32_mse_mask_std_ddof1','repair32_mae','repair32_mae_mask_std_ddof1',
 'stage1_validation_repair_mse','stage1_validation_repair_mae','stage1_epochs','stage2_epochs','native_epochs',
 'stage1_best_epoch','stage2_best_epoch','native_best_epoch','stage1_optimizer_updates','stage2_optimizer_updates','native_optimizer_updates',
 'stage1_full_command_seconds','stage2_full_command_seconds','native_full_command_seconds','full_training_workflow_seconds',
 'all_training_attempts_seconds','final_test_seconds','peak_allocated_GiB','peak_reserved_GiB','gpu','gpu_uuid',
 'batch_sizes','precisions','stage2_loss','stage2_repair_update','author_version','repair_metric_note',
 'checkpoint_sha256','result_path','training_seed_std','drr','gpu_shared_with_other_jobs']


def update_csv(output,state,profile):
    rows=[]
    for reference in csv.DictReader(REFERENCE.open()):
        row={k:reference[k] for k in FIELDS if k in reference}
        row.update(model_type='imputer',status='completed',stage1_seed=2023,stage2_seed=2023,
                   stage2_loss='pure_mse',stage2_repair_update='fully_frozen',author_version='frozen E23-v79 Weather adaptation',
                   gpu_uuid=GPU,batch_sizes='128;128',precisions='float32;float32',
                   repair_metric_note='frozen Stage1 explicit repair',gpu_shared_with_other_jobs='see original run resources')
        rows.append(row)
    for name in ORDER:
        cfg=profile['models'][name]
        base=dict(dataset='Weather',model=name,model_type=cfg['kind'],status=state['models'][name]['status'],
                  start_rate=0.2,block_len=8,seq_len=96,pred_len=96,n_vars=21,stage1_seed=cfg['seed'],
                  stage2_seed=2023 if cfg['kind']=='imputer' else None,test_masks=20,test_windows_per_mask=10444,
                  stage2_loss='pure_mse' if cfg['kind']=='imputer' else None,
                  stage2_repair_update='fully_frozen' if cfg['kind']=='imputer' else None,
                  author_version=cfg['version'],gpu_uuid=GPU,gpu_shared_with_other_jobs=bool(read(output/'plan.json')['concurrent_gpu_jobs_at_launch']))
        finalized=False
        for metric in (['mae','mse'] if cfg['kind']=='imputer' else [cfg['training']['selection_metric']]):
            result_path=output/name/'evaluation'/metric/'twenty-test-draws.json'
            if not result_path.exists():continue
            result=read(result_path);selection=read(result['selection']);resources=result['training_resources']
            row={**base,'status':'completed','selection_metric':metric,'primary':metric==('mae' if cfg['kind']=='imputer' else cfg['training']['selection_metric']),
                 'checkpoint_sha256':result['checkpoint_sha256'],'result_path':str(result_path),
                 'repair_metric_note':result['repair_metric_note']}
            for key,value in result['summary'].items():
                column='forecast_'+key if key in ('mse','mae') else key
                row[column]=None if value is None else value['mean']
                row[column+'_mask_std_ddof1']=None if value is None else value['mask_std_ddof1']
            row.update(full_training_workflow_seconds=resources['full_workflow_seconds'],
                       peak_allocated_GiB=resources['peak_allocated_gib'],peak_reserved_GiB=resources['peak_reserved_gib'],
                       gpu=resources['gpu'],batch_sizes=resources['batch_size'])
            for resource_path in resources['resource_record_paths']:
                resource=read(resource_path);selected=read(resource['selection_manifest'])
                stage='stage1' if selected['stage']=='repair' else 'stage2' if cfg['kind']=='imputer' else 'native'
                row[stage+'_full_command_seconds']=resource['full_workflow_seconds']
                row[stage+'_epochs']=selected['epochs_completed'];row[stage+'_best_epoch']=selected['best_epoch']
                row[stage+'_optimizer_updates']=selected['optimizer_updates']
                if stage=='stage1':
                    row['stage1_validation_repair_mse']=selected['best_validation']['mse']
                    row['stage1_validation_repair_mae']=selected['best_validation']['mae']
            row['precisions']=';'.join(read(p)['precision'] for p in resources['resource_record_paths'])
            attempts=[r for r in state['attempts'] if r['model']==name and r['action']=='train']
            row['all_training_attempts_seconds']=sum(r.get('elapsed_seconds',0) for r in attempts)
            row['final_test_seconds']=next((r['elapsed_seconds'] for r in reversed(state['attempts']) if r['model']==name and r['action']=='evaluate' and r.get('metric')==metric and r.get('returncode')==0),None)
            rows.append(row);finalized=True
        if not finalized:rows.append({**base,'selection_metric':'mae' if cfg['kind']=='imputer' else cfg['training']['selection_metric'],'primary':True})
    CSV.parent.mkdir(parents=True,exist_ok=True)
    tmp=CSV.with_suffix('.csv.tmp')
    with tmp.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=FIELDS,extrasaction='raise');writer.writeheader();writer.writerows(rows)
    tmp.replace(CSV)


def run(output):
    lock=(output/'queue.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=read(output/'plan.json');profile=read(PROFILES)
    if sha(output/'source_seal.json')!=plan['source_seal_sha256']:raise ValueError('source seal changed')
    verify_seal(output)
    state_path=output/'status.json'
    state=read(state_path) if state_path.exists() else dict(models={n:dict(status='queued') for n in ORDER},attempts=[])
    state.update(status='running',pid=os.getpid(),started_utc=utc(),active=None)
    (output/'queue.pid').write_text(str(os.getpid())+'\n')
    env={**os.environ,'CUDA_VISIBLE_DEVICES':GPU,'PYTHONPATH':str(ROOT/'src')+':'+str(ROOT),'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','PYTHONUNBUFFERED':'1'}
    def persist(event=None):
        state['updated_utc']=utc();write(state_path,state);update_csv(output,state,profile)
        if event:
            with (output/'journal.jsonl').open('a') as stream:stream.write(json.dumps(dict(time=utc(),**event))+'\n')
            print(json.dumps(dict(time=utc(),**event)),flush=True)
    persist(dict(event='queue_started',pid=os.getpid(),models=ORDER,gpu_processes=gpu_processes()))

    def execute(name,action,directory,*,stage=None,parent=None,selection=None,metric=None):
        verify_seal(output)
        final=directory/('selection.json' if action=='train' else 'twenty-test-draws.json')
        if final.exists():
            # A stage manifest is finalized before its resource sidecar. Require
            # both so a process killed during final reporting is not misreported.
            if action=='evaluate':return True
            valid=read(final)
            records=[read(p) for p in (directory/'resources').glob('*.json')]
            if any(r.get('status')=='completed' and r.get('mode')=='training' and r.get('selection_manifest_sha256')==sha(final) for r in records):return True
            raise RuntimeError('finalized stage lacks completed resource sidecar: '+str(directory))
        directory.mkdir(parents=True,exist_ok=True)
        command=[python(name),'-u','-m','integrations.weather_baselines_sr020.run',action,'--model',name,
                 '--profiles',str(PROFILES),'--dataset-manifest',str(MANIFEST),'--output',str(directory)]
        if stage:command+=['--stage',stage]
        if parent:command+=['--parent',str(parent)]
        if selection:command+=['--selection',str(selection)]
        previous=[r for r in state['attempts'] if r['model']==name and r['action']==action and r.get('stage')==stage and r.get('metric')==metric]
        for attempt in range(len(previous)+1,3):
            actual=command+(['--resume'] if action=='train' and (directory/'latest.pt').exists() else [])
            log_path=directory/f'{action}-attempt{attempt}.log';started=time.monotonic()
            row=dict(model=name,action=action,stage=stage,metric=metric,attempt=attempt,started_utc=utc(),command=actual,log=str(log_path),gpu_processes_at_start=gpu_processes())
            state['attempts'].append(row)
            with log_path.open('a') as log:
                child=subprocess.Popen(actual,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                row['pid']=child.pid
                state['active']=dict(model=name,action=action,stage=stage,metric=metric,pid=child.pid,log=str(log_path))
                persist(dict(event='child_started',**state['active'],attempt=attempt))
                next_heartbeat=time.monotonic()+HEARTBEAT_SECONDS
                while child.poll() is None:
                    time.sleep(10)
                    row['elapsed_seconds']=time.monotonic()-started
                    write(state_path,{**state,'updated_utc':utc()})
                    if time.monotonic()>=next_heartbeat:
                        tail=log_path.read_text(errors='replace')[-1600:]
                        persist(dict(event='heartbeat',active=state['active'],elapsed_seconds=row['elapsed_seconds'],tail=tail,gpu_processes=gpu_processes()))
                        next_heartbeat=time.monotonic()+HEARTBEAT_SECONDS
                row.update(returncode=child.returncode,finished_utc=utc(),elapsed_seconds=time.monotonic()-started)
                state['active']=None;persist(dict(event='child_finished',model=name,action=action,stage=stage,metric=metric,returncode=child.returncode))
            if child.returncode==0 and final.exists():return True
            if action=='train' and final.exists():raise RuntimeError('training exited after manifest but without successful reporting: '+str(log_path))
        return False

    for name in ORDER:
        if state['models'][name]['status']=='completed':continue
        state['models'][name]=dict(status='running',started_utc=utc());persist(dict(event='model_started',model=name))
        try:
            base=output/name;cfg=profile['models'][name]
            if cfg['kind']=='imputer':
                if not execute(name,'train',base/'stage1',stage='repair'):raise RuntimeError('Stage1 exhausted attempts')
                if not execute(name,'train',base/'stage2',stage='forecast',parent=base/'stage1/selection.json'):raise RuntimeError('Stage2 exhausted attempts')
                choices=[('mae',base/'stage2/selection.json'),('mse',base/'stage2/selection-mse.json')]
            else:
                if not execute(name,'train',base/'training',stage='forecast'):raise RuntimeError('native training exhausted attempts')
                choices=[(cfg['training']['selection_metric'],base/'training/selection.json')]
            for metric,selected in choices:
                if not execute(name,'evaluate',base/'evaluation'/metric,selection=selected,metric=metric):raise RuntimeError('evaluation exhausted attempts: '+metric)
            state['models'][name].update(status='completed',finished_utc=utc())
            persist(dict(event='model_completed',model=name,summary_csv=str(CSV)))
        except Exception as error:
            state['models'][name].update(status='failed',finished_utc=utc(),error=repr(error))
            persist(dict(event='model_failed_continuing',model=name,error=repr(error)))
    state.update(status='completed' if all(r['status']=='completed' for r in state['models'].values()) else 'completed_with_failures',active=None,finished_utc=utc())
    persist(dict(event='queue_finished',status=state['status'],models=state['models']))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['launch','run','status']);parser.add_argument('--output',type=Path,default=DEFAULT_ROOT)
    args=parser.parse_args();output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    if args.action=='status':print(json.dumps(read(output/'status.json'),indent=2));return
    if args.action=='run':run(output);return
    pid_path=output/'queue.pid'
    if pid_path.exists() and alive(pid_path.read_text().strip()):raise RuntimeError('background queue already alive')
    if not (output/'plan.json').exists():prepare(output)
    else:verify_seal(output)
    log=(output/'queue.log').open('a')
    child=subprocess.Popen([sys.executable,'-u','-m','integrations.weather_baselines_sr020.queue','run','--output',str(output)],
        cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT/'src')+':'+str(ROOT)},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    pid_path.write_text(str(child.pid)+'\n');log.close()
    print(json.dumps(dict(pid=child.pid,log=str(output/'queue.log'),status=str(output/'status.json'),csv=str(CSV))),flush=True)


if __name__=='__main__':main()
