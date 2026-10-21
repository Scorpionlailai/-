"""Wait without touching A4, execute exactly once on idle GPU0, publish text evidence."""
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

LOCAL=Path('/home/xyw/v125-shared-prefix-acceptance-20261021')
REPO=Path('/home/xyw/paper-v125-shared-prefix-acceptance')
PKG=REPO/'experiment_results/v125_shared_prefix_acceptance_20261021'
RUN_ID='server_xyw_20261021'
OUTPUT=LOCAL/'runs'/RUN_ID
PUBLISH=PKG/'results'/RUN_ID
A4=Path('/home/xyw/paper/E24/artifacts/v125_a4_seed_recheck_20261021')
BRANCH='acceptance/pamap2-shared-prefix-server-20261021'


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def a4_is_busy(processes):
    return any(x.get('state') not in ('Z','X') and any(
        arg.startswith('/') and arg.endswith('/v125_a4_seed_recheck_20261021/tools/'+name)
        for arg in x.get('argv',x.get('args','').split()) for name in ('supervisor3.py','run3.py')) for x in processes)


def process_snapshot():
    result=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            argv=[a.decode(errors='replace') for a in (p/'cmdline').read_bytes().split(b'\0') if a];args=' '.join(argv)
            status=(p/'status').read_text();state=status.split('State:')[1].split()[0]
        except (OSError,IndexError):continue
        result.append(dict(pid=int(p.name),args=args,argv=argv,state=state))
    return result


def verify_lock():
    record=json.loads((LOCAL/'execution_lock.json').read_text())
    for name,digest in record['source_files'].items():assert sha(LOCAL/'src'/name)==digest,('execution source changed',name)
    for name,digest in record['external_files'].items():assert sha(Path(name))==digest,('external source changed',name)
    for name,digest in record['inventory_files'].items():assert sha(LOCAL/'inventory'/name)==digest,('inventory changed',name)
    return record


def export_text(status,initial=False):
    PUBLISH.mkdir(parents=True,exist_ok=True)
    for folder in ('inventory','src'):
        target=PUBLISH/('source' if folder=='src' else folder);target.mkdir(exist_ok=True)
        for p in (LOCAL/folder).glob('*'):
            if p.is_file() and p.suffix in ('.json','.py','.txt'):shutil.copyfile(p,target/p.name)
    for name in ('execution_lock.json','cpu_coordinate_preflight.json','test_verification.json','command.json','scheduled.json'):
        if (LOCAL/name).exists():shutil.copyfile(LOCAL/name,PUBLISH/name)
    if (LOCAL/'launcher.log').exists():shutil.copyfile(LOCAL/'launcher.log',PUBLISH/'launcher.log')
    if OUTPUT.exists():
        for p in OUTPUT.rglob('*'):
            if p.is_file() and p.suffix in ('.json','.csv','.md','.txt','.log'):
                dest=PUBLISH/'run'/p.relative_to(OUTPUT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
        if (OUTPUT/'report.md').exists():shutil.copyfile(OUTPUT/'report.md',PUBLISH/'report.md')
        if (OUTPUT/'result_manifest.json').exists():shutil.copyfile(OUTPUT/'result_manifest.json',PUBLISH/'result_manifest.json')
    else:
        write(PUBLISH/'result_manifest.json',dict(status='WAITING_FOR_IDLE_GPU',evidence_class='NEW_PAMAP2_FINITE_ACCEPTANCE',
            package_commit='392fc08425371c5a7aad7db47b89440c9548ae7c',assets_verified=9,GPU_acceptance_executed=False,
            all_exact_checks_passed=None,resource_gates_passed=None,formal_model_promoted=False,test_accessed=False,Stage2_run=False,
            missing_fields=['GPU equivalence/performance results pending A4 completion; no values fabricated']))
    write(PUBLISH/'publication_status.json',status)
    if initial:
        (PUBLISH/'README.md').write_text('# 原服务器PAMAP2有限验收\n\n状态：9份清单资产及原源码/协议锁核验通过，CPU适配/梯度检查通过；等待既有A4实验释放GPU。尚未执行GPU性能验收。\n\n唯一预算：A0 seed2023，原双分支各2轮；参考340次、共享340次、恢复重放170次分支更新，共850次。不训练A1或新seed，不调用test批次/评分，不跑Stage2。\n\n三依赖路径位于ablation工作区而非原paper/E24路径，映射和从原设计锁恢复的SHA256见inventory。候选只复用当前batch冻结前缀，保留原外层尺度恢复、独立Adam/RNG、原完整验证及逐支选模。\n\n验收门槛保持任务包原定义：所有数值状态逐值一致、更新段节省≥15%、allocated峰值增幅≤5%、各原推理路径中位耗时回退≤5%。失败或计时不确定如实记录，不自动重跑或扩大范围。原A4和正式消融资产不改动。\n\n源码与计划锁见source及execution_lock.json；完成后将回传四张CSV、完整环境/运行日志及结论。大型checkpoint/验证数组留服务器，不进Git。\n')
    else:
        manifest=json.loads((OUTPUT/'result_manifest.json').read_text()) if (OUTPUT/'result_manifest.json').exists() else {}
        (PUBLISH/'README.md').write_text('# 原服务器PAMAP2有限验收结果\n\n状态：'+manifest.get('status','EXECUTION_FAILED')+'。\n\n来源为交接提交392fc08425371c5a7aad7db47b89440c9548ae7c，9份资产已核验。完整报告见report.md，原始结果/日志及四张CSV见run/。无test评分或Stage2，诊断权重不晋级正式模型。大型权重与验证数组仅保留服务器，路径/SHA详见result_manifest.json。\n\n所有失败、资源门槛未通过及缺测项均保留；没有自动重跑或扩大预算。A4独立实验和experiment_results/v125_ablation未被本验收覆盖。\n')
    manifest={str(p.relative_to(PUBLISH)):sha(p) for p in PUBLISH.rglob('*') if p.is_file() and p.name!='published_file_hashes.json'}
    write(PUBLISH/'published_file_hashes.json',manifest)


def git_publish(message):
    branch=subprocess.run(['git','branch','--show-current'],cwd=REPO,check=True,text=True,capture_output=True).stdout.strip();assert branch==BRANCH
    relative=str(PUBLISH.relative_to(REPO))
    subprocess.run(['git','add','--',relative],cwd=REPO,check=True)
    names=subprocess.run(['git','diff','--cached','--name-only'],cwd=REPO,check=True,text=True,capture_output=True).stdout.splitlines()
    assert all(name.startswith(relative+'/') for name in names),'unrelated staged changes'
    for p in PUBLISH.rglob('*'):
        if p.is_file():assert p.suffix in ('.py','.md','.json','.txt','.csv','.log') and p.stat().st_size<20*1024*1024
    if names:subprocess.run(['git','commit','-m',message],cwd=REPO,check=True)
    commit=subprocess.run(['git','rev-parse','HEAD'],cwd=REPO,text=True,capture_output=True,check=True).stdout.strip()
    subprocess.run(['git','push','-u','origin',BRANCH],cwd=REPO,check=True)
    remote=subprocess.run(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=REPO,text=True,capture_output=True,check=True).stdout.split()[0];assert remote==commit
    return dict(commit_sha=commit,remote_verified=True,branch=BRANCH,
        directory_url=f'https://github.com/Scorpionlailai/-/tree/{commit}/{relative}',published_unix=time.time())


def idle_gpu_probe(env):
    cmd=[sys.executable,'-c',"import torch,json;torch.cuda.set_device(0);free,total=torch.cuda.mem_get_info();print(json.dumps(dict(free=free,total=total,gpu=torch.cuda.get_device_name(0))))"]
    p=subprocess.run(cmd,env=env,text=True,capture_output=True);(LOCAL/'idle_gpu_probe.txt').write_text(p.stdout+p.stderr)
    if p.returncode:return False,None
    line=p.stdout.strip().splitlines()[-1];value=json.loads(line)
    return value['free']/value['total']>.95,value


def main():
    LOCAL.mkdir(parents=True,exist_ok=True)
    with (LOCAL/'waiter.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);verify_lock()
        if (LOCAL/'executed_once.json').exists():raise RuntimeError('one finite execution was already started; no automatic reruns')
        started=time.time();last_notice=0.;deadline=started+12*3600
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',
            PYTHONPATH='/home/xyw/paper-e23-v125-ablation/E24:/home/xyw/paper-e23-v125-ablation/E24/src:'+str(PKG/'code/local_worktree'))
        while True:
            verify_lock();procs=process_snapshot();busy=a4_is_busy(procs)
            state=dict(phase='waiting_for_A4_and_idle_GPU',pid=os.getpid(),A4_busy=busy,wait_seconds=time.time()-started,updated_unix=time.time(),run_id=RUN_ID)
            write(LOCAL/'wait_status.json',state)
            if not busy:
                idle,gpu=idle_gpu_probe(env)
                if idle:
                    # Check again after creating/releasing the probe CUDA context to avoid a known queued A4 task race.
                    if not a4_is_busy(process_snapshot()):break
            if time.time()>deadline:raise RuntimeError('12hour bounded wait exceeded; no GPU acceptance launched')
            if time.time()-last_notice>=240:print(json.dumps(state),flush=True);last_notice=time.time()
            time.sleep(30)
        verify_lock();write(LOCAL/'executed_once.json',dict(started_unix=time.time(),gpu=gpu,prior_A4_active=False,source_lock_sha256=sha(LOCAL/'execution_lock.json')))
        command=[sys.executable,'-u',str(LOCAL/'src/run_acceptance.py'),'--asset-lock',str(LOCAL/'inventory/verified-assets.json'),
            '--arm','A0','--seed','2023','--epochs','2','--batch-size','128','--modes','reference','shared','resume','--output',str(OUTPUT)]
        write(LOCAL/'command.json',dict(argv=command,environment_overrides={k:env[k] for k in ('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','PYTHONUNBUFFERED','CUBLAS_WORKSPACE_CONFIG','PYTHONPATH')},cwd=str(REPO),run_id=RUN_ID))
        OUTPUT.parent.mkdir(parents=True,exist_ok=True)
        with (LOCAL/'runner.log').open('x') as log:
            p=subprocess.Popen(command,env=env,cwd=REPO,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            write(LOCAL/'wait_status.json',dict(phase='finite_acceptance_running',runner_pid=p.pid,pid=os.getpid(),started_unix=time.time(),gpu=gpu))
            # Monitor without reruns; A4 should be terminal. If externally restarted, preserve the run but mark timing contaminated.
            shared_detected=False
            while p.poll() is None:
                if a4_is_busy(process_snapshot()):shared_detected=True
                state=dict(phase='finite_acceptance_running',runner_pid=p.pid,updated_unix=time.time(),gpu_shared_with_A4_detected=shared_detected)
                write(LOCAL/'wait_status.json',state);time.sleep(20)
            code=p.returncode
        OUTPUT.mkdir(parents=True,exist_ok=True);shutil.copyfile(LOCAL/'runner.log',OUTPUT/'run.log')
        if (LOCAL/'idle_gpu_probe.txt').exists():shutil.copyfile(LOCAL/'idle_gpu_probe.txt',OUTPUT/'idle_gpu_probe.txt')
        if not (OUTPUT/'result_manifest.json').exists():
            write(OUTPUT/'result_manifest.json',dict(status='EXECUTION_FAILED',evidence_class='NEW_PAMAP2_FINITE_ACCEPTANCE',exit_code=code,
                all_exact_checks_passed=None,resource_gates_passed=None,test_accessed=False,Stage2_run=False,formal_model_promoted=False,failures=['runner terminated before result manifest; see run.log']))
        for filename in ('metrics.csv','resources.csv','epoch_audit.csv','paired_checks.csv'):
            if not (OUTPUT/filename).exists():
                fields=(PKG/'templates'/filename).read_text().splitlines()[0].split(',')
                if 'explanation' not in fields:fields.append('explanation')
                with (OUTPUT/filename).open('w') as f:
                    writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
                    writer.writerow(dict(explanation='Not measured: process ended before this table was emitted; no automatic rerun; see run.log'))
        if not (OUTPUT/'report.md').exists():
            (OUTPUT/'report.md').write_text('# PAMAP2有限验收中止\n\n运行未产生完整报告，原因见run.log和result_manifest.json；四张CSV中的未测值保留为空。未自动重跑。\n')
        if shared_detected:
            manifest=json.loads((OUTPUT/'result_manifest.json').read_text())
            manifest['status']='TIMING_CONTAMINATED_UNCERTAIN';manifest['resource_gates_passed']=None
            manifest.setdefault('failures',[]).append('A4 externally restarted during acceptance; GPU sharing observed, original no-sharing timing condition not satisfied')
            write(OUTPUT/'result_manifest.json',manifest)
        status=dict(phase='completed' if code==0 else 'failed_no_automatic_rerun',exit_code=code,completed_unix=time.time(),gpu_shared_detected=shared_detected)
        export_text(status);publication=git_publish('Record bounded PAMAP2 shared-prefix acceptance results and exact-state audits')
        write(LOCAL/'publication.json',publication);write(LOCAL/'completion.json',dict(**status,publication=publication,result=json.loads((OUTPUT/'result_manifest.json').read_text())))
        write(LOCAL/'wait_status.json',dict(**status,publication=publication))
        print(json.dumps(dict(**status,publication=publication)),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        write(LOCAL/'waiter_failure.json',dict(time=time.time(),traceback=traceback.format_exc(),pid=os.getpid(),no_automatic_rerun=True));raise
