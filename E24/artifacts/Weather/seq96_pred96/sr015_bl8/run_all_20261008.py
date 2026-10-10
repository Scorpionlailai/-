import subprocess,sys,json,os,time
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path('/home/xyw/paper/E24')
BASE=Path(__file__).resolve().parent
PY='/home/xyw/miniforge3/envs/malr/bin/python'
RUN=BASE/'e24_v79_weather_sr015_20261008/seed_2023/E24-v79'
MANIFEST=BASE/'shared_protocol_v1/dataset_manifest.json'
def record(event,**data):
    value=dict(time_utc=datetime.now(timezone.utc).isoformat(),event=event,controller_pid=os.getpid(),**data)
    with (BASE/'all_models_journal.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')
    (BASE/'all_models_status.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value),flush=True)
def execute(name,command):
    with (BASE/(name+'.log')).open('a') as f:
        p=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        record('started',phase=name,pid=p.pid,command=command)
        while True:
            try:code=p.wait(timeout=3600);break
            except subprocess.TimeoutExpired:record('hourly_report',phase=name,pid=p.pid,log=str(BASE/(name+'.log')))
        record('finished',phase=name,returncode=code)
        return code
(BASE/'all_models.pid').write_text(str(os.getpid())+'\n')
code=execute('E24-v79',[PY,'-u','-m','integrations.e24_sr015.weather_workflow','--dataset-manifest',str(MANIFEST),'--output',str(RUN),'--device','cuda:0'])
if code==0:code=execute('summarize_E24',[PY,'scripts/summarize_weather_v79_sr015.py','--run',str(RUN)])
if code:
    # Keep baseline queue running, but leave E24 values missing if its run failed.
    reference=ROOT/'docs/experiments/Weather_sr015_bl8_E24_v79_seed2023_results.csv'
    if not reference.exists():reference.write_text('dataset,model,selection_metric,primary\n')
    record('E24_failed_baselines_continue',returncode=code)
code=execute('launch_baselines',[PY,'-u','-m','integrations.weather_baselines_sr015.queue','launch'])
record('baselines_launched' if code==0 else 'baseline_launch_failed',returncode=code,BRITS='paused_excluded')
