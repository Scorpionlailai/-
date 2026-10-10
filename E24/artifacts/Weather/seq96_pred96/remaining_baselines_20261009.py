import subprocess,os,json,time
from pathlib import Path
ROOT=Path('/home/xyw/paper/E24');BASE=Path(__file__).resolve().parent
ORDER=['T1','SAITS','S4M','CRIB','CoIFNet','BiTGraph','ChannelTokenFormer']
def job(cmd,log):
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a') as f:
        p=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        print(json.dumps({'pid':p.pid,'command':cmd,'log':str(log)}),flush=True)
        return p.wait()
for tag,pkg in [('sr020_bl8','sr020'),('sr010_bl16','sr010_bl16')]:
    output=BASE/tag/('weather_author_comparison_'+pkg+'_20261008')
    failed=[]
    for name in ORDER:
        env='malr-s4m' if name=='S4M' else 'malr-saits' if name=='SAITS' else 'malr'
        cmd=[f'/home/xyw/miniforge3/envs/{env}/bin/python','-u','-m','integrations.weather_baselines_'+pkg+'.run','preflight','--model',name,'--profiles',str(ROOT/'configs/baselines'/('weather_'+tag+'_author_20261008.json')),'--dataset-manifest',str(BASE/tag/'shared_protocol_v1/dataset_manifest.json'),'--output',str(output/'preflight'/name)]
        if job(cmd,output/'preflight'/name/'preflight.log'):failed.append(name)
    if failed:
        print(json.dumps({'scenario':tag,'failed_preflight':failed}),flush=True);continue
    code=job(['/home/xyw/miniforge3/envs/malr/bin/python','-u','-m','integrations.weather_baselines_'+pkg+'.queue','launch'],output/'launch.log')
    if code:continue
    status=output/'status.json'
    while not status.exists():time.sleep(2)
    while json.loads(status.read_text())['status']=='running':time.sleep(60)
    print(json.dumps({'scenario':tag,'final_status':json.loads(status.read_text())['status']}),flush=True)
