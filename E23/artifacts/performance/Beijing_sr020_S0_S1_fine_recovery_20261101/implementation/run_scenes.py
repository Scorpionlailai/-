"""Sequential four-scene queue on one assigned GPU; one dataset per server."""
import argparse,subprocess,sys,time,json,datetime
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',required=True,choices=['ETTh1','Weather','Exchange-Rate','BeijingAirQuality','PAMAP2','PhysioNet2012'])
    p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--seed',type=int,default=2023)
    p.add_argument('--scenes',nargs='+',default=['sr010_bl8','sr015_bl8','sr020_bl8','sr010_bl16'])
    p.add_argument('--smoke-batches',type=int)
    a=p.parse_args();root=Path(__file__).resolve().parent
    for scene in a.scenes:
        start=time.perf_counter();started=datetime.datetime.now(datetime.timezone.utc).isoformat()
        common=['--dataset',a.dataset,'--scene',scene,'--seed',str(a.seed),'--device',a.device,'--data-root',str(a.data_root.resolve()),'--output-root',str(a.output_root.resolve()),'--variant','R3']
        if a.smoke_batches is not None:common+=['--smoke-batches',str(a.smoke_batches)]
        subprocess.run([sys.executable,str(root/'train.py'),*common,'--stage','both'],check=True,cwd=root)
        subprocess.run([sys.executable,str(root/'evaluate.py'),*common],check=True,cwd=root)
        summary=dict(dataset=a.dataset,scene=scene,seed=a.seed,device=a.device,started_utc=started,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),end_to_end_seconds=time.perf_counter()-start,scope='main+calibration+validation+checkpoint-save+20mask-evaluation+export; includes subprocess data preparation',smoke_only=a.smoke_batches is not None)
        target=a.output_root/('smoke_wallclock' if a.smoke_batches else 'wallclock')/a.dataset/scene/f'seed_{a.seed}'
        target.mkdir(parents=True,exist_ok=True)
        with (target/'attempts.jsonl').open('a') as out:out.write(json.dumps(summary)+'\n')
if __name__=='__main__':main()
