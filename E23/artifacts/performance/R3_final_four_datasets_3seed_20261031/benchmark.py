from pathlib import Path
import sys,os,json,time,statistics,subprocess,argparse
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'implementation'))
from aligned_h_model import install
install()
import torch
from data import load_data
from reuse import load_model_checkpoint
from support import write_json,digest
from variants import library_hash
p=argparse.ArgumentParser();p.add_argument('--dataset');p.add_argument('--scene');p.add_argument('--seed',type=int);a=p.parse_args()
os.environ.update(A64_DATASET=a.dataset,R2_DATA_ROOT=str(ROOT/'data_root'),R2_OUTPUT_ROOT=str(ROOT/'outputs'))
torch.set_num_threads(4);device=torch.device('cuda:0');torch.cuda.set_device(device)
d=load_data(a.dataset,a.scene,ROOT/'outputs/assets');m,info=load_model_checkpoint(d,a.scene,a.seed,'H_GLOBAL',device)
t,seen,starts=next(d.batches('validation',128));t=t.to(device);seen=seen.to(device);values=torch.where(seen,t,0)
before=digest(m);bank=library_hash(m.frozen.repair);times=[]
torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(device)
with torch.no_grad():
 for i in range(30):
  torch.cuda.synchronize();start=time.perf_counter();m(values,seen,starts);torch.cuda.synchronize()
  if i>=5:times.append(time.perf_counter()-start)
assert digest(m)==before and library_hash(m.frozen.repair)==bank
query=subprocess.run(['nvidia-smi','--id='+os.environ['CUDA_VISIBLE_DEVICES'],'--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True)
others=[int(x.strip()) for x in query.stdout.splitlines() if x.strip().isdigit() and int(x.strip())!=os.getpid()]
write_json(ROOT/'outputs/evaluation'/a.dataset/a.scene/f'seed_{a.seed}/H_GLOBAL/resources.json',dict(batch_size=128,p50_batch_ms=statistics.median(times)*1000,warmup=5,timing_samples=25,synchronized=True,exclusive_gpu_verified=query.returncode==0 and not others,competing_gpu_pids=others,gpu=torch.cuda.get_device_name(),physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],torch=torch.__version__,cuda=torch.version.cuda,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,source_model_unchanged=True,library_unchanged=True))
