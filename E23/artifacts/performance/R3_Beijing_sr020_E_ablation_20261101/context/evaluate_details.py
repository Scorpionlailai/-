from pathlib import Path
import os,sys,argparse,time,statistics,subprocess,torch
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'implementation'))
from experiment_model import install,effective_parameters
install()
from data import load_data
from reuse import load_model_checkpoint
from support import write_json,digest
from variants import library_hash
from integrations.pamap2_baselines.run import evaluate
from malr.data.patching import patchify_with_state
p=argparse.ArgumentParser();p.add_argument('--variant',required=True);p.add_argument('--scene',required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args()
os.environ.update(A64_DATASET='BeijingAirQuality',R2_DATA_ROOT='/home/xyw/paper/E23/artifacts/performance/R3_final_three_datasets_3seed_20261030/data_root',R2_OUTPUT_ROOT=str(ROOT/'outputs'))
torch.set_num_threads(4);device=torch.device('cuda:0');torch.cuda.set_device(device)
d=load_data('BeijingAirQuality',a.scene,ROOT/'outputs/assets');m,info=load_model_checkpoint(d,a.scene,a.seed,a.variant,device)
before=digest(m);bank=library_hash(m.frozen.repair)
coverage={k:dict(selected_candidates=0,available_targets=0,damaged_targets=0) for k in ['historical','variable','temporal']}
read_stats=dict(valid_read_tokens=0,damaged_targets=0,nominal_slots=set())
def capture(module,args,kwargs,result):
 pp=patchify_with_state(args[0],args[1],16,8,validity=kwargs.get('valid_mask'));damaged=pp.rho.lt(1)&pp.validity.any(-1)
 read_stats['damaged_targets']+=int(damaged.sum())
 att=result.point_candidate_attention
 if a.variant=='E6':
  assert not result.evidence and att is None
 else:
  assert att['candidate_counts']==(1,1,1)
  read_stats['nominal_slots'].add(sum(att['candidate_counts']))
  read_stats['valid_read_tokens']+=int((att['valid']&damaged[...,None]).sum())
  for name,e in result.evidence.items():
   valid=e.candidates.valid&e.available[...,None]&damaged[...,None];c=coverage[name]
   c['selected_candidates']+=int(valid.sum());c['available_targets']+=int((valid.any(-1)&damaged).sum());c['damaged_targets']+=int(damaged.sum())
hook=m.frozen.repair.core.register_forward_hook(capture,with_kwargs=True)
try:evaluate(m,d,'validation',device,batch_size=128)
finally:hook.remove()
for c in coverage.values():
 c['mean_selected_candidates_per_damaged_patch']=c['selected_candidates']/max(c['damaged_targets'],1)
 c['source_coverage']=c['available_targets']/max(c['damaged_targets'],1)
read_stats['nominal_slots']=sorted(read_stats['nominal_slots']);read_stats['mean_valid_read_tokens_per_damaged_patch']=read_stats['valid_read_tokens']/max(read_stats['damaged_targets'],1)
t,seen,starts=next(d.batches('validation',128));t=t.to(device);seen=seen.to(device);values=torch.where(seen,t,0)
torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();times=[]
with torch.no_grad():
 for i in range(30):
  torch.cuda.synchronize();start=time.perf_counter();m(values,seen,starts);torch.cuda.synchronize()
  if i>=5:times.append(time.perf_counter()-start)
assert digest(m)==before and library_hash(m.frozen.repair)==bank
q=subprocess.run(['nvidia-smi','-i',os.environ['CUDA_VISIBLE_DEVICES'],'--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True)
others=[int(x.strip()) for x in q.stdout.splitlines() if x.strip().isdigit() and int(x.strip())!=os.getpid()]
out=ROOT/'outputs/evaluation/BeijingAirQuality'/a.scene/f'seed_{a.seed}'/a.variant
write_json(out/'resources.json',dict(registered_parameters=sum(p.numel() for p in m.parameters()),effective_parameters=effective_parameters(m),module_parameters={n:sum(p.numel() for p in mod.parameters()) for n,mod in m.frozen.repair.core.named_children()},capacity_hidden=getattr(m.frozen.repair.core,'capacity_hidden',None),has_history=hasattr(m.frozen.repair,'memory'),coverage_validation=None if a.variant=='E6' else coverage,readout_validation=read_stats,p50_batch_ms=statistics.median(times)*1000,batch_size=128,warmup=5,timing_samples=25,synchronized=True,exclusive_gpu_verified=q.returncode==0 and not others,competing_gpu_pids=others,gpu=torch.cuda.get_device_name(),physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],torch=torch.__version__,cuda=torch.version.cuda,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,source_model_unchanged=True,library_unchanged=True))
