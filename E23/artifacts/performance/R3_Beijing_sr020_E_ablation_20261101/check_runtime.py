from pathlib import Path
import sys,os,json,argparse,hashlib
R=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--group',required=True);p.add_argument('--variant',required=True);p.add_argument('--mode',choices=['initial','export'],default='initial');a=p.parse_args()
P=json.loads((R/'plan.json').read_text());G=R/a.group
sys.path.insert(0,str(G/'implementation'))
if a.group=='sources':from ablation_model import install,effective_parameters
else:from experiment_model import install,effective_parameters
install()
import torch,numpy as np
from data import load_data
from variants import make_repair,make_calibrator
from single import FrozenRepair,SingleBranchModel
from support import write_json
os.environ.update(A64_DATASET=P['dataset'],R2_DATA_ROOT=P['data_root'],R2_OUTPUT_ROOT=str(G/'smoke_outputs'))
torch.set_num_threads(4);torch.manual_seed(2023);dev=torch.device('cuda:0');torch.cuda.set_device(dev)
d=load_data(P['dataset'],P['scene'],G/'smoke_outputs/assets')
assert len(d.starts['train'])//128==190
assert list(d.test_seeds)==list(range(2026101301,2026101321))
original=Path(P['reuse_E0'])/'outputs/assets/BeijingAirQuality/sr020_bl8'
for path in original.glob('*'):
 if path.is_file():assert hashlib.sha256(path.read_bytes()).hexdigest()==hashlib.sha256((G/'smoke_outputs/assets/BeijingAirQuality/sr020_bl8'/path.name).read_bytes()).hexdigest(),path.name
# Each dataset loader must reproduce the existing E0 input stream before training.
t,seen,starts=next(d.batches('train',128,epoch=0,seed=2023,max_batches=1))
first=hashlib.sha256(starts.tobytes()+seen.numpy().tobytes()).hexdigest()
if a.mode=='initial':
 repair=make_repair(d,a.variant,2023).to(dev)
 head=make_calibrator(d.n_vars,a.variant,2023).to(dev)
 full=SingleBranchModel(FrozenRepair(repair),head).eval()
else:
 from reuse import load_model_checkpoint
 from train import export_candidate,load_export
 full,info=load_model_checkpoint(d,P['scene'],2023,a.variant,dev,smoke=1)
 file=G/'smoke_outputs/exports'/f'{a.variant}.pt'
 export_candidate(file,full,dict(variant=a.variant,seed=2023,coarse_epoch=info['coarse']['epoch']))
 restored=load_export(file,d,dev)
t,seen,starts=next(d.batches('validation',128));t=t.to(dev);seen=seen.to(dev);values=torch.where(seen,t,0)
with torch.no_grad():
 out=full(values,seen,starts)
 assert torch.isfinite(out).all() and torch.equal(out[seen],values[seen])
 if a.mode=='export':assert torch.equal(out,restored(values,seen,starts))
 if a.mode=='initial':
  changed=torch.where(seen,values,torch.full_like(values,1234))
  assert torch.equal(out,full(changed,seen,starts)),'hidden-value leakage'
  ref=make_repair(d,'E0',2023).to(dev)
  base=SingleBranchModel(FrozenRepair(ref),make_calibrator(d.n_vars,'E0',2023).to(dev)).eval()
  reference=effective_parameters(base)
  if a.variant=='E6':assert abs(effective_parameters(full)/reference-1)<=.01
result=dict(status='passed',mode=a.mode,variant=a.variant,dataset=d.dataset,n_variables=d.n_vars,training_windows=len(d.starts['train']),validation_windows=len(d.starts['validation']),test_windows=len(d.starts['test']),updates_each_stage=19000,mask_files_exact=True,first_batch_fingerprint=first,effective_parameters=effective_parameters(full),registered_parameters=sum(x.numel() for x in full.parameters()),batch128_finite=True,visible_values_exact=True,export_exact=a.mode=='export',validation_only=True)
if a.mode=='initial':result.update(E0_effective_parameters=reference,hidden_value_invariance=True,capacity_hidden=getattr(full.frozen.repair.core,'capacity_hidden',None))
write_json(R/'preflight'/f'{a.variant}_{a.mode}.json',result)
print(json.dumps(result),flush=True)
