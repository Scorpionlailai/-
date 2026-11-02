from pathlib import Path
import argparse,os,sys,json,time,hashlib,fcntl,statistics,subprocess
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'implementation'))
from initialization import FineRecovery,Initialization,neighbor_initial
import torch
from data import load_data
from r2_variants import make_calibrator
from single import apply_head
from support import write_json,save,sha,digest,rng_state,restore_rng
from bootstrap import verify_sources
from integrations.pamap2_baselines.run import evaluate,aggregate,seed_all,resources,synchronize
from integrations.physionet_baselines.losses import hidden_error,mean_or_zero
p=argparse.ArgumentParser();p.add_argument('--variant',choices=['S0','S1'],required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
P=json.loads((ROOT/'plan.json').read_text());OUT=ROOT/('smoke_outputs' if a.smoke else 'outputs')/a.variant;OUT.mkdir(exist_ok=True,parents=True)
os.environ.update(A64_DATASET=P['dataset'],R2_DATA_ROOT=P['data_root'],R2_OUTPUT_ROOT=str(OUT))
lock=(OUT/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
torch.set_num_threads(4);dev=torch.device('cuda:0');torch.cuda.set_device(dev);seed_all(P['seed']);process_start=time.perf_counter()
verify_sources()
for name,h in json.loads((ROOT/'source_lock.json').read_text())['files'].items():assert sha(ROOT/name)==h,name
binding=dict(source_lock=sha(ROOT/'source_lock.json'),plan=sha(ROOT/'plan.json'),variant=a.variant,smoke=a.smoke)
if (OUT/'binding.json').exists():assert json.loads((OUT/'binding.json').read_text())==binding
write_json(OUT/'binding.json',binding)
if not a.smoke:assert json.loads((ROOT/'preflight'/f'{a.variant}.json').read_text())['status']=='passed'
d=load_data(P['dataset'],P['scene'],OUT/'assets');assert len(d.starts['train'])//128==190 and list(d.test_seeds)==P['test_masks']
refassets=Path(P['R3_root'])/'outputs/assets'/P['dataset']/P['scene']
for file in refassets.glob('*'):
 if file.is_file():assert sha(file)==sha(OUT/'assets'/P['dataset']/P['scene']/file.name)
reference=Path(P['R3_root'])/'outputs/runs'/P['dataset']/P['scene']/'seed_2023/H_GLOBAL/calibration'
model=FineRecovery(d,a.variant,P['head_seed']).to(dev);initial=model.initializer;initial_state=digest(initial)
assert sum(x.numel() for x in initial.parameters())==0 and len(initial.state_dict())==0
assert sum(x.numel() for x in model.parameters())==49880
assert all(n.startswith('head.') for n,_ in model.named_parameters())
EPOCHS=2 if a.smoke else 100;MAX=1 if a.smoke else None
if a.smoke:
 model.eval();truth,seen,starts=next(d.batches('validation',128));truth=truth.to(dev);seen=seen.to(dev);values=torch.where(seen,truth,0.)
 ref=make_calibrator(12,'H_GLOBAL',2023).to(dev)
 assert all(torch.equal(v,ref.state_dict()[k]) for k,v in model.head.state_dict().items())
 with torch.no_grad():
  raw=initial(values,seen,starts);pred=model(values,seen,starts)
  delta=float((pred-raw).abs().max());assert delta<=3e-6,delta
  assert torch.equal(pred[seen],values[seen]) and torch.equal(raw[seen],values[seen])
  changed=torch.where(seen,values,torch.full_like(values,12345));assert torch.equal(pred,model(changed,seen,starts))
  # Use original refiner's actual five-class input to verify the interpolation
  # initializer matches its guide arithmetic (including one-sided/no-source).
  zero=Initialization(d,'S0').to(dev).extract(values,seen,starts);capture=[]
  hook=ref.global_dense_refiner.input.register_forward_pre_hook(lambda m,args:capture.append(args[0].detach()))
  apply_head(ref,zero);hook.remove();features=capture[-1];C=d.n_vars
  support=features[...,4*C:5*C];coarse=features[...,:C];interpolated=neighbor_initial(values,seen)
  expected=(interpolated-coarse)*support
  assert torch.allclose(features[...,3*C:4*C],expected,atol=2e-6,rtol=1e-5)
  if a.variant=='S1':
   bb=initial.extract(values,seen,starts);cap=[]
   hh=ref.global_dense_refiner.input.register_forward_pre_hook(lambda m,args:cap.append(args[0].detach()))
   apply_head(ref,bb);hh.remove();suggested_max=float(cap[-1][...,3*C:4*C].abs().max())
  else:suggested_max=None
  # Explicit edge cases: two-sided interpolation, single-sided extension,
  # variable with no legal observations, and an entirely unobserved window.
  syn=torch.zeros(1,96,12,device=dev);mask=torch.zeros_like(syn,dtype=torch.bool)
  syn[0,2,0]=2;syn[0,8,0]=8;mask[0,2,0]=True;mask[0,8,0]=True
  syn[0,20,1]=-3;mask[0,20,1]=True
  interp=neighbor_initial(syn,mask)
  assert torch.equal(interp[0,2:9,0],torch.arange(2,9,device=dev,dtype=syn.dtype))
  assert (interp[0,:2,0]==2).all() and (interp[0,8:,0]==8).all() and (interp[0,:,1]==-3).all() and (interp[0,:,2:]==0).all()
  none=torch.zeros_like(seen);empty=model(torch.full_like(values,12345),none,starts);assert torch.isfinite(empty).all() and empty.abs().max()<3e-6
 write_json(OUT/'interface_checks.json',dict(status='passed',zero_init_output_max_abs_error=delta,head_initialization_exact_R3=True,nominal_trainable_parameters=49880,phase_one_parameters=0,phase_one_network_absent=True,hidden_value_invariant=True,observed_exact=True,interpolation_matches_original_guide=True,edge_cases_passed=True,S1_initial_suggested_delta_max=suggested_max,unchanged_head_structure=True))
 del ref

opt=torch.optim.Adam(model.head.parameters(),lr=.001,weight_decay=0.)
seed_all(P['seed']);start=1;updates=0;seconds=0.;best={'mae':float('inf'),'mse':float('inf')};best_epoch={}
path=OUT/'fine';path.mkdir(exist_ok=True)
if (path/'latest.pt').exists():
 state=torch.load(path/'latest.pt',map_location=dev,weights_only=False);assert state['binding']==binding;model.load_state_dict(state['model'],strict=True);opt.load_state_dict(state['optimizer']);restore_rng(state['rng']);start=state['epoch']+1;updates=state['updates'];seconds=state['seconds'];best=state['best'];best_epoch=state['best_epoch']
if not (path/'complete.json').exists():
 torch.cuda.reset_peak_memory_stats();began=time.perf_counter()
 for epoch in range(start,EPOCHS+1):
  eb=time.perf_counter();count=0;total=0.;fingerprint=hashlib.sha256();model.eval();model.head.train();gradient_names=[]
  for truth,seen,starts in d.batches('train',128,epoch=epoch-1,seed=2023,max_batches=MAX):
   fingerprint.update(starts.tobytes());fingerprint.update(seen.numpy().tobytes());truth=truth.to(dev);seen=seen.to(dev);opt.zero_grad(set_to_none=True)
   batch=initial.extract(torch.where(seen,truth,0.),seen,starts);pred=apply_head(model.head,batch);valid=torch.from_numpy(d.valid_for(starts)).to(dev)
   loss=mean_or_zero(hidden_error(pred,truth,seen,valid).abs())+.25*mean_or_zero(hidden_error(pred[:,-32:],truth[:,-32:],seen[:,-32:],valid[:,-32:]).abs())
   assert torch.isfinite(loss);loss.backward()
   for name,x in model.head.named_parameters():
    if x.grad is not None:
     assert torch.isfinite(x.grad).all(),name
     if count==0 and x.grad.abs().sum()>0:gradient_names.append(name)
   opt.step();count+=1;updates+=1;total+=float(loss.detach())
  assert count==(1 if a.smoke else 190)
  if not a.smoke:
   paired=json.loads((reference/'epochs'/f'{epoch:03}.json').read_text());assert fingerprint.hexdigest()==paired['batch_fingerprint'],epoch
  model.eval();val=evaluate(model,d,'validation',dev,batch_size=128,max_batches=MAX)
  assert digest(initial)==initial_state and len(initial.state_dict())==0
  for metric in ['mae','mse']:
   if val['full96'][metric]<best[metric]:
    best[metric]=val['full96'][metric];best_epoch[metric]=epoch;save(path/f'best_{metric}.pt',dict(head=model.head.state_dict(),epoch=epoch,validation=val,binding=binding))
  elapsed=seconds+time.perf_counter()-began
  row=dict(variant=a.variant,epoch=epoch,updates=updates,updates_this_epoch=count,batch_fingerprint=fingerprint.hexdigest(),loss=total/count,validation=val,best_epoch=best_epoch,epoch_seconds=time.perf_counter()-eb,seconds=elapsed,resources=resources(dev),nonzero_first_batch_gradient_names=gradient_names,initializer_digest=initial_state)
  write_json(path/'epochs'/f'{epoch:03}.json',row)
  save(path/'latest.pt',dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state(),epoch=epoch,updates=updates,seconds=elapsed,best=best,best_epoch=best_epoch,binding=binding))
  write_json(path/'status.json',dict(status='training',**row));print(json.dumps({k:row[k] for k in ['variant','epoch','best_epoch','epoch_seconds']}),flush=True)
 assert updates==EPOCHS*(1 if a.smoke else 190)
 done=dict(status='complete',epochs=EPOCHS,updates=updates,seconds=seconds+time.perf_counter()-began,best_epoch=best_epoch,resources=resources(dev),phase_one_parameters=0,phase_one_training_seconds=0,initializer_digest=initial_state)
 write_json(path/'complete.json',done);write_json(path/'status.json',done)

chosen=torch.load(path/'best_mae.pt',map_location=dev,weights_only=False);model.head.load_state_dict(chosen['head'],strict=True);model.eval()
val=evaluate(model,d,'validation',dev,batch_size=128,max_batches=MAX)
for metric in ['mse','mae']:assert abs(val['full96'][metric]-chosen['validation']['full96'][metric])<1e-7
truth,seen,starts=next(d.batches('validation',128));truth=truth.to(dev);seen=seen.to(dev);values=torch.where(seen,truth,0.)
save(OUT/'model.pt',dict(model=model.state_dict(),variant=a.variant,seed=2023,head_epoch=chosen['epoch'],binding=binding))
restored=FineRecovery(d,a.variant,2023).to(dev);restored.load_state_dict(torch.load(OUT/'model.pt',map_location=dev,weights_only=False)['model'],strict=True);restored.eval()
with torch.no_grad():assert torch.equal(model(values,seen,starts),restored(values,seen,starts))
if a.smoke:
 write_json(ROOT/'preflight'/f'{a.variant}.json',dict(status='passed',validation_only=True,epochs=2,updates_each_epoch=1,export_exact=True,interface=json.loads((OUT/'interface_checks.json').read_text())))
 print('PREFLIGHT_PASSED',a.variant,flush=True);sys.exit(0)

before=digest(model);coarse=[];final=[]
for seed in d.test_seeds:
 coarse.append(evaluate(initial,d,'test',dev,batch_size=128,mask_seed=seed));final.append(evaluate(model,d,'test',dev,batch_size=128,mask_seed=seed));print('MASK_COMPLETE',seed,flush=True)
assert digest(model)==before
write_json(OUT/'results.json',dict(variant=a.variant,dataset=P['dataset'],scene=P['scene'],seed=2023,head_epoch=chosen['epoch'],validation=val,coarse=aggregate(coarse),final=aggregate(final),per_mask=[dict(mask_seed=s,coarse=c,final=f) for s,c,f in zip(d.test_seeds,coarse,final)],test_masks=20,export_exact=True,phase_one_network_absent=True))
latency={}
for name,m in [('initialization_only',initial),('initialization_and_fine',model)]:
 torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();times=[]
 with torch.no_grad():
  for i in range(30):
   synchronize(dev);began=time.perf_counter();m(values,seen,starts);synchronize(dev)
   if i>=5:times.append(1000*(time.perf_counter()-began))
 latency[name]=dict(p50_batch_ms=statistics.median(times),resources=resources(dev))
q=subprocess.run(['nvidia-smi','-i',os.environ['CUDA_VISIBLE_DEVICES'],'--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True)
others=[int(x.strip()) for x in q.stdout.splitlines() if x.strip().isdigit() and int(x.strip())!=os.getpid()]
write_json(OUT/'resources.json',dict(nominal_trainable_parameters=sum(x.numel() for x in model.parameters()),head_parameters=sum(x.numel() for x in model.head.parameters()),phase_one_network_parameters=0,phase_one_training_seconds=0,latency=latency,batch_size=128,warmup=5,timing_samples=25,exclusive_gpu_verified=q.returncode==0 and not others,process_seconds=time.perf_counter()-process_start,scope='imports excluded; data+head training+validation+checkpoint+20mask pre/post test+export+latency included',S1_gate_note=P['S1_guide_gate_boundary'] if a.variant=='S1' else None))
write_json(OUT/'complete.json',dict(status='complete',seed=2023,phase_two_epochs=100,phase_two_updates=19000,paired_epochs_verified=100,test_masks=20,phase_one_training_seconds=0,export_exact=True))
print('EXPERIMENT_COMPLETE',a.variant,flush=True)
