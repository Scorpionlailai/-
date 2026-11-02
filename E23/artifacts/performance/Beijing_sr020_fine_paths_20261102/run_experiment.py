from pathlib import Path
import sys,os,json,time,hashlib,statistics,argparse,fcntl,subprocess
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'implementation'))
from path_model import load_prefix,FineModel,head
import torch
from data import load_data
from single import apply_head
from r2_variants import make_calibrator
from variants import library_hash
from support import write_json,save,sha,digest,rng_state,restore_rng
from bootstrap import verify_sources
from integrations.pamap2_baselines.run import evaluate,aggregate,seed_all,resources,synchronize
from integrations.physionet_baselines.losses import hidden_error,mean_or_zero
p=argparse.ArgumentParser();p.add_argument('--job',choices=['R3_P1','R3_P2','R3_P3','T1_P2'],required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
P=json.loads((R/'plan.json').read_text());backbone,mode=a.job.split('_');OUT=R/('smoke_outputs' if a.smoke else 'outputs')/a.job;OUT.mkdir(exist_ok=True,parents=True)
os.environ.update(A64_DATASET=P['dataset'],R2_DATA_ROOT=P['data_root'],R2_OUTPUT_ROOT=str(OUT));torch.set_num_threads(4);dev=torch.device('cuda:0');torch.cuda.set_device(dev)
lock=(OUT/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
verify_sources()
for name,h in json.loads((R/'source_lock.json').read_text())['files'].items():assert sha(R/name)==h,name
assert sha(P['R3_main_checkpoint'])==P['R3_main_sha256'] and sha(P['T1_checkpoint'])==P['T1_sha256']
if not a.smoke:assert json.loads((R/'preflight'/f'{a.job}.json').read_text())['status']=='passed'
binding=dict(source_lock=sha(R/'source_lock.json'),plan=sha(R/'plan.json'),job=a.job,smoke=a.smoke)
if (OUT/'binding.json').exists():assert json.loads((OUT/'binding.json').read_text())==binding
write_json(OUT/'binding.json',binding);process_start=time.perf_counter();seed_all(2023)
d=load_data(P['dataset'],P['scene'],OUT/'assets');assert len(d.starts['train'])//128==190 and list(d.test_seeds)==P['test_masks']
for file in (Path(P['R3_root'])/'outputs/assets'/P['dataset']/P['scene']).glob('*'):
 if file.is_file():assert sha(file)==sha(OUT/'assets'/P['dataset']/P['scene']/file.name)
frozen,meta=load_prefix(d,backbone,dev,P);model=FineModel(frozen,mode).to(dev);before=digest(frozen);bank=library_hash(frozen.repair) if backbone=='R3' else None
EPOCHS=2 if a.smoke else 100;MAX=1 if a.smoke else None
reference=Path(P['R3_root'])/'outputs/runs/BeijingAirQuality/sr020_bl8/seed_2023/H_GLOBAL/calibration'
write_json(OUT/'coarse_source.json',meta)
# Verify actual parent recovery on a fixed visible-only interface, never fine-tune it.
coarse_validation=evaluate(frozen.repair,d,'validation',dev,batch_size=128,max_batches=MAX)
if not a.smoke:
 if backbone=='R3':expected=meta['validation']
 else:expected=json.loads((Path(P['T1_root'])/'outputs/original_T1_replay.json').read_text())['validation']
 for metric in ['mse','mae']:assert abs(coarse_validation['full96'][metric]-expected['full96'][metric])<5e-6
write_json(OUT/'coarse_validation.json',coarse_validation)

if a.smoke:
 model.eval();truth,seen,starts=next(d.batches('validation',128));truth=truth.to(dev);seen=seen.to(dev);values=torch.where(seen,truth,0.)
 batch=frozen.extract(values,seen,starts)
 ref=make_calibrator(12,'H_GLOBAL',2023).to(dev)
 for k,v in model.head.state_dict().items():assert torch.equal(v,ref.state_dict()[k]),k
 with torch.no_grad():
  init=apply_head(model.head,batch);assert torch.allclose(init,batch.explicit.standardized_sequence,atol=3e-6,rtol=1e-6)
  assert torch.equal(init[seen],values[seen]) and torch.isfinite(init).all()
  changed=torch.where(seen,values,torch.full_like(values,12345));assert torch.equal(model(values,seen,starts),model(changed,seen,starts))
 # Test path arithmetic using the selected trained P0 head, not zero output heads.
 if backbone=='R3':
  file=reference/'best_mae.pt';state=torch.load(file,map_location=dev,weights_only=False);trained=state['head'];expected_reuse=json.loads((Path(P['R3_root'])/'outputs/evaluation/BeijingAirQuality/sr020_bl8/seed_2023/H_GLOBAL/results.json').read_text())
 else:
  file=Path(P['T1_root'])/'outputs/calibration/best_mae.pt';state=torch.load(file,map_location=dev,weights_only=False);trained=state['model'];expected_reuse=json.loads((Path(P['T1_root'])/'outputs/results.json').read_text())
 original=make_calibrator(12,'H_GLOBAL',2023).to(dev);original.load_state_dict(trained,strict=True);original.eval()
 variants={v:head(v).to(dev).eval() for v in ['P0','P1','P2','P3']}
 for v,h in variants.items():h.load_state_dict({k:x for k,x in trained.items() if k in h.state_dict()},strict=True)
 captured={}
 handles=[]
 for v,h in variants.items():
  def cap(mod,args,v=v):captured[v]=args[0].detach().clone()
  handles.append(h.global_dense_refiner.input.register_forward_pre_hook(cap))
 with torch.no_grad():
  outputs={v:apply_head(h,batch) for v,h in variants.items()};old=apply_head(original,batch)
  assert torch.equal(outputs['P0'],old),'P0 numeric path differs from original'
  assert torch.equal(captured['P0'],captured['P1']) and torch.equal(captured['P0'],captured['P3'])
  assert torch.equal(captured['P2'][...,:36],captured['P0'][...,:36]) and torch.count_nonzero(captured['P2'][...,36:])==0
  coarse=batch.explicit.standardized_sequence
  diff=((outputs['P0']-coarse)-((outputs['P1']-coarse)+(outputs['P3']-coarse))).abs().max()
  assert diff<1e-5,float(diff)
 for handle in handles:handle.remove()
 # Reuse audit checks the trained P0 against its actual saved validation values.
 baseline=FineModel(frozen,'P0').to(dev);baseline.head.load_state_dict(trained,strict=True);baseline.eval()
 fullval=evaluate(baseline,d,'validation',dev,batch_size=128)
 for metric in ['mse','mae']:assert abs(fullval['full96'][metric]-state['validation']['full96'][metric])<5e-6
 write_json(OUT/'interface_checks.json',dict(status='passed',job=a.job,head_init_shared_exact=True,P0_output_exact_original=True,P0_input_exact_P1_P3=True,P2_first_three_groups_exact=True,P2_neighbor_groups_zero=True,path_decomposition_max_error=float(diff),hidden_value_invariance=True,visible_exact=True,P0_full_validation_reuse_passed=True,P0_head_sha256=sha(file),P0_validation=fullval,head_registered_parameters=sum(x.numel() for x in model.head.parameters()),constant_input_weights=1536 if mode=='P2' else 0,removed_output_head_parameters=780,coarse_hash=meta['sha256']))
 del variants,original,baseline,ref

opt=torch.optim.Adam(model.head.parameters(),lr=.001,weight_decay=0.);seed_all(2023);start=1;updates=0;seconds=0.;best={'mse':float('inf'),'mae':float('inf')};best_epoch={};path=OUT/'fine';path.mkdir(exist_ok=True)
if (path/'latest.pt').exists():
 s=torch.load(path/'latest.pt',map_location=dev,weights_only=False);assert s['binding']==binding;model.head.load_state_dict(s['head'],strict=True);opt.load_state_dict(s['optimizer']);restore_rng(s['rng']);start=s['epoch']+1;updates=s['updates'];seconds=s['seconds'];best=s['best'];best_epoch=s['best_epoch']
if not (path/'complete.json').exists():
 torch.cuda.reset_peak_memory_stats();began=time.perf_counter()
 for epoch in range(start,EPOCHS+1):
  eb=time.perf_counter();count=0;total=0.;fingerprint=hashlib.sha256();model.eval();model.head.train();gradient_names=[]
  for truth,seen,starts in d.batches('train',128,epoch=epoch-1,seed=2023,max_batches=MAX):
   fingerprint.update(starts.tobytes());fingerprint.update(seen.numpy().tobytes());truth=truth.to(dev);seen=seen.to(dev);opt.zero_grad(set_to_none=True)
   batch=frozen.extract(torch.where(seen,truth,0.),seen,starts);pred=apply_head(model.head,batch);valid=torch.from_numpy(d.valid_for(starts)).to(dev)
   loss=mean_or_zero(hidden_error(pred,truth,seen,valid).abs())+.25*mean_or_zero(hidden_error(pred[:,-32:],truth[:,-32:],seen[:,-32:],valid[:,-32:]).abs())
   assert torch.isfinite(loss);loss.backward()
   for n,x in model.head.named_parameters():
    if x.grad is not None:
     assert torch.isfinite(x.grad).all(),n
     if count==0 and bool(x.grad.abs().sum()>0):gradient_names.append(n)
   if mode=='P2':assert torch.count_nonzero(model.head.global_dense_refiner.input.weight.grad[:,36:])==0
   opt.step();count+=1;updates+=1;total+=float(loss.detach())
  assert count==(1 if a.smoke else 190)
  if not a.smoke:
   paired=json.loads((reference/'epochs'/f'{epoch:03}.json').read_text());assert fingerprint.hexdigest()==paired['batch_fingerprint']
  model.eval();val=evaluate(model,d,'validation',dev,batch_size=128,max_batches=MAX)
  assert digest(frozen)==before and not any(x.grad is not None or x.requires_grad for x in frozen.parameters())
  if backbone=='R3':assert library_hash(frozen.repair)==bank
  for metric in ['mae','mse']:
   if val['full96'][metric]<best[metric]:
    best[metric]=val['full96'][metric];best_epoch[metric]=epoch;save(path/f'best_{metric}.pt',dict(head=model.head.state_dict(),epoch=epoch,validation=val,binding=binding))
  elapsed=seconds+time.perf_counter()-began;row=dict(job=a.job,epoch=epoch,updates=updates,updates_this_epoch=count,batch_fingerprint=fingerprint.hexdigest(),loss=total/count,validation=val,best_epoch=best_epoch,epoch_seconds=time.perf_counter()-eb,seconds=elapsed,resources=resources(dev),frozen_digest=before,bank=bank,nonzero_first_batch_gradient_names=gradient_names)
  write_json(path/'epochs'/f'{epoch:03}.json',row);save(path/'latest.pt',dict(head=model.head.state_dict(),optimizer=opt.state_dict(),rng=rng_state(),epoch=epoch,updates=updates,seconds=elapsed,best=best,best_epoch=best_epoch,binding=binding))
  write_json(path/'status.json',dict(status='training',**row));print(json.dumps({k:row[k] for k in ['job','epoch','best_epoch','epoch_seconds']}),flush=True)
 assert updates==EPOCHS*(1 if a.smoke else 190)
 done=dict(status='complete',epochs=EPOCHS,updates=updates,seconds=seconds+time.perf_counter()-began,best_epoch=best_epoch,resources=resources(dev),frozen_before=before,frozen_after=digest(frozen),library_unchanged=True)
 write_json(path/'complete.json',done);write_json(path/'status.json',done)

chosen=torch.load(path/'best_mae.pt',map_location=dev,weights_only=False);model.head.load_state_dict(chosen['head'],strict=True);model.eval()
val=evaluate(model,d,'validation',dev,batch_size=128,max_batches=MAX)
for metric in ['mae','mse']:assert abs(val['full96'][metric]-chosen['validation']['full96'][metric])<1e-7
truth,seen,starts=next(d.batches('validation',128));truth=truth.to(dev);seen=seen.to(dev);values=torch.where(seen,truth,0.)
file=OUT/'head_export.pt';save(file,dict(head=model.head.state_dict(),backbone=backbone,mode=mode,seed=2023,epoch=chosen['epoch'],coarse=meta,binding=binding))
# Export is intentionally a fine-head artifact bound to the frozen coarse SHA.
export=torch.load(file,map_location=dev,weights_only=False);restored=head(mode).to(dev).eval();restored.load_state_dict(export['head'],strict=True)
with torch.no_grad():
 batch=frozen.extract(values,seen,starts);aa=apply_head(model.head,batch);bb=apply_head(restored,batch);assert torch.equal(aa,bb) and torch.equal(aa[seen],values[seen])
if a.smoke:
 write_json(R/'preflight'/f'{a.job}.json',dict(status='passed',validation_only=True,epochs=2,updates_each_epoch=1,frozen_exact=True,export_exact=True,interface=json.loads((OUT/'interface_checks.json').read_text())))
 print('PREFLIGHT_PASSED',a.job,flush=True);sys.exit(0)

coarse=[];final=[]
for seed in d.test_seeds:
 coarse.append(evaluate(frozen.repair,d,'test',dev,batch_size=128,mask_seed=seed));final.append(evaluate(model,d,'test',dev,batch_size=128,mask_seed=seed));print('MASK_COMPLETE',a.job,seed,flush=True)
assert digest(frozen)==before
write_json(OUT/'results.json',dict(job=a.job,backbone=backbone,mode=mode,seed=2023,head_epoch=chosen['epoch'],validation=val,coarse=aggregate(coarse),final=aggregate(final),per_mask=[dict(mask_seed=s,coarse=c,final=f) for s,c,f in zip(d.test_seeds,coarse,final)],test_masks=20,export_exact=True,coarse_frozen=True,coarse_source=meta))
torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();times=[]
with torch.no_grad():
 for i in range(30):
  synchronize(dev);began=time.perf_counter();model(values,seen,starts);synchronize(dev)
  if i>=5:times.append(1000*(time.perf_counter()-began))
q=subprocess.run(['nvidia-smi','-i',os.environ['CUDA_VISIBLE_DEVICES'],'--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True);others=[int(x.strip()) for x in q.stdout.splitlines() if x.strip().isdigit() and int(x.strip())!=os.getpid()]
write_json(OUT/'resources.json',dict(registered_parameters=sum(x.numel() for x in model.parameters()),head_registered_parameters=sum(x.numel() for x in model.head.parameters()),P2_constant_input_weights=1536 if mode=='P2' else 0,p50_batch128_ms=statistics.median(times),resources=resources(dev),warmup=5,timing_samples=25,exclusive_gpu_verified=q.returncode==0 and not others,process_seconds=time.perf_counter()-process_start,scope='data+coarse replay+fine100+validation+save+20mask pre/post test+export+latency; no coarse training'))
write_json(OUT/'complete.json',dict(status='complete',job=a.job,epochs=100,updates=19000,test_masks=20,paired_training_epochs_verified=100,coarse_frozen=True,export_exact=True))
print('EXPERIMENT_COMPLETE',a.job,flush=True)
