from pathlib import Path
import sys,os,json,time,hashlib,argparse,statistics,fcntl
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'implementation'))
from t1_bridge import T1Repair,CalibratedT1
import torch
from data import load_data
from single import apply_head
from support import write_json,save,sha,digest,rng_state,restore_rng
from bootstrap import verify_sources
from integrations.pamap2_baselines.run import evaluate,aggregate,seed_all,resources,synchronize
from integrations.physionet_baselines.losses import hidden_error,mean_or_zero
P=json.loads((ROOT/'plan.json').read_text())
p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args()
OUT=ROOT/('smoke_outputs' if a.smoke else 'outputs');OUT.mkdir(exist_ok=True)
os.environ.update(A64_DATASET=P['dataset'],R2_DATA_ROOT=P['data_root'],R2_OUTPUT_ROOT=str(OUT))
torch.set_num_threads(4);dev=torch.device('cuda:0');torch.cuda.set_device(dev)
lock=(OUT/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
verify_sources();source=json.loads((ROOT/'source_lock.json').read_text())
for name,h in source['files'].items():assert sha(ROOT/name)==h,name
if not a.smoke:assert json.loads((ROOT/'preflight.json').read_text())['status']=='passed'
d=load_data(P['dataset'],P['scene'],OUT/'assets')
assert len(d.starts['train'])//128==190 and list(d.test_seeds)==P['test_masks']
refassets=Path(P['baseline_R3'])/'outputs/assets'/P['dataset']/P['scene']
for file in refassets.glob('*'):
 if file.is_file():assert sha(file)==sha(OUT/'assets'/P['dataset']/P['scene']/file.name),file.name
reference=Path(P['baseline_R3'])/'outputs/runs'/P['dataset']/P['scene']/'seed_2023/H_GLOBAL'
EPOCHS=2 if a.smoke else 100;MAX=1 if a.smoke else None
binding=dict(source_lock=sha(ROOT/'source_lock.json'),plan=sha(ROOT/'plan.json'),data=d.manifest_sha,smoke=a.smoke)
if (OUT/'binding.json').exists():assert json.loads((OUT/'binding.json').read_text())==binding
write_json(OUT/'binding.json',binding)
seed_all(2023);repair=T1Repair(d).to(dev)
assert sha(P['T1_checkpoint'])==P['T1_checkpoint_sha256']
repair.load_state_dict(torch.load(P['T1_checkpoint'],map_location=dev,weights_only=True),strict=True)
repair.eval()
original_validation=evaluate(repair,d,'validation',dev,batch_size=128,max_batches=MAX)
if not a.smoke:
 archived_rows=[json.loads(x) for x in (ROOT/'original_T1/stage1/curve.jsonl').read_text().splitlines()]
 archived=next(x for x in archived_rows if x['epoch']==90)['validation']
 for metric in ['mse','mae']:
  assert abs(original_validation['full96'][metric]-archived['full96'][metric])<5e-6,('original T1 replay',metric,original_validation,archived)
write_json(OUT/'original_T1_replay.json',dict(status='passed',validation=original_validation,checkpoint_sha256=P['T1_checkpoint_sha256'],full_validation=not a.smoke))
(OUT/'main').mkdir(exist_ok=True)
save(OUT/'main/best_mse.pt',dict(model=repair.state_dict(),epoch=90,validation=original_validation,binding=binding))
write_json(OUT/'main/complete.json',dict(status='reused_original',epochs=100,updates=19000,seconds=2021.0850623520091,selected_epoch=90,checkpoint_sha256=P['T1_checkpoint_sha256']))
if a.smoke:
 truth,seen,starts=next(d.batches('validation',128));truth=truth.to(dev);seen=seen.to(dev);values=torch.where(seen,truth,0.)
 repair.eval();full=CalibratedT1(repair).to(dev).eval()
 with torch.no_grad():
  coarse=repair(values,seen,starts);batch=full.frozen.extract(values,seen,starts);zero=apply_head(full.head,batch)
  delta=float((coarse-zero).abs().max());assert delta<=3e-6,delta
  assert torch.equal(coarse[seen],values[seen]) and torch.equal(zero[seen],values[seen])
  changed=torch.where(seen,values,torch.full_like(values,12345));assert torch.equal(full(values,seen,starts),full(changed,seen,starts))
 assert len(full.head.global_dense_refiner.temporal.layers)==1
 # Calibration must have exactly R3's head state shapes and initial values.
 from r2_variants import make_calibrator
 ref=make_calibrator(12,'H_GLOBAL',2023).to(dev)
 assert all(torch.equal(v,ref.state_dict()[n]) for n,v in full.head.state_dict().items())
 write_json(OUT/'interface_checks.json',dict(status='passed',zero_init_coarse_max_abs_error=delta,hidden_input_invariant=True,visible_exact=True,head_initialization_exact_R3=True,transformer_layers=1,head_parameters=sum(x.numel() for x in full.head.parameters()),T1_config=repair.config))
 del full,ref
 repair.requires_grad_(True)

process_start=time.perf_counter()
for stage in ['calibration']:
 path=OUT/stage;path.mkdir(exist_ok=True)
 if stage=='main':model=repair;optimized=repair;frozen_before=None
 else:
  selected=torch.load(OUT/'main/best_mse.pt',map_location=dev,weights_only=False);repair.load_state_dict(selected['model'],strict=True);repair.eval()
  replay=evaluate(repair,d,'validation',dev,batch_size=128,max_batches=MAX)
  for m in ['mse','mae']:assert abs(replay['full96'][m]-selected['validation']['full96'][m])<=1e-7
  model=CalibratedT1(repair).to(dev);optimized=model.head;frozen_before=digest(repair)
 opt=torch.optim.Adam([x for x in optimized.parameters() if x.requires_grad],lr=.001,weight_decay=0.)
 seed_all(2023);start=1;updates=0;seconds=0.;best={'mse':float('inf'),'mae':float('inf')};best_epoch={}
 if (path/'latest.pt').exists():
  state=torch.load(path/'latest.pt',map_location=dev,weights_only=False);assert state['binding']==binding
  model.load_state_dict(state['model'],strict=True);opt.load_state_dict(state['optimizer']);restore_rng(state['rng']);start=state['epoch']+1;updates=state['updates'];seconds=state['seconds'];best=state['best'];best_epoch=state['best_epoch']
 if (path/'complete.json').exists():continue
 torch.cuda.reset_peak_memory_stats();began=time.perf_counter()
 for epoch in range(start,EPOCHS+1):
  eb=time.perf_counter();count=0;loss_sum=0.;fingerprint=hashlib.sha256()
  if stage=='main':model.train()
  else:model.eval();model.head.train()
  for truth,seen,starts in d.batches('train',128,epoch=epoch-1,seed=2023,max_batches=MAX):
   fingerprint.update(starts.tobytes());fingerprint.update(seen.numpy().tobytes());truth=truth.to(dev);seen=seen.to(dev);opt.zero_grad(set_to_none=True)
   if stage=='main':loss=repair.loss(truth,seen,starts)
   else:
    batch=model.frozen.extract(torch.where(seen,truth,0.),seen,starts);pred=apply_head(model.head,batch);valid=torch.from_numpy(d.valid_for(starts)).to(dev)
    loss=mean_or_zero(hidden_error(pred,truth,seen,valid).abs())+.25*mean_or_zero(hidden_error(pred[:,-32:],truth[:,-32:],seen[:,-32:],valid[:,-32:]).abs())
   assert torch.isfinite(loss);loss.backward()
   for name,x in optimized.named_parameters():
    if x.grad is not None:assert torch.isfinite(x.grad).all(),name
   opt.step();count+=1;updates+=1;loss_sum+=float(loss.detach())
  assert count==(1 if a.smoke else 190)
  if not a.smoke:
   paired=json.loads((reference/stage/'epochs'/f'{epoch:03}.json').read_text());assert paired['batch_fingerprint']==fingerprint.hexdigest(),(stage,epoch)
  model.eval();val=evaluate(model,d,'validation',dev,batch_size=128,max_batches=MAX)
  if stage=='calibration':assert digest(repair)==frozen_before and not any(x.requires_grad or x.grad is not None for x in repair.parameters())
  for metric in (['mse'] if stage=='main' else ['mae','mse']):
   if val['full96'][metric]<best[metric]:
    best[metric]=val['full96'][metric];best_epoch[metric]=epoch
    save(path/f'best_{metric}.pt',dict(model=optimized.state_dict(),epoch=epoch,validation=val,binding=binding))
  elapsed=seconds+time.perf_counter()-began
  row=dict(stage=stage,epoch=epoch,updates=updates,batch_fingerprint=fingerprint.hexdigest(),train_loss=loss_sum/count,validation=val,best_epoch=best_epoch,epoch_seconds=time.perf_counter()-eb,seconds=elapsed,resources=resources(dev),frozen_digest=frozen_before)
  write_json(path/'epochs'/f'{epoch:03}.json',row)
  save(path/'latest.pt',dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state(),epoch=epoch,updates=updates,seconds=elapsed,best=best,best_epoch=best_epoch,binding=binding))
  write_json(path/'status.json',dict(status='training',**row));print(json.dumps({k:row[k] for k in ['stage','epoch','best_epoch','epoch_seconds']},ensure_ascii=False),flush=True)
 assert updates==EPOCHS*(1 if a.smoke else 190)
 done=dict(status='complete',epochs=EPOCHS,updates=updates,seconds=seconds+time.perf_counter()-began,best_epoch=best_epoch,resources=resources(dev),frozen_before=frozen_before,frozen_after=digest(repair) if stage=='calibration' else None)
 write_json(path/'complete.json',done);write_json(path/'status.json',done)

selected=torch.load(OUT/'main/best_mse.pt',map_location=dev,weights_only=False);repair.load_state_dict(selected['model'],strict=True)
full=CalibratedT1(repair).to(dev);cal=torch.load(OUT/'calibration/best_mae.pt',map_location=dev,weights_only=False);full.head.load_state_dict(cal['model'],strict=True);full.eval()
truth,seen,starts=next(d.batches('validation',128));truth=truth.to(dev);seen=seen.to(dev);values=torch.where(seen,truth,0.)
with torch.no_grad():out=full(values,seen,starts)
save(OUT/'model.pt',dict(model=full.state_dict(),seed=2023,T1_epoch=selected['epoch'],head_epoch=cal['epoch'],binding=binding))
restored=CalibratedT1(T1Repair(d),2023).to(dev);restored.load_state_dict(torch.load(OUT/'model.pt',map_location=dev,weights_only=False)['model'],strict=True);restored.eval()
with torch.no_grad():assert torch.equal(out,restored(values,seen,starts)) and torch.equal(out[seen],values[seen])
if a.smoke:
 write_json(ROOT/'preflight.json',dict(status='passed',validation_only=True,calibration_epochs=2,updates_each_epoch=1,original_T1_reused=True,export_exact=True,coarse_frozen=True,interface=json.loads((OUT/'interface_checks.json').read_text())))
 print('PREFLIGHT_PASSED',flush=True);sys.exit(0)

before=digest(full);groups=[];coarse=[]
for seed in d.test_seeds:
 groups.append(evaluate(full,d,'test',dev,batch_size=128,mask_seed=seed));coarse.append(evaluate(repair,d,'test',dev,batch_size=128,mask_seed=seed))
 print('TEST_MASK_COMPLETE',seed,flush=True)
assert digest(full)==before
write_json(OUT/'results.json',dict(dataset=P['dataset'],scene=P['scene'],seed=2023,T1_baseline='original archived checkpoint epoch90, hash verified',old_T1_mse_reference=P['old_T1_mse'],T1_epoch=selected['epoch'],calibration_epoch=cal['epoch'],test_masks=20,coarse=aggregate(coarse),final=aggregate(groups),per_mask=[dict(mask_seed=s,coarse=c,final=f) for s,c,f in zip(d.test_seeds,coarse,groups)],export_exact=True,coarse_frozen=True,binding=binding))
# Whole-batch latency and inference memory, same 5 warmup/25 timing protocol.
latency={}
for name,m in [('T1',repair),('T1_calibrated',full)]:
 m.eval();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();times=[]
 with torch.no_grad():
  for i in range(30):
   synchronize(dev);began=time.perf_counter();m(values,seen,starts);synchronize(dev)
   if i>=5:times.append((time.perf_counter()-began)*1000)
 latency[name]=dict(p50_batch_ms=statistics.median(times),registered_parameters=sum(x.numel() for x in m.parameters()),batch_size=128,resources=resources(dev))
write_json(OUT/'resources.json',dict(latency=latency,calibrator_parameters=sum(x.numel() for x in full.head.parameters()),process_seconds=time.perf_counter()-process_start,timing='same GPU FP32 batch128 5warmup25synchronized, 20mask evaluation included in process time'))
write_json(ROOT/'complete.json',dict(status='complete',single_training_seed=2023,test_masks=20,new_calibration_epochs=100,new_calibration_updates=19000,reused_original_T1_epoch=90,paired_calibration_epochs_verified=100,calibration_frozen=True,export_exact=True))
print('EXPERIMENT_COMPLETE',flush=True)
