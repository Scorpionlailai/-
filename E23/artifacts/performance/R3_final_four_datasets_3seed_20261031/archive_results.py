from pathlib import Path
import json,csv,statistics,hashlib,shutil,gzip,subprocess
SRC=Path('/home/xyw/paper/E23/artifacts/performance/R3_final_three_datasets_3seed_20261030')
REPO=Path('/home/xyw/r3_release_20261030')
DST=REPO/'E23/artifacts/performance/R3_final_four_datasets_3seed_20261031'
DST.mkdir(parents=True,exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def copy(p,rel=None):
 t=DST/(rel or p.relative_to(SRC));t.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,t)
def write(path,obj):
 p=DST/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
def csvout(path,rows):
 p=DST/path;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
for name in ['plan.json','exchange_extension_plan.json','runtime_source_lock.json','hardware.txt','summarize_four_datasets.py','audit_masks.py','prepare_native.py','benchmark.py','run_formal_queue.py','run_exchange_queue.py','wait_then_exchange.py']:
 copy(SRC/name)
for folder in ['summary','audit','preparation','completed']:
 for p in (SRC/folder).glob('*'):
  if p.is_file():copy(p)
for p in SRC.glob('*queue_gpu*.json'):copy(p)
for p in SRC.glob('exchange_start_gate_gpu*.json'):copy(p)
for p in (SRC/'outputs/evaluation').rglob('*.json'):copy(p)
# Preserve exact stage completion/binding evidence; epoch JSON losslessly compressed below.
for name in ['complete.json','binding.json','model.json']:
 for p in (SRC/'outputs/runs').rglob(name):copy(p)
for ds in ['BeijingAirQuality','PhysioNet2012']:
 p=SRC/f'preflight_{ds}.log'
 if p.exists():copy(p,Path('preflight')/p.name)
for name in ['data.py','baseline_physionet_data.py']:
 copy(SRC/'implementation'/name,Path('runtime_data_overlays')/name)
for p in (SRC/'outputs/assets').glob('*/*/manifest.json'):copy(p,Path('data_manifests')/p.relative_to(SRC/'outputs/assets'))
# Exact curves: unpack path and record to reconstruct any original epoch file.
epoch_paths=sorted((SRC/'outputs/runs').glob('*/*/seed_*/H_GLOBAL/*/epochs/*.json'))
assert len(epoch_paths)==9600
with (DST/'training_epochs.jsonl.gz').open('wb') as raw:
 with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as gz:
  for p in epoch_paths:
   gz.write((json.dumps({'path':str(p.relative_to(SRC)),'sha256':sha(p),'record':json.loads(p.read_text())},separators=(',',':'))+'\n').encode())
resources=[];checkpoints=[]
for p in sorted((SRC/'outputs/evaluation').glob('*/*/seed_*/H_GLOBAL/results.json')):
 r=json.loads(p.read_text());ds=r['dataset'];scene=r['scene'];seed=r['seed'];stage_root=SRC/'outputs/runs'/ds/scene/f'seed_{seed}'/'H_GLOBAL'
 main=json.loads((stage_root/'main/complete.json').read_text());cal=json.loads((stage_root/'calibration/complete.json').read_text())
 perf=json.loads((p.parent/'resources.json').read_text());wall=json.loads((SRC/'completed'/f'{ds}_{scene}_seed{seed}.json').read_text())
 assert r['test_masks']==20 and len(r['per_mask'])==20 and r['export_verified'] and main['epochs']==cal['epochs']==100 and cal['frozen_before']==cal['frozen_after']
 assert main['updates']==100*main['updates_per_epoch'] and cal['updates']==100*cal['updates_per_epoch']
 for stage,file,h in [('main',Path(r['info']['coarse']['file']),r['info']['coarse']['sha256']),('calibration',Path(r['info']['head']),r['info']['head_sha256'])]:
  assert sha(file)==h
  checkpoints.append(dict(dataset=ds,scene=scene,training_seed=seed,stage=stage,original_path=str(file),sha256=h,uploaded=False))
 resources.append(dict(dataset=ds,scene=scene,training_seed=seed,batch_size=128,gpu=perf['gpu'],physical_gpu=perf['physical_gpu'],torch=perf['torch'],cuda=perf['cuda'],main_epochs=100,calibration_epochs=100,main_updates=main['updates'],calibration_updates=cal['updates'],main_loop_seconds=main['seconds'],calibration_loop_seconds=cal['seconds'],end_to_end_seconds=wall['wallclock_seconds'],main_peak_allocated_GiB=main['resources']['max_allocated_gib'],main_peak_reserved_GiB=main['resources']['max_reserved_gib'],calibration_peak_allocated_GiB=cal['resources']['max_allocated_gib'],calibration_peak_reserved_GiB=cal['resources']['max_reserved_gib'],p50_batch_ms=perf['p50_batch_ms'],inference_peak_allocated_GiB=perf['peak_allocated_GiB'],inference_peak_reserved_GiB=perf['peak_reserved_GiB'],timing_exclusive_gpu_verified=perf['exclusive_gpu_verified'],coarse_selected_epoch=r['info']['coarse']['epoch'],calibration_selected_epoch=r['info']['epoch']))
assert len(resources)==48
csvout(Path('summary/R3_resources_per_training_seed.csv'),resources)
rs=[]
for ds in ['ETTh1','BeijingAirQuality','PhysioNet2012','Exchange-Rate']:
 for scene in ['sr010_bl8','sr015_bl8','sr020_bl8','sr010_bl16']:
  rr=[r for r in resources if r['dataset']==ds and r['scene']==scene];row=dict(dataset=ds,scene=scene,n_training_seeds=3)
  for field in ['main_loop_seconds','calibration_loop_seconds','end_to_end_seconds','main_peak_allocated_GiB','main_peak_reserved_GiB','calibration_peak_allocated_GiB','calibration_peak_reserved_GiB','p50_batch_ms']:
   row[field+'_mean']=statistics.mean(r[field] for r in rr);row[field+'_std_ddof1']=statistics.stdev(r[field] for r in rr)
  rs.append(row)
csvout(Path('summary/R3_resources_three_seed_summary.csv'),rs)
write(Path('checkpoint_manifest.json'),checkpoints)
copy(Path('/home/xyw/paper/E23/docs/experiments/2026-10-30_R3_final_three_datasets_formal_run.md'),Path('experiment_design.md'))
write(Path('archive_audit.json'),dict(status='passed',models=48,training_stages=96,epochs_per_stage=100,epoch_records=9600,test_masks_each_seed=20,training_seeds=[2023,2024,2025],checkpoint_hashes_verified=96,coarse_frozen_all=True,export_verified_all=True,all_timing_exclusive_gpu_verified=all(r['timing_exclusive_gpu_verified'] for r in resources),no_new_training=True,model_release_commit='59419e9bbc8f2e8833c8265505d0e84fa2b04de4',exchange_masks_commit='836f3c4bb65867a6a21863cb9aaacfd05cfa769a',raw_data_checkpoints_predictions_uploaded=False))
print(DST)
print('archive files',sum(1 for p in DST.rglob('*') if p.is_file()),'MiB',sum(p.stat().st_size for p in DST.rglob('*') if p.is_file())/2**20)
