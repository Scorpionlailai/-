"""Archive concluded Beijing explanation experiments without data/weights/caches."""
from pathlib import Path
import shutil,json,hashlib,gzip
LOCAL=Path('/home/xyw/paper/E23/artifacts/performance')
REPO=Path(__file__).resolve().parents[2]
DEST=REPO/'E23/artifacts/performance'
NAMES=['R3_Beijing_sr020_E_ablation_20261101','Beijing_sr020_T1_R3_calibration_20261101','Beijing_sr020_S0_S1_fine_recovery_20261101','Beijing_sr020_fine_paths_20261102']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name in NAMES:
 source=LOCAL/name;target=DEST/name;target.mkdir(parents=True,exist_ok=True);copied=[];curves=[]
 for file in sorted(source.rglob('*')):
  if not file.is_file() or '__pycache__' in file.parts:continue
  rel=file.relative_to(source);parts=rel.parts;ok=False
  if 'implementation' in parts:
   ok=file.suffix in ['.py','.md','.json','.txt','.yaml','.yml','.csv']
  elif len(parts)==1:
   ok=file.suffix=='.py' or file.name in ['plan.json','source_lock.json','final_report.md','report_final.md','complete.json','queue_gpu0.json','queue_gpu1.json','T1_settings.json']
  elif parts[0] in ['sources','context','point'] and len(parts)==2 and file.suffix=='.py':ok=True
  elif parts[0] in ['preflight','audit']:ok=file.suffix=='.json'
  elif parts[0]=='summary':ok=('final' in file.stem or file.name in ['completion_audit.json','existing_reference_stage_costs.csv','S1_output_range_diagnostic.json','T1_E6_R3_comparison.csv']) and file.suffix in ['.json','.csv']
  elif parts[0]=='original_T1':ok=file.suffix in ['.md','.json','.jsonl','.py']
  elif 'outputs' in parts and not any(x.startswith('smoke') or x.startswith('failed') for x in parts):
   ok=file.name in ['results.json','resources.json','binding.json','complete.json','model.json','status.json','manifest.json','coarse_source.json','coarse_validation.json','original_T1_replay.json','interface_checks.json'] and 'epochs' not in parts
  if ok:
   to=target/rel;to.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(file,to);copied.append(dict(path=str(rel),sha256=sha(file),bytes=file.stat().st_size))
 for folder in sorted(source.rglob('epochs')):
  rel=folder.relative_to(source)
  if 'outputs' not in rel.parts or any(x.startswith('smoke') or x.startswith('failed') for x in rel.parts):continue
  files=sorted(folder.glob('*.json'))
  if not files:continue
  data=b''.join(json.dumps(json.loads(f.read_text()),separators=(',',':')).encode()+b'\n' for f in files)
  path=target/rel.parent/'training_epochs.jsonl.gz';path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(gzip.compress(data,mtime=0))
  curves.append(dict(path=str(path.relative_to(target)),epochs=len(files),sha256=sha(path),uncompressed_source_sha256={f.name:sha(f) for f in files}))
 weights=[]
 for file in sorted(source.rglob('*.pt')):
  rel=file.relative_to(source)
  if any(x.startswith('smoke') or x.startswith('failed') for x in rel.parts) or file.name=='latest.pt':continue
  weights.append(dict(original_path=str(file),relative_to_local_run=str(rel),sha256=sha(file),bytes=file.stat().st_size,uploaded_in_this_archive=False))
 (target/'local_weight_manifest.json').write_text(json.dumps(dict(note='Metadata only. New model weights are not uploaded. Original T1 weight was already published in commit7e6b3e8f, E24/artifacts/beijing_reproduction_20261013/BeijingAirQuality/sr020_bl8/T1/seed_2023/stage1/selected.pt. Other required coarse/head weights remain on original server.',weights=weights),indent=2)+'\n')
 (target/'archive_manifest.json').write_text(json.dumps(dict(original_root=str(source),purpose='Evidence/source archive, not a standalone runnable checkpoint package',files=copied,compressed_training_curves=curves,excluded=['raw data','mask NPZ banks','prediction caches','new checkpoints and optimizer states','smoke checkpoints','runtime logs'],source_lock_note='Original source_lock.json kept verbatim. Some referenced weights are intentionally absent; use manifest hashes to recover them. Archived code retains original absolute plan paths; do not silently bypass source/data bindings when migrating.'),indent=2)+'\n')
 (target/'README.md').write_text(f'# {name}\n\n[最终报告](final_report.md) · [固定实验设计与配置](plan.json) · [归档清单](archive_manifest.json) · [本机权重指纹](local_weight_manifest.json)\n\n这是Beijing sr020_bl8、训练seed2023、20组测试遮挡的解释实验记录。粗恢复器冻结，不是下游预测Stage2。包含实际源代码、逐mask指标、验证/训练元数据、压缩逐轮曲线、资源与核验记录。\n\n本目录不是独立可执行的模型包：未上传新权重、原始数据、NPZ遮挡库、预测缓存、优化器状态。源码和source_lock保留原样，plan中的路径来自原服务器；迁移需要先恢复数据/遮挡和指定粗权重并验证哈希，不能绕过绑定。原T1权重可从本分支已发布的commit7e6b3e8f获取。\n\n最终结论见[跨实验总览](../../../docs/experiments/2026-11-02_Beijing_explanatory_experiments_summary.md)，保留小增益和负结果，不把20组mask当作20个训练seed。\n')
 print(name,'source/record files',len(copied),'training curves',len(curves),'weights metadata only',len(weights),flush=True)
DOCS=REPO/'E23/docs/experiments';DOCS.mkdir(parents=True,exist_ok=True)
for file in sorted(Path('/home/xyw/paper/E23/docs/experiments').glob('2026-11-0[12]*.md')):shutil.copy2(file,DOCS/file.name)
