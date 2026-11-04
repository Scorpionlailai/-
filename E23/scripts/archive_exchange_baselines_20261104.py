"""Publish existing six-model Exchange results; no training or binary assets."""
from pathlib import Path
import json,csv,hashlib,statistics,math,shutil
REPO=Path(__file__).resolve().parents[2]
BASE=Path('/home/xyw/paper/E23/artifacts/Exchange-Rate/seq96_pred96/experiments')
OLD=BASE/'v125_baselines_20261010';NEW=BASE/'csdi_budget_100ep_train1_eval5_20261010'
DEST=REPO/'E23/artifacts/performance/ExchangeRate_baselines_20261104'
DEST.mkdir(parents=True,exist_ok=True)
MODELS=['SAITS','BRITS','CSDI','ImputeFormer','TimesNet','T1'];SCENES=['sr010_bl8','sr015_bl8','sr020_bl8','sr010_bl16'];SEEDS=list(range(2026101001,2026101021))
manifest=[];records=[];permask=[];profile=[];checkpoints=[];draw_count=0

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def copy(p,rel):
 to=DEST/rel;to.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,to);manifest.append(dict(path=str(rel),original_path=str(p),sha256=sha(p),bytes=p.stat().st_size));return str(rel)
def writecsv(path,rows):
 path.parent.mkdir(parents=True,exist_ok=True)
 keys=list(dict.fromkeys(k for row in rows for k in row))
 with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');w.writeheader();w.writerows(rows)

def process_result(path,model,scene,task,selection):
 global draw_count
 d=json.loads(path.read_text());assert d['dataset']=='Exchange-Rate' and d['model']==model
 assert d['model_state_unchanged'] is True
 if task=='forecast':assert d['repair_frozen_verified'] is True
 assert [x['seed'] for x in d['draws']]==SEEDS
 bank=REPO/'data/mask_banks/Exchange-Rate/baselines_20261010'/scene
 m=json.loads((bank/'manifest.json').read_text());bm={x['seed']:x for x in m['masks']}
 assert d['source_csv_sha256']==m['source_sha256']
 assert d['validation_mask_sha256']==bm[2021]['sha256']
 assert sha(bank/bm[2021]['path'])==bm[2021]['sha256']
 for x in d['draws']:
  assert x['mask_sha256']==bm[x['seed']]['sha256']
  assert sha(bank/bm[x['seed']]['path'])==bm[x['seed']]['sha256']
  assert x['windows']==1423
  permask.append(dict(model=model,scene=scene,task=task,selection=selection,training_seed=d['profile_seed'],mask_seed=x['seed'],**{k:v for k,v in x.items() if k!='seed'}));draw_count+=1
 for key,summary in d['summary'].items():
  vals=[x[key] for x in d['draws']]
  assert all(math.isfinite(x) for x in vals)
  assert abs(statistics.mean(vals)-summary['mean'])<1e-10,(model,scene,key)
  assert abs(statistics.stdev(vals)-summary['mask_std_ddof1'])<1e-10
 selected=Path(d['selection']);assert sha(selected)==d['selection_sha256'];sel=json.loads(selected.read_text());assert sel['checkpoint_sha256']==d['checkpoint_sha256']
 tr=d['training_resources']
 row=dict(model=model,scene=scene,task=task,selection=selection,training_seed=d['profile_seed'],stage2_seed=d['stage2_seed'],test_masks=20,test_windows_per_mask=1423,metric_space='train_standardized',mse=d['summary']['mse']['mean'],mae=d['summary']['mae']['mean'],mse_mask_std_ddof1=d['summary']['mse']['mask_std_ddof1'],mae_mask_std_ddof1=d['summary']['mae']['mask_std_ddof1'],stage1_repair_metrics_available=d['stage1_repair_metrics_available'],model_state_unchanged=d['model_state_unchanged'],repair_frozen_verified=d['repair_frozen_verified'],training_workflow_seconds_reported=tr['full_workflow_seconds'],training_stages_seconds_reported=tr.get('training_stages_seconds'),peak_allocated_GiB=tr['peak_allocated_gib'],peak_reserved_GiB=tr['peak_reserved_gib'],gpu=tr.get('gpu'),batch_size_reported=tr.get('batch_size'),hardware_consistent_reported=tr.get('hardware_consistent'),checkpoint_sha256=d['checkpoint_sha256'],dataset_manifest_sha256=d['dataset_manifest_sha256'],original_result=str(path))
 for key in ['repair96_mse','repair96_mae','repair32_mse','repair32_mae']:
  row[key]=d['summary'][key]['mean'];row[key+'_mask_std_ddof1']=d['summary'][key]['mask_std_ddof1']
 row['archived_result']=copy(path,Path('runs')/model/scene/path.parent.name/path.name);records.append(row)
 for p in tr['resource_record_paths']:
  p=Path(p);assert p.is_file(),p;copy(p,Path('resources')/model/scene/p.parent.parent.name/p.name)
 return d

for model in MODELS:
 root=NEW if model=='CSDI' else OLD
 for scene in SCENES:
  run=root/'runs'/model/scene
  selections=[run/'stage1/selection.json',run/'stage2/selection.json',run/'stage2/selection-mse.json']
  for file in selections:
   d=json.loads(file.read_text());copy(file,Path('runs')/model/scene/file.parent.name/file.name)
   if file.parent.name=='stage2':
    assert d['repair_update_policy']=='frozen_v1' and d['repair_initial_sha256']==d['repair_final_sha256'];assert d['stage2_loss']=='forecast_mse_only';assert d['epochs_completed']==100 and d['optimizer_updates']==4000
   if file.name=='selection.json':
    profile.append(dict(model=model,scene=scene,stage=file.parent.name,training_seed=d['training_seed'],profile_seed=d.get('profile_seed'),actual_epochs=d['epochs_completed'],selected_epoch=d.get('best_epoch'),optimizer_updates=d['optimizer_updates'],training=json.dumps(d['training'],sort_keys=True),repair_update_policy=d.get('repair_update_policy'),repair_initial_sha256=d.get('repair_initial_sha256'),repair_final_sha256=d.get('repair_final_sha256'),precision=d['training'].get('precision'),upstream_revision=d.get('upstream_revision'),checkpoint_sha256=d['checkpoint_sha256'],actual_training_windows_per_epoch=d.get('actual_training_windows_per_epoch'),evaluation_samples=d.get('evaluation_samples'),stage2_training_samples=d.get('stage2_training_samples'),reused_stage1_weights=d.get('reused_stage1_weights')))
   checkpoints.append(dict(model=model,scene=scene,selection=str(file.relative_to(root)),checkpoint=d['checkpoint'],checkpoint_sha256=d['checkpoint_sha256'],uploaded_in_this_archive=False))
  process_result(run/'repair_evaluation/twenty-test-draws.json',model,scene,'repair','stage1_rule')
  process_result(run/'forecast_evaluation_mae/twenty-test-draws.json',model,scene,'forecast','mae')
  process_result(run/'forecast_evaluation_mse/twenty-test-draws.json',model,scene,'forecast','mse')
  for file in sorted(run.rglob('*')):
   if file.is_file() and file.name in ['curve.jsonl','training_history.json','history.csv']:
    copy(file,Path('runs')/model/scene/file.relative_to(run))
 for file in sorted((OLD/'runners'/model).rglob('*')):
  if file.is_file() and '__pycache__' not in file.parts and file.suffix in ['.py','.json','.yaml','.yml']:copy(file,Path('runtime/original_runners')/model/file.relative_to(OLD/'runners'/model))
 if model=='CSDI':
  for file in sorted((NEW/'runners/CSDI').rglob('*')):
   if file.is_file() and '__pycache__' not in file.parts and file.suffix in ['.py','.json','.yaml','.yml']:copy(file,Path('runtime/final_csdi_runner')/file.relative_to(NEW/'runners/CSDI'))
for root,label in [(OLD,'original'),(NEW,'csdi_final_budget')]:
 for name in ['profiles.json','accelerated_baseline_overrides.json','budget_change_20261010.json','execution_budget.json','final_audit.json']:
  if (root/name).exists():copy(root/name,Path('provenance')/label/name)
 for file in sorted(root.glob('*.py')):copy(file,Path('runtime')/label/file.name)
writecsv(DEST/'summary/repair_results.csv',[x for x in records if x['task']=='repair'])
writecsv(DEST/'summary/forecast_results_mae.csv',[x for x in records if x['task']=='forecast' and x['selection']=='mae'])
writecsv(DEST/'summary/forecast_results_mse_secondary.csv',[x for x in records if x['task']=='forecast' and x['selection']=='mse'])
writecsv(DEST/'summary/per_mask_metrics.csv',permask)
writecsv(DEST/'summary/training_profiles_and_selection.csv',profile)
(DEST/'checkpoint_manifest.json').write_text(json.dumps(checkpoints,indent=2)+'\n')
# Compact scene-by-model report, primary forecast selection is validation MAE.
lines=['# Exchange-Rate：六个基线的已有结果','','本次只发布已有实验，不重新训练、不重选checkpoint。数据为8变量Exchange-Rate，回溯96、预测96；四场景，同20组测试遮挡。','修复表按原Stage1规则选模；预测主表按验证MAE选择，验证MSE选择作为单独次选表完整保留，不能按测试择优。','各模型沿用其已有配置种子：SAITS/BRITS/CSDI为2023、ImputeFormer为42、TimesNet为2021、T1为502；共享预测头seed2023。每模型单训练seed，mask标准差不是训练seed标准差。','']
for scene in SCENES:
 lines+=['## '+scene,'','| 模型 | 修复MSE | 修复MAE | 预测MSE（验证MAE选模） | 预测MAE |','|---|---:|---:|---:|---:|']
 for model in MODELS:
  rr=next(x for x in records if x['model']==model and x['scene']==scene and x['task']=='repair');ff=next(x for x in records if x['model']==model and x['scene']==scene and x['task']=='forecast' and x['selection']=='mae')
  lines.append(f"| {model} | {rr['mse']:.6f} | {rr['mae']:.6f} | {ff['mse']:.6f} | {ff['mae']:.6f} |")
lines+=['','## 口径与资产边界','','- sr010_bl8、sr015_bl8、sr020_bl8、sr010_bl16分别为起点率/块长0.10/8、0.15/8、0.20/8、0.10/16；起点率不是最终点缺失率。','- 时间划分边界0/5311/6070/7588；测试每mask1423个窗口。指标在训练集标准化尺度，缺失修复仅计人工隐藏坐标，预测计未来目标。','- 预测Stage2为冻结修复器＋共享预测头，100轮、4000更新，纯forecast MSE训练；同时保存验证MAE和MSE选模结果。它与11月Beijing阶段二精恢复不是同一实验。','- BRITS使用已确认TorchScript循环加速、最多200轮、patience20。CSDI使用最终执行记录：sr010_bl8/sr015_bl8复用200轮Stage1，sr020_bl8/sr010_bl16为100轮Stage1，仍50步扩散；预测训练1个样本、修复及预测评价5个样本，采样聚合沿用记录。不能把四场景误记为同一Stage1轮数或旧10样本方案。','- 耗时和显存来自原training_resources及逐阶段资源文件，保留batch/精度/硬件一致性字段；各模型batch和精度不同，hardware_consistent=False不被篡改为True。记录为各进程/阶段实际统计，不冒充同batch或同精度的统一效率排名。','- 旧v125 comparison混合模型表不作为当前R3成绩；本档只发六个基线，R3三seed结果已另行归档。','- 对齐现有[data/mask_banks/Exchange-Rate/baselines_20261010](../../../../data/mask_banks/Exchange-Rate/baselines_20261010/README.md)：测试seed2026101001–2026101020，验证seed2021，逐draw mask SHA与已发布库核对。','- 原记录中github_published=False或awaiting_user_review是旧状态，作为历史provenance保留；用户于2026-11-04明确授权本次上传。','- 包含逐mask指标、选择记录、来源配置、运行脚本和训练资源；不上传数据、预测缓存、checkpoint或优化器状态。checkpoint_manifest仅有原路径/哈希；归档不是独立checkpoint包。','- 部分CSDI选择记录引用原200轮资产，文件哈希与最终评价匹配；旧未完成结果不混入最终主表。']
(DEST/'README.md').write_text('\n'.join(lines)+'\n')
audit=dict(status='passed',dataset='Exchange-Rate',models=MODELS,scenes=SCENES,repair_rows=24,forecast_primary_mae_rows=24,forecast_secondary_mse_rows=24,test_masks_each=20,test_windows_per_mask=1423,per_mask_metric_rows=draw_count,summary_mean_and_sample_std_recomputed=True,test_and_validation_mask_sha_match_published_bank=True,source_csv_sha_match_published_bank=True,evaluation_model_state_unchanged=True,all_forecast_repair_frozen=True,all_forecast_pure_mse_loss=True,all_forecast_100epochs_4000updates=True,single_training_seed_per_model=True,new_training_started=False,new_weights_uploaded=False,publication_requested_date='2026-11-04')
assert draw_count==1440
(DEST/'completion_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
# De-duplicate resource records referenced by all selections.
manifest={x['path']:x for x in manifest}
(DEST/'archive_manifest.json').write_text(json.dumps(dict(source_roots=[str(OLD),str(NEW)],files=list(manifest.values()),excluded=['raw data','prediction caches','weights','optimizer checkpoints','old CSDI incomplete evaluation'],publication='user explicitly requested 2026-11-04'),indent=2)+'\n')
print(json.dumps(audit,indent=2));print('archived_files',len(manifest),'destination',DEST)
