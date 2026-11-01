"""Build a metrics-only R3 archive, independently checking raw completed stages."""
import argparse,csv,hashlib,json,shutil
from pathlib import Path
import numpy as np
DATASETS=['Weather','PAMAP2'];SCENES=['sr010_bl8','sr015_bl8','sr020_bl8','sr010_bl16'];SEEDS=[2023,2024,2025]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def table(p,rows):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def main():
 a=argparse.ArgumentParser();a.add_argument('--source-run',type=Path,required=True);a.add_argument('--output',type=Path,required=True);args=a.parse_args();src=args.source_run;out=args.output;out.mkdir(parents=True,exist_ok=False)
 snapshot=src/'snapshot/E24/versions/r3_20261030'
 for name in ['release_lock.json','vendor_lock.json']:
  for rel,digest in json.loads((snapshot/name).read_text())['files'].items():assert sha(snapshot/rel)==digest,rel
 alignment=json.loads((src/'baseline_alignment.json').read_text())
 for item in alignment['rows']:
  for mask in item['mask_files']:
   f=src/'outputs/assets'/item['dataset']/item['scene']/f"{mask['split']}_{mask['seed']}.npz";assert sha(f)==mask['file_sha256']
   with np.load(f,allow_pickle=False) as a:
    key='observed' if 'observed' in a.files else 'observed_mask';assert hashlib.sha256(a[key].tobytes()).hexdigest()==mask['array_sha256']
 per_seed=[];per_mask=[];stage_rows=[];epochs=[];checkpoints=[];evidence=[]
 for dataset in DATASETS:
  for scene in SCENES:
   reference=None
   for seed in SEEDS:
    root=src/'outputs/runs'/dataset/scene/f'seed_{seed}'/'H_GLOBAL'
    evaluation=src/'outputs/evaluation'/dataset/scene/f'seed_{seed}'/'H_GLOBAL/results.json';r=json.loads(evaluation.read_text())
    assert r['dataset']==dataset and r['scene']==scene and r['seed']==seed and r['variant']=='H_GLOBAL' and not r['smoke_only'] and r['test_masks']==20 and r['export_verified']
    assert len(r['per_mask'])==20
    assert sha(evaluation.parent/'model.pt')==r['export_sha256']
    assert [x['mask_seed'] for x in r['per_mask']]==list(range(2026100801,2026100821) if dataset=='Weather' else range(2026101401,2026101421))
    for metric in ['mse','mae']:
     for label,sub in [('test','final'),('precal_test','coarse')]:
      assert abs(np.mean([x[sub]['full96'][metric] for x in r['per_mask']])-r[label]['full96'][metric+'_mean'])<1e-14
    evidence.append({'relative_source_file':str(evaluation.relative_to(src)),'sha256':sha(evaluation)})
    main_done=None;cal_done=None
    for stage in ['main','calibration']:
     p=root/stage/'complete.json';done=json.loads(p.read_text());assert done['status']=='complete' and done['epochs']==100
     expected=286 if dataset=='Weather' else 85
     assert done['updates_per_epoch']==expected and done['updates']==100*expected
     if stage=='calibration':assert done['frozen_before']==done['frozen_after'] and done['frozen_before'] is not None
     curve=root/stage/'curve.jsonl';records=[json.loads(x) for x in curve.read_text().splitlines()];assert [x['epoch'] for x in records]==list(range(1,101))
     assert all(x['updates_this_epoch']==expected and x['updates_total']==expected*x['epoch'] for x in records)
     fingerprints=[x['batch_fingerprint'] for x in records]
     if reference is None:reference=fingerprints
     else:assert reference==fingerprints,(dataset,scene,seed,stage)
     for x in records:epochs.append({'dataset':dataset,'scene':scene,'seed':seed,'stage':stage,'epoch':x['epoch'],'updates':x['updates_total'],'train_loss':x['train_loss'],'validation_mse':x['validation']['full96']['mse'],'validation_mae':x['validation']['full96']['mae'],'epoch_seconds':x['epoch_seconds'],'cumulative_seconds':x['seconds'],'batch_fingerprint':x['batch_fingerprint']})
     metric='mse' if stage=='main' else 'mae';selected=min(records,key=lambda x:x['validation']['full96'][metric]);assert done['best_epoch'][metric]==selected['epoch']
     resources=done['resources']
     stage_rows.append({'dataset':dataset,'scene':scene,'seed':seed,'stage':stage,'epochs_completed':100,'updates':done['updates'],'updates_per_epoch':expected,'selected_by':'validation_full96_'+metric,'selected_epoch':selected['epoch'],'selected_validation_mse':selected['validation']['full96']['mse'],'selected_validation_mae':selected['validation']['full96']['mae'],'training_loop_seconds_including_validation_save':done['seconds'],'max_allocated_gib':resources['max_allocated_gib'],'max_reserved_gib':resources['max_reserved_gib'],'gpu':resources['gpu'],'python':resources['python'],'torch':resources['torch'],'cuda':resources['cuda'],'batch_size':done['binding']['batch_size'],'dataset_manifest_sha256':done['binding']['dataset'],'frozen_verified':stage=='calibration'})
     evidence.extend([{'relative_source_file':str(f.relative_to(src)),'sha256':sha(f)} for f in [p,curve]])
     if stage=='main':main_done=done
     else:cal_done=done
    info=r['info'];assert info['coarse']['epoch']==main_done['best_epoch']['mse'] and info['epoch']==cal_done['best_epoch']['mae']
    assert info['coarse']['sha256']==main_done['checkpoints']['mse'] and info['head_sha256']==cal_done['checkpoints']['mae']
    assert sha(info['coarse']['file'])==info['coarse']['sha256'] and sha(info['head'])==info['head_sha256']
    job=src/'jobs'/f'{dataset}_{scene}_{seed}_run.json';jobrow=json.loads(job.read_text());assert jobrow['status']=='complete' and not jobrow['smoke']
    test=r['test']['full96'];pre=r['precal_test']['full96']
    per_seed.append({'dataset':dataset,'scene':scene,'seed':seed,'validation_mse':r['validation']['full96']['mse'],'validation_mae':r['validation']['full96']['mae'],'test_mse':test['mse_mean'],'test_mae':test['mae_mean'],'precal_test_mse':pre['mse_mean'],'precal_test_mae':pre['mae_mean'],'effective_parameters':r['effective_parameters'],'registered_parameters':r['registered_parameters'],'main_selected_epoch':info['coarse']['epoch'],'calibration_selected_epoch':info['epoch'],'main_seconds':main_done['seconds'],'calibration_seconds':cal_done['seconds'],'job_wallclock_seconds':jobrow['wallclock_seconds'],'export_verified':True,'test_masks':20})
    checkpoints.append({'dataset':dataset,'scene':scene,'seed':seed,'main_relative_path':str(Path(info['coarse']['file']).relative_to(src)),'main_sha256':info['coarse']['sha256'],'head_relative_path':str(Path(info['head']).relative_to(src)),'head_sha256':info['head_sha256'],'export_sha256':r['export_sha256'],'checkpoints_uploaded':False})
    for item in r['per_mask']:
     row={'dataset':dataset,'scene':scene,'seed':seed,'mask_seed':item['mask_seed']}
     for kind in ['coarse','final']:
      for span in ['full96','recent32']:
       row.update({f'{kind}_{span}_{key}':item[kind][span][key] for key in ['mse','mae','coordinates']})
     per_mask.append(row)
 summary=[]
 for dataset in DATASETS:
  for scene in SCENES:
   rr=[x for x in per_seed if x['dataset']==dataset and x['scene']==scene];assert len(rr)==3
   row={'dataset':dataset,'scene':scene,'training_seeds':3,'test_masks_per_model':20,'effective_parameters':rr[0]['effective_parameters']}
   for k in ['validation_mse','validation_mae','test_mse','test_mae','precal_test_mse','precal_test_mae']:
    row[k+'_mean']=float(np.mean([x[k] for x in rr]));row[k+'_seed_sd']=float(np.std([x[k] for x in rr],ddof=1))
   row['sum_job_wallclock_seconds']=sum(x['job_wallclock_seconds'] for x in rr);summary.append(row)
 previous=json.loads((src/'reports/three_seed_summary.json').read_text())
 for x in summary:
  old=next(t for t in previous if t['dataset']==x['dataset'] and t['scene']==x['scene'])
  for k in ['test_mse_mean','test_mse_seed_sd','test_mae_mean','test_mae_seed_sd']:assert abs(x[k]-old[k])<1e-14
 for name,records in [('per_seed',per_seed),('three_seed_summary',summary),('per_mask',per_mask),('training_stages',stage_rows),('training_epochs',epochs)]:
  table(out/'summary'/f'{name}.csv',records)
  if name!='training_epochs':dump(out/'summary'/f'{name}.json',records)
 dump(out/'checkpoint_manifest.json',checkpoints);dump(out/'source_evidence_hashes.json',evidence)
 for name in ['baseline_alignment.json','release_integrity.json','execution_lock.json']:
  d=json.loads((src/name).read_text())
  dump(out/'audit'/name,d)
 (out/'runtime').mkdir()
 for name in ['adapter.py','baseline_pamap_data.py','worker.py','driver.py','driver_weather_pamap2.py','report.py','audit_protocol.py','protocol.json']:
  shutil.copyfile(src/name,out/'runtime'/name)
 audit={'model':'R3/H_GLOBAL','model_commit':'59419e9bbc8f2e8833c8265505d0e84fa2b04de4','datasets':DATASETS,'scenes':SCENES,'training_seeds':SEEDS,'completed_models':len(per_seed),'complete_training_stages':len(stage_rows),'epochs_per_stage':100,'total_epoch_records':len(epochs),'per_model_test_masks':20,'per_mask_records':len(per_mask),'all_full_budgets':True,'all_main_and_calibration_checkpoints_selected_by_predefined_validation_metric':True,'all_paired_training_batch_fingerprints_match':True,'all_calibration_freeze_checks_pass':True,'all_24_exports_verified':True,'all_168_validation_test_mask_files_equal_baseline':True,'all_published_release_and_vendor_source_hashes_verified':True,'mask_std_not_training_seed_std':True,'exchange_excluded':True,'stage2_executed':False,'inference_latency_measured_in_this_run':False,'excluded_assets':['raw and processed data','mask npz files','checkpoints','repair predictions or input values','training memory contents']}
 dump(out/'completion_audit.json',audit)
 lines=['# R3：Weather、PAMAP2 四场景三种子正式结果','', '归档日期：2026-11-01。24个模型、48个训练阶段全部完成，每阶段100轮。模型固定为完整R3 / H_GLOBAL，代码commit `59419e9bbc8f2e8833c8265505d0e84fa2b04de4`，未搜索架构或挑选seed。','', '每个训练seed先平均20组与其他基线逐值相同的测试遮挡，再对训练seed2023、2024、2025求均值及样本标准差（ddof=1）。结果在训练集全局标准化尺度，仅对隐藏且有原始真值的坐标评分。不运行预测Stage2。','', '|数据集|场景|测试MSE ± seed SD|测试MAE ± seed SD|有效参数|','|---|---|---:|---:|---:|']
 for x in summary:lines.append(f"|{x['dataset']}|{x['scene']}|{x['test_mse_mean']:.6f} ± {x['test_mse_seed_sd']:.6f}|{x['test_mae_mean']:.6f} ± {x['test_mae_seed_sd']:.6f}|{x['effective_parameters']:,}|")
 lines+=['','## 记录与检查','','- [三种子汇总](summary/three_seed_summary.csv)、[逐训练seed](summary/per_seed.csv)、[480组逐mask结果](summary/per_mask.csv)，均包含校准前后诊断。','- [48阶段选模与资源](summary/training_stages.csv)、[4800轮训练记录](summary/training_epochs.csv)、[checkpoint摘要](checkpoint_manifest.json)。权重未上传。','- [完成检查](completion_audit.json)：完整预算、验证选模、冻结、配对训练mask指纹和导出核验均通过。','- [原基线遮挡对齐](audit/baseline_alignment.json)：Weather采用2026100801–2026100820，PAMAP2采用六基线的2026101401–2026101420。发布默认PAMAP2的20261018银行未用于本次正式评价。','- [执行适配与队列代码](runtime/)仅为本次服务器运行来源记录，不是可直接替代发布入口的新模型；原R3架构代码不变。','', '## 口径与限制','', '双RTX4090D运行，Python3.11.17、Torch2.5.1+cu124、CUDA12.4，batch128，Adam lr0.001、weight_decay0。Weather每阶段28600次更新，PAMAP2每阶段8500次。主干按验证full96 MSE选，冻结粗模型/历史库后，校准按验证full96 MAE选；测试不改选。','', '训练循环seconds含轮内验证和保存，不含全部准备；job_wallclock_seconds是该模型工作进程记录的端到端时间。sum_job_wallclock_seconds是串加的模型耗时，不是双卡实验墙钟。显存为各阶段实测allocated/reserved峰值。本次未独立测推理p50，不借用R2或其他服务器的延迟。','', '遮挡组属于同一训练模型内重复评价，不是训练重复；若其他基线只有seed2023，不能把其mask标准差等同本表的训练seed波动。不据均值接近宣称统计等效。','', 'Exchange-Rate按用户要求中途移出本机，历史完成的两个模型未纳入本归档；该数据集由其他服务器结果归档负责。原始数据、mask二进制、checkpoint、逐点预测及训练记忆均不上传。','', 'Git发布缺失的11个协议参考JSON曾按release_lock声明的SHA原样恢复，详见release_integrity；这不改变模型代码。','', '同发布分支上其他四数据集结果见 [既有四数据集归档](../R3_final_four_datasets_3seed_20261031/README.md)，本次未覆盖其内容。']
 (out/'README.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps(audit,ensure_ascii=False))
if __name__=='__main__':main()
