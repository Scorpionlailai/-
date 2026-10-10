"""Persist every prespecified seed and its shared training resource accounting."""
import csv,hashlib,json,tarfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
DOC=Path('/home/xyw/paper/E23/docs/experiments');PREFIX=DOC/'ETTh1_E23_v68_frozen_point_sr015_bl8_20261005'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,text):
 p=Path(p);tmp=p.with_name(p.name+'.tmp');tmp.write_text(text);tmp.replace(p)
def csvwrite(p,rows):
 p=Path(p);tmp=p.with_name(p.name+'.tmp');keys=list(dict.fromkeys(k for row in rows for k in row))
 with tmp.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
 tmp.replace(p)
def main():
 plan=read(ROOT/'plan.json');definition=read(ROOT/'frozen-definition.json');work=Path(plan['work']);assert sha(ROOT/'plan.json')==definition['plan_sha256']
 for relative,h in definition['sources'].items():assert sha(work/relative)==h
 for p,h in definition['protected'].items():assert sha(p)==h
 with (DOC/'ETTh1_E23_v66_point_context_sr015_bl8_20261005.csv').open() as f:rows=list(csv.DictReader(f))
 for row in rows:row.update(cached_reference=True)
 refs={label:read(p) for label,p in plan['cached_references'].items()};scores=refs.copy();completed=[];paired=[];diagnostics=[];newcost=0.;seen=set();seen_resources=set();stage1_resources=set();uuids=set()
 for step in plan['sequence']:
  cell=ROOT/step['name']
  if not (cell/'completed-audit.json').exists():continue
  result=read(cell/'completed-audit.json');spec=result['trial'];lock=result['validation_lock'];score=read(cell/'evaluation/twenty-test-draws.json');seed=spec['training_seed'];s1=read(cell/'stage1/runs/e23-stage1.json');s2=read(cell/'stage2/runs/e23-stage2.json');diag=read(cell/'recovery-diagnostics.json');resources=score['training_resources'];records=[read(p) for p in resources['resource_record_paths']]
  assert s1['epochs_completed']==s2['epochs_completed']==100 and s1['seed']==s2['seed']==seed and len(score['test_draws'])==20
  assert s2['predictor_initial_sha256']==plan['predictor_initial_sha256_by_seed'][str(seed)] and resources['resource_status']=='measured_all_stages' and len(records)==2
  assert read(cell/'parent-choice-before-tests.json')['uses_test_metrics'] is False and score['validation_mask_sha256']==plan['validation_mask_sha256']
  key=resources['resource_record_paths'][1];incremental=records[1]['full_workflow_seconds'] if key not in seen else 0.;seen.update([key]);seen_resources.update(resources['resource_record_paths']);stage1_resources.add(resources['resource_record_paths'][0]);newcost+=incremental;uuids.update(r['device']['uuid'] for r in records)
  configuration=next(r for r in read(ROOT/'preflight.json')['configurations'] if r['training_seed']==seed);stagecost={k:v for r in records for k,v in r['stages'].items()}
  row=dict(scenario='sr015_bl8',start_rate=.15,block_len=8,actual_test_missing_fraction=plan['actual_test_missing_fraction'],model=spec['label'],training_seed=seed,seq_len=96,pred_len=96,test_masks=20,mse=score['test_mean']['mse'],mae=score['test_mean']['mae'],mask_mse_std=score['test_mask_std_ddof1']['mse'],mask_mae_std=score['test_mask_std_ddof1']['mae'],validation_mse=s2['best_val_mse'],validation_mae=s2['best_val_mae'],full_training_seconds=resources['full_workflow_seconds'],peak_allocated_gib=resources['peak_allocated_gib'],peak_reserved_gib=resources['peak_reserved_gib'],gpu=resources['gpu'],training_batch_sizes=f'{s1["stage1_training_batch_size"]},128',stage1_batch_size=s1['stage1_training_batch_size'],stage2_batch_size=128,stage1_epochs=100,stage2_epochs=100,stage1_optimizer_updates=s1['optimizer_updates'],selected_stage2_epoch=s2['best_epoch'],stage1_repair_mse=s1['best_val_repair_mse'],predictor_parameters=35427,predictor_initial_sha256=s2['predictor_initial_sha256'],repair_parameters=configuration['repair_parameters'],front_context_parameters=configuration['front_context_parameters'],stage2_loss=s2['loss'],repair_update_policy=s2['repair_update_policy'],selection_metric=s2['selection_metric'],selection=str(cell/'stage2/runs/e23-stage2.json'),evaluation=str(cell/'evaluation/twenty-test-draws.json'),checkpoint_sha256=s2['checkpoint_sha256'],validation_mask_sha256=score['validation_mask_sha256'],test_bank_sha256=score['test_mask_bank_sha256'],cached_reference=False,stage1_reused=False,actual_new_training_seconds=incremental,repair_stage_seconds=stagecost['repair']['wall_seconds'],forecast_stage_seconds=stagecost['forecast']['wall_seconds'],parent=spec['parent']['label'],architecture_parent=spec['architecture_parent'],only_change=spec['only_change'],accepted_on_validation=lock['accepted_on_validation'],stage2_repair_mse=diag[1]['metrics']['all96']['mse'],stage2_last32_repair_mse=diag[1]['metrics']['last32']['mse'])
  row.update(stage2_policy_name=spec['stage2_policy_name'],stage1_reused=spec['stage2_policy_name']!='legacy')
  rows.append(row);completed.append(result);scores[spec['label']]=score
  for d in diag:diagnostics.append(dict(model=spec['label'],training_seed=seed,stage=d['stage'],selected_epoch=d['selected_epoch'],last32_level_mse=d['last32_level_mse'],**{n+'_'+m:v[m] for n,v in d['metrics'].items() for m in ['mse','mae']}))
  for label in dict.fromkeys(['T1',plan['initial_parent']['label']]):
   baseline=refs[label];delta={m:[] for m in ['mse','mae']}
   for x,y in zip(score['test_draws'],baseline['test_draws']):
    assert (x['mask_seed'],x['mask_sha256'],x['target_elements'])==(y['mask_seed'],y['mask_sha256'],y['target_elements'])
    for m in delta:delta[m].append(x[m]-y[m])
   paired.append(dict(model=spec['label'],baseline=label,training_seed=seed,mse_delta=float(np.mean(delta['mse'])),mae_delta=float(np.mean(delta['mae'])),mse_wins=sum(x<0 for x in delta['mse']),mae_wins=sum(x<0 for x in delta['mae']),test_masks=20))
 draws=[]
 for label,score in scores.items():
  assert score['test_mask_bank_sha256']==definition['bank_sha256'] and score['validation_mask_sha256']==plan['validation_mask_sha256']
  for d in score['test_draws']:draws.append(dict(model=label,training_seed=score['training_seed'],cached_reference=label in refs,**d))
 csvwrite(PREFIX.with_suffix('.csv'),rows);csvwrite(Path(str(PREFIX)+'_all_draws.csv'),draws)
 if paired:csvwrite(Path(str(PREFIX)+'_paired.csv'),paired)
 if diagnostics:csvwrite(Path(str(PREFIX)+'_repair_diagnostics.csv'),diagnostics)
 lines=['# E23 v68：逐点修复器完全冻结对照','',f'{len(completed)}/{len(plan["sequence"])}项选择完成；ETTh1 96→96，起点0.15、块8、实际测试缺失72.9252%，旧20位置。T1维持作者权重；复用v66完整Stage1和seed502，只将Stage2修复器更新策略改为完全冻结。','','| 模型 | 测试MSE | 测试MAE | 验证MSE / MAE | 完整训练分钟 | allocated / reserved GiB |','|---|---:|---:|---:|---:|---:|']
 for row in rows:lines.append(f'| {row["model"]} | {float(row["mse"]):.6f} ± {float(row["mask_mse_std"]):.6f} | {float(row["mae"]):.6f} ± {float(row["mask_mae_std"]):.6f} | {float(row["validation_mse"]):.6f} / {float(row["validation_mae"]):.6f} | {float(row["full_training_seconds"])/60:.2f} | {float(row["peak_allocated_gib"]):.3f} / {float(row["peak_reserved_gib"]):.3f} |')
 lines+=['','原预测头源码、配置及35427参数保持；不同训练种子的初始权重自然不同，SHA分别记录。同seed502初始权重和固定验证及20测试缺失不变。保留v66原混合损失及uniform项，修复器全部冻结；padding/RNG沿用legacy配置，原预测头源码/配置/初始化保持。MAE主与MSE次选择来自同一次训练，按预定验证门槛保留，决定在每个种子测试之前保存。','','完整训练耗时含两阶段准备、训练、验证和保存；测试/诊断另计。一个种子两行共享完整资源，新增耗时去重。位置标准差与训练种子标准差分开报告。全部固定种子披露，开发最佳种子不能替代多种子平均优势。']
 cohort={}
 for policy in plan['stage2_policies']:
  for metric in ['mae','mse']:
   subset=[r for r in rows if r['cached_reference'] is False and r['selection_metric']==metric and r['stage2_policy_name']==policy['name']]
   if len(plan['seeds'])>1 and len(subset)==len(plan['seeds']):cohort[policy['name']+'-'+metric]={m:dict(mean=float(np.mean([r[m] for r in subset])),training_seed_std_ddof1=float(np.std([r[m] for r in subset],ddof=1))) for m in ['mse','mae']}
 if cohort:
  write(ROOT/'seed-cohort-statistics.json',json.dumps(dict(training_seeds=plan['seeds'],selection=cohort,T1_seed=502,seed502_control_excluded_from_new_cohort=True),indent=2)+'\n')
  lines+=['','## 预定四种子的平均值','','| checkpoint选择 | 预测MSE（训练种子std） | 预测MAE（训练种子std） |','|---|---:|---:|']
  for choice,stats in cohort.items():lines.append(f'| {choice} | {stats["mse"]["mean"]:.6f} ± {stats["mse"]["training_seed_std_ddof1"]:.6f} | {stats["mae"]["mean"]:.6f} ± {stats["mae"]["training_seed_std_ddof1"]:.6f} |')
 if (ROOT/'completed-audit.json').exists():
  final=read(ROOT/'completed-audit.json');lines+=['',f'验证保留：**{final["retained_parent"]["label"]}**。新增Stage2训练耗时{newcost/60:.2f}分钟（原Stage1成本仍包含在完整模型成本；新增只计一个Stage2，两份选择共享）。']
  assert all(read(ROOT/'forecast-head-audit.json').values()) and len(uuids)==1
  with tarfile.open(ROOT/'frozen-source.tar.gz') as archive:
   for rel,h in definition['sources'].items():assert hashlib.sha256(archive.extractfile(rel).read()).hexdigest()==h
  write(ROOT/'delivery-audit.json',json.dumps(dict(status='complete',new_stage1_runs=0,new_stage2_runs=len(seen),new_test_draws=len(completed)*20,cached_reference_draws=len(refs)*20,new_full_training_seconds=newcost,physical_gpu_uuid=next(iter(uuids)),source_files_verified=len(definition['sources']),protected_files_verified=len(definition['protected']),original_head_architecture_verified=True,different_seed_initializations_recorded=True,test_used_for_checkpoint_or_automatic_parent_selection=False,retained_parent=final['retained_parent']),indent=2)+'\n')
 lines+=['',f'[CSV]({PREFIX.name}.csv) · [位置明细]({PREFIX.name}_all_draws.csv) · [配对差值]({PREFIX.name}_paired.csv)', '',f'完整记录：`{ROOT}`。']
 write(PREFIX.with_suffix('.md'),'\n'.join(lines)+'\n');print(json.dumps(dict(completed=len(completed),rows=len(rows),new_full_training_seconds=newcost,new_scores=[dict(label=c['label'],test_mean=c['test_mean']) for c in completed])),flush=True)
if __name__=='__main__':main()
