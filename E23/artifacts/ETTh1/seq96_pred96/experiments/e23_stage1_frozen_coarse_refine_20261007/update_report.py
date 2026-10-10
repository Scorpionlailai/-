from pathlib import Path
import json,csv,statistics
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent/'e23_stage1_repair_focus_gpu0_20261007';DOCS=Path('/home/xyw/paper/E23/docs/experiments')
refs=json.loads((BASE/'validation_references.json').read_text());rows=[]
for scene in refs:
 for model,key in [('T1','t1'),('E23-v79','v79')]:
  for region in ['all_hidden','last32_hidden']:
   rows.append({'model':model,'scene':scene['scene'],'region':region,'status':'reference_validation','validation_mse':scene[key][region]['mse'],'validation_mae':scene[key][region]['mae']})
for root in [BASE,ROOT,ROOT.parent/'e23_stage1_frozen_coarse_tail_20261007',ROOT.parent/'e23_stage1_bounded_coarse_tail_20261007',ROOT.parent/'e23_stage1_boundary_coarse_tail_20261007',ROOT.parent/'e23_stage1_boundary_tail_mae_20261007',ROOT.parent/'e23_stage1_repeated_boundary_tail_mae_20261007',ROOT.parent/'e23_stage1_one_sided_boundary_tail_mae_20261007',ROOT.parent/'e23_stage1_one_sided_boundary_gpu1_20261007',ROOT.parent/'e23_stage1_tail_selection_20261008',ROOT.parent/'e23_stage1_local_one_sided_guide_20261008',ROOT.parent/'e23_stage1_mean_and_dense_20261008']:
 plan=json.loads((root/'plan.json').read_text());state=json.loads((root/'queue-state.json').read_text()) if (root/'queue-state.json').exists() else {}
 for entry in plan['models']:
  for scene in plan['scenarios']:
   cell=root/entry['label']/scene['name'];audit=cell/'repair-audit.json';sel=cell/'stage1/runs/e23-stage1.json'
   if not sel.exists():continue
   selection=json.loads(sel.read_text());a=json.loads(audit.read_text()) if audit.exists() else None
   resources=list((cell/'stage1/resources').glob('*.json'));resource=json.loads(resources[-1].read_text()) if resources else {}
   for region in ['all_hidden','last32_hidden']:
    row={'model':entry['label'],'scene':scene['name'],'region':region,'status':'repair_audited' if a else 'stage1_complete','stage1_epochs':selection.get('total_stage1_epochs',selection['epochs_completed']),'best_epoch':selection['best_epoch'],'validation_selector':selection['selection'],'train_scope':'repair_mean_and_dense' if selection.get('expert_parameters_frozen_verified') else 'dense_only','trainable_parameters':selection.get('trainable_parameters',entry['repair_parameters']),'training_seconds':resource.get('full_workflow_seconds'),'parent_training_seconds':selection.get('parent_training_seconds',0),'peak_allocated_gib':resource.get('peak_allocated_bytes',0)/2**30,'peak_reserved_gib':resource.get('peak_reserved_bytes',0)/2**30,'selection_manifest':str(sel),'test_role':'development_20_masks','stage2_training':False}
    if a:
     row.update(validation_mse=a['split_validation']['metrics'][region]['mse'],validation_mae=a['split_validation']['metrics'][region]['mae'])
     for metric in ['mse','mae']:
      row['test_mean_'+metric]=a['test_mean'][region][metric];row['test_std_'+metric]=statistics.stdev([g['metrics'][region][metric] for g in a['test_groups']]);row['t1_test_mean_'+metric]=a['t1_test_mean'][region][metric]
    elif region=='all_hidden':row.update(validation_mse=selection['validation_repair_metrics']['global_mse'],validation_mae=selection['validation_repair_metrics']['global_mae'])
    rows.append(row)
keys=list(dict.fromkeys(k for r in rows for k in r))
with (DOCS/'E23_stage1_repair_iteration_20261007.csv').open('w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
lines=['# E23 Stage1 修复迭代','',f'北京时间：{datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")}','', '只迭代Stage1。验证全窗口缺失坐标MSE选权重；所有修复指标在训练集全局标准化尺度上计算，排除原本可见坐标。20组测试为已使用的开发掩码，标准差表示缺失位置变化，不能代替训练随机种子的标准差。','', '起初GPU0运行E23、GPU1运行E24 Weather；确认Weather进程结束后，v94使用GPU1，与GPU0的v93并行。资源统计为本进程PyTorch峰值；同时运行可能共享CPU与磁盘。v88及后续固定粗修复候选先沿用v79的100轮训练权重，再用100轮学习精修，只更新新增精修参数（v88–v90为79751，v91–v95为80647，v96为81543；v97另开放原有16514个修复均值参数，共97161个训练参数，三专家骨干与历史库冻结）；CSV分别列出本轮全流程耗时与父版本训练耗时，论文正式效率对照需要重新计时完整训练流程。','', '| 模型 | 场景 | 区域 | 验证 MSE / MAE | 开发测试 MSE / MAE | 测试 MSE / MAE 标准差 |','|---|---|---|---:|---:|---:|']
def pair(r,a,b):return '—' if a not in r or b not in r else f'{r[a]:.6f} / {r[b]:.6f}'
for row in rows:lines.append(f'| {row["model"]} | {row["scene"]} | {row["region"]} | {pair(row,"validation_mse","validation_mae")} | {pair(row,"test_mean_mse","test_mean_mae")} | {pair(row,"test_std_mse","test_std_mae")} |')
(DOCS/'E23_stage1_repair_iteration_20261007.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'rows':len(rows),'csv':str(DOCS/'E23_stage1_repair_iteration_20261007.csv')}))
