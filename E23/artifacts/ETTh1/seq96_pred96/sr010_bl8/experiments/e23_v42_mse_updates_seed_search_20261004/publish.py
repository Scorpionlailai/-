"""Publish partial and final tables without modifying the frozen experiment source."""
import csv
import json
from pathlib import Path
from datetime import datetime,timezone
from integrations.forecast_compare.resources import selected_training_resources

ROOT=Path(__file__).resolve().parent
DOCS=Path('/home/xyw/paper/E23/docs/experiments')
PREFIX='ETTh1_P4_E23_v42_MSE_dual_selection_seed_search_20261004'


def read(p):return json.loads(Path(p).read_text())
def csv_write(p,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def collect():
    rows=[];draws=[];actual_resources={}
    cells=[ROOT/'T1_seed502',*sorted(p.parent for p in ROOT.glob('*/trial-definition.json'))]
    for cell in cells:
        t1=cell.name=='T1_seed502'
        trial={'label':'T1','seed':502,'stage1_reused':True} if t1 else read(cell/'trial-definition.json')
        for metric in ['mae','mse']:
            manifest=cell/'training'/('forecast-selection.json' if metric=='mae' else 'forecast-mse-selection.json') if t1 else cell/'stage2/runs'/('e23-stage2.json' if metric=='mae' else 'e23-stage2-mse.json')
            if not manifest.exists():continue
            record=read(manifest)
            first_path=Path(record['repair_selection_manifest'] if t1 else record['stage1_manifest'])
            first=read(first_path)
            resources=selected_training_resources([manifest] if t1 else [first_path,manifest])
            val=record['best_validation'] if t1 else dict(mse=record['best_val_mse'],mae=record['best_val_mae'])
            batch=first['training']['batch_size'] if t1 else first.get('stage1_training_batch_size',first.get('batch_size',128))
            epochs=first['epochs_completed']
            phase='baseline' if t1 else 'geometry' if cell.name.startswith('aligned_') else 'update_budget' if cell.name.startswith('updates') else 'seed_search'
            row=dict(model=trial['label'],trial=cell.name,phase=phase,training_seed=trial['seed'],checkpoint_selection_metric=metric,
                status='validation_finalized',stage1_reused=trial['stage1_reused'],stage1_epochs=epochs,stage1_batch_size=batch,
                stage1_optimizer_updates=(8449//batch)*epochs,stage1_selected_epoch=first['best_epoch'],
                stage1_validation_repair_mse=first['best_validation']['mse'] if t1 else first['best_val_repair_mse'],
                stage2_epochs=record['epochs_completed'],checkpoint_epoch=record['best_epoch'],
                validation_mse=val['mse'],validation_mae=val['mae'],test_mse='',test_mae='',test_mse_std='',test_mae_std='',
                test_mask_groups=0,full_training_seconds=resources['full_workflow_seconds'],
                training_stages_seconds=resources['training_stages_seconds'],peak_allocated_gib=resources['peak_allocated_gib'],
                peak_reserved_gib=resources['peak_reserved_gib'],gpu=resources['gpu'],resource_status=resources['resource_status'],
                prediction_head_parameters=35427,prediction_head_initial_sha256=record['predictor_initial_sha256'],
                selection_manifest=str(manifest),checkpoint_sha256=record['checkpoint_sha256'],test_result='',
                validation_selected_seed=False,geometry_selected='',update_budget_selected='')
            new_seconds=cached_seconds=0.
            for resource_path in resources['resource_record_paths']:
                r=read(resource_path)
                if Path(resource_path).is_relative_to(ROOT) and not r.get('cached_stage1'):
                    new_seconds+=r['full_workflow_seconds'];actual_resources[resource_path]=r['full_workflow_seconds']
                else:cached_seconds+=r['full_workflow_seconds']
            row.update(actual_new_training_seconds=new_seconds,cached_stage1_seconds=cached_seconds,
                stage1_timing_scope='cached repair stage + all original non-stage overhead; excludes old Stage2' if t1 else 'separate measured Stage1 command')
            result_path=cell/f'evaluation_{metric}/twenty-test-draws.json'
            if result_path.exists():
                result=read(result_path)
                row.update(status='tested',test_mse=result['test_mean']['mse'],test_mae=result['test_mean']['mae'],
                    test_mse_std=result['test_mask_std_ddof1']['mse'],test_mae_std=result['test_mask_std_ddof1']['mae'],
                    test_mask_groups=len(result['test_draws']),test_result=str(result_path))
                for d in result['test_draws']:
                    draws.append(dict(model=row['model'],trial=cell.name,training_seed=row['training_seed'],checkpoint_selection_metric=metric,**d))
            if (ROOT/'seed-validation-choice.json').exists():
                choice=read(ROOT/'seed-validation-choice.json')
                row['validation_selected_seed']=choice['selected_cells'][metric]==str(cell)
            if (ROOT/'geometry-validation-choice.json').exists():row['geometry_selected']=read(ROOT/'geometry-validation-choice.json')['selected']
            if (ROOT/'budget-validation-choice.json').exists():row['update_budget_selected']=read(ROOT/'budget-validation-choice.json')['selected']
            rows.append(row)
    return rows,draws,actual_resources


def main():
    rows,draws,actual=collect();complete=(ROOT/'matrix-completed.json').exists()
    csv_write(DOCS/(PREFIX+'.csv'),rows);csv_write(DOCS/(PREFIX+'_all_draws.csv'),draws)
    csv_write(ROOT/'summary.csv',rows);csv_write(ROOT/'all-draws.csv',draws)
    selected_rows=[]
    if (ROOT/'seed-validation-choice.json').exists():
        choices=read(ROOT/'seed-validation-choice.json')
        seed_rows=[r for r in rows if str(ROOT/r['trial']) in choices['seed_cells'].values()]
        csv_write(DOCS/(PREFIX+'_six_seeds.csv'),seed_rows)
        for metric in ['mae','mse']:
            for r in rows:
                if r['checkpoint_selection_metric']==metric and (r['model']=='T1' or r['validation_selected_seed']):
                    selected_rows.append(r)
        csv_write(DOCS/(PREFIX+'_selected_comparison.csv'),selected_rows)
    lines=['# E23-v42：纯 MSE、统一 Stage 2、双指标选模与种子搜索','',
        f"状态：{'全部完成' if complete else '运行中，表格仅包含已完成的训练/评估'}。更新时间（UTC）：{datetime.now(timezone.utc).isoformat()}",'',
        '## 实验口径','',
        '- Stage 2 纯 MSE，Adam，100 轮、batch 128、学习率 1e-4、weight decay 0、FP32；修复器与预测头联合训练，E23 固有 H 库/键保持冻结。',
        '- 原始 PatchTST D16，35,427 参数，结构不变；每个种子的初始预测头一致。E23 补齐位置与 T1 的末值复制一致。',
        '- 同一条训练轨迹分别保存最低验证 MAE、最低验证 MSE 的 checkpoint。两行资源成本相同，不代表训练两次。',
        '- 几何与 Stage 1 预算按 MSE-selected checkpoint 的验证 MSE 选择，平分时比较验证 MAE。最终分别按各自验证指标选一个 E23 种子。',
        '- E23 预先固定种子 [2021, 102, 202, 302, 402, 502]，保留全部结果。Stage 1 对照 128×100=6,600 次更新；增量方案 32×200=52,800 次更新。',
        '- T1 固定 seed=502；该种子来自已有复现最佳测试 MSE，原论文没有公布“最佳种子”。T1 Stage 1 使用同一作者配置与旧权重，只重新训练一次 Stage 2 以保留双指标 checkpoint。',
        '- 同一 ETTh1 P4、96→96、原验证缺失、冻结 20 组测试缺失；sr=0.1 是缺失块起始率，块长 8，重叠后真实缺失比例约 56.7%。',
        '- 本轮属于已观察过旧测试结果后的开发搜索；不能作为未经筛选的多种子平均，旧正式五种子统计保留。','',
        '## 结果','',
        '| 模型/实验 | seed | checkpoint 依据 | epoch | 验证 MSE | 验证 MAE | 测试 MSE | 测试 MAE | 完整训练 s | 显存 allocated/reserved GiB |',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|---|']
    def fmt(x):return '待评估' if x=='' or x is None else f'{x:.6f}'
    for r in rows:
        marker=' ★' if r['validation_selected_seed'] else ''
        lines.append(f"| {r['model']}{marker} | {r['training_seed']} | {r['checkpoint_selection_metric'].upper()} | {r['checkpoint_epoch']} | {fmt(r['validation_mse'])} | {fmt(r['validation_mae'])} | {fmt(r['test_mse'])} | {fmt(r['test_mae'])} | {r['full_training_seconds']:.2f} | {r['peak_allocated_gib']:.3f}/{r['peak_reserved_gib']:.3f} |")
    if complete:
        lines+=['','## 按验证规则选出的最终对照','',
            '| 选模规则 | E23 seed/epoch | E23 测试 MSE/MAE | T1 测试 MSE/MAE | E23 相对 T1 的 MSE/MAE 变化 |',
            '|---|---|---|---|---|']
        differences={}
        for metric in ['mae','mse']:
            e=next(r for r in selected_rows if r['checkpoint_selection_metric']==metric and r['model']!='T1')
            t=next(r for r in selected_rows if r['checkpoint_selection_metric']==metric and r['model']=='T1')
            paired_e=read(e['test_result'])['test_draws'];paired_t=read(t['test_result'])['test_draws']
            differences[metric]=dict(E23_seed=e['training_seed'],E23_epoch=e['checkpoint_epoch'],T1_seed=502,T1_epoch=t['checkpoint_epoch'],
                E23_mean={k:e['test_'+k] for k in ['mse','mae']},T1_mean={k:t['test_'+k] for k in ['mse','mae']},
                E23_minus_T1={k:e['test_'+k]-t['test_'+k] for k in ['mse','mae']},
                relative_change_percent={k:(e['test_'+k]/t['test_'+k]-1)*100 for k in ['mse','mae']},
                paired_mask_wins={k:sum(de[k]<dt[k] for de,dt in zip(paired_e,paired_t)) for k in ['mse','mae']})
            delta=differences[metric]['relative_change_percent']
            lines.append(f"| {metric.upper()} | {e['training_seed']}/{e['checkpoint_epoch']} | {e['test_mse']:.6f}/{e['test_mae']:.6f} | {t['test_mse']:.6f}/{t['test_mae']:.6f} | {delta['mse']:+.2f}%/{delta['mae']:+.2f}% |")
        (ROOT/'selected-comparison.json').write_text(json.dumps(differences,indent=2)+'\n')
        lines+=['','正的变化百分比表示 E23 误差更高。星号/最终对照种子由验证集选出；其他种子的测试结果完整保留于六种子 CSV。']
        lines+=['','## 本轮结论','',
            '- 两种验证规则均选出 E23 seed=102，MAE 选模第 35 轮、MSE 选模第 37 轮。两版测试 MSE、MAE 均高于同口径 T1，20 组缺失中均未胜出。',
            '- 提高 Stage 1 更新次数使验证修复 MSE 从 0.277471 降至 0.260599，但 Stage 2 验证结果更差，未进入种子搜索。',
            '- 本轮所有种子与两种选模结果均未改善之前最佳开发版本的测试指标，本轮设置不替换此前模型。',
            '- 这轮同时更改了损失、修复学习率/保留项、补齐方式与 RNG 重置；没有逐项控制实验，不能将退步单独归因于纯 MSE。',
            '- 已完成 56 项实现测试；最终核对 404 个代码文件、360 个原有文件未变，所有 18 份权重选择与 360 条测试缺失组记录一致。']
    lines+=['','## 资源统计','',
        f'- 本轮已计量实际新训练成本：**{sum(actual.values()):.2f} 秒**（资源文件去重，双指标 checkpoint 和别名不会重复计费）。',
        '- 每行完整训练时间为原 Stage 1 成本加本轮 Stage 2 成本；复用权重仍保留原训练成本。实际新耗时、复用 Stage 1 耗时另列于 CSV。',
        '- T1 原 Stage 1 和 Stage 2 在同一进程测量。本轮将原 repair 阶段实测时间加全部原非阶段开销归给缓存 Stage 1，剔除旧 Stage 2；这项 Stage 1 完整时间是分解值，不是重新训练的测量。',
        '- 显存为 PyTorch allocator 峰值 allocated/reserved，含模型、梯度、优化器及中间张量；完整训练时间包括导入、数据准备、所有训练/验证和权重保存，不含最终测试。',
        '- 增量方案 Stage 1 batch 32 与 T1 作者 Stage 1 batch 16 不同，Stage 2 均 batch 128。','',
        '## 文件','',f'- 汇总：`{PREFIX}.csv`',f'- 全部测试缺失组：`{PREFIX}_all_draws.csv`',
        f'- 计划、代码快照、训练命令、权重、验证锁、资源侧记、逐窗口误差：`{ROOT}`','']
    (DOCS/(PREFIX+'.md')).write_text('\n'.join(lines))
    print(json.dumps(dict(status='complete' if complete else 'running',training_selector_rows=len(rows),test_draws=len(draws),unique_new_training_commands=len(actual),actual_new_training_seconds=sum(actual.values()))))


if __name__=='__main__':main()
