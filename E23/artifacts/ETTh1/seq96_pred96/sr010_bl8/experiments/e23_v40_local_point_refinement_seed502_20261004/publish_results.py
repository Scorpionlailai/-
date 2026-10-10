"""Publish completed point-refiner trials from saved scores and resources only."""
import csv
import hashlib
import io
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
FORMAL=ROOT.parent/'e23_t1_five_seed_standard_20261004'
DOCS=Path('/home/xyw/paper/E23/docs/experiments')
NAME='ETTh1_P4_E23_v40_point_refinement_seed502_20261004'


def read(path):return json.loads(Path(path).read_text())


def build_row(label,evaluation,cell=None):
    data=read(evaluation);resources=data['training_resources'];selection=read(data['selection_manifest'])
    repair=read(selection['repair_selection_manifest'] if label=='T1' else selection['stage1_manifest'])
    records=[read(p) for p in resources['resource_record_paths']]
    stages={}
    for rec in records:
        for stage,details in rec['stages'].items():stages[stage]=stages.get(stage,0)+details['wall_seconds']
    batches={stage:rec['batch_size'] for rec in records for stage in rec['stages']}
    if label=='T1':batches.update(repair=repair['training']['batch_size'],forecast=selection['training']['batch_size'])
    gpu_ids={rec['device'].get('uuid') for rec in records}
    decision=read(cell/'validation-decision.json') if cell else None
    point_metrics=read(ROOT/'selected-repair-diagnostics.json') if (ROOT/'selected-repair-diagnostics.json').exists() else {}
    diagnostic=point_metrics.get(label,{})
    result=dict(model=label,training_seed=502,test_masks=20,mse=data['test_mean']['mse'],mae=data['test_mean']['mae'],
        mask_mse_std=data['test_mask_std_ddof1']['mse'],mask_mae_std=data['test_mask_std_ddof1']['mae'],
        validation_mse=data['validation_replay']['mse'],validation_mae=data['validation_replay']['mae'],
        stage1_validation_repair_mse=repair['best_validation']['mse'] if label=='T1' else repair['best_val_repair_mse'],
        full_training_seconds=resources['full_workflow_seconds'],repair_stage_seconds=stages.get('repair'),
        forecast_stage_seconds=stages.get('forecast'),peak_allocated_gib=resources['peak_allocated_gib'],
        peak_reserved_gib=resources['peak_reserved_gib'],physical_gpu_consistent=len(gpu_ids)==1,
        gpu_uuid=';'.join(sorted(str(x) for x in gpu_ids)),repair_batch_size=batches['repair'],
        forecast_batch_size=batches['forecast'],repair_epochs=repair.get('epochs_completed'),forecast_epochs=selection['epochs_completed'],
        point_refiner_span=selection.get('point_refiner_span',0),refiner_parameters=39041 if cell else 0,
        repair_parameters=174362 if cell else (135321 if label=='E23-v31-B' else 536518),predictor_parameters=35427,
        validation_gate_passed=decision['passed'] if decision else '',
        stage2_repair_mse=diagnostic.get('stage2',{}).get('mse',''),
        stage2_repair_mae=diagnostic.get('stage2',{}).get('mae',''),
        stage2_last32_repair_mse=diagnostic.get('stage2',{}).get('last32_mse',''),
        stage2_before64_repair_mse=diagnostic.get('stage2',{}).get('before64_mse',''),
        stage2_without_refiner_forecast_mse=diagnostic.get('stage2',{}).get('without_refiner_forecast_mse',''),
        stage2_without_refiner_forecast_mae=diagnostic.get('stage2',{}).get('without_refiner_forecast_mae',''),
        predictor_initial_sha256=data['predictor_initial_sha256'],checkpoint_sha256=data['checkpoint_sha256'],
        evaluation=str(Path(evaluation).resolve()),scope='Exploratory seed selected using old test outcomes; mask std is not training-seed std')
    return result,data


def csv_text(rows):
    stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
    writer.writeheader();writer.writerows(rows);return stream.getvalue()


def main():
    pairs=[];pending=[]
    for label in ['E23-v31-B','T1']:
        pairs.append(build_row(label,FORMAL/f'seed_502/{label}/evaluation/twenty-test-draws.json'))
    for trial in ['tail32','full96']:
        cell=ROOT/trial
        if (cell/'completed-audit.json').exists():pairs.append(build_row('E23-v40-'+trial,cell/'evaluation/twenty-test-draws.json',cell))
        else:pending.append(trial)
    protected=read(ROOT/'protected-existing-files.json')
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in protected.items())
    rows=[r for r,d in pairs];draws=[]
    for r,data in pairs:
        for draw in data['test_draws']:
            draws.append(dict(model=r['model'],training_seed=502,
                **{k:draw[k] for k in ['group_id','mask_seed','mask_sha256','mse','mae','windows','target_elements']}))
    report=['# E23 v40：逐点残差精修位置对照','',
        'ETTh1 P4，96→96，seed502。原v31-B + 39,041参数逐点精修，最近32点 vs 全96点输出。状态：'+('全部完成。' if not pending else '部分完成；待完成：'+', '.join(pending)+'。'), '',
        '**seed502依据已见的五种子测试结果选为开发种子，本表不能替代正式五种子统计。** 原v31-B和T1训练/测试记录缓存复用，本轮没有重跑这两个基线。', '',
        '| 模型 | 20组测试MSE | 20组测试MAE | 验证MSE / MAE | Stage1修复MSE | 完整训练分钟 | 峰值 allocated / reserved GiB |',
        '|---|---:|---:|---|---:|---:|---|']
    for r in rows:
        report.append(f"| {r['model']} | {r['mse']:.6f} | {r['mae']:.6f} | {r['validation_mse']:.6f} / {r['validation_mae']:.6f} | {r['stage1_validation_repair_mse']:.6f} | {r['full_training_seconds']/60:.2f} | {r['peak_allocated_gib']:.3f} / {r['peak_reserved_gib']:.3f} |")
    report+=['','## 固定设计','',
        '- v31-B的H/V/T、原PatchTST D16（35,427参数）、Stage1/Stage2训练和选模流程沿用。两阶段各100轮、batch128、FP32、学习率0.001/0.0001。Stage1为全局缺失MSE+0.1检索；Stage2为预测MSE+0.2MAE，第11轮起加0.05均匀修复保留。',
        '- 在原三专家融合、解码、序列拼合后增加按原时间坐标工作的残差支路。输入为当前显式修复数值、可见值、原mask、相对位置、按覆盖数折叠的融合latent和none/H/V/T路由权重。没有缺失真值或未来标签输入。',
        '- 支路由64宽度输入投影、两个kernel5/dilation1和3深度卷积残差块、2倍通道MLP及零初始化标量输出构成，所有变量共享。只修改真实缺失位置，可见值/原mask/补齐规则保持原样。',
        '- 最近32点组与全96点组均读取/计算完整96点上下文，参数和计算方式相同，只改变允许修改的输出位置。本轮检验输出范围，不声称实现计算节省或动态重要性分配。',
        '- 支路增加39,041参数，修复器总参数174,362。原专家和预测头的初始化与seed502基线一致，两支路初始参数SHA也一致。',
        '- 先tail32、后full96；每组从头跑完整两阶段并立刻测试20个共享冻结缺失位置。验证选择和晋级先于测试。', '',
        '## 晋级与局限','',
        '固定晋级条件：按原固定验证MAE选中的checkpoint，其验证MSE严格低于原v31-B的0.7445468891775257，MAE不高于0.5852035470560294。两组都独立与原v31-B比较，测试不参与晋级。']
    for r in rows[2:]:report.append(f"- {r['model']}：{'通过' if r['validation_gate_passed'] else '未通过'}固定验证门槛；20组结果如上。")
    if not pending:
        promoted=[r for r in rows[2:] if r['validation_gate_passed']]
        selected=min(promoted,key=lambda r:(r['validation_mae'],r['validation_mse']))['model'] if promoted else 'E23-v31-B'
        report+=['',f'按固定验证规则，本轮开发选择：**{selected}**。无新增多种子复现，不能据此宣称正式击败T1。']
        (ROOT/'final-selection.json').write_text(json.dumps(dict(selected=selected,
            validation_only=True,formal_paper_statistics_unchanged=True,
            gates={r['model']:r['validation_gate_passed'] for r in rows[2:]}),indent=2)+'\n')
    if (ROOT/'selected-repair-diagnostics.json').exists():
        report+=['','## Stage2 修复与支路干预诊断','',
            '| 模型 | 修复MSE | 最近32点修复MSE | 前64点修复MSE | 关闭精修支路后的验证预测MSE / MAE |',
            '|---|---:|---:|---:|---|']
        for r in rows:
            if r['stage2_repair_mse']=='':continue
            report.append(f"| {r['model']} | {r['stage2_repair_mse']:.6f} | {r['stage2_last32_repair_mse']:.6f} | {r['stage2_before64_repair_mse']:.6f} | {r['stage2_without_refiner_forecast_mse']:.6f} / {r['stage2_without_refiner_forecast_mae']:.6f} |")
        report+=['','关闭支路使用同一个已选Stage2权重和固定预测头，仅移除残差；此项只反映该联合训练模型对支路的适应，不是重新训练后的原v31-B对照。额外验证诊断成本单独保存，未计入原模型训练时间。']
        diagnostics=read(ROOT/'selected-repair-diagnostics.json')
        tail=diagnostics['E23-v40-tail32'];original=diagnostics['E23-v31-B'];full=diagnostics['E23-v40-full96']
        report += ['',
            f"最近32点组在Stage1确实改善该区域：修复MSE为{tail['stage1']['last32_mse']:.6f}，原v31-B为{original['stage1']['last32_mse']:.6f}。但选定Stage2后，该区域MSE升至{tail['stage2']['last32_mse']:.6f}，高于原v31-B的{original['stage2']['last32_mse']:.6f}。在同一选定Stage2权重内部，加入残差后最近32点修复MSE由{tail['stage2']['without_refiner_last32_repair_mse']:.6f}变为{tail['stage2']['last32_mse']:.6f}；预测验证MSE却由{tail['stage2']['without_refiner_forecast_mse']:.6f}降到{tail['stage2']['with_refiner_forecast_mse']:.6f}。这表明支路参与了预测共同适应，但没有保住Stage1的局部数值修复改善。", '',
            f"全窗口组Stage1关闭支路时修复MSE为{full['stage1']['without_refiner_repair_mse']:.6f}，启用后为{full['stage1']['mse']:.6f}；它形成了对精修支路的较强依赖。Stage2关闭支路使验证预测MSE由{full['stage2']['with_refiner_forecast_mse']:.6f}升至{full['stage2']['without_refiner_forecast_mse']:.6f}。因此不能把失败解释为新增支路完全没用；整体共同训练方案没有超过基线。", '',
            '下一步值得优先研究精修支路及其上游输入在Stage2的漂移控制，并单独检验如何保留Stage1对预测敏感区域的恢复质量。当前结果没有证明哪种控制方式有效，本轮未改变原Stage2目标或预测头。']
        cost=read(ROOT/'diagnostic-resources.json')
        report += ['',f"新增验证诊断耗时{cost['diagnostic_seconds']:.2f}秒，峰值 allocated/reserved为{cost['peak_allocated_gib']:.3f}/{cost['peak_reserved_gib']:.3f} GiB；没有新训练或额外测试。"]
    report+=['','## 资源与完整性','',
        '完整训练时间包括入口导入、数据准备、模型/优化器构建、全部训练轮次、每轮验证和选模保存，排除最终测试及本次额外诊断。峰值显存为PyTorch allocated/reserved，不等于整张显卡占用。GPU1 RTX5090，串行训练；E23两阶段batch128 FP32，缓存T1为Stage1 batch16 BF16/Stage2 batch128 FP32。','',
        '实现隔离在`.e23-v40-point/E23`，原共享模型和既有报告保持哈希一致。预检48项通过，更新后canary复查通过；两组真实GPU零残差重放、隐藏值隔离和有限梯度通过。原预测头及Stage2构建、损失、训练函数的AST与v31-B一致。','',
        f'[汇总CSV]({NAME}.csv) · [每组测试明细CSV]({NAME}_all_draws.csv)','',
        f'完整产物：`{ROOT}`。包括固定计划、源代码冻结/归档、配置、命令、日志、checkpoint、20组窗口误差及资源侧车。','']
    if len(rows)>2:
        for filename,content in [(NAME+'.csv',csv_text(rows)),(NAME+'_all_draws.csv',csv_text(draws)),(NAME+'.md','\n'.join(report))]:
            (ROOT/filename).write_text(content);(DOCS/filename).write_text(content)
    state=dict(completed_models=[r['model'] for r in rows],pending=pending,rows=rows,protected_files_unchanged=len(protected))
    (ROOT/'latest-results.json').write_text(json.dumps(state,indent=2)+'\n')
    print(json.dumps({'completed':state['completed_models'],'pending':pending,'rows':rows}),flush=True)


if __name__=='__main__':main()
