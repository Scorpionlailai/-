"""Publish completed seed-502 trials; preserve the five-seed formal comparison."""
import csv
import hashlib
import io
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
FORMAL=ROOT.parent/'e23_t1_five_seed_standard_20261004'
DOCS=Path('/home/xyw/paper/E23/docs/experiments')
NAME='ETTh1_P4_E23_v37_v39_T1_learning_seed502_20261004'

def read(path):return json.loads(Path(path).read_text())

def row(label,evaluation,cell=None):
    data=read(evaluation); resources=data['training_resources']
    if label=='T1':
        selection=read(data['selection_manifest']);repair=read(selection['repair_selection_manifest'])
        repair_mse=repair['best_validation']['mse'];repair_epochs=selection['repair_training']['epochs_completed'] if 'repair_training' in selection else None
    else:
        selection=read(data['selection_manifest'])
        repair=read(selection['stage1_manifest']);repair_mse=repair['best_val_repair_mse'];repair_epochs=repair['epochs_completed']
    resource_records=[read(p) for p in resources['resource_record_paths']]
    stage_times={}
    for rec in resource_records:
        for stage, details in rec['stages'].items():stage_times[stage]=stage_times.get(stage,0)+details['wall_seconds']
    ids={rec['device'].get('uuid') for rec in resource_records}
    batches={stage:rec['batch_size'] for rec in resource_records for stage in rec['stages']}
    if label=='T1':
        batches['repair']=repair['training']['batch_size']
        batches['forecast']=selection['training']['batch_size']
    decision=read(cell/'validation-decision.json') if cell else None
    result=dict(model=label,training_seed=502,test_masks=20,
        mse=data['test_mean']['mse'],mae=data['test_mean']['mae'],
        mask_mse_std=data['test_mask_std_ddof1']['mse'],mask_mae_std=data['test_mask_std_ddof1']['mae'],
        validation_mse=data['validation_replay']['mse'],validation_mae=data['validation_replay']['mae'],
        stage1_validation_repair_mse=repair_mse,
        full_training_seconds=resources['full_workflow_seconds'],
        repair_stage_seconds=stage_times.get('repair'),forecast_stage_seconds=stage_times.get('forecast'),
        peak_allocated_gib=resources['peak_allocated_gib'],peak_reserved_gib=resources['peak_reserved_gib'],
        physical_gpu_consistent=len(ids)==1,gpu_uuid=';'.join(sorted(str(x) for x in ids)),
        repair_batch_size=batches.get('repair'),forecast_batch_size=batches.get('forecast'),
        repair_epochs=repair_epochs or repair.get('epochs_completed'),forecast_epochs=selection.get('epochs_completed'),
        stage1_optimizer_updates=repair.get('optimizer_updates'),
        validation_gate_passed=decision['passed'] if decision else '',
        predictor_initial_sha256=data['predictor_initial_sha256'],checkpoint_sha256=data['checkpoint_sha256'],
        evaluation=str(Path(evaluation).resolve()),scope='Exploratory seed selected using old test outcomes; mask std is not training-seed std')
    if result['repair_epochs'] is None:result['repair_epochs']=read(resources['resource_record_paths'][0]).get('repair_epochs')
    diagnostic=ROOT/'selected-stage2-repair-diagnostic.json'
    diagnostic_labels={'E23-v31-B':'E23-v31-B','E23-v37-batch32':'v37',
                       'E23-v38-shared-features':'v38','E23-v39-two-feature-layers':'v39'}
    metrics={r['model']:r['repair_metrics'] for r in read(diagnostic)} if diagnostic.exists() else {}
    result['stage2_validation_repair_mse']=metrics.get(diagnostic_labels.get(label),{}).get('global_mse','')
    result['stage2_validation_repair_mae']=metrics.get(diagnostic_labels.get(label),{}).get('global_mae','')
    return result,data

def main():
    pairs=[]
    for model in ['E23-v31-B','T1']:
        pairs.append(row(model,FORMAL/f'seed_502/{model}/evaluation/twenty-test-draws.json'))
    pending=[]
    for trial,label in [('v37','E23-v37-batch32'),('v38','E23-v38-shared-features'),('v39','E23-v39-two-feature-layers')]:
        cell=ROOT/trial
        if (cell/'completed-audit.json').exists():pairs.append(row(label,cell/'evaluation/twenty-test-draws.json',cell))
        elif (cell/'deferred.json').exists():pending.append(f"{trial}：{read(cell/'deferred.json')['reason']}")
        else:pending.append(f'{trial}：尚未完成')
    rows=[p[0] for p in pairs]
    shared_hashes=read(ROOT/'protected-existing-files.json')
    changed=[p for p,h in shared_hashes.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
    if changed:raise ValueError(f'Existing shared files changed: {changed}')
    buffer=io.StringIO();writer=csv.DictWriter(buffer,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    draws=[]
    for summary,data in pairs:
        for draw in data['test_draws']:
            draws.append(dict(model=summary['model'],training_seed=502,**{k:draw[k] for k in ['group_id','mask_seed','mask_sha256','mse','mae','windows','target_elements']}))
    detail=io.StringIO();writer=csv.DictWriter(detail,fieldnames=list(draws[0]));writer.writeheader();writer.writerows(draws)
    report=['# E23 v37–v39：seed502 顺序探索','',
        'ETTh1 P4，96→96，原固定验证MAE选模，原PatchTST D16。每项完成两个阶段100轮后，测试同一批20组位置。', '',
        '**seed502 是查看既有五种子测试后选出的开发种子；本表不能替代五种子正式统计。** 原五种子汇总不修改。', '',
        '| 模型 | 测试MSE | 测试MAE | 验证MSE / MAE | Stage1修复MSE | 完整训练秒 | 峰值 allocated / reserved GiB |',
        '|---|---:|---:|---|---:|---:|---|']
    for r in rows:
        report.append(f"| {r['model']} | {r['mse']:.6f} | {r['mae']:.6f} | {r['validation_mse']:.6f} / {r['validation_mae']:.6f} | {r['stage1_validation_repair_mse']:.6f} | {r['full_training_seconds']:.2f} | {r['peak_allocated_gib']:.3f} / {r['peak_reserved_gib']:.3f} |")
    report+=['','## 裁定','',
        '固定晋级规则：选中checkpoint的验证MSE严格降低且MAE不退步，测试不参与晋级。v37仅改变Stage1 batch32；v38增加共享点级时间特征与按特征跨变量传递；v39增加第二个独立特征bank，并逐层刷新。三者都与原v31-B比较；v38同时作为v39的深度消融对照。', '',
        '执行方案有一次明确调整：原先v39以v38通过验证为启动条件；在读取v38最终测试结果前，改为独立运行两层消融，完整检验用户批准的三个方向。方案原件及时间戳调整记录均保留，晋级门槛与选checkpoint规则没有改变。','']
    for r in rows[2:]:report.append(f"- {r['model']}：{'通过' if r['validation_gate_passed'] else '未通过'}固定验证门槛。")
    if len(rows)==5:
        report+=['','**三项完整探索已完成，保留原 E23-v31-B。** 新训练两个阶段各100轮，共60组测试；五种子正式比较保持原样。']
    if (ROOT/'selected-stage2-repair-diagnostic.json').exists():
        report+=['','## 修复改善是否保留到 Stage 2','',
            '以下均为同一固定验证掩码、全局训练标准化尺度的缺失坐标修复MSE；只作诊断，不是额外测试选模。', '',
            '| 模型 | Stage 1 选中修复MSE | Stage 2 选中权重修复MSE |', '|---|---:|---:|']
        for r in rows:
            if r['stage2_validation_repair_mse']!='':report.append(f"| {r['model']} | {r['stage1_validation_repair_mse']:.6f} | {r['stage2_validation_repair_mse']:.6f} |")
        report+=['',
            '- v38微调后修复误差仍低于v31-B，预测MSE却几乎持平；平均修复误差不足以判断预测收益。',
            '- v39在Stage1修复更好，Stage2之后修复和预测均退步；本次两层组合方案未改善最终目标。该结果不能排除其他堆叠或正则化设计。',
            '- v37增加四倍Stage1优化器更新次数，最终预测退步；本次batch32设置不采用。batch改变也影响梯度噪声、对比学习负样本及specialist sampling分组，不能把差异单独归因更新次数。']
    report+=['',*pending,'',
        '资源为完整入口的数据准备、训练、验证、保存时间，排除最终测试；allocated/reserved为PyTorch分配器峰值。v37两个阶段batch为32/128，其他E23为128/128；T1为16 BF16/128 FP32。阶段batch不同会令旧资源聚合器hardware_consistent为false，本表另按GPU UUID核实物理设备一致。', '',
        '所有新模型的原预测头初始化SHA需与seed502基线相同；各版本的训练命令、精确源码、配置、checkpoint、20组窗口误差、验证裁定及资源侧车均保存在artifact。原模型代码与既有CSV校验未变。','',
        f'[汇总CSV]({NAME}.csv) · [20组明细]({NAME}_all_draws.csv)', '',f'产物目录：`{ROOT}`','']
    DOCS.mkdir(exist_ok=True)
    for name,content in [(NAME+'.csv',buffer.getvalue()),(NAME+'_all_draws.csv',detail.getvalue()),(NAME+'.md','\n'.join(report))]:
        (DOCS/name).write_text(content);(ROOT/name).write_text(content)
    state=dict(completed_models=[r['model'] for r in rows],pending=pending,original_files_verified=len(shared_hashes),rows=rows)
    (ROOT/'latest-results.json').write_text(json.dumps(state,indent=2)+'\n')
    print(json.dumps(dict(completed_models=state['completed_models'],pending=pending,summary=str(DOCS/(NAME+'.csv')))))

if __name__=='__main__':main()
