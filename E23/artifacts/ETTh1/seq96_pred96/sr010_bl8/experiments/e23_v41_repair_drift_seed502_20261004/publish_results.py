"""Saved-score summary; never reruns training or evaluation."""
import csv
import hashlib
import io
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
FORMAL=ROOT.parent/'e23_t1_five_seed_standard_20261004'
V40=ROOT.parent/'e23_v40_local_point_refinement_seed502_20261004'
DOCS=Path('/home/xyw/paper/E23/docs/experiments')
NAME='ETTh1_P4_E23_v41_repair_drift_seed502_20261004'


def read(path):return json.loads(Path(path).read_text())
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def csv_text(rows):
    stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
    writer.writeheader();writer.writerows(rows);return stream.getvalue()


def row(label,path,cell=None):
    data=read(path);resources=data['training_resources'];selection=read(data['selection_manifest'])
    stage1=read(selection['repair_selection_manifest'] if label=='T1' else selection['stage1_manifest'])
    records=[read(p) for p in resources['resource_record_paths']];stages={}
    for record in records:
        for stage,details in record['stages'].items():stages[stage]=stages.get(stage,0)+details['wall_seconds']
    new_records=[r for r in records if cell and Path(r['selection_manifest']).is_relative_to(cell)]
    decision=read(cell/'validation-decision.json') if cell else None
    diagnostics=read(ROOT/'selected-repair-diagnostics.json') if (ROOT/'selected-repair-diagnostics.json').exists() else {}
    old_diagnostics=read(V40/'selected-repair-diagnostics.json')
    diag=diagnostics.get(label,old_diagnostics.get(label,{}))
    s1=diag.get('stage1',{});s2=diag.get('stage2',{})
    result=dict(model=label,training_seed=502,test_masks=20,mse=data['test_mean']['mse'],mae=data['test_mean']['mae'],
        mask_mse_std=data['test_mask_std_ddof1']['mse'],mask_mae_std=data['test_mask_std_ddof1']['mae'],
        validation_mse=data['validation_replay']['mse'],validation_mae=data['validation_replay']['mae'],
        repair_update_policy=selection.get('repair_update_policy','joint'),predictor_learning_rate=selection.get('predictor_learning_rate',.0001),
        repair_learning_rate=selection.get('repair_learning_rate',.0001),stage1_reused=bool(cell),
        stage1_repair_mse=stage1['best_validation']['mse'] if label=='T1' else stage1['best_val_repair_mse'],
        stage1_last32_repair_mse=s1.get('last32_mse',''),stage2_repair_mse=s2.get('mse',''),
        stage2_last32_repair_mse=s2.get('last32_mse',''),stage2_before64_repair_mse=s2.get('before64_mse',''),
        stage2_output_drift_mse=diag.get('output_drift_mse',''),repair_state_identical_to_stage1=diag.get('repair_state_identical',''),
        full_training_seconds=resources['full_workflow_seconds'],repair_stage_seconds=stages.get('repair'),forecast_stage_seconds=stages.get('forecast'),
        actual_new_training_seconds=sum(r['full_workflow_seconds'] for r in new_records),
        peak_allocated_gib=resources['peak_allocated_gib'],peak_reserved_gib=resources['peak_reserved_gib'],
        new_stage2_peak_allocated_gib=max((r['peak_allocated_bytes']/1024**3 for r in new_records),default=''),
        new_stage2_peak_reserved_gib=max((r['peak_reserved_bytes']/1024**3 for r in new_records),default=''),
        physical_gpu_consistent=len({r['device'].get('uuid') for r in records})==1,
        gpu_uuid=';'.join(sorted({str(r['device'].get('uuid')) for r in records})),
        stage1_epochs=stage1['epochs_completed'],stage2_epochs=selection['epochs_completed'],selected_stage2_epoch=selection['best_epoch'],
        point_refiner_span=selection.get('point_refiner_span',0),predictor_parameters=35427,
        validation_gate_passed=decision['passed'] if decision else '',
        predictor_initial_sha256=data['predictor_initial_sha256'],checkpoint_sha256=data['checkpoint_sha256'],
        evaluation=str(path),scope='Exploratory seed selected using old test outcomes; mask std is not training-seed std')
    return result,data


def main():
    for path,h in read(ROOT/'protected-existing-files.json').items():assert digest(path)==h, path
    pairs=[row(label,FORMAL/f'seed_502/{label}/evaluation/twenty-test-draws.json') for label in ['E23-v31-B','T1']]
    pairs.append(row('E23-v40-tail32',V40/'tail32/evaluation/twenty-test-draws.json'))
    pending=[]
    for trial in read(ROOT/'plan.json')['trials']:
        cell=ROOT/trial['trial']
        if (cell/'completed-audit.json').exists():pairs.append(row(trial['label'],cell/'evaluation/twenty-test-draws.json',cell))
        else:pending.append(trial['trial'])
    rows=[r for r,d in pairs];draws=[]
    for r,data in pairs:
        for draw in data['test_draws']:draws.append(dict(model=r['model'],training_seed=502,
            **{k:draw[k] for k in ['group_id','mask_seed','mask_sha256','mse','mae','windows','target_elements']}))
    report=['# E23 v41：Stage2修复漂移控制','',
        'ETTh1 P4，96→96，seed502，20组共享冻结测试缺失位置。状态：'+('全部完成。' if not pending else '待完成：'+', '.join(pending)+'。'), '',
        '**seed502依据已见的五种子测试结果选为开发种子，本表是探索结果，不能替代正式五种子统计。** 既有v31-B、T1、v40-tail32结果直接读取；本轮三个模型复用原Stage1权重，只新训练Stage2。','',
        '五种子×20组测试均值中，E23-v31-B的MSE和MAE最低都为seed502（0.408581/0.423311）；T1的MSE最低为seed502（0.388985），MAE最低为seed302（0.416181）。E23若按验证MAE则seed302最佳（0.583533）。开发种子选择依据与正式多种子平均分开记录。','',
        '| 模型 | 测试MSE | 测试MAE | 验证MSE / MAE | 完整训练分钟 | 本轮新增训练分钟 | 峰值 allocated / reserved GiB |',
        '|---|---:|---:|---|---:|---:|---|']
    for r in rows:report.append(f"| {r['model']} | {r['mse']:.6f} | {r['mae']:.6f} | {r['validation_mse']:.6f} / {r['validation_mae']:.6f} | {r['full_training_seconds']/60:.2f} | {r['actual_new_training_seconds']/60:.2f} | {r['peak_allocated_gib']:.3f} / {r['peak_reserved_gib']:.3f} |")
    report+=['','## 固定实验设计','',
        '- base-frozen：冻结原v31-B Stage1修复器；tail32-frozen：冻结v40最近32点精修版的Stage1修复器。修复器保持eval模式，全部参数无梯度，每轮哈希检查，最终加载逐张量验证与Stage1一致。只训练原预测头。',
        '- tail32-slow：同一个v40-tail32 Stage1；修复器学习率降至1e-5，原预测头仍为1e-4。原H库、检索键和key encoder继续冻结。',
        '- 原PatchTST D16结构、35,427参数、seed502初始SHA、显式预测接口、输入mask、归一化和补齐规则不变。本轮改变的是修复器更新策略；不把它描述为与既有联合训练完全相同的Stage2算法。',
        '- Stage2仍为100轮、batch128、FP32、Adam、weight decay0、预测MSE+0.2MAE；第11轮起0.05均匀全局修复MSE保留不变。冻结组中的修复保留项没有可训练修复梯度。',
        '- 每组从原Stage1和原seed502预测头初始化出发，不恢复Stage2优化器状态；按原固定验证MAE选checkpoint，先锁定验证决定，再测试20组位置。',
        '- 晋级门槛：选定验证MSE<0.7445468891775257且MAE≤0.5852035470560294。测试不参与晋级；每组无论是否通过验证门槛都按用户要求测试。',
        '- base-frozen用于检验单独冻结是否有效；tail32-frozen与它对照，检验固定修复输出下新增逐点精修的收益；tail32-slow检验有限共同适应。','']
    if (ROOT/'selected-repair-diagnostics.json').exists():
        diagnostics=read(ROOT/'selected-repair-diagnostics.json')
        old_diagnostics=read(V40/'selected-repair-diagnostics.json')
        report+=['## 固定验证集修复诊断','',
            '| 模型 | Stage1修复MSE | Stage2修复MSE | Stage1最近32点MSE | Stage2最近32点MSE | Stage2相对Stage1输出变化MSE |',
            '|---|---:|---:|---:|---:|---:|']
        for r in rows:
            if r['stage2_repair_mse']=='':continue
            drift=f"{r['stage2_output_drift_mse']:.6f}" if r['stage2_output_drift_mse']!='' else '未测'
            report.append(f"| {r['model']} | {r['stage1_repair_mse']:.6f} | {r['stage2_repair_mse']:.6f} | {r['stage1_last32_repair_mse']:.6f} | {r['stage2_last32_repair_mse']:.6f} | {drift} |")
        report+=['','修复误差和输出变化均只计算原缺失坐标，以训练集全局标准化尺度统计；可见值保持原样。以上为固定验证诊断，未额外训练或测试。']
        slow=diagnostics['E23-v41-tail32-slow'];joint=old_diagnostics['E23-v40-tail32']
        report+=['',
            f"冻结两组的Stage1/Stage2修复权重逐张量相同，固定验证上的输出变化MSE为0。慢速微调组的Stage2全局修复MSE为{slow['stage2']['mse']:.6f}，原v40联合训练为{joint['stage2']['mse']:.6f}；最近32点分别为{slow['stage2']['last32_mse']:.6f}和{joint['stage2']['last32_mse']:.6f}。学习率降低减少了选定模型的修复误差，但最近32点仍高于其Stage1的{slow['stage1']['last32_mse']:.6f}，未完全保住局部修复改善。", '',
            '完全冻结降低了测试MAE，却没有通过验证MSE门槛；慢速微调通过了两项验证门槛。这支持继续研究修复保留与预测适应之间的平衡，不能据此断言修复漂移是全部差距的唯一原因。', '',
            f"额外验证诊断耗时{read(ROOT/'diagnostic-resources.json')['diagnostic_seconds']:.2f}秒，单独记录，没有新训练或新测试。"]
    if not pending:
        promoted=[r for r in rows[3:] if r['validation_gate_passed']]
        selected=min(promoted,key=lambda r:(r['validation_mae'],r['validation_mse']))['model'] if promoted else 'E23-v31-B'
        report+=['','## 结论','',f'按预先固定的验证规则，本轮选择：**{selected}**。']
        for r in rows[3:]:report.append(f"- {r['model']}：{'通过' if r['validation_gate_passed'] else '未通过'}验证门槛；测试MSE相对v31-B变化{r['mse']-rows[0]['mse']:+.6f}，MAE变化{r['mae']-rows[0]['mae']:+.6f}。")
        slow=rows[-1];base=rows[0];t1=rows[1]
        report+=['',
            f"慢速微调版相对原v31-B，测试MSE下降{(base['mse']-slow['mse'])/base['mse']*100:.2f}%，MAE下降{(base['mae']-slow['mae'])/base['mae']*100:.2f}%；与同种子T1的MSE差距由{base['mse']-t1['mse']:.6f}缩至{slow['mse']-t1['mse']:.6f}，缩小{(1-(slow['mse']-t1['mse'])/(base['mse']-t1['mse']))*100:.2f}%。但仍没有超过T1，MAE亦略高于T1。两组冻结版虽测试MAE更低，未通过事先固定的验证MSE要求，不用测试结果重定晋级规则。", '',
            'v41-tail32-slow仅是本轮seed502的开发候选；正式五种子论文对照仍保留原v31-B记录，尚未新增v41多种子复现。']
        (ROOT/'final-selection.json').write_text(json.dumps(dict(selected=selected,validation_only=True,
            gates={r['model']:r['validation_gate_passed'] for r in rows[3:]},formal_paper_statistics_unchanged=True),indent=2)+'\n')
    report+=['','## 时间、显存与可复查产物','',
        '完整训练时间=各自原Stage1全流程实测时间+本轮Stage2全流程实测时间，包括导入、数据准备、全部训练、验证、选模保存，排除最终测试和新增诊断。复用Stage1不会记成本轮新增时间；CSV同时列出本轮新增Stage2时间和Stage2显存。既有三行新增时间为0。','',
        '峰值显存为两阶段PyTorch allocated/reserved最大值，非整卡占用；同一物理GPU1 RTX5090串行训练。冻结版Stage2显存可能下降，但完整模型仍需承担原Stage1峰值。T1作者训练配置仍为Stage1 batch16/BF16、Stage2 batch128/FP32，E23为batch128/FP32。','',
        '实现隔离在`.e23-v41-drift/E23`。25项相关测试、三组真实GPU短步检查通过；冻结repair逐张量与Stage1一致、原预测头和forecast objective验证通过。既有代码与报告逐文件哈希保护。','',
        f'[汇总CSV]({NAME}.csv) · [逐组测试CSV]({NAME}_all_draws.csv)','',f'完整产物：`{ROOT}`。','']
    for name,content in [(NAME+'.csv',csv_text(rows)),(NAME+'_all_draws.csv',csv_text(draws)),(NAME+'.md','\n'.join(report))]:
        (ROOT/name).write_text(content);(DOCS/name).write_text(content)
    (ROOT/'latest-results.json').write_text(json.dumps(dict(rows=rows,pending=pending),indent=2)+'\n')
    print(json.dumps(dict(completed=[r['model'] for r in rows],pending=pending,
        results=[{k:r[k] for k in ['model','mse','mae','validation_mse','validation_mae','full_training_seconds','actual_new_training_seconds','peak_allocated_gib','peak_reserved_gib']} for r in rows])),flush=True)


if __name__=='__main__':main()
