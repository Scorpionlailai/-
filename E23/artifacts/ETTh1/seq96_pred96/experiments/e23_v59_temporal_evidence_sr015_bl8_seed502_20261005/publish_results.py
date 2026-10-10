"""Publish completed conditions while the sequential study runs, then audit all."""
import csv,hashlib,json,tarfile
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
DOCS=Path('/home/xyw/paper/E23/docs/experiments')
PREFIX=DOCS/'ETTh1_E23_v59_temporal_evidence_sr015_bl8_seed502_20261005'
def read(p):return json.loads(Path(p).read_text())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,text):
    p=Path(p);tmp=p.with_name(p.name+'.tmp');tmp.write_text(text);tmp.replace(p)
def csv_write(p,rows):
    p=Path(p);tmp=p.with_name(p.name+'.tmp');keys=list(dict.fromkeys(k for row in rows for k in row))
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    tmp.replace(p)
def main():
    plan=read(ROOT/'plan.json');definition=read(ROOT/'frozen-definition.json');work=Path(plan['work'])
    for rel,h in definition['sources'].items():assert digest(work/rel)==h
    for p,h in definition['protected'].items():assert digest(p)==h
    assert digest(ROOT/'plan.json')==definition['plan_sha256']
    with (DOCS/'ETTh1_E23_v58_recent_mean_sr015_bl8_seed502_20261005.csv').open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==16
    references={label:read(p) for label,p in plan['cached_references'].items()}
    for row in rows:row.update(cached_reference=True,accepted_on_validation='',only_change='cached',parent='')
    all_results=references.copy();completed=[];paired=[];all_draws=[];all_diag=[];new_seconds=0.;uuids=set()
    for step in plan['sequence']:
        cell=ROOT/step['name']
        if not (cell/'completed-audit.json').exists():continue
        result=read(cell/'completed-audit.json');spec=result['trial'];lock=result['validation_lock']
        scores=read(cell/'evaluation/twenty-test-draws.json');s1=read(cell/'stage1/runs/e23-stage1.json');s2=read(cell/'stage2/runs/e23-stage2.json')
        diag=read(cell/'recovery-diagnostics.json');resources=scores['training_resources'];records=[read(p) for p in resources['resource_record_paths']]
        assert s1['epochs_completed']==s2['epochs_completed']==100 and len(scores['test_draws'])==20
        assert resources['resource_status']=='measured_all_stages' and len(records)==2
        assert s2['predictor_initial_sha256']==plan['predictor_initial_sha256']
        assert read(cell/'parent-choice-before-tests.json')['uses_test_metrics'] is False
        uuids.update(r['device']['uuid'] for r in records);new_seconds+=resources['full_workflow_seconds']
        count=next(r['repair_parameters'] for r in read(ROOT/'preflight.json')['configurations'] if r['config_key']==spec['config_key'])
        bystage={name:v for r in records for name,v in r['stages'].items()}
        row=dict(scenario='sr015_bl8',start_rate=.15,block_len=8,actual_test_missing_fraction=plan['actual_test_missing_fraction'],
            model=step['label'],training_seed=502,seq_len=96,pred_len=96,test_masks=20,
            mse=scores['test_mean']['mse'],mae=scores['test_mean']['mae'],mask_mse_std=scores['test_mask_std_ddof1']['mse'],mask_mae_std=scores['test_mask_std_ddof1']['mae'],
            validation_mse=s2['best_val_mse'],validation_mae=s2['best_val_mae'],full_training_seconds=resources['full_workflow_seconds'],
            peak_allocated_gib=resources['peak_allocated_gib'],peak_reserved_gib=resources['peak_reserved_gib'],gpu=resources['gpu'],training_batch_sizes='128',
            predictor_parameters=35427,predictor_initial_sha256=plan['predictor_initial_sha256'],validation_mask_sha256=scores['validation_mask_sha256'],
            test_bank_sha256=scores['test_mask_bank_sha256'],checkpoint_sha256=s2['checkpoint_sha256'],selection=str(cell/'stage2/runs/e23-stage2.json'),
            evaluation=str(cell/'evaluation/twenty-test-draws.json'),cached_reference=False,stage1_epochs=100,stage2_epochs=100,selected_stage2_epoch=s2['best_epoch'],
            stage1_repair_mse=s1['best_val_repair_mse'],repair_stage_seconds=bystage['repair']['wall_seconds'],forecast_stage_seconds=bystage['forecast']['wall_seconds'],
            stage1_batch_size=128,stage2_batch_size=128,stage1_precision='float32',stage2_precision='float32',stage2_loss=s2['loss'],repair_update_policy=s2['repair_update_policy'],
            repair_parameters=count,front_context_parameters=43200,stage1_reused=False,actual_new_training_seconds=resources['full_workflow_seconds'],point_router=spec['point_router'],overlap_fusion=spec['overlap_fusion'],point_candidate_readout=spec['point_candidate_readout'],variable_calibration=spec['variable_calibration'],front_variable_context=spec['front_variable_context'],repair_coordinates=spec['repair_coordinates'],point_variable_readout=spec['point_variable_readout'],historical_key_policy=spec['historical_key_policy'],recent_mean_projection=spec['recent_mean_projection'],temporal_evidence_readout=spec['temporal_evidence_readout'],
            stage1_structure=spec['stage1_structure'],recent_level_weight=1.0 if spec['stage1_structure']=='recent_mean_v2' else 0,
            recent_trend_weight=0.0,parent=spec['parent']['label'],only_change=spec['only_change'],
            accepted_on_validation=lock['accepted_on_validation'],matched_v41_validation_gate=lock['matched_v41_validation_gate'],
            stage2_repair_mse=diag[1]['metrics']['all96']['mse'],stage2_last32_repair_mse=diag[1]['metrics']['last32']['mse'])
        rows.append(row);completed.append(result);all_results[step['label']]=scores
        for d in diag:all_diag.append(dict(model=step['label'],**{k:v for k,v in d.items() if k!='metrics'},**{name+'_'+metric:v[metric] for name,v in d['metrics'].items() for metric in ['mse','mae']}))
        for label in dict.fromkeys([spec['parent']['label'],'E23-v48-point-router','T1']):
            baseline=all_results[label];differences={m:[] for m in ['mse','mae']}
            for a,b in zip(scores['test_draws'],baseline['test_draws']):
                assert (a['mask_seed'],a['mask_sha256'],a['target_elements'])==(b['mask_seed'],b['mask_sha256'],b['target_elements'])
                for m in differences:differences[m].append(a[m]-b[m])
            paired.append(dict(model=step['label'],baseline=label,mse_delta=float(np.mean(differences['mse'])),mae_delta=float(np.mean(differences['mae'])),
                mse_wins=sum(x<0 for x in differences['mse']),mae_wins=sum(x<0 for x in differences['mae']),test_masks=20,
                mse_improvement_pct=(baseline['test_mean']['mse']-scores['test_mean']['mse'])/baseline['test_mean']['mse']*100))
    for label,result in all_results.items():
        assert result['predictor_initial_sha256']==plan['predictor_initial_sha256']
        assert result['test_mask_bank_sha256']==definition['bank_sha256']
        for d in result['test_draws']:all_draws.append(dict(model=label,training_seed=502,scenario='sr015_bl8',cached_reference=label in references,**d))
    csv_write(PREFIX.with_suffix('.csv'),rows);csv_write(Path(str(PREFIX)+'_all_draws.csv'),all_draws)
    if paired:csv_write(Path(str(PREFIX)+'_paired.csv'),paired)
    if all_diag:csv_write(Path(str(PREFIX)+'_repair_diagnostics.csv'),all_diag)
    lines=['# E23 v59：可见时间证据的多头重复读取','',f'状态：{len(completed)}/1个条件完成。ETTh1 96→96，实际缺失72.9252%，训练seed502，每模型20组已有缺失位置。', '',
        '| 模型 | 测试MSE | 测试MAE | 验证MSE / MAE | 完整训练分钟 | allocated / reserved GiB |','|---|---:|---:|---:|---:|---:|']
    for row in rows:lines.append(f"| {row['model']} | {float(row['mse']):.6f} ± {float(row['mask_mse_std']):.6f} | {float(row['mae']):.6f} ± {float(row['mask_mae_std']):.6f} | {float(row['validation_mse']):.6f} / {float(row['validation_mae']):.6f} | {float(row['full_training_seconds'])/60:.2f} | {float(row['peak_allocated_gib']):.3f} / {float(row['peak_reserved_gib']):.3f} |")
    lines+=['','## 预先规定的顺序和判断','','本项在当前保留父版本上仅增加可见时间证据读取器，继承Stage1/2配方。均从头100轮Stage1＋100轮Stage2。预测checkpoint按固定验证MAE选择；验证预测MSE更低且MAE不差时保留新版本。父版本决定在该项测试之前保存。已有20组位置作为开发测试，未用于本轮自动父版本选择；v47开发起点由用户在查看旧测试后指定。']
    for c in completed:
        s=c['trial'];l=c['validation_lock'];lines+=['',f"- {c['label']}：父版本{s['parent']['label']}；仅新增{s['only_change']}。配置point_router={s['point_router']}、overlap_fusion={s['overlap_fusion']}、stage1_structure={s['stage1_structure']}。选中Stage2第{l['selected_epoch']}轮，父版本验证门槛{'通过' if l['accepted_on_validation'] else '未通过'}，原v41严格门槛{'通过' if l['matched_v41_validation_gate'] else '未通过'}。"]
    if (ROOT/'completed-audit.json').exists():
        final=read(ROOT/'completed-audit.json');retained=final['retained_parent']['label']
        retained_scores=all_results[retained]['test_mean'];initial_scores=references['E23-v47-frozen-front']['test_mean'];t1_scores=references['T1']['test_mean']
        mse_gain=(initial_scores['mse']-retained_scores['mse'])/initial_scores['mse']*100
        t1_gap=(retained_scores['mse']/t1_scores['mse']-1)*100
        lines+=['',f"本序列按验证保留：**{retained}**。其20组测试平均MSE相对v47的改善比例为{mse_gain:+.2f}%，相对T1的MSE差距为{t1_gap:+.2f}%。"]
        test_best=min((r for r in rows if r['cached_reference'] is False),key=lambda r:float(r['mse']))
        lines+=['',f"本项新实验的开发测试MSE：**{test_best['model']}**（MSE {float(test_best['mse']):.6f}，MAE {float(test_best['mae']):.6f}）。该描述不改变测试前锁定的父版本判断。"]
    if all_diag:
        lines+=['','## 固定验证集的修复诊断','','所有修复指标在训练集全局标准化尺度上统计；每个窗口内的时间点只计一次。近期水平指标取修复后窗口最近32点的平均误差。', '',
            '| 模型 | Stage1全窗修复MSE | Stage1最近32点MSE | Stage2全窗修复MSE | Stage2最近32点MSE | Stage2近期水平MSE |',
            '|---|---:|---:|---:|---:|---:|']
        for c in completed:
            d1,d2=[d for d in all_diag if d['model']==c['label']]
            lines.append(f"| {c['label']} | {d1['all96_mse']:.6f} | {d1['last32_mse']:.6f} | {d2['all96_mse']:.6f} | {d2['last32_mse']:.6f} | {d2['last32_level_mse']:.6f} |")
    lines+=['','## 实现与资源口径','','在全部原专家、点读出、重叠融合和tail修复后，近期32点作为query，反复读取同变量整窗96点的原真实可见证据。总宽64，8头，每头8维，共两层，带固定多尺度距离先验和学习到的相对位置偏置。source始终按原观测mask限制，空source使用零值null；修复信息不能变成伪观测。增量在稳健相对坐标限幅±2，只更改原真实近期缺失位置；原观测、前64点和coverage保持。新增83969修复参数。若父版本含v58均值校正，则读取器位于均值校正之前。继承原Stage1全局hidden MSE与近期均值监督，不加入预测头或未来标签。','','原PatchTST预测头源码、配置、初始化保持，参数35427；Stage2训练函数、objective和配方与父版本相同。仍冻结原窗口CNN/Hkey/library，修复lr1e-5、预测器lr1e-4，MSE+0.2MAE及原均匀修复保留。T1保持原保存权重和作者参数。','','同一RTX5090顺序运行。完整训练时间包含两个阶段的准备、训练、每轮验证和保存；最终测试与额外诊断另计。显存为训练阶段PyTorch allocated/reserved峰值。20组标准差只反映缺失位置变化，单训练seed不足以代表训练随机性。', '',f'[汇总CSV]({PREFIX.name}.csv) · [测试位置明细]({PREFIX.name}_all_draws.csv) · [配对差值]({PREFIX.name}_paired.csv) · [修复诊断]({PREFIX.name}_repair_diagnostics.csv)', '',f'冻结代码：`{work}`。完整记录：`{ROOT}`。']
    write(PREFIX.with_suffix('.md'),'\n'.join(lines)+'\n')
    if len(completed)==1:
        head_audit=read(ROOT/'forecast-head-audit.json')
        assert all(head_audit.values())
        with tarfile.open(ROOT/'frozen-source.tar.gz') as ar:
            for rel,h in definition['sources'].items():assert hashlib.sha256(ar.extractfile(rel).read()).hexdigest()==h
        assert len(uuids)==1
        audit=dict(status='complete',source_files_verified=len(definition['sources']),protected_files_verified=len(definition['protected']),
            new_stage1_runs=1,new_stage2_runs=1,new_test_draws=20,cached_reference_draws=320,tests_passed=19,gpu_configurations_checked=1,
            source_archive_sha256=digest(ROOT/'frozen-source.tar.gz'),new_full_training_seconds=new_seconds,physical_gpu_uuid=next(iter(uuids)),
            retained_parent=read(ROOT/'completed-audit.json')['retained_parent'],original_predictor_unchanged=True,forecast_head_audit=head_audit,test_used_for_checkpoint_or_automatic_parent_selection=False)
        write(ROOT/'delivery-audit.json',json.dumps(audit,indent=2)+'\n')
        design=DOCS/'2026-10-05_E23_v59_temporal_evidence_design.md'
        write(design,design.read_text().replace('状态：19项检查和GPU预检查通过；从头训练中。','状态：两阶段100轮及20组测试完成；19项检查、GPU预检查和最终审计通过。'))
    print(json.dumps(dict(conditions_completed=len(completed),rows=len(rows),new_full_training_seconds=new_seconds,new_results=[dict(model=r['model'],mse=r['mse'],mae=r['mae'],validation_mse=r['validation_mse'],validation_mae=r['validation_mae'],accepted_on_validation=r['accepted_on_validation']) for r in rows if r['cached_reference'] is False])),flush=True)
if __name__=='__main__':main()
