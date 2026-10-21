"""Publishable text-only records; position dispersion and training dispersion stay separate."""
import csv
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
from integrations.v125_ablation.common import write_json,file_sha256
OUT=Path(__file__).resolve().parents[1]
COARSE=Path('/home/xyw/paper/E24/artifacts/v125_pamap2_coarse_seed_stability_20261019')
DIAG=Path('/home/xyw/paper/E24/artifacts/v125_contribution_diagnostics_20261019')
REPO=Path('/home/xyw/paper-e23-v125-ablation')
DEST=REPO/'experiment_results/v125_ablation'


def csv_write(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        writer.writerows({k:json.dumps(v,sort_keys=True,separators=(',',':')) if isinstance(v,(list,dict)) else v for k,v in row.items()} for row in rows)
    tmp.replace(path)


def immutable_copy(source,target):
    if target.exists():
        assert file_sha256(source)==file_sha256(target),('immutable history changed',str(target))
    else:target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)


def build(final=False):
    DEST.mkdir(parents=True,exist_ok=True)
    with (OUT/'export.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _build(final)


def _build(final):
    audit=json.loads((OUT/'source_audit.json').read_text());per_mask=[];strata=[];resource_rows=[];selections=[]
    for arm in ('A0','A1'):
        for seed in (2023,2024,2025):
            for path in ('coarse','calibration64','calibration256','ensemble'):
                folder=(COARSE if path=='coarse' else OUT)/arm/f'seed_{seed}'
                e=folder/'evaluation' if path=='coarse' else folder/'evaluation'/path
                for f in sorted(e.glob('*/result.json')):
                    item=json.loads(f.read_text());b=item['binding'];m=item['metrics']
                    base=dict(dataset='PAMAP2',scene='sr015_bl8',variant=arm,training_seed=seed,mask_seed=b['mask_seed'],
                        split=b['split'],phase=path,selection_rule=b['selection_metric'],selected_epoch=b['selected_epoch'],
                        checkpoint_sha256=b.get('checkpoint_sha256',b.get('branch_checkpoint_sha256')),
                        coarse_checkpoint_sha256=b.get('coarse_checkpoint_sha256',b.get('checkpoint_sha256')),
                        library_sha256=b['history_snapshot_hash'],data_manifest_sha256=b['data_manifest_sha256'],mask_sha256=b['mask_sha256'])
                    for section in ('full96','recent32'):
                        per_mask.append(dict(**base,section=section,**m[section],windows=m['windows'],records=m['records'],continuous_segments=m['continuous_segments']))
                    for row in item['strata']:strata.append(dict(**base,**row))
            if seed in (2024,2025):
                oldselection=json.loads((COARSE/arm/f'seed_{seed}/coarse/selection.json').read_text())
                epochs=[json.loads(f.read_text()) for f in (COARSE/arm/f'seed_{seed}/coarse').glob('epoch_*.json')]
                resource_rows.append(dict(dataset='PAMAP2',scene='sr015_bl8',variant=arm,training_seed=seed,mask_seed=2023,
                    split='train',phase='coarse',epochs=100,optimizer_updates=8500,training_seconds=oldselection['training_seconds'],
                    incremental_training_seconds_this_round=0.,max_allocated_gib=max(x['resources']['max_allocated_gib'] for x in epochs),
                    max_reserved_gib=max(x['resources']['max_reserved_gib'] for x in epochs),peak_memory_scope='isolated coarse phase',
                    reused=True,selection_rule='validation_full96_mse',checkpoint_sha256=oldselection['checkpoint_sha256']))
            else:
                original=Path(audit['sources'][f'{arm}_{seed}']['coarse_checkpoint']).parent
                selection=json.loads((original/'selection.json').read_text());coarsephase=next(x for x in selection['phases'] if x['phase']=='coarse')
                curve=[json.loads(x) for x in (original/'curve.jsonl').read_text().splitlines()];epochs=[x for x in curve if x['subphase']=='coarse']
                resource_rows.append(dict(dataset='PAMAP2',scene='sr015_bl8',variant=arm,training_seed=seed,mask_seed=2023,split='train',phase='coarse',
                    epochs=100,optimizer_updates=8500,training_seconds=coarsephase['seconds'],incremental_training_seconds_this_round=0.,
                    max_allocated_gib=max(x['resources']['max_allocated_gib'] for x in epochs),max_reserved_gib=max(x['resources']['max_reserved_gib'] for x in epochs),
                    peak_memory_scope='historical cumulative stage1 allocator',reused=True,selection_rule='validation_full96_mse',checkpoint_sha256=audit['sources'][f'{arm}_{seed}']['coarse_sha256']))
            for width in (64,256):
                sfile=OUT/arm/f'seed_{seed}'/f'calibration{width}/selection.json'
                if not sfile.exists():continue
                s=json.loads(sfile.read_text());selections.append(s)
                resource_rows.append(dict(dataset='PAMAP2',scene='sr015_bl8',variant=arm,training_seed=seed,mask_seed=2023,split='train',
                    phase=f'calibration{width}',epochs=s['epochs'],optimizer_updates=s['optimizer_steps'],training_seconds=s['training_seconds'],
                    incremental_training_seconds_this_round=0. if s['reused'] else s['training_seconds'],
                    max_allocated_gib=s['resources']['max_allocated_gib'],max_reserved_gib=s['resources']['max_reserved_gib'],
                    peak_memory_scope=s['resources'].get('peak_memory_scope','isolated phase; includes frozen model allocation'),reused=s['reused'],
                    selection_rule=s['selection'],checkpoint_sha256=s['checkpoint_sha256'],coarse_checkpoint_sha256=s['source_binding']['coarse_sha256'],
                    library_sha256=s['source_binding']['library_sha256'],trainable_parameters=s['trainable_parameters'],
                    loop_seconds=s.get('loop_seconds'),frozen_state_unchanged=s['frozen_state_unchanged']))
    per_seed=[];position_std=[];summary=[]
    group_fields=('dataset','scene','variant','training_seed','split','phase','section','selection_rule')
    identities=sorted({tuple(row[k] for k in group_fields) for row in per_mask})
    for identity in identities:
        take=[x for x in per_mask if tuple(x[k] for k in group_fields)==identity];base=dict(zip(group_fields,identity))
        expected=1 if base['split']=='validation' else 20
        if len(take)!=expected:continue
        per_seed.append(dict(**base,mask_seed=2021 if base['split']=='validation' else 'mean_of20_sealed_positions',positions=len(take),
            mse=float(np.mean([x['mse'] for x in take])),mae=float(np.mean([x['mae'] for x in take])),
            selected_epoch=take[0]['selected_epoch'],checkpoint_sha256=take[0]['checkpoint_sha256'],library_sha256=take[0]['library_sha256']))
        position_std.append(dict(**base,mask_seed=2021 if base['split']=='validation' else '2026101801-2026101820',positions=len(take),
            mse_position_std_ddof1=float(np.std([x['mse'] for x in take],ddof=1)) if len(take)>1 else None,
            mae_position_std_ddof1=float(np.std([x['mae'] for x in take],ddof=1)) if len(take)>1 else None))
    summary_fields=('dataset','scene','variant','split','phase','section','selection_rule')
    for identity in sorted({tuple(row[k] for k in summary_fields) for row in per_seed}):
        take=sorted([x for x in per_seed if tuple(x[k] for k in summary_fields)==identity],key=lambda x:x['training_seed'])
        summary.append(dict(**dict(zip(summary_fields,identity)),training_seed=[x['training_seed'] for x in take],
            n_training_seeds=len(take),mask_seed=2021 if take[0]['split']=='validation' else 'per_seed_mean_of20_sealed_positions',
            mse_mean=float(np.mean([x['mse'] for x in take])),mse_training_std_ddof1=float(np.std([x['mse'] for x in take],ddof=1)) if len(take)>1 else None,
            mae_mean=float(np.mean([x['mae'] for x in take])),mae_training_std_ddof1=float(np.std([x['mae'] for x in take],ddof=1)) if len(take)>1 else None))
    gains=[]
    definitions=[('expert_C0','A1','coarse','A0','coarse'),('expert_C1','A1','ensemble','A0','ensemble'),
                 ('calibration_A0','A0','coarse','A0','ensemble'),('calibration_A1','A1','coarse','A1','ensemble')]
    for split in ('validation','test'):
        for section in ('full96','recent32'):
            for seed in (2023,2024,2025):
                lookup={(x['variant'],x['phase']):x for x in per_seed if (x['split'],x['section'],x['training_seed'])==(split,section,seed)}
                for name,ba,bp,ta,tp in definitions:
                    if (ba,bp) not in lookup or (ta,tp) not in lookup:continue
                    b,t=lookup[ba,bp],lookup[ta,tp]
                    for key in ('mse','mae'):
                        gains.append(dict(dataset='PAMAP2',scene='sr015_bl8',training_seed=seed,mask_seed=2021 if split=='validation' else 'mean_of20_sealed_positions',
                            split=split,section=section,metric=key,gain=name,baseline_variant=ba,baseline_phase=bp,treatment_variant=ta,treatment_phase=tp,
                            baseline_error=b[key],treatment_error=t[key],absolute_gain=b[key]-t[key],relative_gain_percent=(b[key]-t[key])/b[key]*100 if b[key] else None,
                            selection_rule='C0:validation_full96_mse; C1:fixed.5/.5 of validation_full96_mae-selected branches'))
    gain_summary=[]
    for identity in sorted({(x['split'],x['section'],x['metric'],x['gain']) for x in gains}):
        take=sorted([x for x in gains if (x['split'],x['section'],x['metric'],x['gain'])==identity],key=lambda x:x['training_seed'])
        b=np.mean([x['baseline_error'] for x in take]);t=np.mean([x['treatment_error'] for x in take]);values=[x['absolute_gain'] for x in take]
        gain_summary.append(dict(dataset='PAMAP2',scene='sr015_bl8',**dict(zip(('split','section','metric','gain'),identity)),
            training_seeds=[x['training_seed'] for x in take],n_training_seeds=len(take),baseline_mean=float(b),treatment_mean=float(t),
            absolute_gain=float(b-t),relative_gain_percent=float((b-t)/b*100) if b else None,
            paired_difference_std_ddof1=float(np.std(values,ddof=1)) if len(values)>1 else None,paired_differences=values))
    strata_per_seed=[];strata_summary=[];strata_gains=[]
    sfields=('dataset','scene','variant','training_seed','split','phase','section','family','group','selection_rule')
    buckets={}
    for row in strata:buckets.setdefault(tuple(row[k] for k in sfields),[]).append(row)
    for identity,take in sorted(buckets.items()):
        base=dict(zip(sfields,identity));expected=1 if base['split']=='validation' else 20
        if len(take)!=expected:continue
        available=[x for x in take if x['mse'] is not None]
        strata_per_seed.append(dict(**base,mask_seed=2021 if base['split']=='validation' else 'mean_of20_sealed_positions',
            positions_total=expected,positions_nonempty=len(available),
            mse=float(np.mean([x['mse'] for x in available])) if available else None,
            mae=float(np.mean([x['mae'] for x in available])) if available else None,
            coordinates_per_position=[x['coordinates'] for x in take],windows_per_position=[x['windows'] for x in take],
            records_per_position=[x['records'] for x in take],mask_seeds=[x['mask_seed'] for x in take]))
    sf=('dataset','scene','variant','split','phase','section','family','group','selection_rule');sb={}
    for row in strata_per_seed:sb.setdefault(tuple(row[k] for k in sf),[]).append(row)
    for identity,take in sorted(sb.items()):
        available=[x for x in take if x['mse'] is not None];base=dict(zip(sf,identity))
        strata_summary.append(dict(**base,training_seed=[x['training_seed'] for x in take],n_training_seeds_total=len(take),
            n_training_seeds_nonempty=len(available),
            mse_mean=float(np.mean([x['mse'] for x in available])) if available else None,
            mae_mean=float(np.mean([x['mae'] for x in available])) if available else None,
            mse_training_std_ddof1=float(np.std([x['mse'] for x in available],ddof=1)) if len(available)>1 else None,
            mae_training_std_ddof1=float(np.std([x['mae'] for x in available],ddof=1)) if len(available)>1 else None))
    gain_buckets={}
    for row in strata_per_seed:gain_buckets.setdefault(tuple(row[k] for k in ('training_seed','split','section','family','group')), {})[row['variant'],row['phase']]=row
    for identity,lookup in sorted(gain_buckets.items()):
        for name,ba,bp,ta,tp in definitions:
            if (ba,bp) not in lookup or (ta,tp) not in lookup:continue
            b,t=lookup[ba,bp],lookup[ta,tp]
            assert b['coordinates_per_position']==t['coordinates_per_position']
            for key in ('mse','mae'):
                x,y=b[key],t[key]
                strata_gains.append(dict(dataset='PAMAP2',scene='sr015_bl8',**dict(zip(('training_seed','split','section','family','group'),identity)),
                    metric=key,gain=name,baseline_variant=ba,baseline_phase=bp,treatment_variant=ta,treatment_phase=tp,
                    mask_seed=2021 if b['split']=='validation' else 'mean_of20_sealed_positions',
                    baseline_error=x,treatment_error=y,absolute_gain=x-y if x is not None and y is not None else None,
                    relative_gain_percent=(x-y)/x*100 if x and y is not None else None,
                    positions_nonempty=b['positions_nonempty'],positions_total=b['positions_total'],
                    coordinates_per_position=b['coordinates_per_position'],windows_per_position=b['windows_per_position'],records_per_position=b['records_per_position']))
    csv_write(DEST/'stratified_per_seed.csv',strata_per_seed)
    csv_write(DEST/'stratified_summary.csv',strata_summary)
    csv_write(DEST/'stratified_contributions_per_seed.csv',strata_gains)
    complete=len(per_seed)==96 and all((OUT/arm/f'seed_{seed}'/'evaluation_complete.json').exists() for arm in ('A0','A1') for seed in (2023,2024,2025))
    if final:assert complete,'refusing final publication with incomplete models'
    for name,rows in [('summary.csv',summary),('per_seed.csv',per_seed),('per_mask.csv',per_mask),('stratified.csv',strata),
                      ('resources.csv',resource_rows),('position_dispersion.csv',position_std),('contributions_per_seed.csv',gains),('contributions_summary.csv',gain_summary)]:csv_write(DEST/name,rows)
    protocol=dict(schema_version=1,current_round='calibration_confirmation_20261020',**audit['protocol'],training=audit['training'],
        protocol_sha256=audit['protocol_sha256'],training_config_sha256=audit['training_config_sha256'],
        scope='A0/A1 coarse + original64/256 + fixed.5/.5; three modelseeds; noStage2 orA4',
        aggregation='mean20positions per trainingseed, then mean and sampleSD(ddof1) across3 trainingseeds; positionSD stored separately',
        split_independence='train subjects101-106; validation107; test108-109',test_usage='report stability only, never pick epoch/seed/path/weight',
        initialization='own coarse-inherited mean allowed to differ; novel64/256 parameters identical within modelseed acrossA0/A1',
        reuse='strict source coarse/library/protocol/training config hash audit; seed2023 only currently eligible',
        caveats=['phase contribution diagnosis; unequal training cost/capacity','differences are not additive independent causal effects',
                 'three trainingseeds are preliminary stability evidence, not significance','overlapping windows and20masks are not independent training replicates',
                 'historical phase-only GPU memory cannot be recovered; historical cumulative peaks explicitly labeled'])
    write_json(DEST/'protocol.json',protocol)
    for name in ('source_audit.json','design_lock.json','preflight.json','baseline_tests.json','final_verification.json'):
        if (OUT/name).exists():shutil.copyfile(OUT/name,DEST/'provenance'/name)
    immutable_copy(COARSE/'report.md',DEST/'reports/round1_coarse_three_seeds_20261020.md')
    immutable_copy(COARSE/'final_verification_20261020.json',DEST/'provenance/round1_final_verification.json')
    immutable_copy(COARSE/'runtime_revision.json',DEST/'provenance/round1_runtime_revision.json')
    immutable_copy(COARSE/'design_lock.json',DEST/'provenance/round1_design_lock.json')
    immutable_copy(DIAG/'report.md',DEST/'reports/round0_readonly_diagnosis_20261019.md')
    for name in ('diagnostic_design.json','four_cell_gains.csv','existing_key_variant_validation.csv','phase_cost_capacity.csv'):
        immutable_copy(DIAG/name,DEST/'provenance/round0'/name)
    for p in (OUT/'tools').glob('*.py'):
        target=DEST/'provenance/execution_source'/p.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    write_json(DEST/'provenance/source_hashes.json',dict(source_audit_sha256=file_sha256(OUT/'source_audit.json'),
        original_source_lock_sha256=audit['protocol']['source_lock_sha256'],
        execution_sources={p.name:file_sha256(p) for p in (OUT/'tools').glob('*.py')},selection_provenance=selections))
    lines=['# v125 校准贡献确认实验','', '状态：'+('全部完成，等待审核。' if complete else '运行中；部分结果不构成三seed最终结论。'),'',
        'PAMAP2，0.15/8；A0/A1，各seed2023、2024、2025。仅接原64/256校准；各分支100轮、8500步，原MAE损失与验证MAE选模，固定0.5/0.5组合。',
        'C0保留粗阶段原均值层。C1按原forward采用从自身粗均值层继承的分支均值参数和原dense校准；不把A0校准接到A1上，也不改变冻结的公共前端、专家、检索键或原型库。', '',
        '|划分|范围|路径|变体|训练seed数|MSE均值±训练SD|MAE均值±训练SD|','|---|---|---|---|---|---|---|']
    for row in summary:
        def fmt(key):
            m=row[key+'_mean'];sd=row[key+'_training_std_ddof1'];return f'{m:.6f} ± {sd:.6f}' if sd is not None else f'{m:.6f}（仅1次）'
        lines.append(f"|{row['split']}|{row['section']}|{row['phase']}|{row['variant']}|{row['n_training_seeds']}|{fmt('mse')}|{fmt('mae')}|")
    lines+=['','|划分|范围|指标|贡献诊断|训练seed数|绝对收益|相对收益|三对差值|','|---|---|---|---|---|---|---|---|']
    for row in gain_summary:
        lines.append(f"|{row['split']}|{row['section']}|{row['metric']}|{row['gain']}|{row['n_training_seeds']}|{row['absolute_gain']:.6f}|{row['relative_gain_percent']:.3f}%|{[round(v,6) for v in row['paired_differences']]}|")
    if complete:
        lines+=['','跨seed与划分的描述性判断：']
        for split in ('validation','test'):
            for name in ('expert_C0','expert_C1','calibration_A0','calibration_A1'):
                take=[x for x in gains if x['split']==split and x['section']=='full96' and x['gain']==name]
                counts={key:sum(x['absolute_gain']>0 for x in take if x['metric']==key) for key in ('mse','mae')}
                lines.append(f"- {split} {name}：MSE正收益{counts['mse']}/3，MAE正收益{counts['mae']}/3；所有负差值与分层保留。")
        lines+=['三seed只作为初步稳定性证据；专家在粗阶段有效而校准后差距收窄，需要依据上述配对幅度描述，不能据此断言校准独立取代专家。下一结构复验仍优先A4；本轮完成后不自动启动。']
    lines+=['','收益为基准误差减处理后误差；相对收益以基准误差为分母。每seed先平均20位置；跨seed样本SD与位置SD分开保存。这些差值不作严格可加的独立因果解释。',
        '完整的每位置、每seed、每条分支和所有预登记分层（包括空组）均保留。原生缺失不作为真值，观测不变，无patch复制计分。',
        '同seed的继承均值参数来自各自粗权重，可能不同；新增校准参数初值严格一致。校准只训练原member，粗前缀与库逐轮核验冻结。',
        '2023校准严格核验后复用；历史训练时间与本轮新增时间分列。历史显存只能给累计阶段峰值，未伪造独立分支峰值。',
        '阶段贡献诊断的训练成本与容量不同；三seed只能提供初步稳定性证据。A4、其他变体和Stage2持续暂停；不按测试选择seed、epoch或分支。']
    report='\n'.join(lines)+'\n';(OUT/'report.md').write_text(report)
    (DEST/'reports/current_calibration_status.md').write_text(report)
    if complete:
        if not (DEST/'reports/round2_calibration_confirmation_20261020.md').exists():(DEST/'reports/round2_calibration_confirmation_20261020.md').write_text(report)
        immutable_copy(DEST/'protocol.json',DEST/'provenance/round2_protocol.json')
    readonly=DIAG/'report.md'
    readme=['# v125 消融实验记录','', '**当前完成范围**：PAMAP2 0.15/8，A0/A1粗阶段各3训练seed已完成。校准贡献确认：'+('全部完成，等待审核。' if complete else '已批准，运行中。'),'',
        '校准范围为原64/256各100轮、固定0.5/0.5。2023复用须匹配粗权重、库及协议/配置哈希；2024/2025新增。A4、其他变体和Stage2未启动。', '',
        '**已知结论与负结果**：粗阶段三seed在验证和20组测试均有专家收益，测试MSE均值A0=0.335862、A1=0.356587，约5.81%；仍未证明三个专家分别必要、点级读取或串行交互更优。既有去T在验证/测试方向反转，保留旧诊断并列为后续优先复验对象。三seed不等于显著性证据。',
        '校准是否缩小专家收益、是否在某些seed或区域无益/变差，必须连同所有单分支、双分支和分层报告一起审核；不挑测试最佳分支。', '',
        '**文件**：summary.csv为跨训练seed均值/样本SD；per_seed.csv为每训练seed的20位置均值；per_mask.csv含固定验证及每个测试位置；position_dispersion.csv单独保存位置SD；stratified.csv保留全量分层及坐标/窗口/记录数；resources.csv含训练时间、显存及实际更新次数；contributions_*.csv为四类贡献差值；provenance保存配置和来源hash；reports保留各轮历史。', '',
        '**协议与变化**：数据、受试者划分、全局标准化、原生有效掩码和封存20测试位置保持原协议。训练顺序和人工掩码seed2023；模型/校准RNGseed2023/2024/2025。固定验证集选模，测试只报告稳定性。粗阶段按验证MSE，校准分支按验证MAE；双分支固定等权。',
        '粗阶段执行曾因回放日志重复字段停止，修复记录及原设计锁保留；训练权重未受影响，无新增架构。校准继承各自粗均值参数；仅新引入参数要求同seed跨变体一致。历史累计显存与新分支独立峰值明确区分。', '',
        '**待办**：'+('审核校准前后专家增益、每个模型校准收益及全部分层；再决定后续，不自动开新实验。' if complete else '完成受限校准队列、全部回放、最终核验及本轮归档上传；完成后停止。'),
        '下一结构对照优先A4去T；A5/A6/A7暂不扩展多seed。保留原预测头；以后Stage2仍须整个修复器冻结、eval、纯MSE。', '',
        '**统计解释**：先每训练seed等权平均20测试位置，再对三个训练均值取平均和样本SD（ddof=1）；位置SD与训练SD分离。贡献差值是阶段诊断，不是等容量/等预算或严格可加的因果分解。空分组保留，不以零误差填充。', '',
        '**上传边界**：仅文本、CSV、JSON及执行脚本。权重、数据、预测缓存留在实验服务器；源码hash和checkpoint/library hash足以审计来源，完整复现仍需对应本地资产。', '',
        '[当前校准报告](reports/current_calibration_status.md) · [粗阶段三seed报告](reports/round1_coarse_three_seeds_20261020.md) · [最初只读诊断](reports/round0_readonly_diagnosis_20261019.md)']
    (DEST/'README.md').write_text('\n'.join(readme)+'\n')
    state=dict(complete=complete,updated_unix=time.time(),per_mask_rows=len(per_mask),per_seed_rows=len(per_seed),stratified_rows=len(strata),
        resources_rows=len(resource_rows),new_training_branches=sum(not x['reused'] for x in selections),reused_training_branches=sum(x['reused'] for x in selections))
    write_json(OUT/'report_status.json',state)
    if final:
        for p in DEST.rglob('*'):
            if p.is_file():assert p.suffix in ('.md','.csv','.json','.py'),('unexpected publish artifact',str(p))
        # Only the explicitly authorized structured result directory is staged.
        subprocess.run(['git','add','--','experiment_results/v125_ablation'],cwd=REPO,check=True)
        diff=subprocess.run(['git','diff','--cached','--name-only'],cwd=REPO,check=True,text=True,capture_output=True).stdout.splitlines()
        assert all(x.startswith('experiment_results/v125_ablation/') for x in diff),'unrelated staged files'
        if diff:subprocess.run(['git','commit','-m','Record v125 calibration contribution confirmation across three seeds'],cwd=REPO,check=True)
        branch=subprocess.run(['git','branch','--show-current'],cwd=REPO,check=True,text=True,capture_output=True).stdout.strip()
        assert branch=='experiments/e23-v125-ablation-20261018'
        subprocess.run(['git','push','origin','HEAD:refs/heads/'+branch],cwd=REPO,check=True)
        commit=subprocess.run(['git','rev-parse','HEAD'],cwd=REPO,check=True,text=True,capture_output=True).stdout.strip()
        write_json(OUT/'publication.json',dict(commit_sha=commit,branch=branch,complete=True,
            directory_url=f'https://github.com/Scorpionlailai/-/tree/{commit}/experiment_results/v125_ablation',uploaded_unix=time.time()))
    return state

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');a=p.parse_args();print(json.dumps(build(a.final)))
