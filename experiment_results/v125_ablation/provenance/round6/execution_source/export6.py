"""Append A5 to commit-pinned tables, archive round6 and publish only text results."""
import csv
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
from policy6 import OUT,REPO,BASE,COARSE,CAL
from integrations.v125_ablation.common import write_json,file_sha256
DEST=REPO/'experiment_results/v125_ablation'


def read_csv(p):
    with Path(p).open() as f:return list(csv.DictReader(f))


def csv_write(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r));path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader()
        w.writerows({k:json.dumps(v,sort_keys=True,separators=(',',':')) if isinstance(v,(list,dict)) else v for k,v in row.items()} for row in rows)
    tmp.replace(path)


def stat(values):
    values=[float(v) for v in values if v is not None and v!='']
    return dict(mean=float(np.mean(values)) if values else None,std=float(np.std(values,ddof=1)) if len(values)>1 else None,n=len(values))


def phase_info(seed,phase,audit):
    if seed==2023:return audit['A5_seed2023']['phases'][phase]
    p=OUT/'A5'/f'seed_{seed}'/phase/'selection.json'
    if not p.exists():return None
    x=json.loads(p.read_text())
    if 'validation' not in x:
        key=f"epoch_{x['selected_epoch']:03}.json";x['validation']=json.loads((p.parent/key).read_text())['validation']
    x['selection_rule']=x.get('selection_rule',x.get('selection'));return x


def publish(message):
    subprocess.run(['git','add','--','experiment_results/v125_ablation'],cwd=REPO,check=True)
    staged=subprocess.run(['git','diff','--cached','--name-only'],cwd=REPO,text=True,capture_output=True,check=True).stdout.splitlines()
    assert all(x.startswith('experiment_results/v125_ablation/') for x in staged)
    for p in DEST.rglob('*'):
        if p.is_file():assert p.suffix in ('.md','.csv','.json','.py'),str(p)
    if staged:subprocess.run(['git','commit','-m',message],cwd=REPO,check=True)
    branch=subprocess.run(['git','branch','--show-current'],cwd=REPO,text=True,capture_output=True,check=True).stdout.strip()
    assert branch=='experiments/e23-v125-ablation-20261018'
    subprocess.run(['git','push','origin','HEAD:refs/heads/'+branch],cwd=REPO,check=True)
    commit=subprocess.run(['git','rev-parse','HEAD'],cwd=REPO,text=True,capture_output=True,check=True).stdout.strip()
    remote=subprocess.run(['git','ls-remote','origin','refs/heads/'+branch],cwd=REPO,text=True,capture_output=True,check=True).stdout.split()[0];assert remote==commit
    return dict(commit_sha=commit,branch=branch,remote_verified=True,baseline_commit=BASE,
        directory_url=f'https://github.com/Scorpionlailai/-/tree/{commit}/experiment_results/v125_ablation',published_unix=time.time())


def build(coarse_milestone=False,final=False):
    with (OUT/'export.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);return _build(coarse_milestone,final)


def _build(coarse_milestone,final):
    audit=json.loads((OUT/'audit.json').read_text());base=OUT/'baseline_tables'
    per_mask=read_csv(base/'per_mask.csv');per_seed=read_csv(base/'per_seed.csv');summary=read_csv(base/'summary.csv');strata=read_csv(base/'stratified.csv')
    position=read_csv(base/'position_dispersion.csv');sseed=read_csv(base/'stratified_per_seed.csv');ssummary=read_csv(base/'stratified_summary.csv');resources=read_csv(base/'resources.csv')
    newmask=[];newstrata=[];newresources=[]
    for seed in (2023,2024,2025):
        folder=OUT/'A5'/f'seed_{seed}'
        for phase in ('coarse','calibration64','calibration256','ensemble'):
            results={}
            for p in sorted((folder/'evaluation'/phase).glob('*/result.json')):
                item=json.loads(p.read_text());b=item['binding'];results[b['split'],b['mask_seed']]=item
            info=None if phase=='ensemble' else phase_info(seed,phase,audit)
            # Selected validation is reportable before the global test barrier; no new test/strata are generated here.
            if info and ('validation',2021) not in results:
                library=audit['A5_seed2023']['history_snapshot_hash'] if seed==2023 else json.loads((folder/'coarse_complete.json').read_text())['history_snapshot_hash'] if (folder/'coarse_complete.json').exists() else None
                results['validation',2021]=dict(binding=dict(model_seed=seed,split='validation',mask_seed=2021,phase=phase,selected_epoch=info['selected_epoch'],
                    selection_metric=info['selection_rule'],checkpoint_sha256=info['checkpoint_sha256'],coarse_checkpoint_sha256=phase_info(seed,'coarse',audit)['checkpoint_sha256'],
                    history_snapshot_hash=library,data_manifest_sha256=audit['data_manifest_sha256'],
                    mask_sha256=next(m['sha256'] for m in audit['protocol']['masks'] if m['split']=='validation' and m['seed']==2021)),metrics=info['validation'],strata=[])
            for item in results.values():
                b=item['binding'];m=item['metrics'];meta=dict(dataset='PAMAP2',scene='sr015_bl8',variant='A5',training_seed=seed,mask_seed=b['mask_seed'],split=b['split'],phase=phase,
                    selection_rule=b['selection_metric'],selected_epoch=b['selected_epoch'],checkpoint_sha256=b['checkpoint_sha256'],coarse_checkpoint_sha256=b['coarse_checkpoint_sha256'],
                    library_sha256=b['history_snapshot_hash'],data_manifest_sha256=b['data_manifest_sha256'],mask_sha256=b['mask_sha256'],round_id='round6_A5',baseline_commit=BASE)
                for section in ('full96','recent32'):
                    counterpart=next(x for x in per_mask if x['variant']=='A0' and int(x['training_seed'])==seed and x['split']==b['split'] and int(x['mask_seed'])==b['mask_seed'] and x['phase']==phase and x['section']==section)
                    assert m[section]['coordinates']==int(counterpart['coordinates'])
                    newmask.append(dict(**meta,section=section,**m[section],windows=m['windows'],records=m.get('records',counterpart['records']),continuous_segments=m.get('continuous_segments',counterpart['continuous_segments'])))
                newstrata.extend(dict(**meta,**row) for row in item['strata'])
        for phase in ('coarse','calibration64','calibration256'):
            info=phase_info(seed,phase,audit)
            if not info:continue
            if seed==2023:res=info['resources'];cost=info['training_seconds'];newcost=0.;loop=cost
            else:
                if phase=='coarse':
                    epochs=[json.loads(p.read_text()) for p in (folder/phase).glob('epoch_*.json')]
                    res=dict(max_allocated_gib=max(x['resources']['max_allocated_gib'] for x in epochs),max_reserved_gib=max(x['resources']['max_reserved_gib'] for x in epochs),peak_memory_scope='isolated coarse phase')
                    loop=info['training_seconds'];cost=sum(x['seconds'] for x in info['segments'])
                else:res=info['resources'];cost=info['training_seconds'];loop=info['loop_seconds']
                newcost=cost
            newresources.append(dict(dataset='PAMAP2',scene='sr015_bl8',variant='A5',training_seed=seed,mask_seed=2023,split='train',phase=phase,
                epochs=100,optimizer_updates=8500,training_seconds=cost,incremental_training_seconds_this_round=newcost,incremental_training_seconds_round6=newcost,
                loop_seconds=loop,reused=seed==2023,trainable_parameters=info['trainable_parameters'],max_allocated_gib=res['max_allocated_gib'],max_reserved_gib=res['max_reserved_gib'],
                peak_memory_scope=res.get('peak_memory_scope','isolated calibration phase, includes frozen model allocation'),selection_rule=info['selection_rule'],
                checkpoint_sha256=info['checkpoint_sha256'],resource_kind='training',round_id='round6_A5'))
    newseed=[];newposition=[];newsummary=[]
    fields=('dataset','scene','variant','training_seed','split','phase','section','selection_rule')
    for identity in sorted({tuple(x[k] for k in fields) for x in newmask}):
        take=[x for x in newmask if tuple(x[k] for k in fields)==identity];meta=dict(zip(fields,identity));expected=1 if meta['split']=='validation' else 20
        if len(take)!=expected:continue
        newseed.append(dict(**meta,mask_seed=2021 if expected==1 else 'mean_of20_sealed_positions',positions=expected,mse=stat([x['mse'] for x in take])['mean'],
            mae=stat([x['mae'] for x in take])['mean'],selected_epoch=take[0]['selected_epoch'],checkpoint_sha256=take[0]['checkpoint_sha256'],library_sha256=take[0]['library_sha256'],round_id='round6_A5'))
        newposition.append(dict(**meta,mask_seed=2021 if expected==1 else '2026101801-2026101820',positions=expected,
            mse_position_std_ddof1=stat([x['mse'] for x in take])['std'],mae_position_std_ddof1=stat([x['mae'] for x in take])['std'],round_id='round6_A5'))
    sf=('dataset','scene','variant','split','phase','section','selection_rule')
    for identity in sorted({tuple(x[k] for k in sf) for x in newseed}):
        take=sorted([x for x in newseed if tuple(x[k] for k in sf)==identity],key=lambda x:x['training_seed']);meta=dict(zip(sf,identity));a=stat([x['mse'] for x in take]);b=stat([x['mae'] for x in take])
        newsummary.append(dict(**meta,training_seed=[x['training_seed'] for x in take],n_training_seeds=len(take),mask_seed=2021 if meta['split']=='validation' else 'per_seed_mean_of20_sealed_positions',
            mse_mean=a['mean'],mse_training_std_ddof1=a['std'],mae_mean=b['mean'],mae_training_std_ddof1=b['std'],round_id='round6_A5'))
    per_mask+=newmask;per_seed+=newseed;summary+=newsummary;strata+=newstrata;position+=newposition
    sfields=('dataset','scene','variant','training_seed','split','phase','section','family','group','selection_rule');buckets={};newss=[];newsum=[]
    for row in newstrata:buckets.setdefault(tuple(row[k] for k in sfields),[]).append(row)
    for identity,take in sorted(buckets.items()):
        meta=dict(zip(sfields,identity));expected=1 if meta['split']=='validation' else 20
        if len(take)!=expected:continue
        present=[x for x in take if x['mse'] is not None]
        newss.append(dict(**meta,mask_seed=2021 if expected==1 else 'mean_of20_sealed_positions',positions_total=expected,positions_nonempty=len(present),
            mse=stat([x['mse'] for x in present])['mean'],mae=stat([x['mae'] for x in present])['mean'],
            coordinates_per_position=[x['coordinates'] for x in take],windows_per_position=[x['windows'] for x in take],records_per_position=[x['records'] for x in take],mask_seeds=[x['mask_seed'] for x in take]))
    sgroup=('dataset','scene','variant','split','phase','section','family','group','selection_rule');buckets={}
    for row in newss:buckets.setdefault(tuple(row[k] for k in sgroup),[]).append(row)
    for identity,take in sorted(buckets.items()):
        a=stat([x['mse'] for x in take]);b=stat([x['mae'] for x in take])
        newsum.append(dict(**dict(zip(sgroup,identity)),training_seed=[x['training_seed'] for x in take],n_training_seeds_total=len(take),n_training_seeds_nonempty=a['n'],
            mse_mean=a['mean'],mse_training_std_ddof1=a['std'],mae_mean=b['mean'],mae_training_std_ddof1=b['std']))
    sseed+=newss;ssummary+=newsum
    pairs=[]
    for new in newseed:
        matches=[x for x in per_seed if x['variant']=='A0' and int(x['training_seed'])==new['training_seed'] and all(x[k]==new[k] for k in ('split','phase','section'))]
        assert len(matches)==1;old=matches[0]
        for metric in ('mse','mae'):
            a,b=float(old[metric]),float(new[metric]);pairs.append(dict(dataset='PAMAP2',scene='sr015_bl8',training_seed=new['training_seed'],mask_seed=new['mask_seed'],split=new['split'],phase=new['phase'],section=new['section'],metric=metric,
                A0_error=a,A5_error=b,A5_minus_A0=b-a,relative_difference_percent_of_A0=(b-a)/a*100 if a else None,
                interpretation='positive:explicit fusion quality beneficial;negative:qualityconstant better;relative denominator=A0',A0_selection_rule=old['selection_rule'],A5_selection_rule=new['selection_rule'],baseline_commit=BASE))
    pairsummary=[]
    for identity in sorted({(x['split'],x['phase'],x['section'],x['metric']) for x in pairs}):
        take=sorted([x for x in pairs if (x['split'],x['phase'],x['section'],x['metric'])==identity],key=lambda x:x['training_seed']);ds=stat([x['A5_minus_A0'] for x in take]);a=np.mean([x['A0_error'] for x in take]);b=np.mean([x['A5_error'] for x in take])
        pairsummary.append(dict(dataset='PAMAP2',scene='sr015_bl8',**dict(zip(('split','phase','section','metric'),identity)),training_seeds=[x['training_seed'] for x in take],n_training_seeds=len(take),A0_mean=float(a),A5_mean=float(b),
            A5_minus_A0=float(b-a),paired_difference_std_ddof1=ds['std'],relative_difference_percent_of_A0=float((b-a)/a*100) if a else None,
            three_paired_differences=[x['A5_minus_A0'] for x in take],quality_beneficial_seeds=sum(x['A5_minus_A0']>0 for x in take),constant_quality_better_seeds=sum(x['A5_minus_A0']<0 for x in take)))
    spairs=[]
    def number(v):return None if v is None or v=='' else float(v)
    def decode(v):return json.loads(v) if isinstance(v,str) and v.startswith('[') else v
    for new in newss:
        old=next(x for x in sseed if x['variant']=='A0' and int(x['training_seed'])==new['training_seed'] and all(x[k]==new[k] for k in ('split','phase','section','family','group')))
        assert decode(old['coordinates_per_position'])==new['coordinates_per_position']
        for metric in ('mse','mae'):
            a,b=number(old[metric]),number(new[metric])
            spairs.append(dict(dataset='PAMAP2',scene='sr015_bl8',training_seed=new['training_seed'],mask_seed=new['mask_seed'],**{k:new[k] for k in ('split','phase','section','family','group')},metric=metric,
                A0_error=a,A5_error=b,A5_minus_A0=b-a if a is not None and b is not None else None,relative_difference_percent_of_A0=(b-a)/a*100 if a and b is not None else None,
                positions_nonempty=new['positions_nonempty'],positions_total=new['positions_total'],coordinates_per_position=new['coordinates_per_position'],windows_per_position=new['windows_per_position'],records_per_position=new['records_per_position']))
    complete=all((OUT/'A5'/f'seed_{seed}'/'evaluation_complete.json').exists() for seed in (2023,2024,2025)) and (OUT/'capacity_complete.json').exists()
    if final:assert complete and json.loads((OUT/'final_verification.json').read_text())['passed']
    for name,rows in [('summary.csv',summary),('per_seed.csv',per_seed),('per_mask.csv',per_mask),('stratified.csv',strata),('position_dispersion.csv',position),
            ('stratified_per_seed.csv',sseed),('stratified_summary.csv',ssummary),('round6_paired_differences.csv',pairs),('round6_paired_summary.csv',pairsummary),('round6_stratified_differences.csv',spairs)]:csv_write(DEST/name,rows)
    capacity=json.loads((OUT/'capacity.json').read_text()) if (OUT/'capacity.json').exists() else []
    caprows=[{**{k:v for k,v in x.items() if k!='resources'},**x['resources'], 'round_id':'round6_A5'} for x in capacity]
    csv_write(DEST/'capacity.csv',read_csv(base/'capacity.csv')+caprows)
    csv_write(DEST/'round6_capacity.csv',caprows)
    for x in resources:x['incremental_training_seconds_round6']=0.;x['resource_kind']='training';x.setdefault('historical_round',x.get('round_id','previous_round'))
    csv_write(DEST/'resources.csv',resources+newresources)
    comparison_resources=[]
    for seed in (2023,2024,2025):
        for phase in ('coarse','calibration64','calibration256'):
            old=next(x for x in resources if x['variant']=='A0' and int(x['training_seed'])==seed and x['phase']==phase)
            comparison_resources.append({**old,'reused':True,'incremental_training_seconds_round6':0.})
    csv_write(DEST/'round6_resources.csv',comparison_resources+newresources)
    protocol=json.loads((base/'protocol.json').read_text());protocol['current_round']='round6_A5_three_seed_recheck';protocol['round6']=dict(baseline_commit=BASE,variants=['A0','A5'],model_seeds=[2023,2024,2025],scope='PAMAP2 sr015_bl8 coarse+original64/256+fixed.5/.5;noStage2',
        training_order_seed=2023,training_mask_seed=2023,test_seeds=protocol['test_seeds'],all_checkpoints_locked_before_new_test_replay=True,
        paired_difference='A5minusA0;positive=explicit fusion quality beneficial;relative difference denominator=A0',matched_capacity_causal_claim=False,
        intervention='original A5:explicit fusion quality fixed0;all experts/internal quality/ranking/candidate validity retained;not uniform fusion',feature_inventory_sha256=file_sha256(OUT/'quality_features.json'),source_audit_sha256=file_sha256(OUT/'audit.json'),history_pairing_limit=audit['historical_input_pairing_evidence'])
    write_json(DEST/'protocol.json',protocol)
    prov=DEST/'provenance/round6';prov.mkdir(parents=True,exist_ok=True)
    for name in ('audit.json','design_lock.json','calibration_sources.json','calibration_sources_lock.json','all_checkpoint_lock.json','final_verification.json','capacity_complete.json','A1_A4_frozen_manifest.json','quality_features.json','preflight_diagnostic_fix.json','source_review.json'):
        if (OUT/name).exists():shutil.copyfile(OUT/name,prov/name)
    for name in ('quality_features.md',):
        if (OUT/name).exists():shutil.copyfile(OUT/name,prov/name)
    for seed in (2023,2024,2025):
        for phase in ('coarse','calibration64','calibration256','coarse_evaluation','calibration64_evaluation','calibration256_evaluation','ensemble_evaluation'):
            evidence=OUT/'A5'/f'seed_{seed}'/f'{phase}_quality_execution.json'
            if evidence.exists():shutil.copyfile(evidence,prov/f'A5_seed{seed}_{phase}_quality_execution.json')
    # Text-only training traces retain order/mask fingerprints and resources; predictions/weights stay local.
    traces={};evaluations=[]
    for seed in (2024,2025):
        for phase in ('coarse','calibration64','calibration256'):
            stage=OUT/'A5'/f'seed_{seed}'/phase
            rows=[json.loads(p.read_text()) for p in sorted(stage.glob('epoch_*.json'))]
            if rows:traces[f'A5_{seed}_{phase}']=rows
    if traces:write_json(prov/'training_epoch_logs.json',traces)
    for seed in (2023,2024,2025):
        for p in sorted((OUT/'A5'/f'seed_{seed}'/'evaluation').glob('*/*/result.json')):
            item=json.loads(p.read_text());evaluations.append(dict(binding=item['binding'],metrics=item['metrics']))
    if evaluations:write_json(prov/'evaluation_manifest.json',evaluations)
    for p in (OUT/'tools').glob('*.py'):
        t=prov/'execution_source'/p.name;t.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,t)
    for p in (OUT/'plans').glob('*.md'):
        t=prov/'plans'/p.name;t.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,t)
    # Preserve baseline summaries verbatim in a separate archive, plus all earlier reports already tracked.
    snapshot=DEST/'reports/round5_baseline_50a2eb84';snapshot.mkdir(exist_ok=True)
    for name in sorted(audit['baseline_tables_sha256']):
        t=snapshot/name
        if t.exists():assert file_sha256(t)==audit['baseline_tables_sha256'][name]
        else:shutil.copyfile(base/name,t)
    lines=['# 第六轮：PAMAP2 0.15/8，A0 vs A5显式融合质量置常量','',f'基准commit：`{BASE}`。', '状态：'+('全部完成，等待审核。' if complete else '运行中；未齐三seed及测试的项目不作最终结论。'),'',
        '问题：在三专家均保留时，显式融合质量信息能否进一步改善粗恢复与最终修复？先全部粗阶段，再原校准，全部checkpoint锁定后才统一新测试与分层回放。',
        'A0三seed沿用基准；A5 seed2023粗/64/256严格审计后复用，新增2024/2025。T/V/H和公共前端全部保留；仅将原A5规定的显式融合质量特征置0。候选可用性、原生有效性、无来源/null保护、注意力合法性、专家内部质量/检索/排序/消息生成不变。内容驱动权重和内部候选权重仍可变化；本实验不等于均匀融合。原损失与选模不变。新增校准参数按seed对齐；继承的均值参数来自各自粗权重。', '',
        '## 主表：三训练seed均值与样本标准差','', '|划分|阶段|范围|变体|seed数|MSE均值±训练SD|MAE均值±训练SD|','|---|---|---|---|---|---|---|']
    for split in ('validation','test'):
        for phase in ('coarse','calibration64','calibration256','ensemble'):
            for section in ('full96','recent32'):
                for arm in ('A0','A5'):
                    row=next((x for x in summary if x['variant']==arm and x['split']==split and x['phase']==phase and x['section']==section),None)
                    if row is None:continue
                    def fmt(key):
                        m=number(row[key+'_mean']);sd=number(row[key+'_training_std_ddof1']);return f'{m:.6f} ± {sd:.6f}' if sd is not None else f'{m:.6f}（仅1次）'
                    lines.append(f"|{split}|{phase}|{section}|{arm}|{row['n_training_seeds']}|{fmt('mse')}|{fmt('mae')}|")
    lines+=['','## 配对差值：A5−A0，正值表示显式质量信息有益','', '|划分|阶段|范围|指标|seed数|平均差值|相对A0差值|逐seed差值|','|---|---|---|---|---|---|---|---|']
    for x in pairsummary:lines.append(f"|{x['split']}|{x['phase']}|{x['section']}|{x['metric']}|{x['n_training_seeds']}|{x['A5_minus_A0']:.6f}|{x['relative_difference_percent_of_A0']:.3f}%|{[round(v,6) for v in x['three_paired_differences']]}|")
    if complete:
        lines+=['','## 按阶段判断（描述性，不声明显著或质量特征不可替代）','']
        for split in ('validation','test'):
            for phase in ('coarse','ensemble'):
                for metric in ('mse','mae'):
                    x=next(z for z in pairsummary if z['split']==split and z['phase']==phase and z['section']=='full96' and z['metric']==metric)
                    lines.append(f"- {split} {phase} {metric}：显式质量信息有益{x['quality_beneficial_seeds']}/3，置常量更好{x['constant_quality_better_seeds']}/3。")
        mse_test=[x for x in pairsummary if x['split']=='test' and x['section']=='full96' and x['metric']=='mse' and x['phase'] in ('coarse','ensemble')]
        if all(x['constant_quality_better_seeds']==3 for x in mse_test):lines.append('测试粗阶段和最终修复均为3/3置常量更好：提示显式质量可能存在泛化问题，提供简化依据；仍须结合验证方向/MAE/全部分层，不能为保留设计修改定义或挑选分层。')
        elif all(x['quality_beneficial_seeds']==3 for x in mse_test):lines.append('测试粗阶段和最终修复均为3/3保留显式质量更好：提供其增量价值的初步证据；仍须结合验证方向/MAE，不声称显著或不可替代。')
        else:lines.append('跨seed或阶段方向混合，或误差接近：显式质量的必要性尚未稳定证实；完整列出绝对差、相对差与样本标准差，按条件限定作用。')
        lines+=['','## 历史前/中/后32点及缺口长度分层（均保留所有组）','', '|划分|阶段|分组|指标|A0均值|A5均值|A5−A0|逐seed差值|','|---|---|---|---|---|---|---|---|']
        for split in ('validation','test'):
            for phase in ('coarse','ensemble'):
                for family in ('history','invisible_run'):
                    groups=sorted({x['group'] for x in spairs if x['family']==family})
                    for group in groups:
                        for metric in ('mse','mae'):
                            take=[x for x in spairs if x['split']==split and x['phase']==phase and x['section']=='full96' and x['family']==family and x['group']==group and x['metric']==metric]
                            available=[x for x in take if x['A5_minus_A0'] is not None]
                            if available:
                                a=stat([x['A0_error'] for x in available])['mean'];b=stat([x['A5_error'] for x in available])['mean']
                                lines.append(f"|{split}|{phase}|{family}:{group}|{metric}|{a:.6f}|{b:.6f}|{b-a:.6f}|{[round(x['A5_minus_A0'],6) for x in available]}|")
                            else:lines.append(f'|{split}|{phase}|{family}:{group}|{metric}|空组|空组|空组|[]|')
    newcost=sum(float(x['incremental_training_seconds_round6']) for x in newresources)/3600
    lines+=['','## 资源与核验','',f'当前新增训练成本：{newcost:.3f} GPU小时（复用成本另列）。资源明细见round6_resources.csv，本轮推理/参数明细见round6_capacity.csv，capacity.csv保留历轮测量。',
        '保持同构分配参数和公共参数初始化。质量输出在融合处被常量替换，其上游投影可能不再收到主损失梯度；内部消息及检索路径仍训练。分配参数、阶段可训练参数、推理调用参数和实测时间分别报告，不包装为严格机制因果效应。',
        '同一GPU、FP32 eval、batch128，1次预热和5次同步计时；每模型/阶段均报告。历史显存仅有累计阶段峰值时明确标注，不能冒充独立阶段峰值。',
        '2023的历史输入顺序/掩码由同一封存源码和协议支撑，历史实际输入哈希未记录；2024/2025逐轮核验与配对A0实际输入一致。',
        '运行时检查两类融合入口均读取零质量；保留专家消息及原候选对象。固定相同专家输入，仅切换A5标志时内部证据/候选/排序逐值相同；单独污染对外质量报告不改变A5输出。无来源fixture仅允许null，隐藏真值扰动不改变输出；真实batch128主MSE梯度有效。详细特征清单见provenance/round6/quality_features.md。串行状态自然随融合改变，不能声称训练后A0/A5候选数值处处相等。正式训练只做轻量入口/合法掩码监控，完整逐候选合法性检查用于只读隔离审计。新增训练耗时包含轻量监控开销，与历史A0训练耗时的差别不能全部归因于结构计算量；推理计时不包含监控。',
        '原生缺失不作真值，观测值不变，训练全局标准化尺度，96点不重复按patch计分。预登记所有可见率分组及空组也保留在CSV。',
        '先每seed平均20测试位置，再报告三个训练均值/样本SD和三对差值。位置SD独立保存，60组位置不是60次训练。相对差以A0为分母。',
        '三seed仅提供初步稳定性证据。A1/A2/A3/A4均以50a2eb84封存；A6/A7、其他数据集、Stage2均未新增。本轮封存v125双校准与Weather单分支迭代独立；结束后交付报告，不自动启动后续阶段。']
    report='\n'.join(lines)+'\n';(OUT/'report.md').write_text(report)
    reportdir=DEST/'reports';reportdir.mkdir(exist_ok=True);(reportdir/'current_round6_status.md').write_text(report)
    if coarse_milestone:
        assert all((OUT/'A5'/f'seed_{seed}/coarse_complete.json').exists() for seed in (2024,2025))
        target=reportdir/'round6_coarse_validation_checkpoint_20261022.md'
        if not target.exists():target.write_text(report)
    if complete:
        target=reportdir/'round6_A5_three_seeds_20261022.md'
        if not target.exists():target.write_text(report)
    original=(base/'README.md').read_text()
    head=['# v125消融：第六轮A5显式融合质量置常量复验','',f'基准commit：`{BASE}`。当前范围PAMAP2 0.15/8、A0/A5各3seed、粗/64/256/固定双校准。',
        '**当前进度**：'+('全部完成，最终核验后上传，等待审核。' if complete else '运行中；先全部粗阶段、再校准，所有checkpoint锁定后统一新测试/分层。'),
        '**实现与复用**：全部专家及公共前端保持；仅原A5规定的显式融合质量置0，内部质量/排序/候选及合法掩码保持。内容权重仍可变化，不称均匀融合。A0复用基准结果；A5 seed2023严格审计后复用，新增2024/2025。',
        '**判断规则**：A5−A0为正表示显式质量信息有益，负表示置常量更好；不隐藏负结果和跨阶段反转，不宣称容量匹配因果效应或三seed显著性。',
        '**文件更新**：原A0/A1/A2/A3/A4表格保留并追加A5；A1—A4以50a2eb84封存，不再追加训练；round6_paired_*.csv为三对差值/汇总，round6_stratified_differences.csv为全部分层差值；round6_resources.csv区分本轮新增与复用成本，round6_capacity.csv为本轮同GPU推理计时/参数/显存；capacity.csv追加保留历史。',
        '**协议变化**：仅新增A5多seed复验及全checkpoint锁定后测试的调度约束；原数据、掩码、损失、学习率、选模和0.5/0.5均不变。历史输入哈希缺失和历史累计显存限制已注明。',
        '**待办**：'+('审核各seed、验证/测试和粗/校准后的方向与成本，判断显式融合质量的额外价值与成本取舍；不自动开新实验。' if complete else '完成新增A5粗权重、校准、锁定、统一回放、资源测量及最终第六轮报告上传。'),
        'A1—A4封存；A6/A7及其他数据集暂停；Stage2未启动。完整历史报告保留，未上传权重、缓存或数据。',
        '[已封存A1—A4来源证据汇总](reports/source_evidence_A1_A2_A3_A4_20261021.md)',
        '[第六轮报告](reports/current_round6_status.md) · [第二轮校准报告](reports/round2_calibration_confirmation_20261020.md)','','---','','以下保留上一轮审核记录：','',original]
    (DEST/'README.md').write_text('\n'.join(head).rstrip()+'\n')
    status=dict(complete=complete,updated_unix=time.time(),new_training_gpu_hours=newcost,per_mask_rows=len(per_mask),per_seed_rows=len(per_seed),stratified_rows=len(strata),
        A5_complete_models=sum((OUT/'A5'/f'seed_{seed}/evaluation_complete.json').exists() for seed in (2023,2024,2025)),baseline_commit=BASE)
    write_json(OUT/'report_status.json',status)
    if coarse_milestone:
        record=publish('Record round6 A5 coarse validation and resource checkpoint before calibration');write_json(OUT/'coarse_publication.json',record)
    if final:
        record=publish('Record round6 A5 constant-fusion-quality three-seed ablation and all-stage comparisons');write_json(OUT/'publication.json',record)
    return status
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--coarse-milestone',action='store_true');p.add_argument('--final',action='store_true');a=p.parse_args();print(json.dumps(build(a.coarse_milestone,a.final)))
