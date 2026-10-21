"""Append A4 to commit-pinned tables, archive round3 and publish only text results."""
import csv
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
from policy3 import OUT,REPO,BASE,COARSE,CAL
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
    if seed==2023:return audit['A4_seed2023']['phases'][phase]
    p=OUT/'A4'/f'seed_{seed}'/phase/'selection.json'
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
        folder=OUT/'A4'/f'seed_{seed}'
        for phase in ('coarse','calibration64','calibration256','ensemble'):
            results={}
            for p in sorted((folder/'evaluation'/phase).glob('*/result.json')):
                item=json.loads(p.read_text());b=item['binding'];results[b['split'],b['mask_seed']]=item
            info=None if phase=='ensemble' else phase_info(seed,phase,audit)
            # Selected validation is reportable before the global test barrier; no new test/strata are generated here.
            if info and ('validation',2021) not in results:
                library=audit['A4_seed2023']['history_snapshot_hash'] if seed==2023 else json.loads((folder/'coarse_complete.json').read_text())['history_snapshot_hash'] if (folder/'coarse_complete.json').exists() else None
                results['validation',2021]=dict(binding=dict(model_seed=seed,split='validation',mask_seed=2021,phase=phase,selected_epoch=info['selected_epoch'],
                    selection_metric=info['selection_rule'],checkpoint_sha256=info['checkpoint_sha256'],coarse_checkpoint_sha256=phase_info(seed,'coarse',audit)['checkpoint_sha256'],
                    history_snapshot_hash=library,data_manifest_sha256=audit['data_manifest_sha256'],
                    mask_sha256=next(m['sha256'] for m in audit['protocol']['masks'] if m['split']=='validation' and m['seed']==2021)),metrics=info['validation'],strata=[])
            for item in results.values():
                b=item['binding'];m=item['metrics'];meta=dict(dataset='PAMAP2',scene='sr015_bl8',variant='A4',training_seed=seed,mask_seed=b['mask_seed'],split=b['split'],phase=phase,
                    selection_rule=b['selection_metric'],selected_epoch=b['selected_epoch'],checkpoint_sha256=b['checkpoint_sha256'],coarse_checkpoint_sha256=b['coarse_checkpoint_sha256'],
                    library_sha256=b['history_snapshot_hash'],data_manifest_sha256=b['data_manifest_sha256'],mask_sha256=b['mask_sha256'],round_id='round3_A4',baseline_commit=BASE)
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
            newresources.append(dict(dataset='PAMAP2',scene='sr015_bl8',variant='A4',training_seed=seed,mask_seed=2023,split='train',phase=phase,
                epochs=100,optimizer_updates=8500,training_seconds=cost,incremental_training_seconds_this_round=newcost,incremental_training_seconds_round3=newcost,
                loop_seconds=loop,reused=seed==2023,trainable_parameters=info['trainable_parameters'],max_allocated_gib=res['max_allocated_gib'],max_reserved_gib=res['max_reserved_gib'],
                peak_memory_scope=res.get('peak_memory_scope','isolated calibration phase, includes frozen model allocation'),selection_rule=info['selection_rule'],
                checkpoint_sha256=info['checkpoint_sha256'],resource_kind='training',round_id='round3_A4'))
    newseed=[];newposition=[];newsummary=[]
    fields=('dataset','scene','variant','training_seed','split','phase','section','selection_rule')
    for identity in sorted({tuple(x[k] for k in fields) for x in newmask}):
        take=[x for x in newmask if tuple(x[k] for k in fields)==identity];meta=dict(zip(fields,identity));expected=1 if meta['split']=='validation' else 20
        if len(take)!=expected:continue
        newseed.append(dict(**meta,mask_seed=2021 if expected==1 else 'mean_of20_sealed_positions',positions=expected,mse=stat([x['mse'] for x in take])['mean'],
            mae=stat([x['mae'] for x in take])['mean'],selected_epoch=take[0]['selected_epoch'],checkpoint_sha256=take[0]['checkpoint_sha256'],library_sha256=take[0]['library_sha256'],round_id='round3_A4'))
        newposition.append(dict(**meta,mask_seed=2021 if expected==1 else '2026101801-2026101820',positions=expected,
            mse_position_std_ddof1=stat([x['mse'] for x in take])['std'],mae_position_std_ddof1=stat([x['mae'] for x in take])['std'],round_id='round3_A4'))
    sf=('dataset','scene','variant','split','phase','section','selection_rule')
    for identity in sorted({tuple(x[k] for k in sf) for x in newseed}):
        take=sorted([x for x in newseed if tuple(x[k] for k in sf)==identity],key=lambda x:x['training_seed']);meta=dict(zip(sf,identity));a=stat([x['mse'] for x in take]);b=stat([x['mae'] for x in take])
        newsummary.append(dict(**meta,training_seed=[x['training_seed'] for x in take],n_training_seeds=len(take),mask_seed=2021 if meta['split']=='validation' else 'per_seed_mean_of20_sealed_positions',
            mse_mean=a['mean'],mse_training_std_ddof1=a['std'],mae_mean=b['mean'],mae_training_std_ddof1=b['std'],round_id='round3_A4'))
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
                A0_error=a,A4_error=b,A4_minus_A0=b-a,relative_difference_percent_of_A0=(b-a)/a*100 if a else None,
                interpretation='positive:T beneficial;negative:withoutT better;relative denominator=A0',A0_selection_rule=old['selection_rule'],A4_selection_rule=new['selection_rule'],baseline_commit=BASE))
    pairsummary=[]
    for identity in sorted({(x['split'],x['phase'],x['section'],x['metric']) for x in pairs}):
        take=sorted([x for x in pairs if (x['split'],x['phase'],x['section'],x['metric'])==identity],key=lambda x:x['training_seed']);ds=stat([x['A4_minus_A0'] for x in take]);a=np.mean([x['A0_error'] for x in take]);b=np.mean([x['A4_error'] for x in take])
        pairsummary.append(dict(dataset='PAMAP2',scene='sr015_bl8',**dict(zip(('split','phase','section','metric'),identity)),training_seeds=[x['training_seed'] for x in take],n_training_seeds=len(take),A0_mean=float(a),A4_mean=float(b),
            A4_minus_A0=float(b-a),paired_difference_std_ddof1=ds['std'],relative_difference_percent_of_A0=float((b-a)/a*100) if a else None,
            three_paired_differences=[x['A4_minus_A0'] for x in take],T_beneficial_seeds=sum(x['A4_minus_A0']>0 for x in take),withoutT_better_seeds=sum(x['A4_minus_A0']<0 for x in take)))
    spairs=[]
    def number(v):return None if v is None or v=='' else float(v)
    def decode(v):return json.loads(v) if isinstance(v,str) and v.startswith('[') else v
    for new in newss:
        old=next(x for x in sseed if x['variant']=='A0' and int(x['training_seed'])==new['training_seed'] and all(x[k]==new[k] for k in ('split','phase','section','family','group')))
        assert decode(old['coordinates_per_position'])==new['coordinates_per_position']
        for metric in ('mse','mae'):
            a,b=number(old[metric]),number(new[metric])
            spairs.append(dict(dataset='PAMAP2',scene='sr015_bl8',training_seed=new['training_seed'],mask_seed=new['mask_seed'],**{k:new[k] for k in ('split','phase','section','family','group')},metric=metric,
                A0_error=a,A4_error=b,A4_minus_A0=b-a if a is not None and b is not None else None,relative_difference_percent_of_A0=(b-a)/a*100 if a and b is not None else None,
                positions_nonempty=new['positions_nonempty'],positions_total=new['positions_total'],coordinates_per_position=new['coordinates_per_position'],windows_per_position=new['windows_per_position'],records_per_position=new['records_per_position']))
    complete=all((OUT/'A4'/f'seed_{seed}'/'evaluation_complete.json').exists() for seed in (2023,2024,2025)) and (OUT/'capacity_complete.json').exists()
    if final:assert complete and json.loads((OUT/'final_verification.json').read_text())['passed']
    for name,rows in [('summary.csv',summary),('per_seed.csv',per_seed),('per_mask.csv',per_mask),('stratified.csv',strata),('position_dispersion.csv',position),
            ('stratified_per_seed.csv',sseed),('stratified_summary.csv',ssummary),('round3_paired_differences.csv',pairs),('round3_paired_summary.csv',pairsummary),('round3_stratified_differences.csv',spairs)]:csv_write(DEST/name,rows)
    capacity=json.loads((OUT/'capacity.json').read_text()) if (OUT/'capacity.json').exists() else []
    caprows=[{**{k:v for k,v in x.items() if k!='resources'},**x['resources'], 'round_id':'round3_A4'} for x in capacity];csv_write(DEST/'capacity.csv',caprows)
    for x in resources:x['incremental_training_seconds_round3']=0.;x['resource_kind']='training';x['historical_round']='round1_or_round2'
    csv_write(DEST/'resources.csv',resources+newresources)
    comparison_resources=[]
    for seed in (2023,2024,2025):
        for phase in ('coarse','calibration64','calibration256'):
            old=next(x for x in resources if x['variant']=='A0' and int(x['training_seed'])==seed and x['phase']==phase)
            comparison_resources.append({**old,'reused':True,'incremental_training_seconds_round3':0.})
    csv_write(DEST/'round3_resources.csv',comparison_resources+newresources)
    protocol=json.loads((base/'protocol.json').read_text());protocol['current_round']='round3_A4_three_seed_recheck';protocol['round3']=dict(baseline_commit=BASE,variants=['A0','A4'],model_seeds=[2023,2024,2025],scope='PAMAP2 sr015_bl8 coarse+original64/256+fixed.5/.5;noStage2',
        training_order_seed=2023,training_mask_seed=2023,test_seeds=protocol['test_seeds'],all_checkpoints_locked_before_new_test_replay=True,
        paired_difference='A4minusA0;positive=T beneficial;relative difference denominator=A0',matched_capacity_causal_claim=False,
        source_audit_sha256=file_sha256(OUT/'audit.json'),history_pairing_limit=audit['historical_input_pairing_evidence'])
    write_json(DEST/'protocol.json',protocol)
    prov=DEST/'provenance/round3';prov.mkdir(parents=True,exist_ok=True)
    for name in ('audit.json','design_lock.json','calibration_sources.json','calibration_sources_lock.json','all_checkpoint_lock.json','final_verification.json','capacity_complete.json'):
        if (OUT/name).exists():shutil.copyfile(OUT/name,prov/name)
    for p in (OUT/'tools').glob('*.py'):
        t=prov/'execution_source'/p.name;t.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,t)
    # Preserve baseline summaries verbatim in a separate archive, plus all earlier reports already tracked.
    snapshot=DEST/'reports/round2_baseline_b5a72ddb';snapshot.mkdir(exist_ok=True)
    for name in ('summary.csv','per_seed.csv','resources.csv','protocol.json'):
        t=snapshot/name
        if t.exists():assert file_sha256(t)==audit['baseline_tables_sha256'][name]
        else:shutil.copyfile(base/name,t)
    lines=['# 第三轮：PAMAP2 0.15/8，A0 vs A4去T','',f'基准commit：`{BASE}`。', '状态：'+('全部完成，等待审核。' if complete else '运行中；未齐三seed及测试的项目不作最终结论。'),'',
        '问题：公共时间编码不变时，T专家是否稳定改善粗恢复与最终修复？先全部粗阶段，再原校准，全部checkpoint锁定后才统一新测试与分层回放。',
        'A0三seed沿用基准；A4 seed2023粗/64/256严格审计后复用，新增2024/2025。公共前端、H/V、历史库规则和原选模/损失不变。新增校准参数按seed对齐；继承的均值参数来自各自粗权重。', '',
        '## 主表：三训练seed均值与样本标准差','', '|划分|阶段|范围|变体|seed数|MSE均值±训练SD|MAE均值±训练SD|','|---|---|---|---|---|---|---|']
    for split in ('validation','test'):
        for phase in ('coarse','calibration64','calibration256','ensemble'):
            for section in ('full96','recent32'):
                for arm in ('A0','A4'):
                    row=next((x for x in summary if x['variant']==arm and x['split']==split and x['phase']==phase and x['section']==section),None)
                    if row is None:continue
                    def fmt(key):
                        m=number(row[key+'_mean']);sd=number(row[key+'_training_std_ddof1']);return f'{m:.6f} ± {sd:.6f}' if sd is not None else f'{m:.6f}（仅1次）'
                    lines.append(f"|{split}|{phase}|{section}|{arm}|{row['n_training_seeds']}|{fmt('mse')}|{fmt('mae')}|")
    lines+=['','## 配对差值：A4−A0，正值表示T有益','', '|划分|阶段|范围|指标|seed数|平均差值|相对A0差值|逐seed差值|','|---|---|---|---|---|---|---|---|']
    for x in pairsummary:lines.append(f"|{x['split']}|{x['phase']}|{x['section']}|{x['metric']}|{x['n_training_seeds']}|{x['A4_minus_A0']:.6f}|{x['relative_difference_percent_of_A0']:.3f}%|{[round(v,6) for v in x['three_paired_differences']]}|")
    if complete:
        lines+=['','## 按阶段判断（描述性，不声明显著或T不可替代）','']
        for split in ('validation','test'):
            for phase in ('coarse','ensemble'):
                for metric in ('mse','mae'):
                    x=next(z for z in pairsummary if z['split']==split and z['phase']==phase and z['section']=='full96' and z['metric']==metric)
                    lines.append(f"- {split} {phase} {metric}：T有益{x['T_beneficial_seeds']}/3，去T更好{x['withoutT_better_seeds']}/3。")
        mse_test=[x for x in pairsummary if x['split']=='test' and x['section']=='full96' and x['metric']=='mse' and x['phase'] in ('coarse','ensemble')]
        if all(x['withoutT_better_seeds']==3 for x in mse_test):lines.append('测试粗阶段和最终修复均为3/3去T更好：向迭代专家提供简化T的证据，同时保留验证反向与分层负结果；不自动删改架构。')
        elif all(x['T_beneficial_seeds']==3 for x in mse_test):lines.append('测试粗阶段和最终修复均为3/3保留T更好：支持暂保留T；收益条件按下方完整分层说明，不声称不可替代。')
        else:lines.append('跨seed或阶段方向混合：T的作用依赖训练seed/数据条件或校准阶段；分别解释各阶段，不只选择有利结果。')
        lines+=['','## 历史前/中/后32点及缺口长度分层（均保留所有组）','', '|划分|阶段|分组|指标|A0均值|A4均值|A4−A0|逐seed差值|','|---|---|---|---|---|---|---|---|']
        for split in ('validation','test'):
            for phase in ('coarse','ensemble'):
                for family in ('history','invisible_run'):
                    groups=sorted({x['group'] for x in spairs if x['family']==family})
                    for group in groups:
                        for metric in ('mse','mae'):
                            take=[x for x in spairs if x['split']==split and x['phase']==phase and x['section']=='full96' and x['family']==family and x['group']==group and x['metric']==metric]
                            available=[x for x in take if x['A4_minus_A0'] is not None]
                            if available:
                                a=stat([x['A0_error'] for x in available])['mean'];b=stat([x['A4_error'] for x in available])['mean']
                                lines.append(f"|{split}|{phase}|{family}:{group}|{metric}|{a:.6f}|{b:.6f}|{b-a:.6f}|{[round(x['A4_minus_A0'],6) for x in available]}|")
                            else:lines.append(f'|{split}|{phase}|{family}:{group}|{metric}|空组|空组|空组|[]|')
    newcost=sum(float(x['incremental_training_seconds_round3']) for x in newresources)/3600
    lines+=['','## 资源与核验','',f'当前新增训练成本：{newcost:.3f} GPU小时（复用成本另列）。资源明细见round3_resources.csv，推理/参数明细见capacity.csv。',
        '分配参数仍保留同构模块以控制随机初始化；T参数冻结且T专属路由实际禁用。训练可更新参数、推理调用参数和实测时间分开报告，不作为容量匹配的纯机制因果实验。',
        '同一GPU、FP32 eval、batch128，1次预热和5次同步计时；每模型/阶段均报告。历史显存仅有累计阶段峰值时明确标注，不能冒充独立阶段峰值。',
        '2023的历史输入顺序/掩码由同一封存源码和协议支撑，历史实际输入哈希未记录；2024/2025逐轮核验与配对A0实际输入一致。',
        'T运行时方法调用报错哨兵、T候选注意力及串行更新为零、污染T参数不改变输出；公共前端与H/V仍执行。',
        '原生缺失不作真值，观测值不变，训练全局标准化尺度，96点不重复按patch计分。预登记所有可见率分组及空组也保留在CSV。',
        '先每seed平均20测试位置，再报告三个训练均值/样本SD和三对差值。位置SD独立保存，60组位置不是60次训练。相对差以A0为分母。',
        '三seed仅提供初步稳定性证据。A2/A3/A5–A7、其他数据集、Stage2均未新增；结束后仅交付报告，不启动后续阶段。']
    report='\n'.join(lines)+'\n';(OUT/'report.md').write_text(report)
    reportdir=DEST/'reports';reportdir.mkdir(exist_ok=True);(reportdir/'current_round3_status.md').write_text(report)
    if coarse_milestone:
        assert all((OUT/'A4'/f'seed_{seed}/coarse_complete.json').exists() for seed in (2024,2025))
        target=reportdir/'round3_coarse_validation_checkpoint_20261021.md'
        if not target.exists():target.write_text(report)
    if complete:
        target=reportdir/'round3_A4_three_seeds_20261021.md'
        if not target.exists():target.write_text(report)
    original=(base/'README.md').read_text()
    head=['# v125消融：第三轮A4去T复验','',f'基准commit：`{BASE}`。当前范围PAMAP2 0.15/8、A0/A4各3seed、粗/64/256/固定双校准。',
        '**当前进度**：'+('全部完成，最终核验后上传，等待审核。' if complete else '运行中；先全部粗阶段、再校准，所有checkpoint锁定后统一新测试/分层。'),
        '**实现与复用**：T消息、串行状态更新及后续候选读取均禁用；公共时间前端、H/V保持原规则。A0复用基准结果；A4 seed2023严格审计后复用，新增2024/2025。',
        '**判断规则**：A4−A0为正表示T有益，负表示去T更好；不隐藏负结果和跨阶段反转，不宣称容量匹配因果效应或三seed显著性。',
        '**文件更新**：原A0/A1表格保留并追加A4；round3_paired_*.csv为三对差值/汇总，round3_stratified_differences.csv为全部分层差值；round3_resources.csv区分本轮新增与复用成本，capacity.csv为同GPU推理计时/参数/显存。',
        '**协议变化**：仅新增A4多seed复验及全checkpoint锁定后测试的调度约束；原数据、掩码、损失、学习率、选模和0.5/0.5均不变。历史输入哈希缺失和历史累计显存限制已注明。',
        '**待办**：'+('审核各seed、验证/测试和粗/校准后的方向与成本，再决定T简化或保留；不自动开新实验。' if complete else '完成新增A4粗权重、校准、锁定、统一回放、资源测量及最终第三轮报告上传。'),
        'A2/A3/A5–A7及其他数据集暂停；Stage2未启动。完整历史报告保留，未上传权重、缓存或数据。',
        '[第三轮报告](reports/current_round3_status.md) · [第二轮校准报告](reports/round2_calibration_confirmation_20261020.md)','','---','','以下保留上一轮审核记录：','',original]
    (DEST/'README.md').write_text('\n'.join(head)+'\n')
    status=dict(complete=complete,updated_unix=time.time(),new_training_gpu_hours=newcost,per_mask_rows=len(per_mask),per_seed_rows=len(per_seed),stratified_rows=len(strata),
        A4_complete_models=sum((OUT/'A4'/f'seed_{seed}/evaluation_complete.json').exists() for seed in (2023,2024,2025)),baseline_commit=BASE)
    write_json(OUT/'report_status.json',status)
    if coarse_milestone:
        record=publish('Record round3 A4 coarse validation and resource checkpoint before calibration');write_json(OUT/'coarse_publication.json',record)
    if final:
        record=publish('Record round3 A4 no-T three-seed ablation and all-stage comparisons');write_json(OUT/'publication.json',record)
    return status
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--coarse-milestone',action='store_true');p.add_argument('--final',action='store_true');a=p.parse_args();print(json.dumps(build(a.coarse_milestone,a.final)))
