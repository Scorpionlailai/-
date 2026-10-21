"""Summarize already averaged training-seed observations; no model or test access."""
import csv
import itertools
import json
from pathlib import Path
import statistics

ARMS=('A1','A2','A3','A4')
SOURCES={'A1':'all_experts_removed','A2':'H_removed','A3':'V_removed','A4':'T_removed'}
SEEDS=(2023,2024,2025)
DIMENSIONS=(('validation','test'),('coarse','ensemble'),('full96','recent32'))


def build_comparison(per_seed):
    index={}
    for row in per_seed:
        if row['dataset']!='PAMAP2' or row['scene']!='sr015_bl8' or row['variant'] not in ('A0',*ARMS):continue
        if row['phase'] not in DIMENSIONS[1]:continue
        key=(row['variant'],int(row['training_seed']),row['split'],row['phase'],row['section'])
        if key in index:raise ValueError('duplicate training-seed observation: '+str(key))
        if int(row['positions'])!=(20 if row['split']=='test' else 1):raise ValueError('incorrect per-training-seed position count')
        index[key]=row
    expected=set(itertools.product(('A0',*ARMS),SEEDS,*DIMENSIONS))
    if set(index)!=expected:raise ValueError('combined comparison requires exactly3 seeds for every registered arm/split/path/section')
    rows=[]
    for arm,split,path,section,metric in itertools.product(ARMS,*DIMENSIONS,('mse','mae')):
        a0=[index['A0',seed,split,path,section] for seed in SEEDS]
        variant=[index[arm,seed,split,path,section] for seed in SEEDS]
        ref=[float(x[metric]) for x in a0];values=[float(x[metric]) for x in variant];diff=[b-a for a,b in zip(ref,values)]
        am,bm=statistics.mean(ref),statistics.mean(values)
        rows.append(dict(dataset='PAMAP2',scene='sr015_bl8',variant=arm,removed_source=SOURCES[arm],split=split,phase=path,
            section=section,metric=metric,training_seeds=list(SEEDS),n_training_seeds=3,
            A0_mean=am,variant_mean=bm,A0_training_std_ddof1=statistics.stdev(ref),variant_training_std_ddof1=statistics.stdev(values),
            variant_minus_A0=bm-am,difference_training_std_ddof1=statistics.stdev(diff),paired_differences=diff,
            relative_difference_percent_of_A0=100*(bm-am)/am if am else None,
            full_model_better_seeds=sum(x>0 for x in diff),removed_variant_better_seeds=sum(x<0 for x in diff),
            A0_checkpoint_hashes=[x['checkpoint_sha256'] for x in a0],variant_checkpoint_hashes=[x['checkpoint_sha256'] for x in variant],
            A0_library_hashes=[x['library_sha256'] for x in a0],variant_library_hashes=[x['library_sha256'] for x in variant],
            interpretation='variant-A0;positive supports retained source;descriptive3trainingseeds;not60independenttrials;not additive causal contributions'))
    return rows


def export_comparison(destination):
    destination=Path(destination)
    with (destination/'per_seed.csv').open() as f:rows=build_comparison(list(csv.DictReader(f)))
    path=destination/'source_evidence_comparison.csv'
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader()
        for row in rows:writer.writerow({k:json.dumps(v,separators=(',',':')) if isinstance(v,list) else v for k,v in row.items()})
    lines=['# PAMAP2 0.15/8：A1/A2/A3/A4来源证据汇总','',
        '所有对照沿用同一A0三seed。测试先在每训练seed内平均20封存位置，再汇总三次训练均值/样本标准差；每对差值按2023、2024、2025排列。',
        '公共前端时间和跨变量建模保留。A1移除全部专家，A2/A3/A4分别移除H/V/T；均按各自验证选定粗权重和原独立64/256校准、固定0.5/0.5组合。',
        '差值均为变体−A0，正值支持保留该来源，负值表示删除更好。不同删减对照改变路径与成本，不属于容量匹配机制因果实验，不能相加。三seed仅提供初步稳定性证据，MSE/MAE与粗/校准方向冲突均保留。','',
        '|划分|阶段|范围|对照|指标|A0均值±训练SD|变体均值±训练SD|差值|相对A0|三对差值|保留更好seed数|',
        '|---|---|---|---|---|---|---|---|---|---|---|']
    for row in rows:
        lines.append(f"|{row['split']}|{row['phase']}|{row['section']}|{row['variant']}|{row['metric']}|{row['A0_mean']:.6f} ± {row['A0_training_std_ddof1']:.6f}|{row['variant_mean']:.6f} ± {row['variant_training_std_ddof1']:.6f}|{row['variant_minus_A0']:+.6f}|{row['relative_difference_percent_of_A0']:+.3f}%|{[round(x,6) for x in row['paired_differences']]}|{row['full_model_better_seeds']}/3|")
    lines+=['','## 按全96点条件逐项判断','']
    for arm in ARMS:
        lines.append(f'### {arm}：{SOURCES[arm]}')
        lines.append('')
        for row in rows:
            if row['variant']==arm and row['section']=='full96':
                lines.append(f"- {row['split']} / {row['phase']} / {row['metric']}：完整模型更好{row['full_model_better_seeds']}/3，删除更好{row['removed_variant_better_seeds']}/3；平均差{row['variant_minus_A0']:+.6f}（{row['relative_difference_percent_of_A0']:+.3f}%）。")
    lines+=['','所有已登记缺口长度、前中后32点、可见率及其他变量可见程度完整分组仍见stratified.csv和各轮stratified_differences，不按优势筛选。',
        '资源成本应结合各轮resources/capacity解释；A1成本沿用封存记录，A2/A3/A4分别有独立同卡测量，不能将跨轮计时混成完全同时的基准。',
        'A1/A2/A4结果封存复用，A3为本轮复验；完成本轮后仅提交审核，不自动追加种子、变体、结构搜索或Stage2。']
    (destination/'reports/source_evidence_A1_A2_A3_A4_20261021.md').write_text('\n'.join(lines)+'\n')
    return rows
