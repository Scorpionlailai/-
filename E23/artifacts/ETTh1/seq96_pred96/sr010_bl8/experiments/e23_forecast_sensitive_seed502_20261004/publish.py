"""Publish saved validation diagnostics; never load a model or access test data."""
import csv
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent
DOCS=Path('/home/xyw/paper/E23/docs/experiments')
PREFIX='ETTh1_P4_forecast_sensitive_seed502_20261004'
LABELS=['v31B','v38','T1']
DISPLAY={'v31B':'E23-v31-B','v38':'E23-v38','T1':'T1'}
VARIABLES=['HUFL','HULL','MUFL','MULL','LUFL','LULL','OT']


def read(name):
    return list(csv.DictReader((ROOT/name).open()))


def find(rows,**wanted):
    found=[r for r in rows if all(r[k]==str(v) for k,v in wanted.items())]
    assert len(found)==1,(wanted,len(found))
    return found[0]


def main():
    if (DOCS/(PREFIX+'.md')).exists():
        raise FileExistsError('report already exists')
    diag=json.loads((ROOT/'diagnosis.json').read_text())
    follow=json.loads((ROOT/'followup.json').read_text())
    base=read('baseline_head_padding.csv');positions=read('position_interventions.csv')
    joints=read('joint_region_interventions.csv');repair=read('repair_metrics.csv')
    t1=read('T1_native_position_interventions.csv')
    cached=list(csv.DictReader((DOCS/'ETTh1_P4_E23_v37_v39_T1_learning_seed502_20261004.csv').open()))
    print('Cached resource columns',list(cached[0]),flush=True)
    summaries=[]
    for s in LABELS:
        native=find(base,source=s,predictor=s,padding='native')
        repaired=find(repair,model=s)
        oracle=find(joints,model=s,region='all_last32',intervention='oracle',strength=1.)
        half=find(joints,model=s,region='all_last32',intervention='oracle',strength=.5)
        all_oracle=find(base,source=s,predictor=s,padding='oracle_native')
        cached_row=next(r for r in cached if r['model']==('E23-v38-shared-features' if s=='v38' else DISPLAY[s]))
        summaries.append(dict(model=DISPLAY[s],training_seed=502,split='original_fixed_validation',
            scope='exploratory diagnostic; no new training or test evaluation',
            repair_mse=float(repaired['mse']),repair_mae=float(repaired['mae']),
            forecast_mse=float(native['mse']),forecast_mae=float(native['mae']),
            oracle_last32_mse=float(oracle['mse']),oracle_last32_mae=float(oracle['mae']),
            half_oracle_last32_mse=float(half['mse']),half_oracle_last32_mae=float(half['mae']),
            oracle_all_hidden_mse=float(all_oracle['mse']),oracle_all_hidden_mae=float(all_oracle['mae']),
            cached_full_training_seconds=float(cached_row['full_training_seconds']),
            cached_peak_allocated_gib=float(cached_row['peak_allocated_gib']),
            cached_peak_reserved_gib=float(cached_row['peak_reserved_gib']),
            current_diagnostic_total_seconds=diag['diagnostic_seconds']+follow['diagnostic_seconds'],
            current_diagnostic_peak_allocated_gib=max(diag['peak_allocated_gib'],follow['peak_allocated_gib']),
            current_diagnostic_peak_reserved_gib=max(diag['peak_reserved_gib'],follow['peak_reserved_gib']),
            checkpoint_sha256=diag['checkpoint_sha256'][s],artifact_root=str(ROOT)))
    with (ROOT/'summary.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summaries[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(summaries)
    for source,suffix in [('summary.csv','.csv'),('position_interventions.csv','_positions.csv'),
                          ('joint_region_interventions.csv','_joint_regions.csv'),
                          ('T1_native_position_interventions.csv','_T1_native_positions.csv')]:
        target=DOCS/(PREFIX+suffix)
        if target.exists():raise FileExistsError(target)
        shutil.copyfile(ROOT/source,target)

    fig,axes=plt.subplots(3,2,figsize=(11,10),layout='constrained')
    images=[]
    for i,s in enumerate(LABELS):
        rows=[r for r in positions if r['source']==r['predictor']==s and r['intervention']=='oracle']
        repair_map=np.asarray([float(r['repair_mse']) for r in rows]).reshape(6,7).T
        gains=np.asarray([float(r['mean_mse_gain']) for r in (t1 if s=='T1' else rows)
                          if r['intervention']=='oracle']).reshape(6,7).T
        images.append(axes[i,0].imshow(repair_map,aspect='auto',vmin=0,vmax=1.4,cmap='viridis'))
        images.append(axes[i,1].imshow(gains,aspect='auto',vmin=-.02,vmax=.02,cmap='RdBu_r'))
        for j in range(2):
            axes[i,j].set_yticks(range(7),VARIABLES)
            axes[i,j].set_xticks(range(6),['1-16','17-32','33-48','49-64','65-80','81-96'])
            axes[i,j].set_title(f'{DISPLAY[s]}: '+('hidden repair MSE' if j==0 else 'local oracle forecast MSE gain'))
            axes[i,j].set_xlabel('Original history positions')
    fig.colorbar(images[0],ax=axes[:,0],shrink=.7,label='Global standardized repair MSE')
    fig.colorbar(images[1],ax=axes[:,1],shrink=.7,label='Positive = forecast MSE reduced')
    fig.suptitle('Fixed validation, seed 502; frozen own head; native padding retained')
    fig.savefig(ROOT/'repair_vs_forecast_value.png',dpi=180)
    fig.savefig(ROOT/'repair_vs_forecast_value.pdf')
    plt.close(fig)

    lines=['# E23 / T1：seed=502 预测敏感恢复诊断', '',
        '日期：2026-10-04。状态：完成。没有新训练、测试预测或 checkpoint 选择；v31-B 仍为保留版本。', '',
        '## 核心结果', '',
        '固定 v31-B 的原预测头，只干预缺失历史数值；可见值、缺失 mask、query mean/std 和所有权重保持原样。下表是原固定验证集 2,688 窗口的受控干预，**不是新模型测试成绩**。', '',
        '| v31-B 输入条件 | 验证预测 MSE | 验证预测 MAE |',
        '|---|---:|---:|']
    native=find(base,source='v31B',predictor='v31B',padding='native')
    lines.append(f"| 原修复输出 | {float(native['mse']):.6f} | {float(native['mae']):.6f} |")
    for region,kind,strength,label in [
        ('LUFL_last16','oracle',1.,'只将 LUFL 最后16点的缺失值换成真值'),
        ('LUFL_last16','oracle',.5,'只将 LUFL 最后16点的修复误差修正一半'),
        ('all_last32','oracle',1.,'所有变量最后32点的缺失值换成真值'),
        ('all_last32','oracle',.5,'所有变量最后32点的修复误差修正一半'),
        ('all_last32','T1_donor',1.,'只将所有变量最后32点的缺失值换成 T1 估计'),
        ('all_hidden','oracle',1.,'全部96点的缺失值换成真值'),
        ('all_hidden','T1_donor',1.,'全部缺失值换成 T1 估计')]:
        r=find(joints,model='v31B',region=region,intervention=kind,strength=strength)
        lines.append(f"| {label} | {float(r['mse']):.6f} | {float(r['mae']):.6f} |")
    lines += ['',
        '1. 最近32点的恢复具有可测的预测收益；一半修正仍改善 MSE。所有变量最近32点全修正/半修正在验证集三个连续时间段的收益均为正。',
        '2. 局部替换 T1 估计有收益，但全部替换反而退步。整段修复结果不能直接互换，局部修复质量、预测器共同适应和区域交互共同影响预测。',
        '3. 全部真值输入并不保证固定预测头的 MSE 更低，也不是独立训练的 clean 模型或性能上界。该现象支持检查共同适应，不能证明真实系统因果关系或某个新训练方法必然有效。',
        '4. LUFL 最近16点的全/半修正在约68.4%/71.6%有缺失窗口改善预测，其余窗口保留负收益。不能将全局平均排序硬编码成每个窗口都成立的规则。', '',
        '## 三模型自身权重：修复与预测', '',
        '| 模型 | Stage2 缺失历史 MSE | Stage2 缺失历史 MAE | 验证预测 MSE | 验证预测 MAE | 最后32点真值干预后的预测 MSE |',
        '|---|---:|---:|---:|---:|---:|']
    for r in summaries:
        lines.append(f"| {r['model']} | {r['repair_mse']:.6f} | {r['repair_mae']:.6f} | {r['forecast_mse']:.6f} | {r['forecast_mae']:.6f} | {r['oracle_last32_mse']:.6f} |")
    lines += ['',
        'T1 Stage1 插补更准确，但这个 seed 的选定 Stage2 插补误差高于两个 E23；全局修复 MSE 与最终预测排序不一致。此表是验证诊断，T1 在已保存的20测试缺失组中仍优于 E23（完整五种子结论见原正式表）；不能用这里的验证排序替代测试结论。', '',
        '## 相同冻结 v31-B 预测头的交叉搭配', '',
        '| 修复来源 | 统一零补齐后的预测 MSE | 预测 MAE |', '|---|---:|---:|']
    for s in LABELS:
        r=find(base,source=s,predictor='v31B',padding='zero')
        lines.append(f"| {DISPLAY[s]} | {float(r['mse']):.6f} | {float(r['mae']):.6f} |")
    lines += ['',
        'v38 在 LUFL 最近16点的修复 MSE 由 v31-B 的0.697108降至0.541626，最近32点局部修复已有改善，但全部输入 v31-B 的冻结预测头仍没有降低 MSE。v38 的失败不能仅解释为没有修好该区域。交叉搭配仅作适应诊断，不属于经过联合训练的正式新模型。', '',
        '## 末尾补齐位置', '',
        'PatchTST 输入有末尾8个补齐位置。E23 输入为零，T1 复制最后修复值。当前预测器的 validity 约束只作用于缺失 embedding，数值投影仍接收补齐数值。', '',
        '| 模型，自己的固定预测头 | 原方式 MSE | 零补齐 MSE | 复制末值补齐 MSE |', '|---|---:|---:|---:|']
    for s in LABELS:
        metrics=[float(find(base,source=s,predictor=s,padding=m)['mse']) for m in ['native','zero','replicate']]
        lines.append(f'| {DISPLAY[s]} | {metrics[0]:.6f} | {metrics[1]:.6f} | {metrics[2]:.6f} |')
    lines += ['',
        'T1 改为零补齐、E23 改为复制补齐均退步，说明各自适应已有接口；不能据此将 T1 的测试优势归因于补齐，也没有修改正式接口。主区域矩阵为统一零补齐的9个 source/head 组合；T1 原补齐的42区域全/半干预另有独立 CSV，LUFL 最近16点收益仍为正（0.005932/0.003328）。', '',
        '## 下一轮可实施的方向', '',
        '建议在 v31-B 修复器内部新增按原时间坐标工作的局部残差精修支路，先针对所有变量最近32点，使用可见历史、原 mask 和专家/当前修复状态；从零残差起步，可见值严格保留。原预测头架构与 Stage2 目标保持现有 v31-B 配置。', '',
        '与相同参数量的全96点支路做对照，检验收益来自计算位置分配还是仅增加容量。位置范围由本次验证诊断启发，应记为后续开发选择；不能将 LUFL 指标或测试最优 seed 硬编码成普适规则。这个支路尚未实现或训练。', '',
        'v31-C 的位置加权保留损失已做过且未超过 B；此次建议直接增加局部恢复容量和逐点输出能力，不宣称已证明能优于 v31-C/T1。', '',
        '## 统计与资源口径', '',
        '- seed=502 是根据已经看到的正式测试结果挑出的开发 seed。本次只读原固定验证，所有结论属于开发诊断。',
        '- 42区域 = 6个不重叠16点区间 × 7变量；每个原时间坐标的所有有效重叠 patch 副本一并修改，观测、mask、query统计和 padding 保持指定控制。',
        '- 主矩阵588行 = 9个修复来源/预测头组合全真值干预 + 3个自身半干预 + 2个 T1 局部估计替换，共14×42；联合区域另60行；T1原补齐区域另84行。',
        '- 真值仅用于干预，未来标签仅用于评分；不构成可部署预测输入。正收益定义为干预前 MSE 减干预后 MSE，负收益保留。',
        '- 三个连续验证段仅是描述性符号检查，不是独立数据集或独立折；重叠窗口不按独立样本做显著性检验。',
        f"- 原冻结诊断耗时{diag['diagnostic_seconds']:.2f}秒，联合/原补齐追踪诊断{follow['diagnostic_seconds']:.2f}秒；合计{diag['diagnostic_seconds']+follow['diagnostic_seconds']:.2f}秒。GPU1 RTX5090，诊断峰值 allocated/reserved 为{max(diag['peak_allocated_gib'],follow['peak_allocated_gib']):.3f}/{max(diag['peak_reserved_gib'],follow['peak_reserved_gib']):.3f} GiB。", '',
        '| 模型 | 既有完整训练分钟 | 既有训练峰值 allocated / reserved GiB |', '|---|---:|---:|']
    for r in summaries:
        lines.append(f"| {r['model']} | {r['cached_full_training_seconds']/60:.2f} | {r['cached_peak_allocated_gib']:.3f} / {r['cached_peak_reserved_gib']:.3f} |")
    lines += ['',
        '训练资源来自已保存的原选定训练记录，本次没有重新训练；诊断资源单独记录，绘图与报告生成时间不计入 GPU 诊断前向时间。737个既有文件和全部模型/H状态的前后哈希一致；原五种子正式对照没有变更。', '',
        '## 文件', '',
        f'- [汇总 CSV]({PREFIX}.csv)',
        f'- [588行位置干预 CSV]({PREFIX}_positions.csv)',
        f'- [60行联合区域 CSV]({PREFIX}_joint_regions.csv)',
        f'- [T1 原补齐84行位置干预 CSV]({PREFIX}_T1_native_positions.csv)',
        '- [修复误差与局部预测收益热图](../../artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/e23_forecast_sensitive_seed502_20261004/repair_vs_forecast_value.png)',
        '- [热图 PDF](../../artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/e23_forecast_sensitive_seed502_20261004/repair_vs_forecast_value.pdf)',
        f'- 完整根目录：`{ROOT}`。含预先冻结计划、适应性补充计划、脚本、日志、checkpoint身份、9组合/补齐基线、逐窗口收益 NPZ 和完整资源。', '']
    text='\n'.join(lines)
    (ROOT/'report.md').write_text(text)
    (DOCS/(PREFIX+'.md')).write_text(text)
    print('Published',str(DOCS/(PREFIX+'.md')),flush=True)


if __name__=='__main__':
    main()
