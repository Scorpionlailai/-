"""Progress and six-model tables for the subject-isolated PAMAP2 protocol."""
import csv
import json
from pathlib import Path

from .data import SCENES, write_json

MODELS = ('TimesNet','ImputeFormer','BRITS','CSDI','v125','T1')
DATASET = 'PAMAP2'


def read(path):
    path=Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def render(root):
    root=Path(root)
    rows=[]
    for scene in SCENES:
        for model in MODELS:
            folder=root/DATASET/scene/model/'seed_2023'
            status=read(folder/'status.json') or {}
            row=dict(dataset=DATASET,scene=scene,model=model,status=status.get('phase','queued'))
            for label,prefix in [('repair','repair'),('forecast_mae','forecast'),('forecast_mse','secondary_forecast')]:
                record=read(folder/'evaluation'/f'{label}_summary.json')
                if record:
                    for section,scores in record['summary'].items():
                        for key,value in scores.items():
                            if key != 'coordinate_weighted':
                                row[f'{prefix}_{section}_{key}']=value
            plan=read(folder/'plan.json')
            if plan:
                row['parameter_source']=plan['profile']['source']
                row['parameter_status']=plan['profile']['parameter_status']
            rows.append(row)
    columns=sorted(set().union(*(r.keys() for r in rows)))
    with (root/'results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(rows)
    repairs=sum('repair_full96_mse_mean' in r for r in rows)
    forecasts=sum('forecast_forecast_mse_mean' in r for r in rows)
    write_json(root/'review_status.json',dict(expected_runs=24,completed_repairs=repairs,
        completed_forecasts=forecasts,ready_for_final_audit=forecasts==24,test_missing_groups=20,
        ready_for_user_review=(read(root/'final_saved_output_audit.json') or {}).get('passed') is True,
        github_uploaded=False,github_policy='local_only_until_user_review'))
    verified=(read(root/'final_saved_output_audit.json') or {}).get('passed') is True
    lines=['# PAMAP2 六模型复现','',f'进度：修复 {repairs}/24；预测 {forecasts}/24。'+('缓存与冻结状态审核通过，待用户审核。' if verified else '最终报告须完成缓存与冻结状态审核。'),'',
        '数据为PAMAP2的Protocol和Optional全部14份记录、40个有效传感器通道（不含活动标签、时间戳及无效姿态）。101–106号受试者训练、107号验证、108–109号测试，同一人全部记录始终同组。原始100Hz按右闭100ms区间因果聚合至10Hz；回溯96点、预测96点，各9.6秒，切窗步长16点。','',
        '先按文件、活动变化、活动0区间、时间重置和超过0.015秒的原始时间间断分段，再在每段内聚合、填充、生成缺失和切窗。剔除活动0后两侧仍隔离；历史96点和目标96点均须完整位于同一连续段，不跨记录借历史。每段分别前向填充，开头仅用训练受试者的观测中位数占位；原始无观测的聚合坐标不作可见输入、监督或评分。训练标准化器仅拟合训练段填充序列。无真实日历时间，TimesNet/ImputeFormer外生日历通道统一置零。','',
        '六个模型共享四场景与20组测试掩码；掩码在各连续记录段内独立生成，跨模型逐组一致。表中块起点率不等于最终缺失率；报告各组实际坐标数。单训练seed2023，掩码标准差ddof=1不代表训练种子标准差。','',
        'Stage1先保存训练、验证和20组测试的FP32修复缓存；公共PatchTST冻结修复器，固定训练掩码复用缓存，100轮、batch128、Adam lr1e-4、预测MSE；验证MAE为主选择，验证MSE另报，测试不重新调用修复器。','',
        '作者参数来源写入每个plan.json。TimesNet使用作者Weather插补基础配置；ImputeFormer使用作者Air36基础配置（该数据集不是PAMAP2）；BRITS使用作者基础网络及此前授权执行设置。以上均不声称是PAMAP2最佳参数。CSDI为100轮、75/90衰减、50扩散步，训练缓存1次采样，修复评价及最终测试5次采样取中位数。','',
        'v125沿用封存结构：粗修复100轮，64/256校准各100轮，固定1:1；40通道下检索对比损失采用512行分块及梯度检查点，保留全部负样本与原始batch128，数值/梯度等价测试通过；加载历史原型时精确保留原float64键，避免重复归一化引入最后位变化。T1保留作者结构并改输入变量数40，100轮。六模型均从本数据训练，旧数据权重不转移。','',
        '| 场景 | 模型 | 全96修复 MSE / MAE | 近32修复 MSE / MAE | 预测 MSE / MAE | 状态 |',
        '|---|---|---|---|---|---|']
    def pair(r,p):
        return f"{r[p+'_mse_mean']:.6f} / {r[p+'_mae_mean']:.6f}" if p+'_mse_mean' in r else '—'
    for r in rows:
        lines.append('| '+' | '.join((r['scene'],r['model'],pair(r,'repair_full96'),pair(r,'repair_recent32'),pair(r,'forecast_forecast'),r['status']))+' |')
    lines += ['','完整均值、位置标准差、次选分支与参数来源见 [results.csv](results.csv)。当前结果先在本地保留，用户审核后再上传。']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    return rows
