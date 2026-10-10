# E23 v48→v49→v50：逐项优化结果

状态：3/3个条件完成。ETTh1 96→96，实际缺失72.9252%，训练seed502，每模型20组已有缺失位置。

| 模型 | 测试MSE | 测试MAE | 验证MSE / MAE | 完整训练分钟 | allocated / reserved GiB |
|---|---:|---:|---:|---:|---:|
| E23-v41-slow | 0.455761 ± 0.015292 | 0.453554 ± 0.005660 | 0.794754 / 0.607057 | 8.98 | 2.307 / 5.188 |
| T1 | 0.411002 ± 0.013937 | 0.427785 ± 0.004383 | 0.774405 / 0.594832 | 10.85 | 1.779 / 2.699 |
| E23-v46-front-context | 0.446929 ± 0.014109 | 0.444050 ± 0.004880 | 0.816389 / 0.607867 | 9.84 | 2.676 / 5.539 |
| E23-v47-frozen-front | 0.443177 ± 0.014356 | 0.441950 ± 0.004887 | 0.809692 / 0.605520 | 9.73 | 2.676 / 5.539 |
| E23-v48-point-router | 0.438415 ± 0.014375 | 0.438458 ± 0.004890 | 0.797709 / 0.603157 | 10.35 | 2.887 / 5.758 |
| E23-v49-overlap | 0.441748 ± 0.014735 | 0.439822 ± 0.005234 | 0.796707 / 0.601021 | 11.16 | 3.046 / 5.924 |
| E23-v50-recent-state | 0.439688 ± 0.014096 | 0.438508 ± 0.004785 | 0.797656 / 0.601263 | 11.19 | 3.046 / 5.924 |

## 预先规定的顺序和判断

每项在当前保留父版本上只增加一项变化。均从头100轮Stage1＋100轮Stage2。预测checkpoint按固定验证MAE选择；验证预测MSE更低且MAE不差时保留新版本。父版本决定在该项测试之前保存。已有20组位置作为开发测试，未用于本轮自动父版本选择；v47开发起点由用户在查看旧测试后指定。

- E23-v48-point-router：父版本E23-v47-frozen-front；仅新增point_router。配置point_router=pointwise_v1、overlap_fusion=none、stage1_structure=none。选中Stage2第88轮，父版本验证门槛通过，原v41严格门槛未通过。

- E23-v49-overlap：父版本E23-v48-point-router；仅新增overlap_fusion。配置point_router=pointwise_v1、overlap_fusion=confidence_v1、stage1_structure=none。选中Stage2第59轮，父版本验证门槛通过，原v41严格门槛未通过。

- E23-v50-recent-state：父版本E23-v49-overlap；仅新增stage1_structure。配置point_router=pointwise_v1、overlap_fusion=confidence_v1、stage1_structure=recent_state_v1。选中Stage2第56轮，父版本验证门槛未通过，原v41严格门槛未通过。

本序列按验证保留：**E23-v49-overlap**。其20组测试平均MSE相对v47降低0.32%，相对T1的MSE差距为+7.48%。

三项新实验的开发测试最低MSE：**E23-v48-point-router**（MSE 0.438415，MAE 0.438458）。该描述不改变测试前锁定的父版本判断。

## 固定验证集的修复诊断

所有修复指标在训练集全局标准化尺度上统计；每个窗口内的时间点只计一次。近期水平指标取修复后窗口最近32点的平均误差。

| 模型 | Stage1全窗修复MSE | Stage1最近32点MSE | Stage2全窗修复MSE | Stage2最近32点MSE | Stage2近期水平MSE |
|---|---:|---:|---:|---:|---:|
| E23-v48-point-router | 0.397854 | 0.390850 | 0.497061 | 0.545037 | 0.138262 |
| E23-v49-overlap | 0.388773 | 0.387905 | 0.499591 | 0.579766 | 0.159583 |
| E23-v50-recent-state | 0.389829 | 0.386715 | 0.499104 | 0.560624 | 0.148830 |

## 实现与资源口径

逐点路由只重排none/H/V/T之间的数值融合，原patch潜表示路由、各专家内部候选聚合和候选资格保留；不可用专家权重为零。重叠副本置信度采用正权重归一化，原mask/观测/覆盖次数保留，它是学习到的聚合权重。近期损失只加在Stage1训练历史上，最近32点均值与趋势幅度各加权0.2；Stage1仍按全局隐藏MSE选模。

原PatchTST预测头源码、配置、初始化保持，参数35427。本轮Stage2训练函数和损失函数AST与v47完全相同。E23 Stage2继续冻结窗口CNN前端，其他修复参数lr1e-5、预测器lr1e-4，原MSE+0.2MAE和第11轮起0.05均匀修复保留。T1复用作者配置及既有预测扩展，Stage2纯MSE/联合lr1e-4，训练配方差异保留。

同一RTX5090顺序运行。完整训练时间包含两个阶段的准备、训练、每轮验证和保存；最终测试与额外诊断另计。显存为两个训练阶段的PyTorch allocated/reserved峰值。各模型20组标准差只反映缺失位置变化，单训练seed不足以代表训练随机性。

[汇总CSV](ETTh1_E23_v48_v49_v50_point_series_sr015_bl8_seed502_20261005.csv) · [测试位置明细](ETTh1_E23_v48_v49_v50_point_series_sr015_bl8_seed502_20261005_all_draws.csv) · [配对差值](ETTh1_E23_v48_v49_v50_point_series_sr015_bl8_seed502_20261005_paired.csv) · [修复诊断](ETTh1_E23_v48_v49_v50_point_series_sr015_bl8_seed502_20261005_repair_diagnostics.csv)

冻结代码：`/home/xyw/paper/.e23-point-series/E23`。完整权重、日志、资源、源码归档和补丁：`/home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v48_v49_v50_point_series_sr015_bl8_seed502_20261005`。
