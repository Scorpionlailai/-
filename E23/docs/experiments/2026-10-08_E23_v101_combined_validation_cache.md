# v101：Stage1 与 Stage2 固定验证缓存组合验收

## 完成范围

ETTh1 sr015_bl8、seed2023、FP32、batch128，结构和预测头均与v97一致。v100 已完成100轮修复细化，逐轮验证记录和全部选中修复权重与v97相同；本轮以该修复器运行100轮Stage2，只启用固定验证缓存。Stage2修复器完全冻结、纯MSE、原PatchTST配置与初始权重不变。

按验证MAE和MSE选出的两个完整模型状态（全部修复器与预测器张量）均与原v97逐张量完全相同，选中轮次、历史库/key与预测头初始散列相同。因此沿用原同20组缺失位置的测试结果，不重复测试。

## 精度与资源

- 主结果预测MSE **0.398613**、MAE **0.407094**，与v97相同。
- Stage1全窗口hidden修复MSE/MAE仍为0.176235/0.264315，最近32点为0.180031/0.269464。
- 本次Stage2完整实测 **242.562秒**，原 **290.643秒**，减少 **16.54%**。
- 计入父粗修复571.690秒、v100细化263.969秒后，三阶段成本合计 **17.970分钟**；原v97为19.483分钟，减少 **7.76%**。
- 三阶段峰值allocated **4.623GiB**，reserved **6.404GiB**。全流程GPU峰值由粗修复决定，验证缓存未降低它。Stage1 CPU缓存约579MiB，另有Stage2CPU缓存。

该成本合计来自三个分别实测的训练记录，粗修复与细化沿用已有记录，**本轮没有重新跑一个连续的300轮冷启动流程**。各完整计时均包括准备、验证、保存及缓存构建，不含最终测试或独立诊断；单次速度实测，不是多次计时均值。

## 查验

- [精度CSV](/home/xyw/paper/E23/docs/experiments/E23_v101_combined_cache_results_20261008.csv)
- [资源CSV](/home/xyw/paper/E23/docs/experiments/E23_v101_combined_cache_resources_20261008.csv)
- [严格等价验收](/home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v101_combined_validation_cache_20261008/acceptance.json)
- [完整成本及原始来源](/home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v101_combined_validation_cache_20261008/complete_training_cost.json)

v101为执行配置标签，学习结果仍是v97，不能算作一个新的精度改良模型。多变量数据集尚未完成这套组合的完整训练验收。E24后台和封存版本未修改。后续正式效率比较应给各冻结修复基线相同验证缓存机会。
