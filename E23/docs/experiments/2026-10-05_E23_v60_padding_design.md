# E23 v60：预测输入末尾padding单项对齐

状态：Stage1原100轮复用、新Stage2 100轮及20组测试完成；17项检查、GPU预检查和最终审计通过。

v59方案未通过固定验证门槛，按测试前的父版本决定继续从v58出发。本方案在v59评分之前声明；其开发测试领先的v59权重及所有记录保持。此次只改变原预测接口的末尾无效补齐值：从0改为最后一个修复时间点的值，与保存的T1 wrapper一致。原真实有效值、观测mask、validity不变，使用已有forecast_padding实现。

整个修复模型和原PatchTST预测头源码保持，参数量不变，预测头仍35427参数，配置与seed502初始权重相同。Stage2 objective和epoch loop AST保持；优化器、lr、loss、保留、冻结均继承v58，不采用common_stage2的其他改动，不新增输入通道或预测参数。

ETTh1 96→96，起点率0.15、块8/重叠，旧20组实际缺失72.9252%，seed502/validation2021。复用v58已完成100轮的Stage1权重及配对H库；新Stage2从头100轮/batch128，MSE+0.2MAE、epoch11起0.05uniform保留，原CNN/Hkey/library冻结，其他修复lr1e-5、预测器lr1e-4。

按固定验证MAE选checkpoint；只有验证预测MSE比父版低且MAE不差才升级，在任何测试之前保存决定。每版完成旧20组测试，报告平均MSE/MAE及位置标准差。完整模型成本仍计入已测量的原Stage1和新Stage2，actual_new_training_seconds单独记本次新增Stage2成本；训练显存取两阶段allocated/reserved峰值，测试与诊断另计。

17项检查包括：有效预测输入和mask/validity保持、无效补齐值精确匹配T1规则、head权重/参数未变、修复结果未变、带padding身份的checkpoint/验证重放与篡改拒绝、Stage2 objective/epoch loop及原模型源码相同、未知padding拒绝、原可见时间读取和CNN冻结兼容。GPU预检查用父版Stage1原件做两步真实训练样本的新Stage2，正式训练重新初始化。

代码：`/home/xyw/paper/.e23-v60-padding/E23`。
记录：`/home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v60_padding_sr015_bl8_seed502_20261005`。
