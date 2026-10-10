# E23-v52：V候选的变量身份和可见尺度校准

状态：两阶段100轮及20组测试完成；35项检查、GPU预检查和最终审计通过。

用户授权继续迭代直到超过T1，预测头不改。v52父版本由v51在测试之前保存的验证决定确定：通过则基于v51，不通过则维持v49。候选两种配置在v51测试之前预先保存；不根据v51测试分数修改分支。

## 当前信息缺口

原V的source/target embedding用于来源选择、候选先验和质量，但消息MLP本身只输入source局部归一化值、mask、target状态、lag、可见比例和一致性。没有显式输入来源/目标变量身份或各自visible-only归一化尺度；多个变量的标准化形状相似时，这会限制消息的数值解释能力。v51的逐点候选读取保留了候选，但不能直接补回这些缺失特征。

## 唯一新增分支

保留V原硬top4、3个offset、候选资格、原值/mask、原message及source context。在每个原V候选message上添加MLP164→64→64增量：原source/target embedding各64、原可见source patch恢复到训练全局标准化尺度16、原source mask16、source/target可见窗口mean及log(std)共4。std只在取log时clamp到1e-6。

source坐标通过原cache来源索引读取；不使用训练历史隐藏真值或未来目标作为输入。无效候选增量为零。末层零初始化，构造隔离RNG，保持原参数、原消息、原初始修复函数及原预测头初始化。新增14720参数，变量身份embedding复用原参数。变量embeddings随后接受消息分支梯度，其变化也可能影响原top4选择，因此不将结果单独归因于某一种输入特征。

## 固定条件与验收

ETTh1 96→96，原起点率0.15、块长8、允许重叠，20组既有测试位置平均缺失72.9252%。训练seed502、固定验证mask2021；新Stage1和Stage2各100轮，batch128/FP32。Stage1全局隐藏MSE＋0.1检索、specialist exposure0.2；Stage2原预测MSE＋0.2MAE、第11轮起0.05修复保留，冻结前端/H key/库，其他修复lr1e-5、原预测器lr1e-4。

checkpoint仍按固定验证MAE选择；验证预测MSE严格低于父版本且MAE不差才保留，决定在测试前保存。权重锁定后评估同20组测试位置并报告平均MSE/MAE及位置标准差。测试不参与本轮自动分支选择。

同一RTX5090顺序训练，记录完整两阶段时间和allocated/reserved显存，测试与额外诊断另计。若开发结果超过T1，将模型锁定后生成新的同场景位置，与T1保存权重做独立确认。

保存源码、配置、SHA、补丁、测试、资源、权重和诊断；所有以前完成实验的材料保持。隔离代码：`/home/xyw/paper/.e23-v52-variable-calibration/E23`。实验根：`/home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v52_variable_calibration_sr015_bl8_seed502_20261005`。
