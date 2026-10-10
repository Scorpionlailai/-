# E23 v67：完全冻结时的纯MSE对照

沿用用户之前明确要求的纯MSE方向，不修改原预测头源码、配置、35427参数或同seed初始化。在已经完成的同架构完全冻结控制中，按固定验证预测MAE挑控制（MSE及预定种子顺序打破平局），复用其Stage1、库、键和训练seed；不按开发测试挑Stage1。整个修复器依旧frozen_v1，预测MSE+.2MAE改为MSE；删除uniform保留项，此项在修复器冻结时是常数、对修复器/头均无梯度。保持原legacy padding和RNG，不引入common-stage2额外设置变动。

仅新增一个完整100轮Stage2、batch128、lr1e-4；保存MAE主/MSE次checkpoint及测试前的验证父版决定。评估旧20位置，报告完整原Stage1+新增Stage2成本及显存；增量训练只计新增Stage2，选择共享训练。10项检查包含真实纯MSE/混合冻结往返、同seed头初始化、原padding/RNG身份以及逐参数修复器保持。
