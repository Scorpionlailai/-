# E23-v99：局部混合精度与执行优化

## 预先固定的设计

目标是在 v97 结构和完整训练预算下减少耗时/显存。先进行单个 ETTh1 sr015_bl8 场景试验，验证/测试和预测头仍使用 FP32。保留 v97、v98 原始源码、权重及报告。

- v97 完整结构：专家宽度64、八头候选读出总宽度64、mean+dense 修复细化；504885修复参数，97161细化可训练参数。
- 从头粗修复100轮、细化100轮、预测100轮；训练种子2023，batch128，GPU0。E24 在 GPU1 原流程运行。
- 粗修复/细化仅在训练 forward 中对 window context 与 point candidate readout 启用 BF16 autocast，输出返回 FP32。历史查询、原型库、损失、参数/梯度和Adam仍按原精度。验证和测试不启用该范围。
- 只编译 point candidate readout，保留全部参数对象和 state_dict 键；首次编译与不同模式所需编译均计入资源侧车的完整时间。
- Stage1 按固定验证全 hidden 全局标准化 MSE 选模；细化损失 full hidden MSE +0.25 tail32 hidden MSE +0.05 tail32 hidden MAE。
- Stage2 全部修复器冻结、纯MSE，原PatchTST与初始权重SHA不变；MAE主选、MSE副选。全部FP32，使用已验证的固定验证缓存。
- 选模后统一报告已有20组开发缺失位置的修复全窗口/最近32点、预测MSE/MAE以及缺失位置标准差。这不是新盲测，也不是多训练种子。
- 完整资源必须将父粗修复、细化和Stage2三个独立工作流相加；峰值取三阶段max。不能仅报告细化+预测成本。

## 前置检查

查询投影的等价拆分原型：Weather B128 forward/backward微测没有提速（174.41→174.84ms），不进入本轮完整训练。局部 BF16 原型：174.32→153.16ms，峰值10.30→9.03GiB；修复损失0.042731→0.042774，梯度相对L2差0.0301，观测值不变。BF16确实改变数值，不能作为零精度损失的替换，因此需要从头训练和FP32选模。以上仅为关闭dropout、无优化器的模块微测，不能直接外推完整训练收益。

预检查确认有限输出/梯度、观测保持、八头64维和原预测头初始SHA。运行前首次预检查因缺少隔离目录data软链接失败，修复路径后通过；未产生训练结果，也未更改实验配置。

## 可查验文件

- 计划与命令、冻结源码散列、权重、资源侧车、测试记录：`/home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/experiments/e23_v99_selective_bf16_20261008`。
- 隔离代码：`/home/xyw/paper/E23/versions/e23-v99-selective-bf16-20261008/E23`。
- 局部混合精度帮助器：`integrations/e23/selective_precision.py`。
- CLI 新参数仅用于本隔离实验：`--repair-compute-precision selective_bf16_v1 --compile-repair-readout`。
- 检测周期300秒；用户阶段汇报约600秒。结果尚在运行，不能宣称它优于v97或T1。
