# 运行入口说明

新E24模型的正式训练入口是 `python -m integrations.e24.run`，它强制纯MSE及完整冻结修复器。`check_e24_inheritance.py` 只读取父版权重验收，不进行训练。

其余脚本从v79来源保留，供查验作者配置、指标/资源接口和历史适配。尤其 `run_author_aligned_p4.py` / `run_t1_p4.py` 的旧插补预测调度不能直接当作E24规范下的冻结Stage2运行：需要使用匹配的冻结适配器、相同公共头初始化和新数据集执行manifest，并通过规范验收。`integrations/forecast_compare/t1_frozen.py` 提供现有只读T1修复包装器。

公共Stage2的正式基线执行矩阵将在多数据集适配阶段建立；当前没有启动任何基线重训。
