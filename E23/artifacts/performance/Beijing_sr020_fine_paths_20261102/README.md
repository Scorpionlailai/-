# Beijing_sr020_fine_paths_20261102

[最终报告](final_report.md) · [固定实验设计与配置](plan.json) · [归档清单](archive_manifest.json) · [本机权重指纹](local_weight_manifest.json)

这是Beijing sr020_bl8、训练seed2023、20组测试遮挡的解释实验记录。粗恢复器冻结，不是下游预测Stage2。包含实际源代码、逐mask指标、验证/训练元数据、压缩逐轮曲线、资源与核验记录。

本目录不是独立可执行的模型包：未上传新权重、原始数据、NPZ遮挡库、预测缓存、优化器状态。源码和source_lock保留原样，plan中的路径来自原服务器；迁移需要先恢复数据/遮挡和指定粗权重并验证哈希，不能绕过绑定。原T1权重可从本分支已发布的commit7e6b3e8f获取。

最终结论见[跨实验总览](../../../docs/experiments/2026-11-02_Beijing_explanatory_experiments_summary.md)，保留小增益和负结果，不把20组mask当作20个训练seed。
