# 独立实现审查

审查者：/root/v70_tv_history_review。只读；无文件修改、GPU工作或完整继承测试重跑。

没有Critical、Important或Minor问题。实现可继续已批准实验，最终准确性和资源表尚未审查。

- TV查询发生在T/V预览之后；仅完全缺失且有真实窗口上下文的有效patch替换候选，部分可见patch保留原候选。
- top4、0.7阈值、原mask及候选成员结束时间不晚于query的筛选保持。
- teacher输入为逆归一化回训练全局尺度的回溯真值，padding validity保留，仅进入Stage1对齐损失。
- 硬检索query detached；对齐梯度到查询/T/V，teacher及key encoder不接收此损失的梯度。
- 禁用专家与specialist预览遮罩有效；旧初始化/RNG、35427参数预测器保持；TV参数可strict保存/加载，配置身份拒绝错配。
- Stage2完整冻结包括新查询模块；优化器更新后修复hash不变。

证据：8项新增CPU测试通过；额外CPU探针验证完整模型hidden-NaN不变、空库仍保留对齐provenance、末尾padding/global teacher尺度、禁用/单专家遮罩、序列化与冻结。1229个独立锁定文件哈希一致：522源码、574受保护、129参考、3controller和plan（去重后计数）。

范围：最终成绩、资源CSV及完成manifest将在训练结束后复核；继承全套未重跑。库center/support可能包含更晚训练成员属于继承行为，本实验保持原策略，选定成员仍满足原end-before-query条件。未据此宣称整个库统计严格在线因果。
