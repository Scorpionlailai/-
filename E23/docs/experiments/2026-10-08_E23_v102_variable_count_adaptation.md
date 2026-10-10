# E23-v102：变量数量适配与 V 候选预算准备（2026-10-08）

## 本轮完成的范围

先完成有界的尺寸适配，沿用v97模型结构与两阶段规则。**尚未进行V4/V8的正式训练或准确率比较，也未实现动态top-p。**

1. RecentMeanProjector 去除七变量限制，recent-band统计拼接按6C计算，输入按13C+3计算。
2. GlobalDenseRefiner 输入35→5C、输出7→C；主干64维、两层时间Transformer和损失不变。参数数目77952+385C，C=7/21/64分别80647/86037/102592。
3. 训练、模型构建和原型库读取 `protocol.bounds.name` 对应数据集，不再固定ETTh1。动态训练缺失生成器读取实际训练长度与变量数，修复原8640行/7变量固定值。
4. `load_e23_settings(..., allow_variable_scaling=True)` 明确启用多变量试验，并允许V预算4或8。默认调用仍使用原ETTh1协议锁定，不能静默改变封存实验。
5. 修复参数/特征记录随C更新；候选数记录按C−1截断。新增数据集seal散列字段和末批验证选项，默认保留旧ETTh1行为。
6. Weather读取已有封存manifest，验证5175窗口全部保留，最后一批55窗口；训练窗口仍原batch128丢弃不足末批。预测头配置未修改，V4与V8初始预测头SHA完全一致。

代码入口为 `/home/xyw/paper/E23/versions/e23-v102-variable-count-20261008/E23`，软链接到隔离Git worktree `/home/xyw/.cache/codex-worktrees/e23-v102-variable-count-20261008`。工作树基于旧git提交并载入完整v100源码快照，因此git diff包含继承的历史变更；本轮真实差异以 [changes_vs_v100.patch](/home/xyw/paper/E23/artifacts/performance/e23_v102_variable_count_20261008/changes_vs_v100.patch) 为准。原E23/E24运行源码与封存权重未修改。

## 行为兼容性与验证证据

- 先执行失败测试：21/64变量在旧均值模块入口拒绝，多变量配置入口缺失，Weather loader缺少协议字段。
- 新增14项测试全部通过：7/21/64变量、三个均值模式、有限梯度、观测保持、dense无来源窗口不新增残差、多变量配置显式开启、Weather训练mask尺寸和全部验证窗口。
- 用保存的v97已训练均值+dense权重严格加载旧/新模块，在非零残差下输出逐张量完全一致；不是只验证零初始化。仅检查该后缀，没有完整验证集预测重测。
- 完整套件基线：91 failed /789 passed /1 skipped /2 errors；最终：91 failed /803 passed /1 skipped /2 errors。全部失败/错误标识集合完全相同，无新增失败。原失败包括 `tests/integration/test_e23_config.py`、`test_e23_stage1.py`、`test_e23_stage2.py` 中的迁移路径/协议约束，以及 `tests/s4m/test_s4m_provenance.py` 的历史checkout来源检查；各具体测试标识完整列于 [test_summary.json](/home/xyw/paper/E23/artifacts/performance/e23_v102_variable_count_20261008/test_summary.json)。因此不能声称完整测试套件通过。

## 固定候选4/8的运行成本

以下是**合成输入、batch16、完整修复forward/backward、无优化器、空H原型库**的单次微测。每种设置热身后3步取中位数，保持FP32。它不包含正式训练开销，也不反映训练后准确率，不应用于与T1的正式资源表。

| C | 配置V来源数 | 含三个时间偏移的候选槽数 | 单步ms | allocated GiB |
|---:|---:|---:|---:|---:|
| 7 | 4（有效4） | 12 | 22.97 | 0.484 |
| 7 | 8（有效6） | 18 | 22.92 | 0.516 |
| 21 | 4（有效4） | 12 | 31.22 | 1.384 |
| 21 | 8（有效8） | 24 | 33.62 | 1.585 |
| 64 | 4（有效4） | 12 | 67.41 | 4.181 |
| 64 | 8（有效8） | 24 | 74.50 | 4.799 |

21变量时V4→V8耗时约增加7.7%、显存约增加14.5%；64变量约增加10.5%/14.8%。七变量最多有6个非自身来源，所谓V8实际只能取6，应明确标注，不能声称使用了8个来源。固定预算8不增加模型参数，但扩大候选激活张量。

## 已准备的真实数据配置

- [Weather V4配置](/home/xyw/paper/E23/artifacts/performance/e23_v102_variable_count_20261008/weather_v102_V4.yaml)
- [Weather V8配置](/home/xyw/paper/E23/artifacts/performance/e23_v102_variable_count_20261008/weather_v102_V8.yaml)
- [真实Weather入口检查](/home/xyw/paper/E23/artifacts/performance/e23_v102_variable_count_20261008/weather_protocol_preflight.json)：各使用4个固定验证输入完成修复/预测前向，有限输出、观测保持，未运行优化器或测试评分。
- [候选成本CSV](/home/xyw/paper/E23/docs/experiments/E23_v102_variable_candidate_cost_20261008.csv)
- 源码、检查脚本及配置归档：[source_and_checks.tar.gz](/home/xyw/paper/E23/artifacts/performance/e23_v102_variable_count_20261008/source_and_checks.tar.gz)。

通用构建和训练函数现在能接收已加载的多变量协议；旧 `integrations.e23.run` CLI仍是锁定ETTh1入口，Weather不能直接套用该CLI。检查脚本只读使用E24的封存数据loader，未另建或修改缺失位置。

## 后续实验和限制

正式候选数量比较需在同一个多变量数据集、同一训练种子和缺失协议下分别从头训练V4/V8；Stage1修复全96/近32和Stage2预测均报告，Stage2冻结修复器、纯MSE，原预测头保持一致。该训练比较尚未开始。先根据验证指标评价候选质量，再决定是否设计有上限的动态候选。

这次尺寸适配能运行不同C，但不是跨数据集零样本权重迁移：输入/输出投影、变量embedding和位置参数随C变化，多变量必须重新训练。全变量统计拼接与变量注意力仍随C增长（其中一些为二次成本）；还没有解决百变量上的全部效率问题。变量数变大时，优先扩大低成本搜索池，再用有限最终候选预算，而不是直接把所有来源送入逐点读出。
