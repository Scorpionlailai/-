# E23-v103：有预算限制的动态 V 候选

## 状态与范围

2026-10-08，独立实现并验证动态 V，Weather 0.10/8 的 Stage 1 正式训练已启动。**此文的候选分布与微基准不是新模型的修复或预测结果。** ETTh1 封存模型及 E24 源码、训练任务均未修改。

- 实验代码：`/home/xyw/.cache/codex-worktrees/e23-v103-dynamic-candidates-20261008`。
- 用户目录入口：`/home/xyw/paper/E23/versions/e23-v103-dynamic-candidates-20261008/E23`。
- 实现、检查、配置、任务及资源记录：`/home/xyw/paper/E23/artifacts/performance/e23_v103_dynamic_candidates_20261008`。
- `training_job.json` 保存真实 PID、日志、输出路径；最终完成状态必须查进程、训练日志、selection 和 resources，不能只看这份启动记录。

## 机制

策略名为 `window_context_nucleus32_p90_cap16_v1`。

1. 使用原有可训练 source/target embedding 的全局关系，对每个目标变量排除自身，预选最多32个来源。Weather 有21个变量，因此搜索池为其余20个。
2. 对目标 patch 及其相邻来源 patch，使用全局关系、可见比例、共同可见位置的一致性评分：`prior + 0.5 * visible_fraction + 0.5 * consistency`。隐藏值不参与评分。
3. 对有可见信息的来源做 top-p=0.9 选择，再施加最多16个来源的上限。无可见来源时输出空证据。
4. 为支持批量计算保留固定容量的槽位，以布尔 mask 标记实际来源；只让有效候选进入 V 的消息 MLP。后续上下文投影、逐点读取仍有容量相关的开销，不能声称整个模型的耗时与实际候选数成比例。
5. 每个来源依然最多读取3个 patch，最大48个候选位置。V 参数的形状不变，可以严格加载原 V 权重。
6. 当变量数大于33时，训练期间按 epoch 与目标变量确定性地轮换搜索池末位，探索一个池外来源；测试时关闭。探索仍受32/16的预算限制，不强制扩大最终核集。

**达到16个上限时，保留的概率质量可能小于0.9。** 记录未截断候选数、实际数量、是否触顶、截断后的概率质量。没有把截断后的概率重新归一化成“已经覆盖90%”来报告。

样本相关的比较由完整变量两两比较改为目标数 × 至多32个来源；全局 embedding 的关系矩阵仍是 C×C。其他模块的变量相关开销也仍存在。

## 隔离与不变量

- H 保持原设置；T、融合、预测头均不因本次动态选择而改变。
- 原固定候选路径保持原行为；默认 ETTh1 协议入口不能直接启用此策略，需要显式 `allow_variable_scaling=True`。
- 已用真实 Weather 数据验证一次 Stage 2 更新：优化器只更新预测器，修复器无梯度且权重哈希不变，损失为纯 MSE。
- PatchTST 初始化哈希与原版相同：`fbbe8cde92c6417f1b5f67db888e20437dae434709b9e2adc834ce91118f6561`。

## 已完成检查

- 相关测试：39 passed，包括候选数随置信度变化、无来源、自身排除、隐藏值与 NaN 排除、梯度、参数兼容、配置预算、多变量尺寸及真实 Weather 数据协议。
- 全量测试：808 passed，91 failed，1 skipped，2 errors。失败标识与 v102 相同，没有新增失败；旧问题仍未解决，不能说全量通过。
- Weather 真实训练批次 batch128 的前向、反向检查通过，空 H 库下峰值 allocated 为14.683 GiB；这不代表完整训练峰值。
- Stage 1 验证记录新增动态来源直方图、平均数量、触顶比例、保留概率质量，以及近32点修复 MSE/MAE。

## 用已有 Weather 权重检查动态性

读取已有 Weather v79、0.10/8 的 V 权重，只做来源选择诊断。没有更新权重，没有测试新模型精度。

| 项目 | 数值 |
| --- | ---: |
| 验证窗口 | 5,175 |
| 需要修复的目标变量 patch | 1,174,018 |
| 每目标 patch 的平均来源数 | 8.091346 |
| 实际来源数范围 | 1–16 |
| 因16上限发生截断的比例 | 0.014651% |
| 平均保留概率质量 | 0.915247 |

明细：`weather_trained_prior_candidate_distribution.json`。数量是每个目标变量 patch 的来源数，不是整个窗口所有专家的数量。

## 合成资源微基准

同一 GPU 0（RTX 5090）、batch16、完整修复器前向与反向、无优化器、空 H 库，取预热后三次耗时的中位数。两种候选策略的参数量相同。

| 变量数 | V 策略 | 步耗时 ms | 峰值 allocated GiB |
| ---: | --- | ---: | ---: |
| 21 | 固定4 | 31.55 | 1.382 |
| 21 | 动态、上限16 | 39.65 | 2.056 |
| 64 | 固定4 | 67.59 | 4.184 |
| 64 | 动态、上限16 | 92.14 | 6.339 |

动态版相较固定4扩大了预算，因此没有更快、更省显存。64变量的随机未训练关系在非空样本上普遍触顶；真实已训练 Weather 的关系则显示数量变化。没有64变量已训练模型的动态性证据。明细为 `candidate_cost.csv` 和 `scaling_checks.json`。

## 正式 Stage 1 试验

为了单独判断动态 V 的作用，本轮在 **Weather 已有 v79 的结构**上只替换 V 选择策略和来源上限，不把 v97 的新 dense residual 同时加入。

- 数据：Weather，21变量；沿用 E24 已封存数据划分与标准化。
- 场景：起点率0.10、块长8；沿用原 `shared_protocol_v1/dataset_manifest.json`，不重新生成 mask。
- 回溯96、预测96；修复 patch16、stride8，其他设置与已有 Weather v79 相同。
- 随机种子2023，batch128，从头训练100轮，Adam lr0.001，weight decay0。
- Stage 1 主目标全局标准化空间的隐藏位置 MSE；保留原 retrieval 与 recent-mean 辅助项，不使用未来预测标签。
- 按固定验证集全局隐藏修复 MSE 选权重。验证与测试保留末尾不足 batch 的窗口。
- 基准：已有 Weather v79 固定4来源的100轮 Stage 1 记录，路径在 `training_job.json`。
- 训练完成后，自动在现有20组开发测试 mask 上报告修复96点与近32点 MSE/MAE 的均值及 mask 标准差。这些是已有开发测试位置，不能作为新的盲测或多训练种子证据。
- 此时尚未安排 Stage 2 正式训练；后续若做，必须冻结全部修复器、使用纯预测 MSE、预测头保持原配置。
- `TrainingResources` 记录 imports/data/training/validation/save 的全流程时间及 allocated/reserved 峰值。训练完成后的20组修复评估另记评估时间，不混入训练耗时。

输出：`/home/xyw/paper/E23/artifacts/Weather/seq96_pred96/sr010_bl8/e23_v103_dynamic_candidates_20261008/seed_2023/E23-v103/stage1`。

## 复查与复现

运行入口为诊断目录内的 `train_weather_stage1.py`，GPU0。该入口拒绝覆盖已有 Stage1 输出，因此不应在当前任务运行时再次启动。

检查启动任务：读取 `training_job.json` 后用 `ps` 核对 PID，查看 `weather_stage1_training.log`。完整训练输出选择文件 `stage1/runs/e23-stage1.json`、resources sidecar，以及自动评估输出 `stage1_repair_20_masks.json`。

`frozen_source.json` 记录代码、训练配置、入口及只读数据加载器 SHA256；`changes_vs_v102.json` 列出实际改动；`source_and_checks.tar.gz` 封存代码与启动前检查，未包含未完成的训练权重。

## 续跑监测与对照

续跑时未重启训练或更改冻结代码。`watch_v103.py` 独立每300秒读取训练进程、验证历史和最终输出，生成：

- `monitor_status.json`：实际已完成轮数、当前最佳验证指标、训练/评估/完成/异常状态。
- `matched_epoch_validation.csv`：v103 与已有固定4来源模型相同 epoch 的验证 MSE。
- `stage1_comparison.csv`：固定4、T1、v103 的修复96/32点 MSE/MAE、20组 mask 标准差、Stage1 全流程耗时、allocated/reserved 显存。v103 未完成的测试与资源项留空，不能填估算值。

T1 使用其已有作者配置记录：seed502、batch16，最多300轮并按原早停规则实际训练120轮、最佳第90轮；固定4与v103为 seed2023、batch128、固定100轮。两者训练预算不同，表中明确列出，不宣称完全相同预算。

现有20组 Weather 测试的修复对照如下，来源均为**冻结 Stage1 修复器**的下游评估记录，已确认两者使用同一数据 manifest 与20个 mask SHA256：

| 模型 | 修复96 MSE | 修复96 MAE | 修复32 MSE | 修复32 MAE |
| --- | ---: | ---: | ---: | ---: |
| Weather-v79 固定4 | 0.045973 | 0.067763 | 0.046785 | 0.073846 |
| T1 | 0.047406 | 0.064644 | 0.048624 | 0.066960 |

这是已有模型的完整测试结果。v103 的中途验证数值不能与这组测试数值直接比较。最终合并时，监测脚本还会核对 v103 的20个 mask SHA256、窗口数以及隐藏坐标数，并检查评估未改变修复权重。


## 正式完成结果

Stage1完成100轮，按固定验证全窗口修复MSE选择第93轮；同20组Weather 0.10/8开发测试评估已完成。以下均为修复指标，**尚未运行Stage2，不存在v103预测结果**。

| 模型 | 修复96 MSE | 修复96 MAE | 修复32 MSE | 修复32 MAE |
| --- | ---: | ---: | ---: | ---: |
| 固定4 Weather-v79 | 0.045973 | 0.067763 | 0.046785 | 0.073846 |
| T1 | 0.047406 | 0.064644 | 0.048624 | 0.066960 |
| v103 动态V | 0.044039 | 0.065166 | 0.045115 | 0.071264 |

动态版四项修复测试均值均优于固定4来源；相对T1，两个修复MSE更低，但两个MAE仍更高。只能说本场景测试修复MSE改善，不能宣称全面超过T1或预测改善。固定验证修复MSE0.178645、MAE0.096531；近32点0.179049/0.101818。验证MSE仍高于T1，测试与验证不能混为同一个结论。

Stage1完整命令耗时159.772分钟，峰值allocated23.408GiB、reserved30.671GiB；同场景固定4为128.210分钟、allocated20.062GiB。扩大动态预算并未带来效率提升。最终评估148.268秒单列，不计入训练时间。

最终验证平均来源数6.152692，保留概率质量0.921526，触顶截断比例0。20组mask标准差、精度和资源原始值见 `stage1_comparison.csv`、`stage1_repair_20_masks.json` 和resources记录。v97队列已按用户要求暂停，不会自动启动。

## Stage2 正式启动

用户授权进行预测后，已用第93轮修复权重启动Stage2：Weather 0.10/8，seed2023、batch128、100轮、原PatchTST初始化与配置、修复器全冻结、纯预测MSE、验证MAE主选及验证MSE副选。启动前已严格核对Stage1 manifest/checkpoint身份与原预测头初始化SHA。启用固定验证缓存，不更改训练损失或模型。v97队列保持暂停。

`prediction_job.json`、`weather_stage2_training.log` 保存PID与日志；`watch_prediction.py` 每300秒读取状态。完成后自动评估两份权重的原20组开发测试mask，主结果汇入 `forecast_comparison.csv`，原始记录位于 `E23-v103/evaluation_mae` 和 `evaluation_mse`。`complete_training_cost.json` 合计Stage1和Stage2完整命令时间；最终测试耗时单列。启动不代表已有预测结果。

## Stage2 最终结果

预测训练100轮及两套20组开发测试评估完成。验证MAE与MSE均选择第63轮，得到相同测试结果；修复器冻结已经核验。

| 模型 | 预测 MSE | 预测 MAE |
| --- | ---: | ---: |
| Weather-v79 固定4 | 0.199519 | 0.247416 |
| T1 | 0.191598 | 0.231173 |
| v103 动态V | 0.197220 | 0.242822 |

v103相较固定4的预测MSE改善1.15%、MAE改善1.86%，仍落后T1。20组mask标准差：MSE0.001142、MAE0.000732。单训练种子、现有开发mask，不是盲测或多训练种子结果。测试修复MSE更低未转化为超越T1的预测。

Stage2完整命令55.235分钟，allocated峰值7.767GiB。Stage1+Stage2完整命令成本合计215.007分钟，整体allocated/reserved峰值23.408/30.672GiB；最终测试耗时不计入训练。最终原始JSON和预测对照见 `evaluation_mae/twenty-test-draws.json`、`forecast_comparison.csv`。继承的E24评估器输出模型名称原先为硬编码E24-v79，现已将最终JSON/CSV标签纠正为E23-v103，未改变指标、权重或选择；修正记录见 `evaluation_label_correction.json`。
