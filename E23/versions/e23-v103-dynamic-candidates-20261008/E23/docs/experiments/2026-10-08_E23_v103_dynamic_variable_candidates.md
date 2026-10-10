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
