# 作者配置对齐与三组测试缺失位置（2026-10-02）

## 已确认的实验口径

每个模型只训练一次，使用原来的固定验证缺失位置选定一个 checkpoint。测试采用原组与新增两组缺失位置，同一个 checkpoint 评测全部三组，报告平均 MSE、MAE 和三组间标准差（`ddof=0`）。新增位置只用于测试，不用于训练、验证、选模或调参。训练耗时和显存只计一次完整训练，不把三组测试当作三次训练。

任务固定为 ETTh1、7 变量、96→96，切分边界 0/8640/11520/14400，仅由训练集拟合全局标准化参数。缺失块起点率 0.10、长度 8、允许重叠，评测每组相同的 2688 个窗口。模型 RNG 与训练缺失生成器 seed 2021 是共同实验控制；数据加载使用现有 P4 接口、4 个 CPU 线程，各方法训练 batch 和计算精度按作者配置。插补方法继续接共享 PatchTST 预测接口，Stage 2 固定 100 轮、batch 128、Adam 1e-4、forecast MSE，按原固定验证组的 forecast MAE 选模。

没有公开 ETTh1 专用最终配置时，按用户确认的缺省规则采用作者公开基础配置或运行示例，并标明版本。论文明确给出的对应参数优先，其余采用对应脚本或作者默认。必要的任务适配在配置与 CSV 中逐项记录，不把基础配置称作作者的 ETTh1 最优配置。

## 冻结的测试组

| 组 | Seed | 评测窗口实际缺失比例 |
| --- | ---: | ---: |
| 原组 | 2021 | 56.671073% |
| 新组 1 | 2026100201 | 56.680042% |
| 新组 2 | 2026100202 | 56.955793% |

原组文件逐字节复用原 test mask。生成器固定块起点数，块可能重叠，因此 0.10 不是最终缺失比例。验证文件未变化。完整身份和 SHA 见 [掩码 CSV](ETTh1_P4_test_mask_groups_2026-10-02.csv)；文件位于 `artifacts/ETTh1/seq96_pred96/sr010_bl8/test_mask_bank_20261002/`，由 `manifest.json` 与 `manifest.sha256` 校验。

## 已落实的作者配置

实际训练入口读取 [版本化配置](../../configs/baselines/etth1_p4_author_20261002.json)（配置实际路径为 `configs/baselines/etth1_p4_author_20261002.json`）；完整参数、代码 revision、来源和任务适配见 [配置 CSV](ETTh1_P4_author_configuration_alignment_2026-10-02.csv)。

| 模型版本 | 主要参数变化 | 作者训练上限 / batch / 精度 | 验证指标与早停 |
| --- | --- | --- | --- |
| CRIB：论文 + 发布默认 | attention 2 层、4 heads、宽度 32 | 10 / 256 / FP16 AMP | forecast MAE；无早停 |
| CoIFNet：论文表 III | λ=0.2、hidden 256 | 100 / 512 / FP32 | forecast MAE；patience 10 |
| BiTGraph：论文 + ETTh1 发布通道 | 3 blocks；两个 top-k 操作均限制为 min(10,7)=7 | 200 / 32 / FP32 | forecast MAE；无早停 |
| ChannelTokenFormer：ETTh1 脚本 | keep_prob=1、宽度 256、2 层，恢复 type1 LR 调整 | 10 / 32 / FP32 | forecast MSE；patience 3 |
| T1：ETTh1 脚本 | 发布默认 mask embedding=false；type3 实际为固定 LR | 300 / 16 / BF16 | hidden reconstruction MSE；patience 30 |
| SAITS-base | 2 层、宽度 256、FFN 128、4 heads、dropout .1 | 10000 / 128 / FP32 | hidden imputation MAE；patience 30 |
| 原作者 BRITS + README 示例参数 | hidden 108、impute weight .3；保留原始求和 ORT 与 .1 一致性损失 | 1000 / 64 / FP32 | hidden imputation MAE；无早停 |
| S4M：ETTh1 论文参数 + 发布示例 | 宽度 256、LR .005；示例 depth 8；type1 LR 调整 | 10 / 16 / FP32 | forecast MSE；patience 3 |

依据：[CRIB Appendix D](https://arxiv.org/html/2509.23494#A4)、[CoIFNet Table III](https://arxiv.org/html/2506.13064v2#S5.SS1)、[BiTGraph §5.1](https://proceedings.iclr.cc/paper_files/paper/2024/file/7f3b25e39e0f0330435d1ffc1d6fa193-Paper-Conference.pdf)、[CTF 作者 ETTh1 脚本](https://github.com/jinkwan1115/ChannelTokenFormer/blob/b1c100effb9c27cd25de07e91de7f9005d18d5e7/scripts_regular/CTF_ETTh1.sh)、[T1 论文与发布代码](https://arxiv.org/html/2602.21043)、[SAITS-base](https://arxiv.org/html/2202.08516)、[BRITS 原作者仓库](https://github.com/caow13/BRITS)、[S4M 附录](https://arxiv.org/html/2503.00900)。具体运行参数还由本地固定 revision 的脚本逐项核实。

需要保留的来源说明：

- CRIB 发布默认 3 层与论文 2 层冲突，采用论文 2 层；新增固定验证 checkpoint 选择属于 P4 接入，作者原训练入口没有该过程。
- CoIFNet 论文配置表的 λ=.2、敏感性段落的 .3、代码默认 .1 不同，本轮固定采用配置表 .2。
- BiTGraph 论文 top-k=10 超过本任务 7 个变量，图构建与 `nconv` 都改为 7。仅在新适配器加载内存中的代码时更改，原源码文件和旧入口不变。验证用本文标准化尺度，而非发布代码逆标准化尺度。
- T1 的 ETTh1 脚本未开启 mask embedding，其默认 false 与论文文字不同，准确标注为发布脚本配置。
- SAITS 没有找到唯一 ETTh1 专用配置，采用明确公开的 SAITS-base。恢复模块输入改为作者的训练集全局标准化尺度；验证在每轮末进行，取代原配置中的按训练 step 验证频率。
- BRITS README 参数示例对应 RITS_I，不是 ETTh1 的 BRITS 最优配置。本轮明确将该公开参数组用于原作者 BRITS。仅适配序列/变量数、移除无用旧导入和修正 tensor device；P4 无分类标签，因此 label weight=0。预测入口不传入隐藏真值和未来标签；共享预测头保留可见值。
- S4M 论文提供深度候选 {2,4,8}，没有找到唯一选定深度，按公开运行示例固定 8。本轮不进行新搜索。当前专用环境没有 S4 结构化 CUDA kernel 或 pykeops，使用发布代码的 PyTorch 后备实现，资源结果必须注明该后端。原核心三份文件按发布 git blob 校验。原模型推理仍会随机抽取 topM 历史条目，验证和每个测试组固定该随机状态为 2021，使组间差异由缺失位置产生。

## E23 v12 的已有 checkpoint 三组结果

直接复用已锁定的 v12 checkpoint；原组读取已有测试 JSON，只新增两组评测。没有新训练或选模，H 原型库与检索 key 状态保持一致。

| 汇总 | MSE | MAE |
| --- | ---: | ---: |
| 三组平均 | 0.427462825429 | 0.431355525397 |
| 三组标准差（ddof=0） | 0.004205234829 | 0.000677963866 |

逐组结果见 [E23 v12 三组 CSV](ETTh1_P4_E23_v12_three_test_draws_2026-10-02.csv)。这份旧 checkpoint 的完整训练耗时和训练显存没有历史记录，不能从本次评测推算。正式资源对照将为同一冻结 v12 配置从头训练一次，包含全部 Stage 1 和 Stage 2。

## 新正式对照的入口

八个基线已完成真实 CUDA 训练步检查；T1、SAITS、BRITS 覆盖两个阶段。smoke 明确标为 smoke，不允许进入正式测试汇总。另使用真实 CoIFNet CPU 两轮训练验证 checkpoint 保存、固定验证重放和防覆盖。

配置修正使用新的输出目录，保留旧结果。正式队列按 E23 v12、CRIB、CoIFNet、BiTGraph、CTF、T1、SAITS、BRITS、S4M 顺序在同一 GPU 0 运行，包含训练和三组测试。训练入口自动记录资源，checkpoint 和 manifest 在三组评测前后必须不变。队列保存命令、日志、进度和全部源码校验值；训练期间源码变化或命令失败会停止，避免混入不同版本。

```bash
PATH=/home/xyw/miniforge3/envs/malr/bin:$PATH PYTHONPATH=src:. \
python scripts/run_author_p4_suite.py --output artifacts/author_p4_20261002 --dry-run
```

正式启动去掉 `--dry-run`；用现有 `scripts/run_background.py` 启动可保留 PID、日志及退出状态。新汇总 [正式对照 CSV](ETTh1_P4_author_three_test_draws_2026-10-02.csv) 随完成的模型逐项更新，未完成行保留状态且误差和资源为空。报告字段包含各组三组指标、平均/标准差、实际缺失比例、全流程耗时、各阶段耗时、allocated/reserved 峰值 GiB、GPU/UUID、实际轮数、各阶段 batch/precision、参数量和配置版本。两阶段耗时相加，显存取阶段峰值；测试耗时不混入训练时间。

S4M 使用 `malr-s4m` 环境；其余使用 `malr`。全部方法遵循自己的作者训练上限与早停，BRITS 的 1000 轮无早停以及 SAITS 的 10000 轮上限均如实保留。

## 验证与正式队列交接

独立代码审查修正了 BRITS delta 输入：使用原作者 `input_process.py` 的当前时刻 mask，缺失块边界回归序列 `[1,0,0,1]` 的 delta 为 `[1,2,3,1]`。正式 evaluator 在评分前强制校验三个预声明 seed；协议 YAML 和 v12 YAML 均纳入队列冻结，选模记录额外绑定协议配置 SHA。

最终全量检查：**785 passed、1 skipped、8 warnings，24.59 秒**，日志 `/tmp/e23-author-alignment-final-suite.log`。唯一 skip 是未设置 `S4M_OFFICIAL_ROOT` 的旧 bootstrap 检查；本轮 S4M 实际运行已校验三个受保护核心 git blob。CoIFNet 和 S4M 另外完成临时一轮配置的完整 train→固定验证重放→三组测试→资源 CSV 检查，输出留在 `/tmp`，不进入正式表。

正式队列已启动，输出根目录 `artifacts/author_p4_20261002/`。后台 supervisor PID **1490961**，队列 child PID **1490962**；运行时间和最终退出状态以 `artifacts/jobs/author_p4_20261002/status.json` 为准。每个模型的 `progress.json`、`command_*.json`、`command_*.log` 保存在各自子目录，最终训练、选模、测试和资源全部持久保存。表格由队列随完成的模型自动更新，无需人工抄录。运行中修改冻结的 Python 或 YAML 会导致队列在下一个命令前停止；排查应先看对应日志，不重复已完成的实验。
