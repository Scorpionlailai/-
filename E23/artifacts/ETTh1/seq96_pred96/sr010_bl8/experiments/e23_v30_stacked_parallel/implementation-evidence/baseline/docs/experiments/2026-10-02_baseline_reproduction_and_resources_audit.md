# ETTh1 P4 基线复现与训练资源核查（2026-10-02）

## 结论

对照以本工作的缺失回溯窗口预测任务为准：统一数据切分、96→96、7 变量、缺失生成规则、训练数据可用范围和评测尺度。在这个协议下接入已发表方法并比较，是本文要完成的实验。原论文任务、预测长度和缺失设置不同，可以解释论文数字与当前数字之间的差异，不是当前统一任务对照无效的理由。

需要修正的是尚未对齐的模型与训练配置。采用作者针对相应数据集公布的配置作为固定基线，无需为每个基线额外开展参数搜索。优先采用论文明确给出的对应实验配置；论文未给出的项目采用对应数据集官方脚本，再采用全局默认。论文与代码冲突时记录选用来源，避免混合不同实验的参数。输入长度、输出长度、变量数、数据切分和缺失设置遵循本文协议；为使模型在该协议下可运行的必要改动逐项记录。

E23 没有使用外部预训练权重。它的 Stage 1 恢复训练与 Stage 2 预测训练都在本任务训练集内从头进行，两阶段均计入训练成本。各基线保留其作者推荐的训练流程与目标，插补方法按共享预测头的任务接口接入。

保留最优 E23 v12 和所有已选定测试结果。补充报告不重跑旧实验；用户随后确认开展配置修正后的新正式对照，采用新输出目录，从头训练一次，先按原固定验证组锁定 checkpoint，再测试原组与两组新增位置。最新配置、掩码和运行状态见 [作者配置与三组测试记录](2026-10-02_author_configuration_and_three_test_draws.md)。

## 统一报告口径

| 字段 | 定义 |
| --- | --- |
| MSE / MAE | 未来 96 步、全部 7 个变量的误差，使用仅由训练集拟合的标准化尺度 |
| `full_workflow_seconds` | 一次成功训练命令从入口计时开始，覆盖主要导入、数据准备、模型及优化器构建、全部轮次、逐轮验证、checkpoint 和选模记录保存；截止到资源记录写入前 |
| `training_stages_seconds` | 各阶段完整训练函数的耗时之和，包含该函数内的模型准备、训练、验证和保存 |
| `legacy_training_loop_seconds` | 旧选模记录中的 `training_seconds`，只作有范围说明的历史数据；不能替代完整命令耗时 |
| `peak_allocated_gib` | 所有必要训练阶段中最大的 PyTorch CUDA allocated 峰值，含参数、梯度、Adam 状态、中间张量及验证峰值 |
| `peak_reserved_gib` | 对应 CUDA 缓存分配器 reserved 峰值，单独列出 |
| GPU / batch / precision | 记录设备名称、设备索引与 UUID、批大小、精度、AMP、Torch/CUDA/cuDNN 和 TF32 设置 |

两阶段模型从头训练的耗时必须相加；显存取阶段最大值，不能相加。共享 Stage 1 checkpoint 的后续变体，若报告从头训练成本，也必须计入它依赖的 Stage 1。最终独立测试、人工等待、失败尝试及超参数搜索不计入单次成功训练时间，搜索成本需要单独说明。旧记录中未知的显存和全流程时间保持空值，不能由 checkpoint 大小、推理显存或最佳轮数推算。

CUDA 显存字段是本进程 PyTorch 分配器统计，不是整张卡的占用；不含 CUDA context 和分配器之外的原生库内存。CPU 模式的 GPU 显存为缺失值。正式对照应在同一硬件上逐个运行，采用各方法作者推荐的训练批大小与精度，并逐项记录；S4M 现有批大小为 16，其余旧结果为 128。配置对齐后，CoIFNet 将采用 512，T1 插补阶段将采用 16，不能把这些方法之间的资源差异解释为相同 batch 或精度下的差异。ChannelTokenFormer 为正确处理逐样本掩码，会把一个有效 batch 拆成多个可见性分组；实际 forward 的子批更小，耗时含适配开销。

资源记录独立保存于训练输出目录的 `resources/*.json`，绑定选模 manifest 的 SHA-256，不改变历史 checkpoint 或选模 manifest。失败和 smoke 记录不会作为完整训练资源纳入表格。

## 现有结果和可追溯耗时

完整字段见 [ETTh1_P4_extended_baselines_2026-10-02.csv](ETTh1_P4_extended_baselines_2026-10-02.csv)。下表时间来自旧记录，含训练循环与验证，所有必要阶段相加；旧硬件信息不完整，不用这张表宣称速度优势。

| 模型 | Test MSE | Test MAE | 已知训练循环合计（秒） | 阶段轮次 |
| --- | ---: | ---: | ---: | --- |
| E22 | 0.443195 | 0.433248 | 3812.72 | 100 + 100 |
| SAITS + PatchTST | 0.440584 | 0.436958 | 352.07 | 100 + 100 |
| S4M | 0.985358 | 0.753972 | 1404.21 | 100 |
| E23 v12 | 0.425613 | 0.431006 | 345.03 | 100 + 100 |
| CRIB | 0.612933 | 0.552574 | 121.78 | 100 |
| CoIFNet | 0.457392 | 0.430951 | 74.57 | 100 |
| BiTGraph | 0.536847 | 0.513945 | 125.99 | 100 |
| ChannelTokenFormer | 0.550205 | 0.524615 | 2654.57 | 100 |
| T1 + PatchTST | 0.430288 | 0.434663 | 636.12 | 100 + 100 |
| BRITS + PatchTST | 0.556675 | 0.539466 | 1905.54 | 100 + 100 |

旧的训练峰值显存和完整命令耗时均未保存。E23 v12 已有 `227643392` 字节（约 0.212 GiB）的 **推理** allocated 峰值记录，不能放入训练显存列。

## 已核实的差异

### P4 的 0.10 不是最终缺失 10%

`start_rate=0.10` 控制缺失块起点数：对每个变量，从合法起点中无放回抽取 `round(0.10 × 支持时间轴长度)` 个位置，块长 8，允许块重叠。它不是最终缺失率。用独立起点近似并忽略边界时，缺失率约为 `1-(1-0.10)^8 = 56.9533%`；实际生成器固定起点数，因此以保存掩码的实测值为准。读取锁定掩码并按实际评测窗口统计，validation 和 test 均为 **56.6711%**；整条支持时间轴为 **56.7204%**。

论文表头同时说明 start rate、block length 和实际 missing fraction。这是本文统一协议的描述，不要求复用各基线论文的缺失设置。共享多组掩码用于检验结果对随机缺失位置的稳定性，所有组均保留并汇报。

### 原论文任务和训练配置

| 模型 | 原论文或官方代码 | 当前适配 | 判断 |
| --- | --- | --- | --- |
| CRIB | 论文 Appendix D：24→24，按完整数据集 60/20/20 切分，attention 2 层 | 96→96；固定 8640/11520/14400 边界；模型采用发布代码的 3 层配置 | 96→96 和本文切分继续固定；将层数等模型参数对齐论文，不要求搜索长预测最优配置 |
| BiTGraph | 论文：未来 24 步，60/20/20，batch 32，全局描述使用 3 个 block | 96→96；batch 128；采用发布代码 ETTh1 的小模型（2 层、subgraph 5） | 核对数据集专用配置与论文描述的关系，采用作者推荐配置；邻居数等参数若超过 7 变量的合法范围，记录必要限制 |
| CoIFNet | 论文表 III：batch 512、损失权重 λ=0.2；块缺失最多 10 个时间点 × 5 个变量 | batch 128、λ=0.1（发布代码默认）；逐变量块长 8；固定 P4 切分 | 对齐论文表 III 的 λ=0.2 和 batch=512；缺失设置继续使用本文协议 |
| ChannelTokenFormer | 论文包括异步、不同变量采样周期、频率决定 patch 和缺失情形；常规 ETTh1 脚本有学习率调整 | 同步 7 变量、patch 24、P4 mask、keep_prob 0.6、固定学习率 1e-4；按可见性分组 | 核对对应 ETTh1 官方配置，恢复相应训练策略；同步任务与逐样本 mask 正确性继续遵循本文协议 |
| T1 | 主要评估插补；块缺失实验为 5% 点缺失 + 0.15% 起点概率、24–96 长度块；ETTh1 发布脚本最多 300 轮、batch 16，并使用 lradj type3 / bf16 | Stage 1 100 轮、batch 128、P4 块缺失、固定学习率、FP32；再接共享 PatchTST 联合预测训练 100 轮 | 恢复对应官方模型和插补训练配置；保留共享预测接口，统一报告未来预测误差 |
| SAITS / BRITS | 原始核心任务是插补 | 接共享 PatchTST 并训练预测；BRITS 用 SAITS 仓库实现而非 BRITS 原作者仓库 | 在本文任务中作为插补模块对照；明确实现来源并核对作者模型配置 |
| S4M | 发布示例使用其他回溯长度和训练配置 | P4 96→96，batch 16，official profile 搜索后按验证集选定 | 核对原始入口、模型配置与输入输出尺度，确保本文任务中的适配正确 |

论文依据：[CRIB Appendix D](https://arxiv.org/html/2509.23494#A4)、[BiTGraph §5.1](https://proceedings.iclr.cc/paper_files/paper/2024/file/7f3b25e39e0f0330435d1ffc1d6fa193-Paper-Conference.pdf)、[CoIFNet §V-A](https://arxiv.org/html/2506.13064v2#S5.SS1)、[ChannelTokenFormer 实验与附录](https://arxiv.org/html/2506.08660)、[T1 §4 与附录](https://arxiv.org/html/2602.21043)。实现依据是各 `artifacts/upstream/` 固定 revision 的配置与训练脚本，以及 `integrations/forecast_compare/` 适配器；具体 revision 见已有 [基线记录](ETTh1_P4_extended_baselines_2026-10-02.md)。

CoIFNet 的公开代码与论文还有一个区别：代码 ETTh1 test 边界延伸到第 24 个月，当前固定 P4 只到第 20 个月，正式对照继续采用本文固定切分。当前时间特征处理的 `day-of-month % 7` 虽命名像星期信息，但它与作者 `CoIFNetTask._process_timestamps_feat` 一致，暂不能据此认定是适配 bug。论文损失权重敏感性段落还提到 0.3 最佳，与表 III 的 0.2、代码默认 0.1 不同；本轮固定采用论文实验配置表 III 的 0.2 并记录来源，不据当前测试结果选权重。

T1 当前 `imputation_use_mask_embedding=false` 与论文描述的 mask-aware embedding 不完全一致。该选项不是解释当前接近 E23 结果的依据；应核实作者对应实验使用的设置，按该配置复现。

S4M 的已有 Clean 诊断 MSE 为 **1.145705**，P4 Missing 为 **0.985358**，均偏高，而且两个诊断是各自训练、选模后的模型。这提示问题可能涉及整体配置或任务适配，不能只归因于缺失难度，也不能从这个顺序判断缺失改善了同一个模型。当前尚未定位到具体根因。

### 统一任务中的公平对照原则

当前十个模型的对照误差统一在 ETTh1 P4 预测任务下报告。六个新增基线使用同一 source CSV、同一锁定 validation/test mask、同一 train-only 标准化、96→96、7 变量、seed 2021，所有预测阶段跑满 100 轮。预测原生基线保留各自训练目标；T1、BRITS 使用与 E23 同结构的预测头。可见值不会被 T1/BRITS 插补结果改写。

E23 的模型设计迭代与作者已发表模型的推荐配置都作为各自方法的一部分。基线采用作者公布的配置，无需在本文任务上额外搜索更优参数。公平性重点是相同训练数据可用范围、相同缺失输入、相同预测目标、正确的实现与一致的评测；各方法训练流程和成本如实报告。本文以作者推荐配置作为固定对照，研究它们在本文任务中的表现。

旧 SAITS 按验证 MSE 选模，新增六个模型与 E23 v12 按验证 MAE 选模。后续正式对照固定统一的选模规则，或明确保留作者规则并报告差异；不以测试结果选择 checkpoint。

在统一任务、多组共同缺失掩码下，若 E23 相比作者配置的基线取得稳定提升，可以支持本文方法在该任务中的有效性。多个数据集和训练 seed 会进一步扩展结论范围。现有记录是单 seed、单数据集、单缺失抽样；历史 ETTh1 test 已在研发期间被观察，后续不再依据测试组表现调整配置，多组结果全部保留。

## 下一轮配置对齐顺序

1. **CRIB**：按论文配置恢复层数等参数，保留本文 96→96 与共享缺失输入。
2. **CoIFNet**：采用论文表 III 的 λ=0.2、batch=512，保留本文任务协议；设备不支持的设置明确记录。
3. **BiTGraph**：核实 ETTh1 专用配置与论文配置的来源，采用作者推荐模型；记录变量数引起的必要参数限制。
4. **ChannelTokenFormer**：采用对应数据集官方模型和训练策略，保留逐样本掩码正确性，并记录分组适配开销。
5. **T1**：恢复相应官方插补模型、batch、轮次上限、学习率策略、精度和 mask embedding 配置；共享预测接口继续使用本文设置。
6. **BRITS / SAITS / S4M**：核对作者模型与训练设置，特别检查 S4M 输入输出尺度和 mask 接口。

上述核查已落实到版本化配置与新入口，八个基线完成真实训练步 smoke，不声称必然改善性能。新正式对照仅在测试阶段使用三组掩码，报告平均 MSE、MAE、标准差、完整训练耗时和显存。原任务数字不直接与本文 96→96 的 P4 预测指标混合。

## 使用方式

E23 `integrations.e23.run`、T1 和 BRITS 训练 CLI 自动记录资源，独立测试 JSON 也会附带 `training_resources`，汇总所有已绑定的训练阶段。CRIB、CoIFNet、BiTGraph、ChannelTokenFormer 的旧选模记录校验整个训练入口的 SHA-256，因此这些入口保持训练时的精确源码，通过下述同进程 wrapper 记录新训练资源，再用报告脚本汇总。没有记录的老结果仍报告缺失值。

汇总已有记录或新实验：

```bash
PYTHONPATH=src:. /home/xyw/miniforge3/envs/malr/bin/python scripts/update_p4_resource_report.py
```

新实验可用 `--manifest-map paths.json --table input.csv --output output.csv`，其中 JSON 为 CSV 模型名到按顺序排列的所有必要阶段 manifest 路径列表。

E22、SAITS、S4M、CRIB、CoIFNet、BiTGraph、ChannelTokenFormer 等历史入口可使用 `scripts/run_measured_training.py`，在目标模型原有解释器、工作目录和 PYTHONPATH 下运行原命令，不改其受 hash 校验的源文件。S4M 使用其专用环境；示意：

```bash
python /home/xyw/paper/E23/scripts/run_measured_training.py \
  --model SAITS --stage repair --device cuda:0 --batch-size 128 \
  --output /absolute/new-run/stage1 \
  --selection-manifest /absolute/new-run/stage1/runs/saits-stage1.json \
  --module integrations.forecast_compare.saits_run -- \
  train-repair [原入口所需参数]
```

wrapper 的阶段耗时覆盖整个目标训练命令，因此含数据准备；不要将其误标成单纯 GPU 运算时间。它只适用于在当前进程执行训练的 Python 入口，不能包一层只负责启动训练子进程的 supervisor/search 脚本来测子进程显存。S4M 应直接包实际的 `s4m/run.py` 训练命令。精度/AMP 选项只记录既有设置，不改变原训练代码。

CRIB 旧入口的测量调用示例（用于新输出目录，现有选模结果不重复训练）：

```bash
PYTHONPATH=src:. /home/xyw/miniforge3/envs/malr/bin/python scripts/run_measured_training.py \
  --model CRIB --stage forecast --device cuda:0 --batch-size 128 \
  --output /absolute/new-run/crib \
  --selection-manifest /absolute/new-run/crib/crib-selection.json \
  --script scripts/run_crib_p4.py -- \
  --train --device cuda:0 --output /absolute/new-run/crib
```

配置对齐版应采用新的训练入口或明确版本化的配置；不能修改历史 manifest 的源码 hash 来允许新版代码冒充旧训练实现。

## 验证记录

资源相关检查使用真实 CPU 和 CUDA 训练步，检查含 Adam 状态的显存峰值、两阶段耗时相加、显存取最大值、失败记录、旧记录缺失值和 manifest 绑定。资源接入过程中另执行过 CoIFNet 一个训练 batch 和一个 validation batch 的 smoke，该 smoke 不进入结果表；最终为保持历史源码身份，CoIFNet 等四个旧入口采用 wrapper。wrapper 使用独立临时训练程序检查，确认源码和 selection 不变、资源 sidecar 正确绑定。新增四个回归用例验证旧入口能够通过历史 selection 的源码检查，同时仍拒绝损坏的 checkpoint。

测试运行时把 `malr/bin` 放入 PATH，使 `git` 可见；同时将三个旧测试的掩码引用改为迁移后的 E22 路径。修复源码身份兼容后，全量测试：**773 passed，1 skipped**（24.02 秒）。跳过的是 `tests/s4m/test_s4m_patch_contract.py::test_official_patch_is_applied_and_python_syntax_is_valid`：它需要设置 `S4M_OFFICIAL_ROOT`，本次默认未设置。已有 weight_norm 弃用和 CuBLAS 确定性提示不影响本次通过结果。
