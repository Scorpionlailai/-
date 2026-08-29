# S4M 公平基线适配设计

日期：2026-08-29

状态：待用户确认后进入实现计划

范围：第一阶段仅 ETTh1，`seq_len=96`、`pred_len=96`、`start_rate=0.10`

## 1. 目标

在不修改现有 PatchTST、MALR 恢复模块和实验结果的前提下，引入官方 S4M 作为独立基线，并回答一个单一问题：

> 在完全相同的数据、划分、训练缺失过程、验证/测试固定掩码和评价空间下，专门面向缺失多变量时间序列的 S4M，与普通 PatchTST 在缺失输入上的预测性能相比如何？

第一轮只完成 ETTh1 的公平比较。确认协议和实现无误后，再以同一适配方式运行 Weather。

## 2. 比较对象与主要结论口径

第一轮至少运行四个结果：

| 骨干 | 输入 | 用途 |
|---|---|---|
| PatchTST | clean | 已有的完整数据性能上界参考 |
| PatchTST | missing | 现有普通预测模型的缺失基线 |
| S4M | clean | 测量 S4M 骨干自身在完整输入上的能力 |
| S4M | missing | 专门缺失方法的核心对照 |

主比较是 `S4M missing` 对 `PatchTST missing` 的测试集标准化空间 MSE/MAE。为了避免把骨干本身的强弱误认为缺失处理能力，同时报告每个骨干从 clean 到 missing 的绝对退化量与相对退化率：

```text
absolute_degradation = metric_missing - metric_clean
relative_degradation = (metric_missing - metric_clean) / metric_clean * 100%
```

不把 S4M 论文表格中的数值作为直接胜负依据，因为论文采用的划分、掩码、训练实现和评估条件与本项目不完全相同；论文数值只用于量级检查。

## 3. 固定的公平协议

### 3.1 数据与划分

- 数据文件：复用当前项目下载并校验过的 ETTh1 CSV，不另存一份数据。
- 训练段：索引 `[0, 8640)`，即一年数据。
- 验证段：索引 `[8640, 11520)`。
- 测试段：索引 `[11520, 14400)`。
- 回看窗口：`seq_len=96`。
- 预测长度：`pred_len=96`。
- 训练窗口、验证窗口和测试窗口的边界规则复用 MALR 当前实现。
- 随机种子：第一版固定为 `2021`。

### 3.2 标准化

- 只用训练段的观测值拟合每个变量的均值和标准差。
- 验证集和测试集只应用训练统计量，不重新拟合。
- 训练、选模和最终报告均使用 `train_standardized` 空间。
- 不额外报告原始量纲指标。

### 3.3 缺失协议

- 缺失类型只有 `variable missing`：每个变量独立随机选择缺失块起点，然后遮挡长度为 `block_len` 的连续值。
- 第一轮固定 `start_rate=0.10`；`block_len` 保持配置可调，首轮沿用当前 S4M 风格设置。
- 不强制不同缺失块互斥，允许重叠；实际 `final_rate` 由生成后的掩码统计。
- 训练掩码按 epoch 动态生成，且 PatchTST 与 S4M 必须使用相同的 epoch/seed 派生规则。
- 验证和测试掩码固定，S4M 直接读取 MALR 已生成的 `.npz`，不得自行重新采样。
- 结果清单必须记录验证/测试掩码 SHA-256、实际 `final_rate` 和种子。

### 3.4 严格禁止泄露

- 标准化统计量只来自训练段。
- 早停和超参数选择只看固定验证集。
- 测试集只能在训练完成、最佳 checkpoint 已确定后评估。
- 缺失输入可以使用对应的完整历史窗口作为训练监督，但完整值不得进入模型输入或 S4M 的辅助统计量。
- 预测目标在训练时作为监督使用，不能用于输入侧填补、memory 检索或掩码特征构造。

## 4. S4M 来源与隔离策略

### 4.1 官方来源

- 论文：S4M: S4 for Multivariate Time Series Forecasting with Missing Values，ICLR 2025。
- 官方仓库：`https://github.com/WINTERWEEL/S4M`
- 适配时固定上游提交：`a718823addd3606e763dfc261174e0135b2535f4`

官方仓库当前未提供明确的 LICENSE 文件。因此：

- 不把其源代码复制到本项目的受版本控制目录。
- 用户在本机把官方仓库检出到被 `.gitignore` 排除的 `external/S4M`。
- 适配器通过配置的 `s4m_root` 动态导入官方实现。
- 每个运行清单记录仓库 URL、提交哈希和本地代码状态；提交不匹配时默认拒绝正式运行。

### 4.2 独立环境

S4M 使用独立 Conda 环境 `s4m-fair`，避免其依赖影响当前 `malr` 环境：

- 使用 Windows 可用的最小依赖清单，不直接使用官方 Linux/CUDA 环境导出文件。
- 不卸载、不更新系统 CUDA 驱动。
- 环境中的 PyTorch CUDA runtime 只是 Python 环境依赖，不改动系统 CUDA 安装。
- 适配器需要以 editable 方式安装当前 MALR 包，以复用数据、掩码和指标协议。

## 5. 代码边界与组件设计

S4M 是独立基线，不进入 `Stage1Model`，也不新增 `ModelVariant.S4M`。建议新增以下边界：

```text
src/malr/baselines/s4m/
├── adapter.py       # 官方 S4M 的动态导入和统一 forward 接口
├── data.py          # MALR batch -> S4M 输入及辅助张量
├── trainer.py       # S4M 独立训练、验证、早停和 checkpoint
├── evaluator.py     # 固定测试掩码下的 MSE/MAE
└── provenance.py    # 上游 commit、环境和代码状态校验
```

CLI 使用独立入口，避免与现有 `train --variant ...` 混淆：

```powershell
malr train-baseline --model s4m --config <config> --dataset ETTh1 --artifacts <dir> --device cuda
malr evaluate-baseline --model s4m --config <config> --dataset ETTh1 --artifacts <dir> --run-id <id> --device cuda
```

现有 PatchTST/MALR 训练命令、checkpoint 格式和模型代码不因 S4M 适配而改变。

## 6. 数据适配细节

官方 S4M 的前向输入包含：

```text
seq_x, seq_x_mask, max_idx, min_idx, max_value, min_value
```

适配层负责从 MALR 的标准化窗口和观测掩码构造这些张量：

1. `seq_x_mask` 与 MALR 的 `observed_mask` 保持相同语义和位置。
2. 标准化后缺失位置以训练均值对应的 `0` 作为占位值，真实值不会进入 `seq_x`。
3. `max_idx/min_idx/max_value/min_value` 仅由当前输入窗口内的已观测值计算。
4. 当某变量在输入窗口内没有任何观测值时，使用显式的安全哨兵和全缺失标记；不得回看验证/测试窗口之外的数据取得值。
5. 适配结果需要在合成数据上与官方 `Dataset_Custom4` 的同义输出逐项比对，确认索引方向、mask 语义和张量形状一致。
6. 预测标签使用同一 MALR 窗口生成器产生的 clean future target；两种模型必须评估相同的样本集合。

## 7. 训练配置

官方论文、README 和代码存在若干差异，例如论文描述 Adam，而当前官方实验代码使用带 momentum 的 SGD；Weather 学习率在论文和 README 示例中也不一致。第一版采用明确、可复现的“论文优先”配置，并把差异写入运行清单。

ETTh1 正式配置：

| 参数 | 值 |
|---|---:|
| optimizer | Adam |
| learning_rate | 0.005 |
| batch_size | 16 |
| d_model | 256 |
| d_ff | 256 |
| dropout | 0.1 |
| candidate_e_layers | 2, 4, 8 |
| max_epochs | 15 |
| early_stopping_patience | 3 |
| selection_metric | validation MSE |
| seed | 2021 |

S4M memory 参数使用官方 Weather/README 所公开的基础组合作为首版固定值：

| 参数 | 值 |
|---|---:|
| per_mem_size | 5 |
| topK | 20 |
| thres1 | 0.95 |
| thres2 | 0.6 |
| M | 50 |
| memory_size | 200 |
| momentum | 0.99 |
| short_len | 32 |
| num_pattn | 30 |
| num_dG | 1 |
| encoder_type | 3 |
| memnet | 13 |
| mem_type | 3 |

训练分两级：

1. 冒烟测试：`e_layers=2`、1 epoch，只验证数据、前后向、显存和 checkpoint 流程。
2. 正式比较：分别训练 `e_layers in {2,4,8}`，只用固定验证 MSE 选择最佳配置，然后对测试集评估一次。

不基于测试集调整层数、学习率、阈值或 memory 参数。

## 8. 工件与可追溯性

S4M 工件单独存放，例如：

```text
artifacts/s4m_fair_etth1_sr010/
├── checkpoints/
├── runs/
└── evaluations/
```

每个 run manifest 至少记录：

- 数据集、划分边界、`seq_len`、`pred_len`。
- `start_rate`、`block_len`、验证/测试实际 `final_rate`。
- 数据 CSV、固定掩码、配置和 checkpoint 的 SHA-256。
- 标准化方式和训练统计量校验值。
- Python、PyTorch、CUDA runtime、GPU 名称。
- S4M 仓库 URL、固定 commit、本地 dirty 状态。
- 全部超参数、候选层数、最佳 epoch 和最佳验证 MSE。
- 论文/README/代码差异说明和本次采用值。
- 测试 MSE、MAE、训练耗时和可训练参数量。

## 9. 验证与测试

实现必须增加以下自动化检查：

### 9.1 单元测试

- train-only scaler 不读取验证/测试值。
- 固定 mask 读取后位置、shape、hash 和 `final_rate` 不变。
- 动态训练 mask 在同一 seed/epoch 下可重复，在不同 epoch 下发生变化。
- 缺失位置填充为 0，已观测位置保持标准化值。
- S4M 辅助张量不读取被遮挡值。
- 窗口起止位置与现有 MALR 数据集一致。
- 上游 commit 校验能拒绝非固定提交。

### 9.2 官方适配一致性测试

- 使用小型合成 CSV 和确定性 NaN/mask，比较适配器与官方 `Dataset_Custom4` 的 mask、索引辅助量和数值辅助量。
- 若官方实现对全缺失变量没有定义稳定行为，则用独立安全分支处理，并在 manifest 中记录。

### 9.3 集成测试

- CPU 上运行一个小 batch 的 forward/backward，loss 和梯度均为有限值。
- 1 epoch 冒烟训练能够保存、加载 checkpoint 并复现相同预测。
- 训练过程中不创建测试指标文件；只在显式 evaluate 命令中访问测试集。
- 原有 MALR 测试必须继续通过。

## 10. 正式结果验收标准

只有同时满足下列条件，S4M 与 PatchTST 的结果才进入比较表：

1. 两者使用同一个源 CSV 哈希和完全相同的时间划分。
2. 两者的验证/测试 mask SHA-256 完全一致。
3. 两者的测试样本数量、窗口索引和 clean target 完全一致。
4. 两者均在 `train_standardized` 空间计算 MSE/MAE。
5. S4M 只由验证 MSE选择 checkpoint 和 `e_layers`。
6. S4M 上游 commit 与固定哈希一致，运行环境写入 manifest。
7. `final_rate` 由实际 mask 统计并在比较表中显示。
8. 不因 S4M 的表现而修改测试掩码或测试期超参数。

第一轮结果表采用：

| Model | Input | MSE | MAE | Clean→Missing ΔMSE | Clean→Missing ΔMAE | Final rate | Mask hash |
|---|---|---:|---:|---:|---:|---:|---|
| PatchTST | clean | 已有 | 已有 | — | — | 0 | — |
| PatchTST | missing | 已有 | 已有 | 计算 | 计算 | 记录 | 记录 |
| S4M | clean | 新运行 | 新运行 | — | — | 0 | — |
| S4M | missing | 新运行 | 新运行 | 计算 | 计算 | 同上 | 同上 |

## 11. 非目标

第一轮明确不做：

- 不复现论文的 70/10/20 划分作为主结果。
- 不先运行 Weather。
- 不更改 MALR/ours 的恢复架构、原型库或损失。
- 不把 S4M memory 融入 MALR。
- 不使用测试集进行调参。
- 不复制未明确授权的上游源代码到本仓库。
- 不安装官方 Linux CUDA 环境，也不改动本机系统 CUDA。

## 12. 实施顺序

用户确认本设计后，下一步单独编写可执行实施计划。计划将按以下依赖顺序拆解：

1. 来源固定、`.gitignore` 和环境清单。
2. 配置 schema 与 provenance 校验。
3. 数据/掩码适配及一致性测试。
4. S4M 动态导入和模型 adapter。
5. 独立 trainer、checkpoint 与 evaluator。
6. CLI、冒烟测试和用户运行命令。
7. 正式 ETTh1 比较表生成。
