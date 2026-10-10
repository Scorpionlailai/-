# ETTh1 实验进度与回溯台账

最后更新：2026-09-20  
项目目录：`/home/xyw/missing_aware_latent_recovery/.superpowers/sdd/2026-08-31-residual-mlp-recovery/checkout`  
运行环境：Conda `malr`，Python 3.10.21，PyTorch 2.7.1+cu128，CUDA 12.8，NVIDIA GeForce RTX 5090  
统一随机种子：2021

本文档用于回答三个问题：做过什么、结果如何、如何找到当时的结果与参数。所有分数均来自已经落盘的 run/evaluation 文件；没有测试结果时明确写为“未测”，不拿验证集结果代替测试集结果。

E22 在 P4～P8 不同缺失条件下的统一结果表：`docs/experiments/E22_ETTh1_missing_scenarios.xlsx`；同数据的可移植版本：`docs/experiments/E22_ETTh1_missing_scenarios.csv`。表格顶部只列一次共用 Clean，后续每个协议使用各自 Missing 计算 DRR。

## 1. 阅读规则

### 1.1 指标

- MSE、MAE 越低越好，均在训练集标准化后的数值空间计算。
- 相对 Missing 改善率：`(Missing - Ours) / Missing × 100%`。
- DRR（退化恢复率）：`(Missing - Ours) / (Missing - Clean) × 100%`。
- DRR = 0% 表示没有修复 Missing 带来的退化；DRR = 100% 表示回到 Clean；DRR < 0% 表示比 Missing 更差。
- 当 `Missing - Clean` 很小时，DRR 会非常敏感。因此 seq336 的 DRR 只能和同协议结果比较。

### 1.2 可比性

只有完全相同的 dataset、seq_len、pred_len、start_rate、block_len、数据切分、mask 和 seed 才能直接比较。本文把实验分为 P1～P8，禁止跨协议直接判断模型优劣。

### 1.3 回溯状态

- **完整**：run、evaluation 和 checkpoint 均存在，可继续训练或复查。
- **验证-only**：只完成验证，未产生正式测试结果。
- **结果-only**：结果仍在，但精确 checkpoint 已被后续实验覆盖，不能严格恢复该模型。
- **待运行**：代码和方案已经准备好，尚无训练结果。

## 2. 实验协议索引

| 协议 | 数据与预测长度 | 缺失参数 | 实际最终缺失率 | Clean MSE / MAE | Missing MSE / MAE | 结论用途 |
|---|---|---|---:|---:|---:|---|
| P1 | ETTh1，96 → 96 | start_rate=0.10，block_len=5 | 40.9130% | 0.391574 / 0.406288 | 0.461589 / 0.442178 | 早期模块与 grouped memory |
| P2 | ETTh1，336 → 96 | start_rate=0.10，block_len=5 | 41.1647% | 0.373231 / 0.397912 | 0.397190 / 0.415134 | 检查长序列失效问题 |
| P3 | ETTh1，336 → 96 | start_rate=0.15，block_len=5 | 56.0634% | 0.373231 / 0.397912 | 0.428657 / 0.432460 | 放大 seq336 的缺失压力 |
| P4（当前主线） | ETTh1，96 → 96 | start_rate=0.10，block_len=8 | 56.7204% | 0.391587 / 0.406349 | 0.519669 / 0.466895 | 放大 Clean/Missing 矛盾，开发两阶段 TVH |
| P5（E22 缺失压力测试） | ETTh1，96 → 96 | start_rate=0.15，block_len=8 | 73.1087% | 0.391587 / 0.406349 | 0.600958 / 0.511243 | 只改变起始缺失率，比较 E22 零样本迁移与同结构重训 |
| P6（E22 轻缺失测试） | ETTh1，96 → 96 | start_rate=0.10，block_len=5 | 40.9130% | 0.391587 / 0.406349 | 0.461711 / 0.442272 | 只缩短缺失块，比较 E22 零样本迁移与同结构重训 |
| P7（E22 更轻缺失测试） | ETTh1，96 → 96 | start_rate=0.10，block_len=3 | 27.0305% | 0.391587 / 0.406349 | 0.424323 / 0.424508 | 继续缩短缺失块，检查收益与 T 的迁移/重训差异 |
| P8（E22 极高缺失测试） | ETTh1，96 → 96 | start_rate=0.20，block_len=8 | 82.9157% | 0.391587 / 0.406349 | 0.721772 / 0.553815 | 只提高起始缺失率，检查 E22 在更恶劣缺失下的边界 |

## 3. 当前结论快照

1. seq336 下 Clean 与 Missing 的差距很小，现有恢复模块没有取得正收益；H-only 最接近 Missing，但仍未改善。
2. 把缺失压力提高到 start_rate=0.15 仍未解决 seq336 问题。因此 E14--E22 开发阶段固定 seq96、block_len=8 验证模块逻辑；P5/P6 是 E22 定版后的缺失协议压力测试，不是继续修改 seq336。
3. E17 的静态恢复数据交接证明“先修复、再预测”有效，但 Stage 2 无法纠正已经固化的错误恢复；它目前只作为历史对照，不再是 P4 最优。
4. E19--E21 逐步验证了冻结评分、冻结 teacher 和 `{none,H,V}` 软融合。E21 中 H/V 都有贡献，但旧 T 不参与预测，因此仍未充分利用当前窗口的时序上下文。
5. **E22 是当前 P4 协议最优版本**：测试 MSE / MAE 为 **0.443195 / 0.433248**，相对 Missing 改善 **14.72% / 7.21%**，DRR 为 **59.71% / 55.57%**。它同时刷新 P4 的 MSE、MAE 与两项 DRR。
6. E22 保留 E21 的 H/V 候选与软融合，在其后新增独立的全回溯动态 T；Stage 2 对每个 batch 在线显式修复，并把恢复值与 observation mask 一同交给独立 PatchTST 预测器。
7. E22 推理消融证明三类关系均有预测贡献：移除 H、T、V 后测试 MSE 分别退化到 `0.470573`、`0.498556`、`0.502524`。因此当前结论不是“某个单模块取代其余模块”，而是 H/V 基础修复与动态 T 的协同有效。
8. “当前最优”严格限定为 P4：ETTh1、seq96→96、start_rate=0.10、block_len=8、seed=2021、相同数据切分和固定 test mask；不能据此宣称跨数据集、跨缺失协议或多随机种子全局最优。
9. P5 起始缺失率从 0.10 提至 0.15，实际最终缺失率达 73.11%。原 E22 不重训的测试 MSE / MAE 为 `0.504612 / 0.467289`；同结构从头重训为 `0.508583 / 0.472684`；均优于 P5 Missing 的 `0.600958 / 0.511243`。重训验证更好但测试略差，不能宣称重训提高了跨缺失率泛化，也不能把 P4/P5 绝对分数直接排名。
10. P6 保持 `start_rate=0.10`，把 `block_len` 从 8 缩至 5，实际缺失率降到 40.91%。原 E22 不重训测试 `0.409095 / 0.417777`，同结构从头重训 `0.405926 / 0.413958`，均优于新 Missing 的 `0.461711 / 0.442272`。重训测试略好，但验证集反而不如零样本迁移；这是单种子结果，不能据此推断重训收益具有普遍性。
11. P7 再把 `block_len` 缩至 3，实际缺失率为 27.03%。原 E22 零样本测试 `0.409385 / 0.416107`，同结构从头重训 `0.385426 / 0.402017`，均优于 P7 Missing 的 `0.424323 / 0.424508`。重训在本协议的测试分数也略低于 Clean，但 Clean–Missing 差距仅 `0.032736 / 0.018159`，超过 100% 的 DRR 不能单独解读为“修复优于真实数据”或跨协议优势。零样本移除 T 有益，重训移除 T 有害，说明 T 对缺失模式变化敏感，重新适配后才稳定贡献。
12. P8 将 `start_rate` 提至 0.20，保持 `block_len=8`，实际缺失率为 82.92%。原 E22 零样本测试 `0.616536 / 0.511924`，同结构从头重训 `0.609522 / 0.509541`，均优于 P8 Missing 的 `0.721772 / 0.553815`。重训比零样本略好，但仍远离 Clean；本协议的 DRR 为 `34.00% / 30.02%`。重训版移除 H/T/V 均使测试误差明显上升，说明三类关系在极高缺失下仍有贡献；这是单种子、单数据集结果，不能外推为普遍鲁棒性。

## 4. 完整实验台账

### P1：seq96 → 96，start_rate=0.10，block_len=5

| ID | 方法 / 关键修改 | Ours run ID | Val MSE / MAE | Test MSE / MAE | 相对 Missing 改善 MSE / MAE | DRR MSE / MAE | 状态 |
|---|---|---|---:|---:|---:|---:|---|
| E01 | replacement filter + recovery loss | `ETTh1-ours-1797868fd4b5` | 0.816656 / 0.608563 | 未测 | — | — | 验证-only |
| E02 | replacement filter，无 recovery loss | `ETTh1-ours-2e5be777c4ac` | 0.791507 / 0.603458 | 0.454575 / 0.445413 | +1.26% / -0.92% | +8.45% / -11.55% | 完整 |
| E03 | fast，无 recovery loss | `ETTh1-ours-81516073c2b4` | 0.806629 / 0.609981 | 未测 | — | — | 验证-only |
| E04 | imputation | `ETTh1-ours-a4f198f9a513` | 0.787747 / 0.601313 | 0.456444 / 0.442889 | +0.86% / -0.35% | +5.73% / -4.35% | 完整 |
| E05 | imputation，wide + strict | `ETTh1-ours-8d4cbd808b90` | 0.791255 / 0.603548 | 0.459952 / 0.445317 | +0.09% / -0.90% | +0.63% / -11.27% | 完整 |
| E06 | grouped prototype memory | `ETTh1-ours-7559d1137756` | 0.769151 / 0.589825 | **0.438385 / 0.433786** | **+5.03% / +1.90%** | **+33.14% / +23.38%** | 完整；P1 最好 |
| E07 | online memory | `ETTh1-ours-ddb2fd5ea105` | 0.813696 / 0.605215 | 0.446167 / **0.433631** | +3.34% / +1.93% | +22.03% / +23.82% | 完整 |

P1 结论：grouped prototype memory 明显有效；MSE 最好为 E06。E07 的 MAE 略低于 E06，但 MSE 更差，不能据此认定整体超过 E06。

### P2：seq336 → 96，start_rate=0.10，block_len=5

| ID | 方法 / 关键修改 | Ours run ID | Val MSE / MAE | Test MSE / MAE | 相对 Missing 改善 MSE / MAE | DRR MSE / MAE | 状态 |
|---|---|---|---:|---:|---:|---:|---|
| E08 | grouped memory，无 Huber | `ETTh1-ours-c26507cca839` | 0.771857 / 0.596060 | 0.401488 / 0.421808 | -1.08% / -1.61% | -17.94% / -38.75% | 完整 |
| E09 | grouped memory + Huber | `ETTh1-ours-bccae6a66937` | 0.766862 / 0.593509 | 0.399279 / 0.419253 | -0.53% / -0.99% | -8.72% / -23.92% | 完整 |
| E10 | multiscale temporal edge + Huber | `ETTh1-ours-1ce5792e6b51` | 0.768058 / 0.597312 | 0.401093 / 0.419607 | -0.98% / -1.08% | -16.29% / -25.97% | 完整 |
| E11 | historical-only + Huber | `ETTh1-ours-b5d02ad96bb4` | 0.785222 / 0.602913 | **0.397597 / 0.415387** | -0.10% / -0.06% | -1.70% / -1.47% | 完整；最接近 Missing |

P2 结论：所有版本均未超过 Missing。长 seq 本应提供额外信息，但当前局部关系与恢复方式没有利用好这些信息；此问题暂缓，不与 P4 同时修改。

### P3：seq336 → 96，start_rate=0.15，block_len=5

| ID | 方法 / 关键修改 | Ours run ID | Val MSE / MAE | Test MSE / MAE | 相对 Missing 改善 MSE / MAE | DRR MSE / MAE | 状态 |
|---|---|---|---:|---:|---:|---:|---|
| E12 | grouped memory，提高缺失压力 | `ETTh1-ours-2f94df55cb48` | 0.801600 / 0.613285 | 0.431426 / 0.437949 | -0.65% / -1.27% | -5.00% / -15.89% | 完整 |
| E13 | grouped memory + gated | `ETTh1-ours-bfb49280b1fb` | 0.873536 / 0.649199 | 未测 | — | — | 验证劣于 Missing，未进入测试 |

P3 结论：只增加缺失强度不能修复 seq336 的结构性问题。

### P4：seq96 → 96，start_rate=0.10，block_len=8（当前主线）

| ID | 方法 / 关键修改 | Ours run ID | Val MSE / MAE | Test MSE / MAE | 相对 Missing 改善 MSE / MAE | DRR MSE / MAE | 状态 |
|---|---|---|---:|---:|---:|---:|---|
| E14 | relation-v2 初始版 A | `ETTh1-ours-6fc34c940fd7` | 0.901246 / 0.640452 | 0.508881 / 0.460611 | +2.08% / +1.35% | +8.42% / +10.38% | 结果-only；checkpoint 被 E15 覆盖 |
| E15 | relation-v2 初始版 B | `ETTh1-ours-3901c37fae2f` | 0.882131 / 0.633164 | 0.503069 / 0.458458 | +3.19% / +1.81% | +12.96% / +13.93% | 完整 |
| E16 | 首个两阶段 continuation：Stage 1 30 epoch + forecast | `ETTh1-ours-0973e2755ffd` | 0.884343 / 0.633402 | 0.501352 / 0.456929 | +3.52% / +2.13% | +14.30% / +16.46% | 完整 |
| E17 | 静态 restored-data handoff：先导出修复值，再训练新预测器 | Stage 1 `ETTh1-ours-223368fc7e86`；Stage 2 `ETTh1-ours-b78bff6faef6` | S1 repair 0.707632 / 0.563942；S2 0.857974 / 0.623282 | 0.475595 / 0.446251 | +8.48% / +4.42% | +34.41% / +34.10% | 完整；历史静态修复对照 |
| E18 | 正式预训练 + 监督预测微调：S1 100 epoch；S2 原始 masked data + mask，forecast-only | Stage 1 `ETTh1-ours-51c31a2edc13`；Stage 2 `ETTh1-ours-6085fc7417cf` | S1 repair 0.601440 / 0.518443；S2 0.805145 / 0.610608 | 0.481877 / 0.457186 | +7.27% / +2.08% | +29.51% / +16.04% | 完整；历史正式两阶段版本 |
| E19 | E18 + Stage 2 冻结 gain/uncertainty 路由评分头 | `ETTh1-ours-e3d14b2276fc` | 0.815025 / 0.611406 | 0.473404 / 0.450233 | +8.90% / +3.57% | +36.12% / +27.52% | 完整；历史评分冻结版本 |
| E20 | 完整冻结 Stage 1 路由 teacher；学生消息与预测路径 forecast-only 微调 | `ETTh1-ours-f456446e5937` | 0.855924 / 0.626438 | 未测 | — | — | 验证-only；未通过 E19 MSE 门槛 |
| E21 | 冻结 Stage 1 teacher 候选/先验 + forecast-MSE 学习 H/V/none 软融合；T 预测权重为 0 | `ETTh1-ours-f34280265468` | **0.778489 / 0.601875** | 0.474366 / 0.456770 | +8.72% / +2.17% | +35.37% / +16.72% | 完整；E22 的 H/V 基线 |
| **E22** | **E21 H/V 受保护基线 + 独立全回溯动态 T + 在线显式修复 + mask-aware PatchTST** | Stage 1 `ETTh1-ours-8dda1f69aa67`；Stage 2 `ETTh1-ours-dea387986927` | S1 repair 0.451677 / 0.447030；S2 0.782639 / **0.596018** | **0.443195 / 0.433248** | **+14.72% / +7.21%** | **+59.71% / +55.57%** | **完整；当前 P4 最优** |

#### E17 的准确身份

之前记录的 `Ours 0.475595 / 0.446251，DRR 34.41% / 34.10%` 就是 **E17**，不是 E18。它使用静态恢复数据交接：Stage 1 修复数据并导出，Stage 2 使用恢复值和 mask 训练一个新的预测器，Stage 1 参数没有直接加载进 Stage 2。

#### E18 路由诊断

| 路由 | 预测 gain | 实际反事实 gain |
|---|---:|---:|
| H | +2.073922 | -0.122836 |
| T | -1.319296 | -0.292736 |
| V | +1.347703 | -0.303139 |

E18 的 H 选择率为 30.1205%，V 选择率为 69.8795%，none 为 0%，路由选择错误率为 56.6218%。移除 T 后测试分数完全不变；移除 V 后变为 0.507535 / 0.462658，与 H-only 相同。这说明 Stage 2 的 forecast-only 梯度改变了路由评分含义，Stage 1 学到的“修复收益”没有稳定保留下来。

#### E19 完成结果与设计局限

- Stage 1：直接复用 E18 的 100-epoch checkpoint，不重跑。
- Stage 2 损失：只计算 forecast MSE，不重新加入 route loss。
- 冻结：H/T/V gain 与 uncertainty 评分头、retrieval encoder、patch decoder、online memory/EMA/statistics。
- 训练：PatchTST core、mask projection、H/T/V message path、graph recovery。
- 路由评分在 Stage 2 中 detach；预测梯度仍能沿被选中的消息与恢复路径反向传播。
- 可训练参数 40,436；冻结参数 6,374。
- 完成 run：`ETTh1-ours-e3d14b2276fc`，最佳验证 MSE / MAE 为 0.815025447448 / 0.611405675373（epoch 94）。
- 正式测试 MSE / MAE 为 0.473403907958 / 0.450233105900，DRR MSE / MAE 为 36.1216928793% / 27.5193510012%。
- Stage 2 checkpoint SHA-256：`540403a875dfdb110e1a91b19c10a89dac34c11babe6d90469adf7389ecf8eb6`。
- 局限：虽然末端评分头参数冻结，其输入的 base latent 和候选消息仍由可训练模块产生，因而路由评分函数并不完全固定；E20 专门隔离这一变量。

#### E20 验证结果与硬路由局限

- teacher 模式：`frozen_stage1`；完整快照并冻结 Stage 1 的 patch projection 与 H/T/V 关系算子。
- Stage 1 来源：`artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v1/checkpoints/ETTh1_ours_stage1.pt`，SHA-256 `02af95b9b2196e00c5c180927c9a231b9d5698e910ce421929bffd2b761e7da4`。
- Stage 2 损失仍严格为 `forecast_mse_only`；不加入 route、repair、gain、memory、consistency 或 distillation loss。
- teacher 固定候选接纳、聚合计划、gain、uncertainty、availability 与 H/T/V/none 路由；学生继续训练关系消息、图恢复、mask projection 与 PatchTST。
- 新增只读诊断：`without_historical`、`route_repair_regret`、`route_repair_regret_positive_rate`，并保留 without-T/V 与 historical-only。
- artifact 目录：`artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v3_frozen_teacher_router`；已预置并核验 E18 固定 masks（validation `cbe7c520afded9002f714c5ed8e67d62c92b97fa7aa9b1e6e4711db785fd846c`，test `f7974ff9b4228d69fe33d940e3722cd1a069766e2fa1387c6240757d5a8762d8`）。
- run：`ETTh1-ours-f456446e5937`；最佳验证 MSE / MAE 为 `0.855924246263 / 0.626438361447`（epoch 24，共 100 epoch）。
- teacher checksum 与不变性检查通过，但最佳验证 MSE 未严格小于 `0.8150254474`，所以未运行正式测试。
- 验证消融：full `0.855924 / 0.626438`；without-H `0.899949 / 0.642961`；without-T `0.850244 / 0.626368`；without-V `0.911843 / 0.653404`。H/V 对预测有用，T 在此设置下无正收益，且 teacher 硬选路不能让 H/V 协同。

#### E21 训练结果与设计

- teacher 模式：`frozen_stage1_hv_fusion`；完整 Stage 1 teacher 保持冻结，只作为 H/V 候选拓扑与 detached 先验。
- 预测融合：由可训练 `ForecastHVFusion` 对 `{none,H,V}` 软加权，返回四路权重中 T 精确为 0；图恢复和 teacher 硬路由仍用于候选与诊断。
- Stage 2 损失：仍严格为 `forecast_mse_only`，不加 route、repair、gain、memory、consistency 或 distillation loss。
- 参数更新：融合器、H/V 学生消息、图恢复、mask projection 和 PatchTST 由预测 MSE 更新；teacher 与 T 预测候选参数冻结。
- artifact 目录：`artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v4_hv_forecast_fusion`。
- Stage 1 复用：`two_stage_v1/checkpoints/ETTh1_ours_stage1.pt`，SHA-256 `02af95b9b2196e00c5c180927c9a231b9d5698e910ce421929bffd2b761e7da4`。
- 固定 masks 已从 E20 原样复制到 E21 目录：validation SHA-256 `cbe7c520afded9002f714c5ed8e67d62c92b97fa7aa9b1e6e4711db785fd846c`，test SHA-256 `f7974ff9b4228d69fe33d940e3722cd1a069766e2fa1387c6240757d5a8762d8`。
- 回归验证：E21 relation/teacher/fusion/CLI 合并定向测试 `124 passed`；全量测试 `491 passed, 1 skipped`，唯一 skip 为预存的 official S4M checkout 未 bootstrap。
- Stage 2 run：`ETTh1-ours-f34280265468`；最佳验证 MSE / MAE 为 `0.778488870651 / 0.601874789413`（epoch 86，共 100 epoch）；训练耗时约 41.2 分钟。
- E21 checkpoint SHA-256：`633d821f373c9f2afd817db8705fd6599fe9f0f65a3e14528a2380b82bd91c17`。
- teacher checksum 与不变性检查均通过；最佳验证 MSE 严格小于 `0.8150254474`，已授权进入测试集。
- 测试：MSE / MAE 为 `0.474365506853 / 0.456770187603`，相对 Missing 改善 `8.72% / 2.17%`，DRR 为 `35.37% / 16.72%`；checkpoint、固定 test mask 与 baseline fingerprint 均已记录。
- 测试消融：full / without-T `0.474366 / 0.456770`；without-H `0.600211 / 0.519290`；without-V / historical-only `0.502999 / 0.463473`。H 是主要贡献，V 在软融合中仍有明确辅助作用，T 不贡献预测收益。
- 与 E19 比较：E21 测试 MSE `0.474366`，仅高 `0.000962`（约 `0.20%`）；但 MAE `0.456770`，比 E19 高 `0.006537`（约 `1.45%`），因此 E21 暂时不能称为全面超过 E19。

#### E22 当前最优结果与设计

- 方案：保持 E21 的 H/V 候选、关系消息和 `{none,H,V}` 软融合；把新 `DynamicTemporalContextRepair` 作为独立模块接在 H/V 基础恢复之后，以 `{none,local,global}` 动态门控产生 missing-only 有界残差。
- Stage 1 run：`ETTh1-ours-8dda1f69aa67`；最佳修复验证 MSE / MAE 为 `0.451677033141 / 0.447030060194`（epoch 72，共 100 epoch）。
- Stage 1 checkpoint SHA-256：`17dbf0c834d12149e5a12e5521a068c384f2f4cf9a8a335ae3e7b24e3cd522ab`。
- Stage 2 run：`ETTh1-ours-dea387986927`；最佳预测验证 MSE / MAE 为 `0.782638682800 / 0.596017590996`（epoch 100，共 100 epoch）。验证 MSE 比 E21 高约 `0.53%`，但正式测试显著改善。
- Stage 2 checkpoint SHA-256：`9d96d5c74b72e6b3a5b517d59504ca73db3550a6d12181e4bf74ab084f103c24`。
- 正式测试 MSE / MAE：`0.443195460875 / 0.433248216062`；相对 Missing 改善 `14.7159% / 7.2065%`；DRR `59.7069% / 55.5723%`。
- 相对 E21，测试 MSE / MAE 分别下降 `6.5709% / 5.1496%`。
- 测试消融：without-H `0.470573 / 0.447909`；without-T `0.498556 / 0.459510`；without-V `0.502524 / 0.474723`；H-only `0.514073 / 0.468811`。完整模型严格优于所有单项移除结果。
- 新 T 使用率：global `73.2394%`，local `15.8551%`，合计 selected `89.0944%`。这些是新动态 T 的诊断；旧兼容路由中的 `route_temporal_rate=0` 不代表新 T 未启用。
- 归档目录：`artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context`。
- 架构交接文档：`docs/superpowers/specs/2026-09-16-e22-current-best-architecture-handoff.md`。
- 协议级当前最优指针：`artifacts/ETTh1/seq96_pred96/sr010_bl8/CURRENT_BEST.json`。

### P5：seq96 → 96，start_rate=0.15，block_len=8（E22 缺失压力测试）

除 `missing.start_rate` 外，配置与 P4 E22 一致；Clean 基线复用，Missing 基线重新训练。零样本迁移直接加载 P4 E22 checkpoint 与在线状态，在 P5 固定 mask 上评估，不做参数更新。重训从头运行 Stage 1 和 Stage 2，不从 P4 checkpoint 初始化。两者共享 P5 validation/test mask。

| ID | 方法 | 模型 / run ID | Val MSE / MAE | Test MSE / MAE | 相对 P5 Missing 改善 MSE / MAE | P5 DRR MSE / MAE | 状态 |
|---|---|---|---:|---:|---:|---:|---|
| E23-Z | P4 E22 零样本迁移，不重训 | 来源 Stage 2 `ETTh1-ours-dea387986927` | 0.894785 / 0.640409 | **0.504612 / 0.467289** | **+16.03% / +8.60%** | **+46.02% / +41.90%** | 完整；P5 零样本对照 |
| E23-R | E22 同结构在 P5 从头重训 | Stage 1 `ETTh1-ours-584d9a9215b9`；Stage 2 `ETTh1-ours-40454be537d2` | S1 repair 0.611537 / 0.510407；S2 **0.868966 / 0.631087** | 0.508583 / 0.472684 | +15.37% / +7.54% | +44.12% / +36.76% | 完整；P5 重训对照 |

P5 的 Missing-Clean 测试 MSE 差距为 `0.209371`，大于 P4 的 `0.128082`。重训在验证集比零样本迁移降低 MSE 约 `2.89%`，但在测试集高 `0.79%`，MAE 高 `1.15%`。两种设置的完整模型均优于移除 H/T/V 的消融；零样本和重训 without-T 测试 MSE 分别为 `0.589839`、`0.588851`。P4 `CURRENT_BEST.json` 保持原样。

### P6：seq96 → 96，start_rate=0.10，block_len=5（E22 轻缺失测试）

除 `missing.block_len` 从 8 缩短到 5 外，配置与 P4 E22 一致。P6 生成的掩码数组和元数据与旧 P1 逐点相同，归档文件 SHA-256 不同；旧 P1 的模型配置和基线缓存不同，所以 P6 重新训练 Missing，Clean 复用当前缓存。零样本迁移使用原 P4 E22 Stage 2 checkpoint 与在线状态；重训从随机初始化 Stage 1，再由本次 Stage 1 初始化 Stage 2。两种结果共享 P6 固定 validation/test mask。

| ID | 方法 | 模型 / run ID | Val MSE / MAE | Test MSE / MAE | 相对 P6 Missing 改善 MSE / MAE | P6 DRR MSE / MAE | 状态 |
|---|---|---|---:|---:|---:|---:|---|
| E24-Z | P4 E22 零样本迁移，不重训 | 来源 Stage 2 `ETTh1-ours-dea387986927` | **0.737442 / 0.574080** | 0.409095 / 0.417777 | +11.40% / +5.54% | +75.03% / +68.19% | 完整；P6 零样本对照 |
| E24-R | E22 同结构在 P6 从头重训 | Stage 1 `ETTh1-ours-678af27e9ada`；Stage 2 `ETTh1-ours-6c83b6d2bf14` | S1 repair 0.306230 / 0.364522；S2 0.748429 / 0.580192 | **0.405926 / 0.413958** | **+12.08% / +6.40%** | **+79.55% / +78.82%** | 完整；P6 重训对照 |

P6 的 Missing-Clean 测试 MSE 差距仅 `0.070124`，DRR 分母较小，须结合绝对 MSE/MAE 一起看。重训测试 MSE / MAE 比零样本迁移分别低 `0.003169 / 0.003819`（相对零样本约 `0.77% / 0.91%`），但零样本验证集反而更好；单种子结果不足以断言重训稳定占优。重训版移除 H、T、V 的测试 MSE 分别为 `0.414086`、`0.437469`、`0.416849`；完整模型 `0.405926`，三者在本协议均有贡献，其中 T 的消融差距最大。旧兼容路由 `route_temporal_rate=0` 不表示新动态 T 未启用。

### P7：seq96 → 96，start_rate=0.10，block_len=3（E22 更轻缺失测试）

除 `missing.block_len` 从 8 缩短到 3 外，配置与 P4 E22 完全一致。实际最终缺失率 `27.0305%`。Clean 基线复用当前缓存，Missing 在 P7 固定掩码下重训。E25-Z 直接加载 P4 E22 Stage 2 checkpoint 与在线状态，不更新参数；E25-R 从随机初始化重新训练 100 轮 Stage 1，再用本次 Stage 1 checkpoint 初始化 100 轮 Stage 2。两者与 Missing 共享 P7 test mask。

| ID | 方法 | 模型 / run ID | Val MSE / MAE | Test MSE / MAE | 相对 P7 Missing 改善 MSE / MAE | P7 DRR MSE / MAE | 状态 |
|---|---|---|---:|---:|---:|---:|---|
| E25-Z | P4 E22 零样本迁移，不重训 | 来源 Stage 2 `ETTh1-ours-dea387986927` | 0.720712 / 0.563373 | 0.409385 / 0.416107 | +3.52% / +1.98% | +45.63% / +46.26% | 完整；P7 零样本对照 |
| E25-R | E22 同结构在 P7 从头重训 | Stage 1 `ETTh1-ours-0ecf2d73e5d1`；Stage 2 `ETTh1-ours-57a01c8261ee` | S1 repair 0.185864 / 0.287735；S2 0.717175 / 0.564436 | **0.385426 / 0.402017** | **+9.17% / +5.30%** | **+118.82% / +123.86%** | 完整；P7 重训对照 |

P7 Missing–Clean 的测试差距仅为 MSE `0.032736`、MAE `0.018159`。E25-R 相对 Missing 的绝对改善为 MSE `0.038897`、MAE `0.022491`；相对 Clean 分别低 `0.006161 / 0.004332`，但这只是当前单次训练/测试的数值比较，不能直接证明“恢复值比真实值更好”。E25-Z 移除 T 后测试 MSE 从 `0.409385` 降到 `0.399420`；E25-R 移除 T 则从 `0.385426` 升到 `0.402443`。E25-R 移除 H/V 后 MSE 分别为 `0.386603 / 0.392484`，完整模型仍最好；H 的边际差距较小。旧兼容 `route_temporal_rate=0` 不是动态 T 关闭，新 T 的 selected rate 接近 100%。

### P8：seq96 → 96，start_rate=0.20，block_len=8（E22 极高缺失测试）

仅将 P4 E22 的 `missing.start_rate` 从 0.10 提高到 0.20；其他配置、模型、seed 和数据切分不变。实际最终缺失率为 `82.9157%`。Clean 复用既有缓存，Missing 在 P8 固定 mask 下重新训练。E26-Z 加载 P4 E22 Stage 2 checkpoint 和在线状态，不更新参数；E26-R 从随机初始化训练 100 轮 Stage 1，再加载本次 Stage 1 checkpoint 训练 100 轮 Stage 2。三者共享 P8 test mask。

| ID | 方法 | 模型 / run ID | Val MSE / MAE | Test MSE / MAE | 相对 P8 Missing 改善 MSE / MAE | P8 DRR MSE / MAE | 状态 |
|---|---|---|---:|---:|---:|---:|---|
| E26-Z | P4 E22 零样本迁移，不重训 | 来源 Stage 2 `ETTh1-ours-dea387986927` | 1.037186 / 0.700469 | 0.616536 / 0.511924 | +14.58% / +7.56% | +31.87% / +28.41% | 完整；P8 零样本对照 |
| E26-R | E22 同结构在 P8 从头重训 | Stage 1 `ETTh1-ours-544afa887949`；Stage 2 `ETTh1-ours-3add7bd3de91` | S1 repair 0.722008 / 0.570901；S2 0.994953 / 0.681371 | **0.609522 / 0.509541** | **+15.55% / +7.99%** | **+34.00% / +30.02%** | 完整；P8 重训对照 |

P8 Missing–Clean 的测试差距为 MSE `0.330184`、MAE `0.147466`，不属于小分母协议。E26-R 相对 Missing 的绝对改善为 `0.112250 / 0.044274`；相对 E26-Z 进一步降低 MSE `0.007015`、MAE `0.002383`（约 `1.14% / 0.47%`），幅度不大。E26-R 移除 H/T/V 后测试 MSE 分别为 `0.692354 / 0.692549 / 0.731978`，均高于完整模型 `0.609522`；零样本的对应移除结果也均更差。新动态 T 的候选/选中率约 `58.97%`，旧兼容路由的 `route_temporal_rate=0` 不代表新 T 未启用。跨 P4/P5/P8 可比较缺失压力趋势，但不能直接按绝对误差给不同协议的模型排名。

## 5. 关键文件与 checkpoint 回溯

### 5.1 P4 公共基线缓存

这些基线不随 Ours 实验变化，后续同一 P4 协议应复用，不要每次重训。

| 类型 | 指标 | fingerprint / SHA-256 | 文件 |
|---|---:|---|---|
| Clean | 0.391587319017 / 0.406349116172 | fingerprint `b5932e46c76ee3e89ccbc51bec219b521bb1c8b4ed085d063a8c1b89102148fe` | `artifacts/ETTh1/seq96_pred96/clean/b5932e46c76ee3e89ccbc51bec219b521bb1c8b4ed085d063a8c1b89102148fe/baseline.json` |
| Clean checkpoint | — | `89587143b9685b83246233c31da7427f044a43b292ff0ecb67443b837962f618` | 同一 fingerprint 目录内 checkpoint |
| Missing | 0.519669280842 / 0.466894917477 | fingerprint `292570de0560129a4bd96568f13fc19875e06e80fd7041a7e7f8d06a89710f10` | `artifacts/ETTh1/seq96_pred96/sr010_bl8/missing/292570de0560129a4bd96568f13fc19875e06e80fd7041a7e7f8d06a89710f10/baseline.json` |
| Missing checkpoint | — | `011458cccae5025b6c294cdd9c0ef444246843d42f2f7501930032b0c759facf` | 同一 fingerprint 目录内 checkpoint |
| Validation mask | — | `cbe7c520afded9002f714c5ed8e67d62c92b97fa7aa9b1e6e4711db785fd846c` | P4 mask cache |
| Test mask | — | `f7974ff9b4228d69fe33d940e3722cd1a069766e2fa1387c6240757d5a8762d8` | P4 mask cache |
| ETTh1 CSV | — | `f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066` | `data/raw/ETTh1.csv` |

### 5.2 关键实验目录

| 实验 | artifact 根目录 | 关键 checkpoint / 记录 |
|---|---|---|
| E06 | `artifacts/etth1_sr010_grouped_memory` | `checkpoints/ETTh1_ours.pt`、对应 runs/evaluations |
| E06 汇总表 | `/home/xyw/missing_aware_latent_recovery/.superpowers/sdd/2026-09-11-关系专用时序变量边恢复/ETTh1_sr010_grouped_memory_2026-09-07.xlsx` | P1 历史表格 |
| E15 | `artifacts/etth1_seq96_bl8_v2` | `ETTh1-ours-3901c37fae2f` 对应 checkpoint；E14 checkpoint 已被覆盖 |
| E16 | `artifacts/etth1_seq96_bl8_v2_two_stage` | `ETTh1-ours-0973e2755ffd` 对应 runs/evaluations/checkpoint |
| E17 | `artifacts/etth1_seq96_bl8_v3_direct_repair` | Stage 1 SHA `a1153809d5050102ed9a23feb6c27677a241a8e67bfa30a1e28b8cc62117492a`；Stage 2 SHA `2d520b318fa9a5728cc0c81ae0b93e71e535b594aab337b67354f706dabc037b` |
| E17 对比 | `artifacts/etth1_seq96_bl8_v3_direct_repair/evaluations/comparisons/ETTh1-ours-b78bff6faef6.json` | Clean/Missing/Ours/DRR 汇总 |
| E18 | `artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v1` | `checkpoints/ETTh1_ours_stage1.pt` 与 `ETTh1_ours_stage2.pt` |
| E19 | `artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v2_frozen_scores` | run `ETTh1-ours-e3d14b2276fc`；`checkpoints/ETTh1_ours_stage2.pt`，SHA `540403a875dfdb110e1a91b19c10a89dac34c11babe6d90469adf7389ecf8eb6` |
| E20（验证-only） | `artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v3_frozen_teacher_router` | run `ETTh1-ours-f456446e5937`；Stage 2 checkpoint SHA `7692f51c7cb3b3215bba8284f0522f98d9330cd20d50c6fa231652dde909cf89`；未通过验证门槛 |
| E21 | `artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v4_hv_forecast_fusion` | run `ETTh1-ours-f34280265468`；Stage 2 SHA `633d821f373c9f2afd817db8705fd6599fe9f0f65a3e14528a2380b82bd91c17`；正式测试已完成 |
| **E22（当前 P4 最优）** | `artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context` | Stage 1 run `ETTh1-ours-8dda1f69aa67`，SHA `17dbf0c834d12149e5a12e5521a068c384f2f4cf9a8a335ae3e7b24e3cd522ab`；Stage 2 run `ETTh1-ours-dea387986927`，SHA `9d96d5c74b72e6b3a5b517d59504ca73db3550a6d12181e4bf74ab084f103c24` |
| E23-Z（P5 零样本迁移） | `artifacts/ETTh1/seq96_pred96/sr015_bl8/experiments/e22_dynamic_temporal_retrain` | 复用 P4 E22 Stage 2 checkpoint；`stress_evaluations/ETTh1-ours-dea387986927_validation.json` 和 `_test.json` |
| E23-R（P5 从头重训） | `artifacts/ETTh1/seq96_pred96/sr015_bl8/experiments/e22_dynamic_temporal_retrain` | Stage 1 run `ETTh1-ours-584d9a9215b9`，SHA `5f11bbca1e250284011e42288c5b07f6db5df19826ae3b73896fd12916c2e7a7`；Stage 2 run `ETTh1-ours-40454be537d2`，SHA `747d7a2013264b575f6fb9d0f271d4707af19c8974e9d58e3ae4b812ae2880e1` |
| E24-Z（P6 零样本迁移） | `artifacts/ETTh1/seq96_pred96/sr010_bl5/experiments/e22_dynamic_temporal_low_missing` | 复用 P4 E22 Stage 2 checkpoint；`stress_evaluations/ETTh1-ours-dea387986927_validation.json` 和 `_test.json` |
| E24-R（P6 从头重训） | `artifacts/ETTh1/seq96_pred96/sr010_bl5/experiments/e22_dynamic_temporal_low_missing` | Stage 1 run `ETTh1-ours-678af27e9ada`，SHA `58743c24e13031b66130c4cb0f309b6690eaccf18309dbdf876a1a5b99d4bf8b`；Stage 2 run `ETTh1-ours-6c83b6d2bf14`，SHA `e3c07479f5c423dd76d87daf58b03a4cc2bebad29395cdd523494eadf07eecde` |
| E25-Z（P7 零样本迁移） | `artifacts/ETTh1/seq96_pred96/sr010_bl3/experiments/e22_dynamic_temporal_very_low_missing` | 复用 P4 E22 Stage 2 checkpoint；`stress_evaluations/ETTh1-ours-dea387986927_validation.json` 和 `_test.json` |
| E25-R（P7 从头重训） | `artifacts/ETTh1/seq96_pred96/sr010_bl3/experiments/e22_dynamic_temporal_very_low_missing` | Stage 1 run `ETTh1-ours-0ecf2d73e5d1`，SHA `7037df390f88c5adb53b4038b2a0cb92122092bc7adc3bac17643a6cf58ccc35`；Stage 2 run `ETTh1-ours-57a01c8261ee`，SHA `1b9e25d613769310da5f28fe42b0749bdbeb1498097abc4be52348351d6b3a01` |
| E26-Z（P8 零样本迁移） | `artifacts/ETTh1/seq96_pred96/sr020_bl8/experiments/e22_dynamic_temporal_extreme_missing` | 复用 P4 E22 Stage 2 checkpoint；`stress_evaluations/ETTh1-ours-dea387986927_validation.json` 和 `_test.json` |
| E26-R（P8 从头重训） | `artifacts/ETTh1/seq96_pred96/sr020_bl8/experiments/e22_dynamic_temporal_extreme_missing` | Stage 1 run `ETTh1-ours-544afa887949`，SHA `ff98ab9e90a67a85f23185d06e984117af77d52baccc6938458c2bd4a4c08ca3`；Stage 2 run `ETTh1-ours-3add7bd3de91`，SHA `07f04935efa1e3b8376212616cd7d77e09a307c869905d06ba7eee9718f11390` |

E18 Stage 1 checkpoint SHA-256：`02af95b9b2196e00c5c180927c9a231b9d5698e910ce421929bffd2b761e7da4`。  
E18 Stage 2 checkpoint SHA-256：`22932ea1965b73c000080af995d758d693a9eada30db3a373180f073b6e75271`。

### 5.3 P5 固定掩码与基线

| 类型 | 指标 / SHA-256 | 文件 |
|---|---|---|
| Clean（复用 P4） | MSE / MAE `0.391587319017 / 0.406349116172` | `artifacts/ETTh1/seq96_pred96/clean/b5932e46c76ee3e89ccbc51bec219b521bb1c8b4ed085d063a8c1b89102148fe/baseline.json` |
| Missing（P5 重训） | MSE / MAE `0.600957924547 / 0.511242583225`；fingerprint `a191cfee1454c64ec1a77e1a06a8f72014fe59814cabac0dcea1bf3e33d28586` | `artifacts/ETTh1/seq96_pred96/sr015_bl8/missing/a191cfee1454c64ec1a77e1a06a8f72014fe59814cabac0dcea1bf3e33d28586/baseline.json` |
| Validation mask | `914aaff4e3de6aacce2323fab477f37af48b65bd2934cee31be6fc4debe47bb6` | `artifacts/ETTh1/seq96_pred96/sr015_bl8/experiments/e22_dynamic_temporal_retrain/masks/ETTh1_validation.npz` |
| Test mask | `a023ee1da642229e3169ba817f9103c2170d62510b7063c7b28f9f6304158f1e` | `artifacts/ETTh1/seq96_pred96/sr015_bl8/experiments/e22_dynamic_temporal_retrain/masks/ETTh1_test.npz` |

P5 配置为 `configs/stage1_seq96_dynamic_temporal_repair_sr015.yaml`；配置 SHA-256（规范化配置哈希）为 `8e2b261d845757375b48bf9e904d95ea0683238db38c3bad95148a95ab299a45`。E23-Z 与 E23-R 的测试记录均指向同一个 test mask SHA-256。E23-Z 记录来源 checkpoint SHA-256 `9d96d5c74b72e6b3a5b517d59504ca73db3550a6d12181e4bf74ab084f103c24`；E23-R 记录自身 checkpoint SHA-256 `747d7a2013264b575f6fb9d0f271d4707af19c8974e9d58e3ae4b812ae2880e1`。

### 5.4 P6 固定掩码与基线

| 类型 | 指标 / SHA-256 | 文件 |
|---|---|---|
| Clean（复用当前缓存） | MSE / MAE `0.391587319017 / 0.406349116172` | `artifacts/ETTh1/seq96_pred96/clean/b5932e46c76ee3e89ccbc51bec219b521bb1c8b4ed085d063a8c1b89102148fe/baseline.json` |
| Missing（P6 重训） | MSE / MAE `0.461710942846 / 0.442271714848`；fingerprint `5f479539fbe76647106028fab868cc18cc68372bb1ba206b5334e3498291a288` | `artifacts/ETTh1/seq96_pred96/sr010_bl5/missing/5f479539fbe76647106028fab868cc18cc68372bb1ba206b5334e3498291a288/baseline.json` |
| Validation mask | `349217bfdd78d4d695b9144e6de5608c608de3a945ab2f1f1ee1a04f6151b2d7` | `artifacts/ETTh1/seq96_pred96/sr010_bl5/experiments/e22_dynamic_temporal_low_missing/masks/ETTh1_validation.npz` |
| Test mask | `c7e77ea374b42cfeb0adb4751cec9714fbd44e77c0be2e0972b760aaf5b084b9` | `artifacts/ETTh1/seq96_pred96/sr010_bl5/experiments/e22_dynamic_temporal_low_missing/masks/ETTh1_test.npz` |

P6 配置为 `configs/stage1_seq96_dynamic_temporal_repair_bl5.yaml`，文件 SHA-256 `6501f54d6b89de3c75532cac7fb67d015dc7ee978e71ff144a49c1f7463a39b1`，规范化配置哈希 `83fb64eef524d646c7b5d8a4cbc24a4ef7b8903c2d66e8c08ef3b165b9a6194c`。E24-Z 与 E24-R 的测试记录均指向同一个 test mask SHA-256。E24-Z 使用 P4 E22 checkpoint SHA-256 `9d96d5c74b72e6b3a5b517d59504ca73db3550a6d12181e4bf74ab084f103c24`；E24-R 使用新 checkpoint SHA-256 `e3c07479f5c423dd76d87daf58b03a4cc2bebad29395cdd523494eadf07eecde`。`make-eval-masks` 会写入所指定目录，因此各协议必须保持独立 artifact 根目录，不能混放或重复生成到旧实验目录。

### 5.5 P7 固定掩码与基线

| 类型 | 指标 / SHA-256 | 文件 |
|---|---|---|
| Clean（复用当前缓存） | MSE / MAE `0.391587319017 / 0.406349116172` | `artifacts/ETTh1/seq96_pred96/clean/b5932e46c76ee3e89ccbc51bec219b521bb1c8b4ed085d063a8c1b89102148fe/baseline.json` |
| Missing（P7 重训） | MSE / MAE `0.424323343636 / 0.424507923948`；fingerprint `8744ccc618c72ee893766a37872ee5bb507b83ac6c8d852e1f87cfdd25e40aab` | `artifacts/ETTh1/seq96_pred96/sr010_bl3/missing/8744ccc618c72ee893766a37872ee5bb507b83ac6c8d852e1f87cfdd25e40aab/baseline.json` |
| Validation mask | `89c7cdf7e6dc22bf36286822b9cb89a8b316ae58cc0baea46d2efaf859061a21` | `artifacts/ETTh1/seq96_pred96/sr010_bl3/experiments/e22_dynamic_temporal_very_low_missing/masks/ETTh1_validation.npz` |
| Test mask | `1e87e9cf42845c73187be02aea83acdc53e0055ff1d47987b48c27f2c6d1ff5d` | `artifacts/ETTh1/seq96_pred96/sr010_bl3/experiments/e22_dynamic_temporal_very_low_missing/masks/ETTh1_test.npz` |

P7 配置为 `configs/stage1_seq96_dynamic_temporal_repair_bl3.yaml`，文件 SHA-256 `1d9bf013466793377a1d55f789a0c94001731186d53e669119ea5b193bb90e1a`，规范化配置哈希 `d29827a5773e06f15f05228bd0193a3416cd21cf2525d8f46f9659bb85eb2edb`。E25-Z、E25-R 与 Missing 的测试记录均指向该规范化配置哈希和 test mask SHA-256。E25-Z 来源 checkpoint SHA-256 为 `9d96d5c74b72e6b3a5b517d59504ca73db3550a6d12181e4bf74ab084f103c24`；E25-R Stage 2 manifest 明确指向本次 Stage 1 checkpoint SHA-256 `7037df390f88c5adb53b4038b2a0cb92122092bc7adc3bac17643a6cf58ccc35`，teacher checksum/invariance 检查均通过。各协议目录互相独立，P4 `CURRENT_BEST.json` 未改变。

### 5.6 P8 固定掩码与基线

| 类型 | 指标 / SHA-256 | 文件 |
|---|---|---|
| Clean（复用当前缓存） | MSE / MAE `0.391587319017 / 0.406349116172` | `artifacts/ETTh1/seq96_pred96/clean/b5932e46c76ee3e89ccbc51bec219b521bb1c8b4ed085d063a8c1b89102148fe/baseline.json` |
| Missing（P8 重训） | MSE / MAE `0.721771635977 / 0.553815052353`；fingerprint `7d078924dcf1ec18dae74b6e01cc96b8564152a0a76728052cdd9d3354613391` | `artifacts/ETTh1/seq96_pred96/sr020_bl8/missing/7d078924dcf1ec18dae74b6e01cc96b8564152a0a76728052cdd9d3354613391/baseline.json` |
| Validation mask | `8cda5409c8297b488910f2f2ebaaf4bd3752252e9563b0e46cb8665996c09835` | `artifacts/ETTh1/seq96_pred96/sr020_bl8/experiments/e22_dynamic_temporal_extreme_missing/masks/ETTh1_validation.npz` |
| Test mask | `6c52e0e6214bca2356f356ff2c8bf1a11b1b96609a32831a82578a799cfd1bc7` | `artifacts/ETTh1/seq96_pred96/sr020_bl8/experiments/e22_dynamic_temporal_extreme_missing/masks/ETTh1_test.npz` |

P8 配置为 `configs/stage1_seq96_dynamic_temporal_repair_sr020.yaml`，文件 SHA-256 `dd66b4afdd1f1db8dca25747bc0467a43253fc7144989a3fadb03b7f0ca94924`，规范化配置哈希 `7fb53bd53787c65ce83d7aa9b1bd02fe410186b5338e1011f0c414a672d729cc`。E26-Z、E26-R 与 Missing 的测试记录已核对共同指向该配置哈希与 test mask SHA-256；E26-Z 来源为 P4 E22 checkpoint SHA-256 `9d96d5c74b72e6b3a5b517d59504ca73db3550a6d12181e4bf74ab084f103c24`；E26-R Stage 2 manifest 指向本次 Stage 1 checkpoint SHA-256 `ff98ab9e90a67a85f23185d06e984117af77d52baccc6938458c2bd4a4c08ca3`，teacher checksum/invariance 检查均通过。生成 mask 前的 83.92% 估值错误地使用了 2880 步，而正式固定 mask 包括额外 96 步历史支持区间、长度为 2976；实际记录 `82.9157%` 为准。P4 `CURRENT_BEST.json` 仍限定并指向 P4 E22，不因 P8 改写。

早期 Windows 运行记录中的绝对路径仍可能是 Windows 路径。原始 JSON 不应修改；如需重新评估，应复制 manifest 到新实验目录并迁移路径，保留原记录作为证据。

## 6. 两阶段定义（当前版本）

### Stage 1：自监督修复预训练

目标：让 H/V 基础关系、新的动态 T、图恢复和显式数值恢复学习“如何从可见上下文恢复人为遮挡位置”，不进行预测任务。

当前损失：

`L_stage1 = 1.0 × L_repair + 0.20 × L_gain + 0.05 × L_memory`

- `L_repair`：在 overlap aggregation 后的唯一序列缺失坐标上计算 SmoothL1，是唯一数值重建主损失。
- `L_gain`：只校准 H/V 候选的真实修复收益；新 T 不进入旧 gain/hard-route 目标。
- `L_memory`：约束历史原型/检索表示，使训练积累的历史原型保持可用。
- forecast loss 和旧 route loss 不参与 Stage 1，避免重复优化同一选路目标。

### Stage 2：监督预测微调

输入不是 Stage 1 对训练集导出的静态恢复结果，而是每个样本的原始 masked values 与 observation mask。模型加载 Stage 1 学到的参数，在每个 batch 内依次执行 H/V 基础恢复、新动态 T 修正、显式 overlap aggregation、观测值硬覆盖和重新切 patch。

`L_stage2 = L_forecast_mse`

恢复序列重新切成 patch 后，mask projection 会标记哪些输入来自真实观测、哪些来自模型恢复。Stage 2 的 forecast PatchTST、mask projection，以及在线修复路径中的 H/V 学生消息、graph、H/V fusion、新 T、repair projection/decoder 由预测 MSE 微调；Stage 1 teacher 和在线 memory/statistics 状态保持只读。

## 7. E22 定版时的代码状态与验证

当前 shell 没有可用的 `git` 可执行文件，因此 E22 不能用 commit/tag 标记。当前版本改用 checkpoint、mask、配置和关键实现文件 SHA-256 共同标记：

| 文件 | SHA-256 |
|---|---|
| `configs/stage1_seq96_dynamic_temporal_repair.yaml` | `6805b1c785e884411e85895104b927c815a75dea96d09a53728c916bcf61c9e5` |
| `src/malr/models/dynamic_temporal_context.py` | `8c458be2475ad129b3316524fe20906759eaee8e65579e520634025c70e7bf6b` |
| `src/malr/models/explicit_repair.py` | `835606b293b05bf31645582d6c72659e9bf8e6ca9cdea724d0cd36040661cbb9` |
| `src/malr/models/dynamic_context_model.py` | `0cdd5ede2dd4f44ccd6bb80a83f46fd6ea597eb8f59b1870a35c4a80d7f500df` |
| `src/malr/cli.py` | `de30645f31227c1c691abbcbdd73028ec2d572b3e7c97663fb3cd37b08c28b16` |
| E22 Stage 1 checkpoint | `17dbf0c834d12149e5a12e5521a068c384f2f4cf9a8a335ae3e7b24e3cd522ab` |
| E22 Stage 2 checkpoint | `9d96d5c74b72e6b3a5b517d59504ca73db3550a6d12181e4bf74ab084f103c24` |

2026-09-16 新鲜全量测试结果：`502 passed, 1 skipped`；唯一 skip 为 official S4M checkout 尚未 bootstrap。H/V 受保护实现的更多哈希详见 E22 架构交接文档。

2026-09-19 为 P5 压力测试新增独立配置和 `stress-evaluate` 命令，并对新协议使用同协议 Missing 验证门槛；因此上表 `src/malr/cli.py` 哈希是 **E22 定版快照**，不是现在磁盘上的哈希。P4 E22 的 checkpoint、mask 与 `CURRENT_BEST.json` 未改动。P5 实验后全量测试：`509 passed, 1 skipped`；唯一 skip 仍为 official S4M checkout 未 bootstrap。

2026-09-20 为 P6 轻缺失测试扩展 `stress-evaluate`，允许与来源配置只差更短的 `missing.block_len`，但禁止同时改动起始率或架构。P6 实验后全量测试：`516 passed, 1 skipped`；唯一 skip 同上。P6 三份测试记录（零样本、重训、Missing）的配置哈希与 test mask SHA-256 相同，checkpoint SHA-256 均与实际文件匹配。

2026-09-20 P7 更轻缺失测试仅新增配置 `configs/stage1_seq96_dynamic_temporal_repair_bl3.yaml` 及其协议测试；将该配置中的 `block_len: 3` 替回 `8` 后，与 P4 E22 配置逐字节一致。E25-Z、E25-R、Missing 的配置哈希、test mask SHA-256 已核对相同；Stage 2 manifest 指向本次 Stage 1 checkpoint，checkpoint 文件哈希一致；P4 最优标记仍指向 E22。新鲜全量测试：`517 passed, 1 skipped`，唯一 skip 同上。

2026-09-20 P8 极高缺失测试仅新增配置 `configs/stage1_seq96_dynamic_temporal_repair_sr020.yaml` 及其协议测试；把配置中的 `start_rate: 0.20` 替回 `0.10` 后，与 P4 E22 配置逐字节一致。E26-Z、E26-R、Missing 的配置哈希与 test mask SHA-256 已核对相同；两阶段 checkpoint 链、teacher 不变性和 P4 最优标记均已核对。新鲜全量测试：`518 passed, 1 skipped`，唯一 skip 同上。

注意：哈希只能识别代码，不能恢复已丢失的代码内容。恢复 Git 后，应给每次正式实验记录 commit hash；恢复前不要覆盖已完成实验目录。

## 8. 回到某次实验时怎么做

1. 先在第 4 节通过协议和结果找到实验 ID。
2. 在第 5 节定位 artifact 根目录。
3. 读取 `runs/<run-id>.json` 确认 config、训练阶段和 checkpoint；读取 `evaluations/<run-id>.json` 获取正式测试结果。
4. 用 `sha256sum <checkpoint>` 和台账/manifest 中的 SHA-256 核对权重。
5. 需要继续训练时，永远新建 experiment 目录，不要复用旧目录或覆盖旧 checkpoint。
6. 如果只是查历史结果，直接读取 evaluation JSON，不重新训练 Clean/Missing。

通用查看命令：

```bash
cd /home/xyw/missing_aware_latent_recovery/.superpowers/sdd/2026-08-31-residual-mlp-recovery/checkout
conda run -n malr python -m json.tool artifacts/ETTh1/seq96_pred96/sr010_bl8/CURRENT_BEST.json
conda run -n malr python -m json.tool artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context/evaluations/ETTh1-ours-dea387986927.json
sha256sum artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context/checkpoints/ETTh1_ours_stage2.pt
```

E22 已完成训练和正式测试。只查看结果时不要重新运行 Clean、Missing 或 evaluation，直接读取落盘文件：

```bash
cd /home/xyw/missing_aware_latent_recovery/.superpowers/sdd/2026-08-31-residual-mlp-recovery/checkout
conda run -n malr python -m json.tool \
  artifacts/ETTh1/seq96_pred96/sr010_bl8/CURRENT_BEST.json

conda run -n malr python -m json.tool \
  artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context/evaluations/ETTh1-ours-dea387986927.json
```

需要复现实验时，应复制 E22 配置和固定 masks 到新的 experiment 目录，并保留当前 E22 artifact 目录只读。

## 9. 后续实验登记规范

目录固定为：

`artifacts/<dataset>/seq<seq_len>_pred<pred_len>/sr<start_rate>_bl<block_len>/experiments/<experiment_id>/`

每次实验完成后必须追加以下信息：

```text
实验 ID：
日期：
协议：dataset / seq_len / pred_len / start_rate / block_len / seed / mask SHA
相对上一版本唯一改动：
Stage 1 run ID / checkpoint SHA / best epoch / validation：
Stage 2 run ID / checkpoint SHA / best epoch / validation：
Test MSE / MAE：
相对 Missing 改善：
DRR：
路由选择率与真实反事实 gain：
artifact 根目录：
代码 commit；无 Git 时记录关键文件 SHA：
结论：保留、淘汰或待复查：
```

同一 experiment 目录一旦完成即视为只读。新逻辑、新损失、新冻结策略或新超参数都使用新目录；否则即使 run JSON 还在，也可能像 E14 一样丢失精确 checkpoint。
