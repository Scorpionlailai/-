# E23 v28: Stage 1 修复目标与选模尺度对齐

日期：2026-10-03。状态：两阶段各100轮完成；通过预先固定的验证门槛并完成三组测试。

## 诊断与假设

在已选定 v12 的固定验证集上，query_std < 0.01 的位置只占缺失坐标的 0.672678%，却贡献窗口归一化平方误差的 98.991745%。同一模型换算到训练集全局标准化尺度，修复 MSE 为 0.368823644，此类位置只贡献平方误差的 0.286937%。T1 Stage 1 的同尺度 MSE 为 0.154001358。

数值主导尚不能证明这些位置决定了历轮 checkpoint 的排序。v28 记录每轮两种尺度的验证误差，以检查排序是否不同，并检验训练目标与预测评价尺度对齐是否提高最终预测。

## 本轮修正

配置：`configs/e23_multi_expert_p4_width64_global_stage1_v28.yaml`。相对 v12，新增 `training.stage1_objective: global_mse_v1`。

- Stage 1：隐藏且有效的位置上，计算 `((repair_normalized - target_normalized) * query_std)^2` 的平均值。query_std 来自当前可见点并停止梯度；query_mean 在误差中抵消。目标尺度是训练集全局标准化尺度，不是物理原始单位。训练总损失为该 MSE 加 0.1 倍原检索辅助损失。
- Stage 1：按同尺度固定验证 MSE 选择最早的最优 checkpoint。另存窗口归一化 MSE、全局 MAE、小标准差位置比例及两种平方误差贡献；每轮写入 `stage1/runs/e23-stage1-history.json`。
- 新 manifest 与 checkpoint 显式保存 `stage1_objective` 和 `stage1_selection_metric_space`，并传播到 Stage 2。旧配置默认 `window_smooth_l1`，原 checkpoint 身份规则保留。
- 修复仍使用 v12 H/V/T、宽度64和四来源 V；预测仍使用原宽度16的 PatchTST。两阶段各100轮、batch128、seed2021、Adam学习率0.001/0.0001。Stage 2 仍用 MSE + 0.2 MAE，按原固定验证 MAE 选模。

本轮联合改变 Stage 1 的损失形状及损失/选模尺度；结果不能单独归因于其中一项。

## 固定评价与资源口径

ETTh1 P4：96→96、start_rate0.10、block_len8、允许块重叠；训练每轮动态缺失，验证使用原固定位置。三组位置只用于锁定后的测试。

晋级条件在训练前固定：Stage 2 选定 checkpoint 的验证 MAE 严格低于 v12 的 **0.5882120694703256**，且 MSE 不高于 **0.7454477286392869**。选定权重须完整重放固定验证集并复现指标，检查 H 状态不变。通过后测试现有三组位置，报告均值及位置间标准差；不通过则保留 v12。

完整训练时间包含数据准备、模型与优化器建立、全部训练轮次、验证及保存，分阶段记录；最终测试和资源 sidecar 写入不计入。显存报告 PyTorch 峰值 allocated/reserved GiB。因为用户的 BRITS 正在物理 GPU0 上运行，本轮使用空闲 GPU1（同型号 RTX5090）。资源记录保留真实设备 UUID；与原 GPU0 的耗时对照注明物理设备和并发主机任务差异。

输出根目录：`artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/e23_v28_global_stage1/`。

## 结果

### Stage 1

按全局尺度选择第99轮：隐藏修复 MSE **0.305815909**，MAE **0.372775650**；相对 v12 同尺度 MSE 0.368823644 降低17.0834%。对应窗口归一化 MSE 为107.088755782，只作为诊断。

同一100轮训练记录若按窗口指标选择，会选第26轮（窗口 MSE 106.324854247），该轮全局修复 MSE 为0.359006021。因此，本轮中两种评分确实选择不同轮次；这不能直接证明旧 v12 的其他轮次会得到怎样的全局评分。

选定第99轮中，小标准差位置仍占缺失坐标的0.672678%，贡献窗口平方误差的99.057963%，但对正式全局尺度平方误差只贡献**0.346875%**。修正消除了原评分中的逆方差放大，而这些坐标仍参加训练和评价。

Stage 1 实测全流程195.342589秒，峰值 allocated1.625506 GiB / reserved5.306641 GiB，物理GPU1 RTX5090。

### 代码验证

最初5个新回归用例在实现前观察到预期失败。修正后57个E23相关测试通过；完整测试在正确 PATH 下790通过、1跳过、8个已有警告。首次全量命令因 PATH 缺少 git，四个 S4M provenance 用例（wrong_commit、modified_protected_core、allows_only_declared_patch_paths、unrelated_dirty_path）出现 FileNotFoundError，配置环境后重跑通过。

独立审查没有Critical/Important问题，并只读成功加载旧v12 Stage1及旧v3 Stage1/Stage2。按审查建议增补两种指标排名相反的选模回归，6个新目标测试通过；另以进程内错误选模变体确认该回归能失败，生产文件未改变。

增补回归与三组评价标签支持后，最终完整测试为 **791通过、1跳过、8个已有警告**，日志 `/tmp/e23-v28-full-suite-final.log`。独立复核确认新标签与配置身份元数据不改变测试 mask、权重或评分过程。

### Stage 2 与预测结果

Stage 2 按固定验证 MAE 选择第85轮，MSE **0.7402293330687785**、MAE **0.5855096656989619**，均优于 v12 的0.7454477286392869 / 0.5882120694703256。重放全部2,688验证窗口、1,806,336预测坐标精确复现指标，H key 哈希前后一致。

依照预先设定的验证规则，v28通过门槛，再以同一 checkpoint 对冻结三组测试缺失位置评分。三组均值与位置间总体标准差如下：

| 模型 | 平均MSE | MSE标准差 | 平均MAE | MAE标准差 | 完整训练秒 | 峰值allocated GiB | 峰值reserved GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v12 原记录 | 0.427462825 | 0.004205235 | 0.431355525 | 0.000677964 | 347.362774 | 1.625697 | 5.306641 |
| v28 本轮 | 0.422192317 | 0.005189469 | 0.432326501 | 0.000657312 | 361.496901 | 1.625506 | 5.306641 |
| T1 用户手动重跑 | 0.402273434 | 0.007522139 | 0.425507725 | 0.002545625 | 643.480174 | 1.779408 | 2.699219 |

v28三组测试各自的MSE都低于v12，各自的MAE都高于v12；平均MSE降低 **1.232975%**，平均MAE上升 **0.225099%**。仍未超过T1的两项指标。修复目标与选模的逆方差放大已修正，但不能据此宣称预测两项指标全面提升，也不能将较低修复MSE视为必然提高预测的证据。

v28是本轮按预先验证规则选出的新候选；测试上，v28与v12分别在MSE和MAE占优。本轮未根据测试改选checkpoint或再搜索参数。

对Stage2选定权重的只读修复诊断：固定验证集全局尺度隐藏修复MSE **0.496757501**、MAE **0.484997685**，优于v12对应的0.518288958 / 0.493436010。它们仍比v28 Stage1的修复误差大，说明预测联合训练改变了数值修复表示；不能仅凭这一观察判断这种变化对预测是利是弊。另存 `stage2_repair_diagnostic.json`，H key前后相同。

资源分阶段：Stage1全流程195.342589秒，Stage2全流程166.154312秒，合计361.496901秒；训练阶段计时合计359.728716秒。资源sidecar与各自selection SHA绑定。v28使用物理GPU1，另两行使用物理GPU0；主机同时存在用户BRITS训练，时间差应在这一条件下解释。

完整汇总 CSV：`ETTh1_P4_E23_v12_v28_T1_comparison_2026-10-03.csv`；逐缺失位置 CSV：`ETTh1_P4_E23_v12_v28_T1_test_groups_2026-10-03.csv`。原v12和T1采用已保存结果，本轮只训练v28。

### 产物与重现入口

输出根目录下包含 Stage1/Stage2 的checkpoint、manifest、资源sidecar、训练日志、每轮修复验证历史、`stage1_scale_diagnostic.json`、`validation_replay.json`、`three-test-draws.json`、代码冻结哈希、对照JSON与最终`status.json`。

训练入口仍是 `python -m integrations.e23.run train-repair` 和 `train-forecast`，两步均指定新配置 `configs/e23_multi_expert_p4_width64_global_stage1_v28.yaml`；预测阶段使用 `--selection-metric mae --forecast-loss mse_plus_0p2_mae`。三组评价使用 `scripts/evaluate_e23_three_test_draws.py`，显式传 `--model-label E23_v28`。旧配置默认原训练目标，以保持旧记录可重现；后续采用新修正的实验应从v28配置派生。
