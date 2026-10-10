# Weather 实验台账

## 目标与目录规则

在 Weather 上从头训练已定版的 E22 架构，检验其跨数据集适用性；不迁移 ETTh1 权重，不改 H/V/T 实现。E22 的 ETTh1 最优记录仍由 `artifacts/ETTh1/seq96_pred96/sr010_bl8/CURRENT_BEST.json` 标记，架构说明见 `docs/superpowers/specs/2026-09-16-e22-current-best-architecture-handoff.md`。

Weather 结果按 `artifacts/Weather/seq{seq}_pred{pred}/` 组织：`clean/<fingerprint>/` 为可复用 Clean 缓存；`sr{rate}_bl{block}/missing/<fingerprint>/` 为对应缺失条件的 Missing 缓存；`sr{rate}_bl{block}/experiments/<实验编号>/` 保存固定 mask、两阶段 checkpoint、运行清单、逐 epoch 日志和测试结果。失败尝试也保留编号，不复用或覆盖已产出的 checkpoint。

首个协议为 seq96→pred96、`variable_missing`、`start_rate=0.10`、`block_len=8`、seed 2021。Weather 文件为 `data/raw/weather.csv`，SHA-256 `34ee981d07313e51da2a50bb600072c8ae4a69cb4b0651f4cb93a069d7a2ba63`，52,696 行、21 个变量；固定切分为 train `[0,36887)`、validation `[36887,42157)`、test `[42157,52696)`。沿用当前加载器对 `-9999` 的保留规则，指标空间为训练集标准化后的数值。

## W01：E22 原始 Weather profile，显存预检未通过

- 配置：`configs/stage1_seq96_dynamic_temporal_repair_weather.yaml`；文件 SHA-256 `4ad4e2d2235c28f4a88a03ab5794512b4e633f926c913c4efcff57b01ab3073c`；规范化配置哈希 `38e09cec43903f42a3216a27a801bcc2f76794837ca5d0cfc4da3c7a1a057b3f`。
- 唯一的 ETTh1→Weather 模型规格变动：`n_vars=21`，Weather PatchTST `d_model=128`、`n_heads=16`、`d_ff=256`、3 层、dropout 0.2。W01 训练 batch=128、上限 100 epoch；E22 的 H/V/T、损失、路由和修复逻辑未改。
- 固定 mask 根目录：`artifacts/Weather/seq96_pred96/sr010_bl8/experiments/w01_e22_dynamic_temporal/masks/`。验证 mask 内容校验值 `df53315df5988b8d2c2848ccf5f5c2e609eb7a423a500513ce9ad19d53f43eb6`、文件 SHA-256 `49aa35e9bc9e8cc71424a92792d92ff5b2bb2066bb46c28efbcd5f8264a6ad0c`，实际缺失率 56.9325%；测试 mask 内容校验值 `8de2c4fcb9a13918a3bdec9cbc3a5ba544deebfde32d03fa9033dd7b61661ee5`、文件 SHA-256 `8656da3ad5bca822ad4306c56273282ad1396af5144b8c9c14b44ae8c8584a4e`，实际缺失率 56.7976%。CLI 的 `checksum` 是 mask 内容和元数据的校验值，不是 `.npz` 文件的 SHA-256；正式运行记录使用后者。
- 数据读取、模型创建，以及真实 Weather batch=2 的 Stage 1 前向/反向计算通过。正式 Stage 1 在 epoch 1 的 V 候选投影处 CUDA OOM；`torch.cat` 需要再分配约 15.08 GiB，当时卡上只有 14.08 GiB 空闲。完整栈见 `runs/Weather_ours_stage1.console.log`。
- 同时启动的 Clean 训练为避免与后续缩小 batch 的配置不一致，在约第 7 个 epoch 手动中断。`checkpoints/Weather_clean.pt` 是不完整训练的临时产物，**不得作为 W01 正式 Clean 基线、不得评估或复用**。W01 没有完整 Stage 1/2、没有可用 Clean/Missing 缓存，也没有正式测试结果。

### 显存探针与后续决策

使用相同的 21 变量、96 步输入及 E22 Stage 1 前向/反向，batch=32 的 PyTorch 峰值 reserved 约 27.53 GiB；batch=16 为 13.88 GiB。Stage 2 带冻结教师与预测头的 batch=16 探针 reserved 约 7.84 GiB。单张 RTX 5090 约 31.36 GiB；batch=32 的 Stage 1 余量不足以稳妥覆盖完整训练。因此 W02 采用 batch=16；这只是单 batch 探针及正式训练的初期进展，不等于整次实验已完成。

## W02：E22 Weather batch=16，Stage 1 中断后从最佳权重运行 Stage 2

- 配置：`configs/stage1_seq96_dynamic_temporal_repair_weather_bs16.yaml`；文件 SHA-256 `eb1c1f16dfed72aae8e41183a9f544d3521da66621e29efcb44b1b5a780f8fa7`；规范化配置哈希 `a1452ca1defb171a6fe04bae64355b9a7c4be0cd8cbe99c307d1fc3af706b557`。自动化测试确认其解析结果与 W01 仅差 Weather 训练 batch 128→16；Clean、Missing、Stage 1、Stage 2 都使用该同一配置。
- 实验根目录：`artifacts/Weather/seq96_pred96/sr010_bl8/experiments/w02_e22_dynamic_temporal_bs16/`。验证、测试固定 mask 的内容校验值、文件 SHA-256 和实际缺失率都与 W01 完全一致。
- 2026-09-20 Stage 1 在 GPU 0 完成 epoch 30、运行到 epoch 31 batch 2050 时意外退出；日志中没有 Python 异常，原因尚未确证，也没有 Stage 1 run manifest。现有 `checkpoints/Weather_ours_stage1.pt` 是有效的最佳 epoch 28 权重：修复验证 MSE `0.22771730711301025`、MAE `0.11444932379809869`，SHA-256 `c5dcf2c9e67a35bf509a006c4e17754bd1a93e6d487cdce780405b5850f3294b`。它**不是完成 100 epoch 的 Stage 1**。
- W02 Clean 在 GPU 1 完成 100 epoch，run ID `Weather-clean-8c15d60d1a35`；最佳 epoch 41 的验证 MSE `0.42273233058593324`、MAE `0.2822331598604204`。2026-09-21 已正式测试并缓存。逐 epoch 日志分别位于 `runs/Weather_ours_stage1.console.log` 和 `runs/Weather_clean.console.log`。
- W02 Clean 缓存 fingerprint 为 `b48fcaf26e2f7f592f24584c5576f3128eb3fd79c86b534e6a554828c076aefb`；Missing 缓存 fingerprint 为 `a8b87d449653be87b92ec174fca271df8f60ca7e2a354aef74313fd4abca426e`。两者已完成训练与正式评估，`baseline.json` 和可复用 checkpoint 均已生成；W01 的中断 checkpoint 不可替代它们。
- 依当前实验决策，Stage 2 于 2026-09-20 23:14 CST 从上述 Stage 1 最佳 epoch 28 checkpoint 单独启动，沿用 E22 的 `--stage2-route-teacher frozen_stage1_hv_fusion`、batch=16。已完成独立的 100 epoch，最佳 epoch 66：验证 MSE `0.4131230730243945`、MAE `0.2887709428182755`；run ID `Weather-ours-6df17cd6b981`，checkpoint SHA-256 `19fc71573c69017cfe6f07c96f48ac9678f41949c28df281368100ce0695aa28`。冻结路由教师的 checksum 和不变性校验通过。此结果必须标注为“Stage 1 最佳 epoch 28 初始化”，不能表述为完整 100 epoch 预训练后的结果。

### W02 正式测试结果（2026-09-21）

三者使用同一规范化配置哈希 `a1452ca1defb171a6fe04bae64355b9a7c4be0cd8cbe99c307d1fc3af706b557` 和训练集标准化指标空间。Missing/Ours 的测试 mask SHA-256 均为 `8656da3ad5bca822ad4306c56273282ad1396af5144b8c9c14b44ae8c8584a4e`，实际缺失率 `56.79763583853852%`；Clean 输入无缺失。所有 checkpoint 文件 SHA-256 与运行记录一致。

| 方法 | Run ID | 测试 MSE | 测试 MAE | DRR-MSE | DRR-MAE |
| --- | --- | ---: | ---: | ---: | ---: |
| Clean | `Weather-clean-8c15d60d1a35` | 0.172345 | 0.214705 | — | — |
| Missing | `Weather-missing-2e7aecf7f4dd` | 0.217515 | 0.254445 | 0% | 0% |
| Ours | `Weather-ours-6df17cd6b981` | 0.173257 | 0.220932 | 97.98% | 84.33% |

Missing 完成 100 epoch，最佳 epoch 24：验证 MSE `0.538950702530013`、MAE `0.36613076158141417`；checkpoint SHA-256 `384678f7cb808cd9e230a16f15379e906cf5cb94c39a880d309ccaa82f97f490`。Ours 相对 Missing 的测试 MSE 降低 `20.35%`、MAE 降低 `13.17%`；相对 Clean 仍有 MSE `0.53%`、MAE `2.90%` 的退化。DRR 使用 `(Missing - Ours) / (Missing - Clean) × 100%` 计算。

Ours 的测试推理消融：去掉 T 时 MSE/MAE 为 `0.186543/0.233766`，去掉 V 为 `0.196405/0.238016`，去掉 H 为 `0.172593/0.218842`。因此本次 Weather 测试中 T、V 对完整模型有正向贡献，H 的边际贡献为负；这是单次测试的推理消融，不应外推为所有数据集上的模块结论。原始记录位于本实验的 `runs/`、`evaluations/`，Clean/Missing 缓存分别位于 `artifacts/Weather/seq96_pred96/clean/` 和 `artifacts/Weather/seq96_pred96/sr010_bl8/missing/`。

在仓库 checkout 根目录查看完整训练与测试日志：

```bash
tail -n 50 artifacts/Weather/seq96_pred96/sr010_bl8/experiments/w02_e22_dynamic_temporal_bs16/runs/Weather_missing_workflow.console.log
```

## 后续结果填写约定

每个正式实验记录：配置文件和哈希、源数据和固定 mask 哈希、Clean/Missing/Stage 1/Stage 2 run ID 与 checkpoint 哈希、验证最优 epoch、测试 MSE/MAE、相对改善和 DRR-MSE/DRR-MAE、H/T/V 消融。测试前不从验证集数值推断测试集收益；只有同配置、同 mask 的 Clean/Missing/Ours 才进入同一比较表。
