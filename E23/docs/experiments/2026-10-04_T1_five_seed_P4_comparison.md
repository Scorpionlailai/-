# v31-B / T1：五次独立训练与二十组测试缺失位置

## 状态

五次独立训练对照已全部完成，2026-10-04 12:38（北京时间）通过最终完整性核验。共 10 次完整训练、200 组测试结果；五个种子下原预测头初始化均匹配，代码/配置/数据/掩码 SHA 未变，旧模型与 E22 表格保留。

正式均值：v31-B MSE **0.414918**、MAE **0.430711**；T1 MSE **0.396421**、MAE **0.421283**。五个种子的配对指标均为 T1 更低，MSE/MAE 差异的配对交叉 bootstrap 95% 区间均大于零。在此 ETTh1 P4 协议下，不能继续主张 v31-B 准确性超过 T1。完整训练时间均值为 404.64 秒与 751.83 秒；显存 allocated/reserved 分开列出。

[正式汇总与差异区间](ETTh1_P4_E23_v31B_T1_five_seed_twenty_masks_20261004.md) · [汇总 CSV](ETTh1_P4_E23_v31B_T1_five_seed_twenty_masks_20261004.csv) · [逐种子 CSV](ETTh1_P4_E23_v31B_T1_five_seed_twenty_masks_20261004_per_seed.csv)

2026-10-04 10:55（北京时间）已进入正式训练，顺序为每个种子的 v31-B、T1；种子依次为 102、202、302、402、502。10:47 的初次协调器启动在路径校验处退出，未进入训练；已修正并保存失败日志。

本文件记录实验方案。最终结果以产物目录中的 `comparison_statistics.json` 的 `status=complete` 为准，不能把部分运行的均值当成最终五次结果。

产物目录：

`/home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/e23_t1_five_seed_standard_20261004`

隔离代码目录：`/home/xyw/paper/.e23-t1-standard/E23`。

## T1 原论文怎么做

[T1 附录 A.2.2](https://arxiv.org/html/2602.21043#A2.SS2) 明确规定五个随机种子 102、202、302、402、502，汇报均值和标准差。其完整数据实验在 40% 点缺失下训练同一个模型，再测试 10%、30%、50%、70% 点缺失，以及 5% 点缺失叠加 0.15% 起始概率、长度 24–96 的块缺失。

本轮先对齐独立训练次数和种子，在已有 P4 场景中比较。原论文的训练和测试缺失分布列为后续补充实验；当前五种子 P4 训练不宣称已完成该补充。预测任务继续使用 96 点回溯窗口、96 点预测长度。

官方发布脚本存在需要明确记录的差异：固定版本 `823b76aab57a2e60eb5029caab06fdfec8b19256` 的 `run.py` 固定种子 2021、默认 `itr=1`；`T1_ETTh1.sh` 按各个缺失率分别启动训练。统计重复采用论文明确公布的五个种子，模型与优化器采用已冻结的官方 ETTh1 参数。[官方脚本](https://github.com/Oppenheimerdinger/T1/blob/823b76aab57a2e60eb5029caab06fdfec8b19256/scripts/imputation/T1_ETTh1.sh)

## 本轮正式协议

| 项目 | 设置 |
|---|---|
| 数据与任务 | ETTh1，原训练/验证/测试切分，96→96，训练集统计量标准化 |
| 模型 | 已选定 E23-v31-B，官方 ETTh1 参数的 T1 |
| 独立训练 | 每个模型从头完整训练 5 次，包括两个阶段；无 checkpoint 复用 |
| 训练种子 | 102、202、302、402、502；控制初始化、样本顺序和训练掩码 |
| 缺失场景 | P4：块起始概率 0.10、块长度 8、允许重叠；0.10 不是点缺失比例 |
| 验证 | 原固定验证掩码；Stage 2 只按验证 MAE 选 checkpoint |
| Stage 1 选模 | 两者按缺失坐标修复 MSE；E23 使用修正后的全局标准化尺度 |
| 测试掩码 | 20 组新位置，种子 2026100401–2026100420；两模型、五次运行共享 |
| 测试范围 | 固定原先的 2688 个窗口，每组 1,806,336 个未来坐标 |
| 预测头 | 原 PatchTST D16，35,427 参数；同种子两模型初始状态 SHA 相同 |
| E23 训练 | 原 v31-B：两个阶段各 100 轮，修复宽度 64、单轮 T→V→H |
| E23 Stage 2 损失 | 预测 MSE + 0.2 MAE，第 11 轮起加 0.05 均匀修复 MSE；原方案保持 |
| T1 修复训练 | 官方 ETTh1：Adam，lr=0.001，batch=16，BF16，最多 300 轮，patience=30，恒定学习率 |
| T1 预测训练 | 原共享 PatchTST 接口，100 轮，batch=128，FP32，lr=0.0001，预测 MSE |
| 设备 | 同一张 RTX 5090：GPU 1，顺序训练；保留 GPU UUID 与实际阶段配置 |

T1 论文的主任务是插补，本表的主指标是未来预测 MSE/MAE。论文原表的插补数字不能直接作为这项预测实验的指标。两者使用相同预测头；E23 保留其已选定的训练损失，因此本表不标为“训练目标完全相同”。

## 如何统计

### 最优版本、最优权重与训练种子

本轮使用已选定的最佳模型版本 v31-B。原来的 seed=2021 最优权重完整保留，正式重复实验不会覆盖它。

正式表的训练种子在实验前按 T1 论文确定；每次运行可以按验证集选择最佳 checkpoint，但不按测试结果选择最有利的种子。部署时可以按事先确定的验证规则选择一个模型；论文的五次均值与标准差仍需全部汇报。如果另外报告最佳单次，两种方法必须使用同样的验证选择规则，另列为单次结果。

### 统计口径

每个模型形成 `5 个训练种子 × 20 个测试掩码` 的结果矩阵。

1. **主均值**：每个 checkpoint 的 20 组测试指标先求均值，再对 5 次训练求均值。
2. **训练标准差**：5 个 checkpoint 的掩码均值之间的样本标准差，`ddof=1`。
3. **缺失位置标准差**：每个掩码在 5 次训练上的均值之间的样本标准差，`ddof=1`。
4. **配对差异**：相同训练种子、相同测试掩码的 `E23−T1`。负数代表 E23 更好。
5. **差异区间**：训练种子和掩码两个轴分别重采样，保留模型配对，10,000 次，固定 bootstrap 种子 20261004，输出 95% 区间。100 个单元不按独立样本处理。[交叉数据 bootstrap 方法](https://arxiv.org/abs/0712.1111)
6. **资源**：每次完整训练的入口耗时，包括导入、数据准备、所有训练阶段、验证、checkpoint 保存；排除最终测试。汇报时间均值/标准差、各阶段时间、峰值 allocated/reserved GiB 的均值和最大值；附批大小、精度与设备信息。

同时保存每个掩码下各测试窗口的误差与 `target_start`，以便后续分析时识别重叠窗口，避免把滑窗坐标误当成独立数据。

五次训练的区间估计仍有限；20 组掩码检验的是这个固定 ETTh1 测试时间段对缺失位置变化的稳定性。它不会增加独立测试时间段，也不能单凭这张表证明所有数据集、所有缺失场景上最优。跨数据集和 T1 点/长块缺失分布仍需另行评估。

## 冻结与检查

- `trial-definition.json` 与 SHA 封条：运行次数、种子、配置、代码、掩码与统计口径均在训练前冻结。
- `frozen-source.tar.gz`：完整代码快照；`baseline-source.tar.gz` 保存本次修改前的基础。
- `implementation-audit.json`：模型、预测头、损失实现均与选定版本一致；只修改现有入口的训练种子处理，新增测试与报告流程。
- `preflight-validated.json`：CUDA 前向/反向有限、模型不读缺失真值或未来标签、原预测头初始化匹配。早期 `preflight.json` 记录的是前向后 BN buffer 哈希，初始化核查以 validated 文件为准。
- 61 项相关测试通过，包括训练批次配对、验证掩码不变、掩码文件篡改拒绝、五个种子下的原预测头初始化匹配、冻结方案读取与资源报告，以及避免重复掩码虚假缩窄训练不确定性。
- 每次测试前重放固定验证分数、锁定 checkpoint 和父阶段 SHA；测试期间 H 库与 key 只读；最终校验相同物理 GPU。

## 输出文件

- `comparison_summary.csv`：正式模型汇总；至少两组配对训练完成后产生，最终须核对 `independent_training_runs=5`。
- `per_training_seed.csv`：每次训练的 MSE、MAE、资源、阶段轮数及 checkpoint 路径。
- `per_training_seed_and_mask.csv`：完整 200 行测试明细。
- `comparison_statistics.json`：完成状态、分离的标准差、配对差异和区间。
- `seed_<seed>/<model>/evaluation/twenty-test-draws.json`：单次训练的 20 组结果、验证重放和训练资源引用。
- `logs/` 与 `coordinator.log`：逐阶段日志。

运行中的主协调器 PID 保存在 `coordinator-pid.json`。只在原协调器已停止时续跑：

```bash
cd /home/xyw/paper/.e23-t1-standard/E23
export PATH=/home/xyw/miniforge3/envs/malr/bin:$PATH
export PYTHONPATH=src:.
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
python scripts/run_e23_t1_five_seed.py \
  --device cuda:1 \
  --root /home/xyw/paper/E23/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/e23_t1_five_seed_standard_20261004
```

完整阶段可续用，部分阶段会拒绝覆盖，需要检查失败原因。所有旧版三组测试记录保持原样。
