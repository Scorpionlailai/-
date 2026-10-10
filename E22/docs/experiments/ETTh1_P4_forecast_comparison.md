# ETTh1 P4 预测对照实验运行记录

本实验只比较未来 96 点预测。固定 ETTh1、历史 96 点、缺失起始率 `0.10`、块长 `8`、随机种子 `2021`。验证和测试均复用 E22 已保存的固定掩码，测试各包含 2,688 个有效窗口。MSE、MAE 在训练段标准化空间计算；共同参照 DRR 的分母固定为已缓存 PatchTST Missing 减 Clean。当前是单种子探索性对照，不是独立确认实验。

比较组：缓存 PatchTST Clean/Missing、归档 E22 `ETTh1-ours-dea387986927`、官方 SAITS 接同结构 mask-aware 预测器、官方 S4M。SAITS Stage 1 以 P4 块掩码进行 ORT+MIT（各权重 1）插补预训练，Stage 2 联合微调插补器和预测器但只用未来预测 MSE。S4M 按官方 profile 搜索 `e_layers=2,4,8`；其 Clean 只作独立诊断，不参与共同 DRR。

锁定输入：CSV SHA-256 `f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066`；验证掩码 `cbe7c520afded9002f714c5ed8e67d62c92b97fa7aa9b1e6e4711db785fd846c`；测试掩码 `f7974ff9b4228d69fe33d940e3722cd1a069766e2fa1387c6240757d5a8762d8`。SAITS 上游提交 `660b87f19c1277065f314f24134f646229e89ca9`；S4M 上游提交 `a718823addd3606e763dfc261174e0135b2535f4`。S4M 本地 checkout 的受保护文件有 CRLF 行尾差异，预检按行尾归一化后的 Git blob 哈希验证，语义改动仍拒绝。

## 已启动与环境

SAITS 隔离环境是 `/home/xyw/miniforge3/envs/malr-saits`，S4M 隔离环境是 `/home/xyw/miniforge3/envs/malr-s4m`；未修改 `malr` 的 PyTorch/CUDA。最初的 SAITS v1 已完成但缺少不可变配置身份，现保留作归档，不参与最终对照。配置身份修复后，SAITS Stage 1/2 v2 和选定 checkpoint 的测试均已完成；S4M 实时输出搜索 v2 仍在后台。`finish_after_s4m_20260929` 后台接续任务已启动：只有搜索退出码为 0 且两个验证选型文件均存在，才会依次启动 S4M Missing 测试、S4M Clean 测试和最终报告；每步使用独立的 job 目录、日志和退出码，失败时停止接续。S4M v1 的部分 checkpoint 和失败/中断日志也保留，不参与选型。S4M 虽传入 `CUDA_VISIBLE_DEVICES=1`，物理卡映射以 `nvidia-smi` 为准，不假设它一定是显示的 GPU 1。监控命令：

```bash
cd /home/xyw/missing_aware_latent_recovery/.superpowers/sdd/2026-08-31-residual-mlp-recovery/checkout
tail -f artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/forecast_comparison_v1/jobs/saits_stage1_v2_20260929/console.log
tail -f artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/forecast_comparison_v1/jobs/saits_stage2_v2_20260929/console.log
tail -f artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/forecast_comparison_v1/jobs/saits_test_v2_20260929/console.log
tail -f artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/forecast_comparison_v1/jobs/s4m_search_unbuffered_20260929/console.log
tail -f artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/forecast_comparison_v1/jobs/finish_after_s4m_20260929/console.log
```

同一目录下 `status.json` 给出子进程 PID 和退出码。若显示 `running`，仍应以 `ps -p <child_pid> -o pid,etime,stat,cmd` 核实进程。不要再次启动同名 Stage 1。

SAITS v2 Stage 1 的最佳验证插补 MAE 为 `0.5978319887024138`（第 79 轮），Stage 2 的最佳验证预测 MSE 为 `0.7909441513269126`（第 18 轮）。选定 checkpoint 在固定测试集上的预测 MSE/MAE 为 `0.4405843816917229/0.4369578956206099`，相对共同 PatchTST Clean/Missing 参照的 DRR-MSE/DRR-MAE 为 `61.74554013960881%/49.44524840896372%`。这是初步结果；S4M 测试及完整对照报告尚未完成。

## 后续后台命令

以下命令仅是各步骤的复现模板；当前 SAITS Stage 1/2/测试 v2 已完成，S4M 搜索 v2 与自动接续任务已启动，**不要重复启动它们或手动抢先运行下方的 S4M 测试/报告命令**。接续任务会在上一任务成功并产生验证选型文件后，依次启动余下步骤。若要独立复现，所有 job 目录必须是新目录，若已有同名目录则改用新的后缀。任务结束可查看 `status.json`；实时进度用 `tail -f <job目录>/console.log`。Stage 1、Stage 2 和测试每个 batch 或 epoch 都会打印进度。

```bash
cd /home/xyw/missing_aware_latent_recovery/.superpowers/sdd/2026-08-31-residual-mlp-recovery/checkout
P4_ROOT="$PWD"
P4_BASE="$P4_ROOT/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/forecast_comparison_v1"
P4_MASKS="$P4_ROOT/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context"
P4_CONFIG="$P4_ROOT/configs/stage1_seq96_dynamic_temporal_repair.yaml"
P4_SAITS=/home/xyw/missing_aware_latent_recovery/external/SAITS
P4_S4M=/home/xyw/missing_aware_latent_recovery/external/S4M
```

SAITS Stage 2（必须等 Stage 1 `status.json` 为 `finished` 且 `exit_code=0`）：

```bash
/home/xyw/miniforge3/envs/malr/bin/python scripts/run_background.py \
  --job-dir "$P4_BASE/jobs/saits_stage2_v2_20260929" -- \
  /usr/bin/env PYTHONPATH="$P4_ROOT:$P4_ROOT/src:$P4_SAITS" \
  /home/xyw/miniforge3/envs/malr-saits/bin/python -u -m integrations.forecast_compare.saits_run train-forecast \
  --config "$P4_CONFIG" --mask-artifacts "$P4_MASKS" \
  --stage1-manifest "$P4_BASE/saits/stage1_p4_v2/runs/saits-stage1.json" \
  --output "$P4_BASE/saits/stage2_p4_v2" --saits-root "$P4_SAITS" --device cuda
tail -f "$P4_BASE/jobs/saits_stage2_v2_20260929/console.log"
```

SAITS 选定 checkpoint 测试（必须等 Stage 2 完成）：

```bash
/home/xyw/miniforge3/envs/malr/bin/python scripts/run_background.py \
  --job-dir "$P4_BASE/jobs/saits_test_v2_20260929" -- \
  /usr/bin/env PYTHONPATH="$P4_ROOT:$P4_ROOT/src:$P4_SAITS" \
  /home/xyw/miniforge3/envs/malr-saits/bin/python -u -m integrations.forecast_compare.saits_run evaluate \
  --config "$P4_CONFIG" --mask-artifacts "$P4_MASKS" \
  --selection "$P4_BASE/saits/stage2_p4_v2/runs/saits-stage2.json" \
  --output "$P4_BASE/saits/evaluations/saits-test-v2.json" --device cuda
tail -f "$P4_BASE/jobs/saits_test_v2_20260929/console.log"
```

S4M 冒烟通过后，再启动官方 profile 的 100 epoch 搜索。桥接按验证 MSE 从三个缺失候选中选一个，并训练单独的 S4M Clean 参考；训练掩码为按 epoch 生成的整段时间线掩码，而 E22/SAITS 为按样本生成，固定验证和测试完全一致。

```bash
/home/xyw/miniforge3/envs/malr/bin/python scripts/run_background.py \
  --job-dir "$P4_BASE/jobs/s4m_search_unbuffered_20260929" -- \
  /usr/bin/env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 PYTHONPATH="$P4_ROOT:$P4_ROOT/src" \
  /home/xyw/miniforge3/envs/malr-s4m/bin/python -u -m integrations.forecast_compare.s4m_p4 \
  --config "$P4_CONFIG" --mask-artifacts "$P4_MASKS" --s4m-root "$P4_S4M" \
  --output "$P4_BASE/s4m/official_p4_v2" \
  --python /home/xyw/miniforge3/envs/malr-s4m/bin/python --phase search
tail -f "$P4_BASE/jobs/s4m_search_unbuffered_20260929/console.log"
```

等 `selection-official.json` 和 `selection-official-clean.json` 均存在、搜索任务退出码为 0 后，分别运行选定模型的测试，不能用测试误差重新选型：

```bash
/home/xyw/miniforge3/envs/malr/bin/python scripts/run_background.py \
  --job-dir "$P4_BASE/jobs/s4m_missing_test_20260929" -- \
  /usr/bin/env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 PYTHONPATH="$P4_ROOT:$P4_ROOT/src" \
  /home/xyw/miniforge3/envs/malr-s4m/bin/python -u -m integrations.forecast_compare.s4m_p4 \
  --config "$P4_CONFIG" --mask-artifacts "$P4_MASKS" --s4m-root "$P4_S4M" \
  --output "$P4_BASE/s4m/official_p4_v2" \
  --python /home/xyw/miniforge3/envs/malr-s4m/bin/python --phase evaluate-missing
tail -f "$P4_BASE/jobs/s4m_missing_test_20260929/console.log"
```

```bash
/home/xyw/miniforge3/envs/malr/bin/python scripts/run_background.py \
  --job-dir "$P4_BASE/jobs/s4m_clean_test_20260929" -- \
  /usr/bin/env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 PYTHONPATH="$P4_ROOT:$P4_ROOT/src" \
  /home/xyw/miniforge3/envs/malr-s4m/bin/python -u -m integrations.forecast_compare.s4m_p4 \
  --config "$P4_CONFIG" --mask-artifacts "$P4_MASKS" --s4m-root "$P4_S4M" \
  --output "$P4_BASE/s4m/official_p4_v2" \
  --python /home/xyw/miniforge3/envs/malr-s4m/bin/python --phase evaluate-clean
tail -f "$P4_BASE/jobs/s4m_clean_test_20260929/console.log"
```

全部测试完成后再汇总。命令读取缓存 Clean/Missing 和归档 E22，不会重跑它们；输入身份或选定 checkpoint 对不上会报错：

```bash
/home/xyw/miniforge3/envs/malr/bin/python scripts/run_background.py \
  --job-dir "$P4_BASE/jobs/report_20260929" -- \
  /usr/bin/env PYTHONPATH="$P4_ROOT:$P4_ROOT/src" \
  /home/xyw/miniforge3/envs/malr/bin/python -u -m integrations.forecast_compare.report \
  --clean-cache "$P4_ROOT/artifacts/ETTh1/seq96_pred96/clean/b5932e46c76ee3e89ccbc51bec219b521bb1c8b4ed085d063a8c1b89102148fe/baseline.json" \
  --missing-cache "$P4_ROOT/artifacts/ETTh1/seq96_pred96/sr010_bl8/missing/292570de0560129a4bd96568f13fc19875e06e80fd7041a7e7f8d06a89710f10/baseline.json" \
  --e22-evaluation "$P4_MASKS/evaluations/ETTh1-ours-dea387986927.json" \
  --saits-evaluation "$P4_BASE/saits/evaluations/saits-test-v2.json" \
  --s4m-output "$P4_BASE/s4m/official_p4_v2" \
  --output "$P4_BASE/reports/comparison_p4_v2"
tail -f "$P4_BASE/jobs/report_20260929/console.log"
```

汇总产物为 `comparison.json`、`comparison.csv`、`comparison.md`。主表同时列出 MSE、MAE、DRR-MSE、DRR-MAE；S4M DRR 仅是共同 PatchTST 参照归一化分数，不是其自身恢复退化率。
