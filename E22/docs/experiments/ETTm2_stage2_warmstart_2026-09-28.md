# ETTm2 Stage 2：从最佳第 22 轮追加 78 轮

## 实验口径

原实验：`artifacts/ETTm2/seq96_pred96/sr010_bl8/experiments/e22_transfer_v1`。
原 Stage 2 完成 54 轮，在第 55 轮中断；只保存了最佳模型，没有最新权重、优化器或 RNG 状态。
日志确认最佳模型来自 Stage 2 第 **22** 轮，验证 MSE `0.13468428507024685`，MAE `0.2515647481507066`。
该 checkpoint 中的 `epoch=85` 是冻结的 Stage 1 历史库版本，**不是 Stage 2 的训练轮数**。

本次从该最佳模型追加 78 轮，日志显示第 **23–100** 轮；优化器重新创建。
这是 **22 轮权重基础上的 warm-start + 78 轮**，不等同于不间断训练 100 轮，也不是恢复第 55 轮现场。
旧实验的第 23–54 轮计算已经发生，但其最后权重未保存，不在新权重的训练链路中。

新实验目录：

`artifacts/ETTm2/seq96_pred96/sr010_bl8/experiments/e22_transfer_v1_stage2_warmstart_e22_plus78`

保持原配置、batch、浮点设置、缺失参数、训练目标及 H/V/T 结构不变。
`--init-checkpoint` 仍指向原 Stage 1，负责重建模型结构和冻结 teacher；随后完整加载原 Stage 2 的学生、预测器、融合及 mask projection 权重。
加载时校验配置/训练数据/验证 mask、Stage 1 来源、teacher checksum 和冻结 memory/EMA/变量关系统计。
历史库现有加载器会重新归一化 `clean_key`：实际 Stage 1/2 的 1,178 个检索向量中，329 个存在舍入差异，最大绝对差 `5.551115123125783e-17`，其余历史库字段逐项完全一致。
续训校验只对 `clean_key` 使用 `rtol=0, atol=1e-14`，要求结构、样本标识、原始样本值、EMA、变量统计和 teacher 精确一致；训练结束恢复最佳模型后再次检查。没有修改 H 的编码器或历史库实现。
训练掩码与 shuffle 的 epoch 索引从 22 开始，保持第 23 轮的原定索引；随机数生成器重新设种，不恢复原中断位置。

续训前先重新验证原最佳权重，并把它保存为新实验的初始最佳候选，只有验证 MSE 更低才替换。
新 checkpoint 的 `training_epoch` 单独记录 Stage 2 轮数；`epoch` 继续保留在线历史库版本。
manifest 的 `stage2_warm_start` 保存来源 SHA256、新增轮数和优化器/RNG 未恢复标记。
clean、missing 和测试 mask 复用；本次不重新训练或评估基线。

## 后台运行规范

工作目录：`/home/xyw/missing_aware_latent_recovery/.superpowers/sdd/2026-08-31-residual-mlp-recovery/checkout`。

`scripts/run_background.py` 使用独立会话、脱离终端的标准输入输出、忽略 SIGHUP 的 supervisor。
启动命令立即返回，关闭 VS Code / SSH 不会因终端挂断而停止任务。
不需要再加 `&` 或 `tee`。

通用用法（`--job-dir` 每次换一个未存在的目录）：

```bash
PYTHONPATH=.:src /home/xyw/miniforge3/envs/malr/bin/python scripts/run_background.py \
  --job-dir artifacts/jobs/唯一任务名 -- \
  /home/xyw/miniforge3/envs/malr/bin/python -u -m malr.cli <子命令及参数>
```

每个 job 目录记录：

- `launch.json`：命令、工作目录、创建时间。
- `pid`：supervisor PID，**不是训练子进程 PID**。
- `status.json`：运行状态、supervisor/child PID、结束时间和退出码。
- `console.log`：程序原始输出，包含每轮和 batch 进度。

查看日志用 `tail -f JOB目录/console.log`，退出查看按 Ctrl+C **不会停止训练**。
`status.json` 中 `state=finished` 且 `exit_code=0` 表示正常完成；非零为错误或被信号终止。
如需停训练，先核验 `child_pid` 对应的命令，再对训练子进程发 SIGTERM，让 supervisor 记录结束状态；不要只杀 supervisor。
服务器重启、OOM 或管理员终止不在终端脱离的保障范围；出现异常要结合进程实际存活情况检查，不能只相信遗留的 `running` 状态。

## 启动参数

在新实验目录准备原验证和测试 mask 的逐字节副本后，使用：

```bash
OLD=artifacts/ETTm2/seq96_pred96/sr010_bl8/experiments/e22_transfer_v1
NEW=artifacts/ETTm2/seq96_pred96/sr010_bl8/experiments/e22_transfer_v1_stage2_warmstart_e22_plus78
PYTHONPATH=.:src /home/xyw/miniforge3/envs/malr/bin/python scripts/run_background.py \
  --job-dir "$NEW/jobs/stage2_plus78" -- \
  /home/xyw/miniforge3/envs/malr/bin/python -u -m malr.cli train \
  --config configs/datasets/ettm2/sr010_bl8.yaml --dataset ETTm2 \
  --artifacts "$NEW" --variant ours --stage forecast \
  --init-checkpoint "$OLD/checkpoints/ETTm2_ours_stage1.pt" \
  --stage2-route-teacher frozen_stage1_hv_fusion \
  --warm-start-checkpoint "$OLD/checkpoints/ETTm2_ours_stage2.pt" \
  --warm-start-epoch 22 --additional-epochs 78 --device cuda
```

该命令只应启动一次。目录已存在时启动器会拒绝，防止误覆盖日志；先检查已有任务是否仍在运行。
