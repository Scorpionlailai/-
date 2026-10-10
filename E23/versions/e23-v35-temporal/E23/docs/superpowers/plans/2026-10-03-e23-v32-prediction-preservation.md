# E23 v32 Prediction Preservation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** 实现Stage1信息保持并完成A/B两阶段纯MSE预测对照和三组测试。

**Architecture:** 新增独立训练辅助预测器，前20轮只在完整训练历史上学习，第21轮起冻结并将预测一致性梯度传给修复器。正式预测器从头训练，辅助模型独立归档。

**Tech Stack:** Python、PyTorch、pytest、现有measured entrypoint与三组测试工具。

**Spec:** `docs/superpowers/specs/2026-10-03-e23-v32-prediction-preservation-design.md`

## Global Constraints

遵循spec全部值：Stage1/2各100轮，batch128/FP32/seed2021、lr0.001/0.0001、单轮修复D64、预测D16；辅助seed2053、20轮、系数0.1。Stage1全局修复MSE选模，Stage2纯MSE训练/验证MAE选模；既有三组test-only masks。正式源冻结，保留dirty基线与用户CSV，因另一会话并发修改，转入paper/e23-v32-preservation独立工作副本，恢复精确dirty基线；详情见ledger。

## Review Focus

1. 教师冻结仍须让修复输入有梯度，而完整目标和教师无梯度：Task1真实patch与梯度测试。
2. clean分支不得改mask/query统计或padding；原坐标重叠副本一致：Task1输入及恒等测试。
3. 辅助训练不得消耗修复器RNG、更新修复器或沿用Stage2初始化：Task1 RNG、Task2完整21轮及Stage2初值测试。
4. 模式/教师/系数与checkpoint错配不得静默接受：Task2加载器篡改测试。
5. 资源含辅助训练、验证不更新、原mask不变、选模不看测试：Task3报告身份和选择测试，Task4完整重放。

### Task 1: 辅助预测训练与一致性

**Files:** Create `integrations/e23/prediction_preservation.py`, `tests/integration/test_e23_prediction_preservation.py`。
**Interfaces:** preservation_identity(mode)->dict；build_probe(protocol,device,seed)->MaskAwareForecastModel；predict_sequence(probe,sequence,batch)->Tensor；consistency_loss(probe,repaired,batch)->Tensor；train_clean_probe_step(probe,optimizer,batch)->float；freeze_probe(probe)。

- [x] 写真实patch恒等、缺失梯度/teacher无梯度、mask/stats固定、小std有限、辅助构建/训练RNG不变测试，运行观察RED。
- [x] 最小实现，运行该文件GREEN；probe只在clean训练分支更新，冻结显式检查。

### Task 2: Stage1集成及阶段身份

**Files:** Modify `integrations/e23/train.py`, `integrations/e23/run.py`, `scripts/evaluate_e23_three_test_draws.py`; extend Task1 tests。
**Interfaces:** train_repair(...,stage1_preservation='none')；manifest/payload携带preservation_identity；阶段2传递并校验父级身份。

- [x] 写21轮真实小批集成测试：warmup20/active1、冻结SHA不变、独立教师、相同预测头初始SHA及身份篡改拒绝，观察RED。
- [x] 接入Stage1循环、每轮损失/成本/覆盖日志、辅助checkpoint保存；Stage2仅传递身份，不改变目标。
- [x] 运行新测试和Stage1/2相关测试GREEN；记录增量。

### Task 3: 固定矩阵、诊断和报告

**Files:** Create `configs/e23_multi_expert_p4_width64_prediction_preservation_v32.yaml`, `scripts/run_e23_v32_trial.py`, `scripts/report_e23_v32_trial.py`, `scripts/diagnose_e23_v32.py`, `tests/integration/test_e23_v32_reporting.py`。
**Interfaces:** runner --root --device --mode train|report；A/B独立完整两阶段，锁定后立即三组测试；report读取JSON不重训。

- [x] 写矩阵固定设置、严格验证门槛、非有限与错误身份拒绝测试，观察RED。
- [x] 实现runner/report/teacher和repair验证诊断，保存模型比较和6行新test CSV；包含全部实测资源。
- [x] GREEN相关测试；全套pytest和batch128真实CUDA预检；独立审查尝试因额度限制未执行；执行者完成本地检查并明确披露限制，修复测试路径问题后冻结代码。

### Task 4: 正式训练、测试、归档

- [x] 顺序A/B各Stage1/2满100轮，每5分钟检查，各自锁定验证后立即测试3组。
- [x] 核对代码/mask/checkpoint/辅助身份、完整成本、CSV逐值；按验证规则判断并写最终报告。
- [x] 更新计划、归档SHA核对后的执行ledger及增量；选择性提交owned文件，保留当前分支。
