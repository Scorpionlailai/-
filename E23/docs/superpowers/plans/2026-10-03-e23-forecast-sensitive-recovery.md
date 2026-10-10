# E23 v31 Forecast-Sensitive Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 实现并验证单轮三专家Stage2位置收益加权修复约束，完成A/B/C公平对照及三组测试。

**Architecture:** 原修复器和预测器结构不变，新增仅训练使用的局部干预收益计算、权重归一化和修复保留损失。C在每轮开始冻结上轮末模型作为教师，监督当前batch前16窗口的缺失位置；A/B作为原预测目标与均匀修复约束对照。

**Tech Stack:** 现有Python、PyTorch、pytest、YAML、CSV；不新增依赖。

**Spec:** `docs/superpowers/specs/2026-10-03-e23-forecast-sensitive-recovery-design.md`，用户已确认。

## Global Constraints

- cwd `/home/xyw/paper/E23`；保留现有dirty基础和用户CSV，先归档增量前基线与SHA，不能盲提交基础文件。
- 模型为v29-r1单轮H/V/T、修复宽度64、原预测器宽度16、显式输入，无新增推理参数。
- ETTh1 P4 96→96、7变量、patch16/stride8；原H/V/T策略、原mask bank和三个测试seed2021/2026100201/2026100202不变。
- 一次本轮从头Stage1训练100轮；A/B/C各Stage2训练100轮，共用本次选定Stage1。seed2021、batch128、FP32、lr0.001/0.0001、weight_decay0。
- Stage1保持global_mse_v1+0.1 retrieval、specialist exposure0.2；Stage2主目标预测MSE+0.2 MAE。
- 三条件前10轮只用预测目标；B/C第11轮起加0.05修复损失。C教师来自上轮末、整轮冻结、eval、无梯度，不用验证选定教师。
- 六个不重叠16点时间区域×7变量=42；C每batch前16个打乱窗口算全部42区域真值恢复收益，其余窗口权重1。
- 正收益映射原时间坐标；s为窗口缺失坐标上正收益均值，raw_w=1+clamp(gain/max(s,1e-6),0,3)，归一化至每窗口缺失权重均值1。无缺失、全非正收益时权重1。
- 修复error=(repaired_norm-clean_norm)*detach(query_std)，按原序列缺失坐标计算，不重复patch覆盖。权重stop-gradient。
- 每条件按原验证MAE选模；晋级要求验证MSE<0.7499365438950035且MAE≤0.5840456761470457；多个通过按MAE、MSE、额外训练时间、A/B/C顺序。
- 每条件锁定重放后立刻测试三组一次；测试不影响后续配置。训练每5分钟检测。完整时间包含教师、准备、训练、验证、保存，诊断及测试另记。
- 每条件报告共享Stage1成本+该条件Stage2成本，另列共享Stage1及实际矩阵总成本，不把共同阶段执行三次。
- root `artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/e23_v31_forecast_sensitive`；正式训练前冻结源/config/script/test哈希。
- 默认PATH=/home/xyw/miniforge3/envs/malr/bin:$PATH，PYTHONPATH=src:.，OMP_NUM_THREADS=4，MKL_NUM_THREADS=4。

## Review Focus

1. 重叠patch及末尾padding：必须一致替换原坐标全部副本且只改真实缺失值，归Task1真实patch测试。
2. 全负收益、无缺失、很小query_std：权重/全局损失有限，归Task1归一化与尺度测试。
3. 教师快照与训练model共享参数或模式：冻结教师不影响学生，轮内固定、轮间更新，归Task2状态测试。
4. 标签从收益计算泄漏到推理或验证权重：只有train batch进入教师收益，无推理评分器，归Tasks2/3调用与身份测试。
5. 共享Stage1导致成本重复计费或各Stage2初始化不同：资源绑定和重置RNG/optimizer须验证，归Task3矩阵与Task4资源检查。

---

### Task 1: 局部收益、位置权重和修复损失

**Files:** Create `integrations/e23/forecast_sensitive.py`; Create `tests/integration/test_e23_forecast_sensitive.py`。

**Interfaces:** `region_oracle_gains(teacher:E23ForecastModel,batch:P4Batch,*,subset:int=16)->Tensor` 返回[min(16,B),42]；`gains_to_weights(gains:Tensor,hidden:Tensor)->Tensor` 返回[B,T,C]；`weighted_repair_loss(repaired:Tensor,target:Tensor,hidden:Tensor,query_std:Tensor,weights:Tensor)->Tensor` 返回全局尺度scalar。teacher调用无梯度，收益使用MSE。

- [x] **Step 1:** 写失败测试：受控预测器中只恢复缺失值，原mask/stats/observed/padding不变；原时间80–96所有重叠副本一致；无缺失替换预测恒等。再测试全负/全零/很小正收益权重有限、均值1、正下限、未覆盖窗口权重1和权重无梯度。
- [x] **Step 2:** 运行 `python -m pytest tests/integration/test_e23_forecast_sensitive.py -q` 保存RED；确认失败原因是新接口缺失。
- [x] **Step 3:** 实现三个接口，复用patchify_preserving_values与固定mask/query统计量，按真实96点序列映射42区域；每组最多7个替换并行以控制显存。教师只处理前16窗口。损失使用全部batch缺失坐标，空缺失返回可微0。
- [x] **Step 4:** 增加全1权重严格等价原global_mse_v1、尺度极小值、不重复覆盖、修复梯度存在且教师/权重无梯度测试；运行Task1文件GREEN。
- [x] **Step 5:** 保存任务增量及测试日志，选择性提交新文件。

### Task 2: Stage2目标、教师生命周期与身份

**Files:** Modify `integrations/e23/train.py`, `integrations/e23/run.py`; Extend Task1 tests; Create `tests/integration/test_e23_forecast_sensitive_training.py`。

**Interfaces:** 在train_forecast新增可选 `repair_retention:Literal['none','uniform_v1','forecast_gain_v1']='none'`；none为A，uniform_v1为B，forecast_gain_v1为C。新增模式记录warmup10、repair_weight0.05；C另记teacher previous_epoch_final、subset16、regions42、权重规则。旧调用默认行为不变。

- [x] **Step 1:** 写失败测试：前10轮不加约束，第11轮加入；冻结教师快照与学生参数地址不同、eval且requires_gradFalse；轮内参数不变，下一轮复制前轮末状态；Stage2冻结H key/库不变。
- [x] **Step 2:** 运行新增测试保存RED，随后实现显式训练模式派发；保持原预测目标，B/C使用Task1损失。C第11轮开始从前轮末构建/刷新教师，不使用best checkpoint，实际额外耗时单独累计。
- [x] **Step 3:** 完善manifest/checkpoint加载校验，新CLI允许模式参数；校验训练模式、warmup/系数及C教师身份一致，旧权重没有字段时保持原逻辑。指标仍为原固定验证预测MSE/MAE，增加选定修复指标及实际覆盖记录。
- [x] **Step 4:** GREEN运行新增训练测试和旧 `test_e23_stage2.py`, `test_e23_iterative_training.py`, `test_e23_global_repair_objective.py`；验证训练未来标签不成为repair/predictor输入，验证/测试没有教师标注调用。
- [x] **Step 5:** 保存任务增量与日志；基础dirty文件不整文件盲提交。

### Task 3: 固定实验矩阵与资源报告

**Files:** Create `scripts/run_e23_v31_trial.py`, `scripts/report_e23_v31_trial.py`, `tests/integration/test_e23_v31_reporting.py`; Create `configs/e23_multi_expert_p4_width64_forecast_sensitive_v31.yaml`，参数与v29-r1一致。

**Interfaces:** runner `--root PATH --device cuda:1 --mode train|report`，train先共同Stage1，然后A/B/C各自Stage2、验证重放及三组测试。report只读manifest/resources/test结果。模式身份区分三个条件，不以相同结构混用权重。

- [x] **Step 1:** 写RED测试：三条件同Stage1 SHA、同预测器初始化SHA、相同训练RNG重置、独立optimizer，100轮与固定顺序；已有完成输出拒绝覆盖。
- [x] **Step 2:** 实现runner，调用现有measured entrypoint和三组测试工具；记录GPU UUID/并发、每阶段资源、共享Stage1成本、各条件完整成本和实际矩阵成本。测试评估器接入新模式身份但保持既有协议检查。
- [x] **Step 3:** 写选模严格双指标边界、多个候选排序、错误模式/配对SHA/非有限成绩/错mask/缺资源拒绝测试；实现report读取A/B/C及保存v29-r1/T1参考，不重新训练参考。
- [x] **Step 4:** GREEN报告与runner测试，CSV记录9行新测试组、条件比较、均值/总体std、修复与预测指标、成本、教师覆盖及耗时；保留共享训练口径。
- [x] **Step 5:** 保存增量和日志，选择性提交新文件。

### Task 4: 集成验证、审查与训练冻结

**Files:** root下baseline、incremental patch、preflight、verification、review与code-freeze证据；必要真实CUDA预检工具保存在root中。

- [x] **Step 1:** 运行全套pytest，保存结果；只修复实际失败，不扩大无关改动。确认旧权重可加载、用户CSV SHA保持原值。
- [x] **Step 2:** GPU短预检batch128 FP32，B/C均验证forward/backward、教师无梯度、学生更新及H状态不变；测峰值和教师额外耗时，不用于正式模型初始化。
- [x] **Step 3:** 使用执行技能要求的完整实现独立审查，核查标签路径、快照生命周期、patch覆盖、初始化、资源归属与选模。解决实际Critical/Important，保存证据。
- [x] **Step 4:** 冻结源/config/test/script文件及哈希，保存dirty基线和任务累计patch；训练期间不修改实现/config。确认正式GPU空闲再开始。

### Task 5: 完整训练、逐条件测试和归档

**Files:** root/common_stage1、A/B/C各自stage2/replay/resources/test及CSV/report；`docs/experiments/E23_v31_forecast_sensitive_P4_trial.md`。

- [x] **Step 1:** 运行共同Stage1满100轮，记录全局验证修复选模与H/key库SHA；验证本次权重全新、未复用诊断教师。
- [x] **Step 2:** 顺序运行A/B/C Stage2各100轮，按用户要求每5分钟检测；各自锁定验证MAE选定权重，完整重放验证并检查H状态不变。
- [x] **Step 3:** 每条件完成后立即用同一权重测试三个固定位置一次，检查2688窗口/1806336预测坐标和mask SHA；后续条件不读取先前测试结果。
- [x] **Step 4:** 运行report，逐值核对三条件原JSON与CSV，检查共享Stage1与各自Stage2成本、教师成本和实际矩阵成本；按验证规则晋级或保留旧版，分别报告B对A、C对B。
- [x] **Step 5:** 完成报告、final-status和哈希校验，更新已完成checkbox，归档执行ledger/测试/审查证据；最终汇报准确性、成本和局限，不将位置干预称为真实变量因果发现。

## Self-review 与执行方式

五项Review Focus均有明确测试归属；规格中的标签、教师、权重、两阶段共享成本、测试顺序和报告要求由Tasks1–5覆盖。计划采用本会话逐任务执行，接口有依赖，不逐任务派发实现代理；在正式训练前保留一次完整实现审查。Tasks1–5已完成；本轮从头共同Stage1及A/B/C各100轮，按验证规则选择B，已完成各条件三组测试、资源报告与证据归档。
