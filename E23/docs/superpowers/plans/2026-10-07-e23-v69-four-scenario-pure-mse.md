# E23-v69 Four Scenario Pure MSE Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在v69基础上找到统一配置，使四个缺失场景同时超过纯MSE冻结T1。

**Architecture:** 先重新测量双方纯MSE冻结Stage2；再以v69为基础逐项增强修复器，第一候选仅增加已有跨变量全局水平均值分支。最终预测头始终保持原样。

**Tech Stack:** Python/PyTorch、既有E23训练CLI和T1适配器、RTX5090。

**Spec:** /home/xyw/paper/E23/docs/superpowers/specs/2026-10-07-e23-v69-four-scenario-pure-mse-design.md

## Global Constraints

- Stage2纯MSE、冻结全部修复参数及缓冲状态、Adam lr1e-4/batch128/100轮/FP32，seed2023。
- 不改变预测头；同初始SHA；MAE主选/MSE次选，20组共享测试不用于选checkpoint。
- 三个率8块场景加.10/16；T1 Stage1作者配置seed502；保留双方既有padding。

## Review Focus

- 混合损失误标为纯MSE：检查实际objective及manifest。
- 冻结参数后BatchNorm仍更新：检查train/eval与完整状态SHA。
- .15/8漏跑或读错mask：四场景manifest/hash与场景参数逐一验证。
- 新旧checkpoint/损失混表：结果绑定selection SHA、明确版本。
- 缓存Stage1时间虚报新两阶段时间：资源记录单列复用和实测范围。

### Task 1: Pure MSE frozen baseline

- [x] 隔离源代码、建立四场景计划和已选Stage1引用。
- [x] CUDA预检修复状态与头初始SHA/梯度，原权重哈希保护。
- [x] 双方各四次100轮Stage2，同轨迹MAE/MSE checkpoint；各完成后20组测试。
- [x] 生成对照CSV与资源报告，复核checkpoint/validation/mask与冻结状态。

### Task 2: Single global-level branch on v69

- [x] 新配置只改变recent_mean_projection为global_level_mean_v2；训练等待基线全部完成。
- [x] 运行已有针对全局水平访问、隐藏payload隔离、零初始化及head保持的测试，CUDA预检。
- [x] 四场景相同配置重新Stage1/2各100轮；同Stage2约束和共享测试。
- [x] 四场景开发20组双指标胜过T1；全部24行双选择对照和资源已核查、报告。
- [ ] 锁定共同候选，封存新20组/场景仅确认一次，无重训和测试选模。
