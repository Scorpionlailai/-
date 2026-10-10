### Task 8: r1/r3 完整训练、验证选模及最终报告

**Files:** Create `scripts/report_e23_v29_trial.py`; Test `tests/integration/test_e23_v29_selection.py`; Create `docs/experiments/E23_v29_iterative_TV_H_P4_trial.md` 和 `ETTh1_P4_E23_v28_v29_T1_comparison_2026-10-03.csv`；可晋级时 Create 相应逐组结果 CSV。

**Interfaces:** `choose_v29_candidate(records:Sequence[Mapping[str,object]], *, reference_mse:float=0.7402293330687785, reference_mae:float=0.5855096656989619) -> Mapping[str,object]|None`，只消费已经验证重放的 Stage2 记录，拒绝非有限指标、无身份或未完成100/100轮记录。返回通过双门槛者中按 MAE/MSE/rounds 排序的候选。

- [ ] **Step 1:** 写 `test_v29_selection_uses_only_validation_and_both_gates`、`test_v29_selection_ties_prefer_fewer_rounds`、`test_v29_selection_rejects_incomplete_or_nonfinite_records`，在记录中放入与排序相反的假 test 指标仍不影响选择；运行确认选择函数缺失导致失败，随后实现选择/报告并使三测试通过。
- [ ] **Step 2:** 检查两个新输出根目录尚无完成记录；冻结当前代码与两配置。顺序执行 r1 Stage1→Stage2、r3 Stage1→Stage2，每步保存日志/manifest/resource。每个模型必须独立从头训练100/100轮，不能复用 preflight 或另一个模型的权重。

训练命令采用以下模板，分别把 rounds替换为1、3后顺序执行；`E23_V29_CONFIG/E23_V29_ROOT` 为任务变量，不能改系统 HOME：

```bash
export PATH=/home/xyw/miniforge3/envs/malr/bin:$PATH
export PYTHONPATH=src:.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
E23_V29_CONFIG=configs/e23_multi_expert_p4_width64_iterative_tv_h_r1_v29.yaml
E23_V29_ROOT=artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/e23_v29_iterative_tv_h_r1
python -m integrations.e23.run train-repair --protocol-config configs/stage1_seq96_dynamic_temporal_repair.yaml --e23-config "$E23_V29_CONFIG" --migration-catalog ../E22/migration/paper_catalog.jsonl --mask-artifacts ../E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context --output "$E23_V29_ROOT/stage1" --device cuda:1
python -m integrations.e23.run train-forecast --protocol-config configs/stage1_seq96_dynamic_temporal_repair.yaml --e23-config "$E23_V29_CONFIG" --migration-catalog ../E22/migration/paper_catalog.jsonl --mask-artifacts ../E22/artifacts/ETTh1/seq96_pred96/sr010_bl8/experiments/two_stage_v5_dynamic_temporal_context --stage1-manifest "$E23_V29_ROOT/stage1/runs/e23-stage1.json" --output "$E23_V29_ROOT/stage2" --device cuda:1 --selection-metric mae --forecast-loss mse_plus_0p2_mae --forecast-input explicit_v1
python scripts/replay_e23_p4_validation.py --e23-config "$E23_V29_CONFIG" --stage2-manifest "$E23_V29_ROOT/stage2/runs/e23-stage2.json" --output "$E23_V29_ROOT/validation_replay.json" --device cuda:1
```

- [ ] **Step 3:** 每个选定权重重放全部2688验证窗口、1806336预测坐标；比较 manifest 至1e-6，核对 H key/bank未改变。以选定 Stage1/2 权重记录各轮全局修复 MSE/MAE、状态更新幅度、预门控和最终路由统计。冻结代码检查未变；逐轮诊断不用于额外选模。
- [ ] **Step 4:** 调用 choose_v29_candidate，先写 `validation_selection.json`，其中保留两模型验证指标、checkpoint SHA、门槛判定和唯一候选。没有候选时直接进入最终报告，不执行新测试。
- [ ] **Step 5:** 若有候选，仅对该模型调用既有三组评价脚本一次；把下例的 config/root/model-label 改为预先选定者，保持现有 test mask bank 不变。

```bash
python scripts/evaluate_e23_three_test_draws.py --e23-config "$E23_V29_CONFIG" --selection "$E23_V29_ROOT/stage2/runs/e23-stage2.json" --test-mask-bank artifacts/ETTh1/seq96_pred96/sr010_bl8/test_mask_bank_20261002/manifest.json --model-label E23_v29_r1 --output "$E23_V29_ROOT/three-test-draws.json" --device cuda:1
```

- [ ] **Step 6:** 生成 Markdown/CSV：包含 v28、T1既有结果及r1/r3验证、修复、训练时间/显存。只有被选中并实际测试的版本填写三组测试均值/标准差；未测试者空值并明确标记。资源绑定实际selection SHA，最终测试耗时不算训练，不用单组 Missing 参考计算三组 DDR。
- [ ] **Step 7:** 复核 epoch数、两个实验目录、配置/code/hash、资源、测试seed/window/count及CSV行数；保存最终 status。最终汇报区分验证选择、测试表现和固定权重诊断，明确是否超过T1，以及一轮与三轮的实际成本。

## Execution Handoff

推荐在当前会话按任务顺序实施：八项任务共享专家、路由和加载接口，连续实现能减少交接成本，完整训练前保留独立审查。

计划自查项：八个任务覆盖 spec 的缓存、串行依赖、共享迭代、约束、身份、兼容、资源和评价；上述五类 Review Focus 均有明确归属与测试；任务间 cache/返回类型及配置名保持一致。

当前仅计划完成，所有执行复选框仍未勾选。按 writing-plans 技能的交接要求，用户审阅此计划并确认执行方式后，使用 executing-plans 或 subagent-driven-development 开始实现。
