### Task 6: 实验矩阵、逐层/组诊断与报告

**Files:** Create `scripts/run_e23_v30_trial.py`, `diagnose_e23_v30_stack.py`, `report_e23_v30_trial.py`; Create `tests/integration/test_e23_v30_reporting.py`。

**Interfaces:** runner CLI `--root PATH --device cuda:1 --mode train|evaluate|report`；train按l1_g8/l4_g1/l4_g8执行两阶段并锁定验证，不自动开始test；evaluate须确认全部三个权重已锁定/重放/code freeze匹配后各执行一次三组测试；report只读取已产出证据。

`choose_v30_candidate(records:Sequence[Mapping],*,reference_mse:float=0.7499365438950035,reference_mae:float=0.5840456761470457)->Mapping|None`仅使用验证MSE严格更低/MAE不更高者，排序MAE→MSE→repair参数数→固定矩阵次序；原v28门槛另记。`validate_v30_diagnostics(record:Mapping,diagnostic:Mapping)->None`校验模型/层/组/config/checkpoint/hash、validation split、train_standardized尺度、H状态不变。

diagnostic CLI `--e23-config PATH --stage1-manifest PATH --stage2-manifest PATH --output PATH --device cuda:1`；各stage按层解码，Stage2各层接同一原预测器记录预测MSE/MAE，汇总组门控/路由/差异，并执行逐组关闭诊断。逐组关闭改变全部数据流，不用来改选checkpoint或正式深度。

- [ ] **Step 1:** 写选择测试：严格MSE/MAE边界、tie顺序、fake test指标不影响选择；拒绝非有限/未完成100+100epoch/不匹配SHA记录。写diagnostic错checkpoint、错层/组、变更H SHA拒绝；运行RED。

边界fixture使用已完整构造的合法记录，核心断言：

```python
record["best_val_mse"] = 0.7499365438950035
record["best_val_mae"] = 0.5840456761470457
assert choose_v30_candidate([record]) is None  # MSE必须严格改善
```

- [ ] **Step 2:** 写runner dry-run测试，断言三个config/目录顺序、100/100、原mask路径、验证选模参数、全部锁定后才测试、已有完成输出拒绝覆盖；资源绑定各自选定manifest，不复用其他模型资源。实现选择、诊断及runner/report。
- [ ] **Step 3:** 报告CSV含v28/T1/v29-r1/v29-r3既有记录及三个新模型；逐组三组CSV含固定seed、窗口/坐标数、MSE/MAE，std为总体位置方差。新模型无证据时相应指标留空，不能填旧版成绩。
- [ ] **Step 4:** GREEN报告/脚本CLI测试；逐层最终预测与正常forward一致，诊断的层/组数量准确，不用单组Missing参考计算三组DDR。保存增量；保证所有可运行脚本在代码冻结前已实现。

