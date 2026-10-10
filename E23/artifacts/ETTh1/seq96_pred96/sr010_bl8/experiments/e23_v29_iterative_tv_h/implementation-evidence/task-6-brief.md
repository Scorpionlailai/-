### Task 6: 配置、两阶段身份及评价元数据

**Files:** Modify `integrations/e23/config.py`, `train.py`, `scripts/replay_e23_p4_validation.py`, `scripts/evaluate_e23_three_test_draws.py`; Create 两个 spec 指定 YAML；Create/Test `tests/integration/test_e23_iterative_training.py`; Extend `tests/integration/test_e23_config.py`。

**Interfaces:** E23Settings 增加 `recovery_schedule:str='parallel_v1'`、`recovery_rounds:int=1`。`_trial_identity` 仅给非默认模式增加两字段，Stage1/2 保存和加载完整传播。`build_repair_model` 传入 schedule/rounds。评价脚本新增结果元数据两字段，不能改变 mask 或评分。

- [ ] **Step 1:** 写新配置 r1/r3解析测试，严格拒绝 bool轮数、rounds2、parallel3、旧多头/后聚合/adapter组合、window_smooth_l1搭配新模式；保留旧配置解析。运行新增用例确认新键被旧校验拒绝。
- [ ] **Step 2:** 写 `test_iterative_two_stage_training_preserves_schedule_and_global_selection`，沿用 `tests/integration/test_e23_stage1._protocol_settings/_batch`，以 monkeypatch batch 做 Stage1两轮/Stage2一轮；断言正式尺度、最终状态评分、manifest/checkpoint两字段一致；写 `test_identical_shape_checkpoint_with_wrong_rounds_is_rejected`，改变身份字段但保持权重形状仍拒绝。
- [ ] **Step 3:** 实现受限配置、构建和身份传播，保持 Stage1 global_mse_v1 与 Stage2 mse_plus_0p2_mae 不变。history_adapter 非 none 搭配迭代明确拒绝。三组结果 profile 显式包含 schedule/rounds，新标签使用 E23_v29_r1 或 E23_v29_r3。
- [ ] **Step 4:** 运行 `python -m pytest tests/integration/test_e23_config.py tests/integration/test_e23_global_repair_objective.py tests/integration/test_e23_iterative_training.py tests/integration/test_e23_stage1.py tests/integration/test_e23_stage2.py -q`；wrong-round身份、两阶段训练、极小标准差的全局评分全部通过。
- [ ] **Step 5:** 保存增量与证据；按工作区约束提交 `Record iterative recovery identity across training and evaluation`。

