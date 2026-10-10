### Task 3: T 的可见来源缓存与状态条件消息

**Files:** Modify `src/malr/models/e23/temporal.py`; Create/Test `tests/unit/test_e23_temporal_context.py`。

**Interfaces:** `TemporalSourceCache` 持有 `source_encoded:[B,C,N,D]` 和原可见/有效来源张量。新增构造参数 `contextual: bool=False`；`prepare_sources(inputs: ExpertInputs) -> TemporalSourceCache`；`from_cache(inputs: ExpertInputs, cache: TemporalSourceCache, *, round_embedding: Tensor | None=None) -> ExpertEvidence`。

来源特征为 `source_encoded + source_context_projection(LayerNorm(current_source_state))`，投影输出零初始化；查询条件使用当前 state 与 round_embedding。方向带、来源可见性及一致性仍来自原可见数据。

- [ ] **Step 1:** 沿用 `tests/unit/test_e23_temporal.py` 的样例几何，写 `test_temporal_context_reads_updated_source_states_without_reencoding`：受控设置上下文投影为非零，改变原有效来源的 state 后消息改变，三次 from_cache 的原 source_encoder 调用数仍为1；写 `test_temporal_context_ignores_hidden_and_invalid_raw_values`，改变隐藏/无效槽有限数值不改变结果；断言无效槽输入梯度为零。
- [ ] **Step 2:** 运行 `python -m pytest tests/unit/test_e23_temporal_context.py -q`，确认新接口缺失导致失败。
- [ ] **Step 3:** 实现 cache 和新消息路径，距离带不变、全局候选允许随 state 改变；保留旧 forward 的一次原来源编码行为。
- [ ] **Step 4:** 运行 `python -m pytest tests/unit/test_e23_temporal.py tests/unit/test_e23_temporal_context.py -q`；来源可见性、无证据回退、缓存计数及状态投影梯度通过。
- [ ] **Step 5:** 保存增量与证据；按工作区约束提交 `Condition temporal evidence on evolving repair states`。

