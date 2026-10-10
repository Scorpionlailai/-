### Task 4: V 读取 T 更新后的目标及来源状态

**Files:** Modify `src/malr/models/e23/variable.py`; Create/Test `tests/unit/test_e23_variable_context.py`。

**Interfaces:** `VariableSourceCache` 包含原选定四来源及 offsets、source_values/source_observed、quality、prior/logits、candidate_valid、索引 provenance。新增构造参数 `contextual: bool=False`；`prepare_sources(inputs: ExpertInputs, *, epoch:int, explore:bool) -> VariableSourceCache`；`from_cache(inputs: ExpertInputs, cache: VariableSourceCache, *, round_embedding: Tensor | None=None) -> ExpertEvidence`。

候选消息是原 MLP 消息加原选定来源的当前 state 投影；候选分数加 [-0.5,0.5] 的目标/来源 state 匹配项。来源投影和匹配初始影响为零。上下文模式仅走 scalar_v1/window_context_topk_v1，旧 v26/v27 保留。

- [ ] **Step 1:** 写 `test_variable_context_reads_both_updated_target_and_source_states`，分别扰动目标、已选来源，受控启用新投影后两者均改变消息；写 `test_variable_context_keeps_source_ids_and_original_validity`，跨轮 source ids 相同、候选不超过4×3、原不可见来源不复活；写 `test_variable_context_single_channel_has_zero_available_evidence`。
- [ ] **Step 2:** 运行 `python -m pytest tests/unit/test_e23_variable_context.py -q`，确认缺接口导致失败。
- [ ] **Step 3:** 实现来源缓存与重新形成消息，候选匹配 q 输出零初始化，避免每轮重做离散来源选择或训练探索轮换；质量只以原观测及本轮权重汇总。
- [ ] **Step 4:** 运行 `python -m pytest tests/unit/test_e23_variable.py tests/unit/test_e23_variable_context.py -q`，新梯度、权重及既有来源选择/聚合测试通过。
- [ ] **Step 5:** 保存增量与证据；按工作区约束提交 `Pass temporal repair context into variable evidence`。

