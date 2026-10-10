### Task 2: H 候选缓存与逐轮历史消息

**Files:** Modify `src/malr/models/e23/historical.py`; Create/Test `tests/unit/test_e23_historical_cache.py`。

**Interfaces:** `HistoricalCandidateCache` 包含 `normalized_raw:[B,C,N,K,P]`、`raw_quality:[B,C,N,K,5]`、`base_logits:[B,C,N,K]`、`valid`、`available`、`provenance`。增加构造参数 `contextual: bool=False`；增加 `prepare_candidates(inputs: ExpertInputs) -> HistoricalCandidateCache` 和 `from_cache(inputs: ExpertInputs, cache: HistoricalCandidateCache, *, round_embedding: Tensor | None=None) -> ExpertEvidence`。

新模式才实例化状态匹配投影。匹配项为 `0.5*tanh(dot(q(state),k(candidate))/sqrt(D))`，q 输出零初始化；消息沿用原 message_projector，candidate validity 和原可见质量始终保留。

- [ ] **Step 1:** 用现有 `test_e23_historical._inputs/_memory` 写 `test_h_cache_queries_once_and_messages_depend_on_current_state`：真实索引 query spy 计数为1，三次 from_cache 不再查询，改变 state 后 message 改变；写 `test_h_cache_rebuilds_after_mask_or_snapshot_change`、`test_h_cache_empty_and_unobserved_queries_stay_unavailable`；验证原未来成员 starts12 被排除。
- [ ] **Step 2:** 运行 `python -m pytest tests/unit/test_e23_historical_cache.py -q`，确认缓存接口不存在导致失败。
- [ ] **Step 3:** 实现 cache/from_cache。空库或无可检索目标返回固定 K 槽且全无效的缓存；缓存保留训练张量，诊断 provenance 不作为训练输入。旧 forward 留原路径，旧构造不生成新 state_dict 键。
- [ ] **Step 4:** 运行 `python -m pytest tests/unit/test_e23_historical.py tests/unit/test_e23_historical_cache.py -q`；候选权重有效槽和为1，无效槽为0，状态匹配不复活无效候选，message_projector 梯度有限且非零。
- [ ] **Step 5:** 保存增量与通过证据；按工作区约束提交 `Cache historical candidates within iterative forwards`。

