### Task 1: 可复用的专家修正、预门控与最终仲裁

**Files:** Modify `src/malr/models/e23/router.py`; Test `tests/unit/test_e23_router.py`。

**Interfaces:** 在 `ExpertRouter` 增加：

- `expert_delta(inputs: ExpertInputs, name: str, item: ExpertEvidence) -> Tensor`
- `expert_logit(inputs: ExpertInputs, name: str, item: ExpertEvidence, delta: Tensor, *, round_embedding: Tensor | None = None) -> Tensor`
- `none_logit(inputs: ExpertInputs, *, round_embedding: Tensor | None = None) -> Tensor`
- `preview_gate(inputs: ExpertInputs, name: str, item: ExpertEvidence, delta: Tensor, *, allowed: Tensor, round_embedding: Tensor | None = None) -> Tensor`
- `route_precomputed(inputs: ExpertInputs, evidence: Mapping[str, ExpertEvidence], deltas: Mapping[str, Tensor], *, round_embedding: Tensor | None = None) -> RouterOutput`

embedding 形状为 [D]，只进入打分条件，不能改变返回状态的 base。预门控为 `sigmoid(expert_logit-none_logit)*allowed*damaged*item.available`。

- [ ] **Step 1:** 写 `test_precomputed_router_matches_legacy_output_and_gradients`：固定 seed42，沿用 `_inputs/_evidence`，比较显式计算修正后的仲裁与旧 forward 的 latents、weights、梯度；写 `test_preview_gate_is_zero_when_disallowed_or_undamaged`，精确断言零；保留原 `test_router_scores_experts_in_one_batched_call`。
- [ ] **Step 2:** 运行 `python -m pytest tests/unit/test_e23_router.py -q`，确认新方法缺失导致新测试失败，记录失败原因。
- [ ] **Step 3:** 按接口拆分路由器计算。旧 forward 继续一次批量调用 shared_score；预计算修正不再在最终仲裁重算，检查专家名称、形状和有限性。
- [ ] **Step 4:** 重跑同一文件；全部通过，原 no-evidence/base 和批量调用断言保持通过。
- [ ] **Step 5:** 保存本任务增量与测试结果；仅对已核对的本轮变更提交 `Refactor expert arbitration for iterative recovery`。

