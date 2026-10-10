### Task 1: 分离历史候选前端与独立H消息

**Files:** Create `src/malr/models/e23/historical_candidates.py`, `historical_messages.py`; Modify `historical.py`; Create `tests/unit/test_e23_historical_provider.py`。

**Interfaces:** `HistoricalCandidateProvider(patch_length:int,d_model:int,retrieval_dim:int,variable_count:int,*,topk:int=4,similarity_threshold:float=0.7,top_p:float|None=None,candidate_cap:int=8,retrieval_strategy:str='one_group')`拥有key_encoder/retrieval/index，提供`refresh_snapshot(snapshot)`、`prepare_candidates(inputs:ExpertInputs)->HistoricalCandidateCache`。`HistoricalMessageEncoder(patch_length:int,d_model:int,*,contextual:bool)`仅拥有消息/质量/状态匹配参数，`from_cache(inputs,cache,*,round_embedding:Tensor|None=None)->ExpertEvidence`。为保留旧键，采用共享的初始化/消息计算函数；旧HistoricalExpert不能把原顶层message_projector改注册到新子路径。

- [ ] **Step 1:** 捕获当前tracked/untracked源代码、配置、测试、脚本的基线及SHA；存入root/implementation-evidence，排除artifacts、缓存及.git，另存用户CSV与git status。沿用已批准当前分支；读取using-git-worktrees技能并据现有授权判断是否需独立scratch，不能从HEAD丢失dirty基础。
- [ ] **Step 2:** 写`test_provider_has_only_retrieval_parameters`、`test_provider_cache_matches_existing_historical_cache`、`test_message_encoder_matches_existing_message_and_gradients`：沿用`tests/unit/test_e23_historical._inputs/_memory`，复制同名消息权重，比较真实候选、消息/质量、权重与梯度；断言message encoder没有key/EMA/index。

核心断言：

```python
assert provider.state_dict()
assert all(name.startswith("key_encoder.") for name in provider.state_dict())
assert "key_encoder" not in dict(message.named_modules())
torch.testing.assert_close(actual.message, expected.message, rtol=0, atol=0)
```

- [ ] **Step 3:** 运行`python -m pytest tests/unit/test_e23_historical_provider.py -q`，记录缺少新类的RED。
- [ ] **Step 4:** 实现provider/消息分离，保留旧类导入、构造参数和state_dict键；缓存训练张量不来自detached provenance。
- [ ] **Step 5:** 运行新测试与`tests/unit/test_e23_historical.py tests/unit/test_e23_historical_cache.py tests/integration/test_e23_model.py`；检查空库、未来成员排除、top-p与旧H政策不变，保存GREEN及任务增量。

