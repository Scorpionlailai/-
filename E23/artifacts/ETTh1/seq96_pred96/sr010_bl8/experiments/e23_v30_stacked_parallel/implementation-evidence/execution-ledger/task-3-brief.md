### Task 3: 组门控、拼接投影及精确回退

**Files:** Create `src/malr/models/e23/stacked_fusion.py`; Create `tests/unit/test_e23_stacked_fusion.py`。

**Interfaces:** `GroupFusionOutput(latents:Tensor,group_weights:Tensor,projected_deltas:Tensor)`；`StackedGroupFusion(d_model:int,group_count:int).forward(base:Tensor,deltas:Tensor,routes:Tensor,rho:Tensor,available:Tensor)->GroupFusionOutput`。base为`[B,C,N,D]`，deltas为`[B,C,N,G,D]`，routes为`[B,C,N,G,4]`，available为`[B,C,N,G]`；返回group_weights为`[B,C,N,G+1]`，projected_deltas为`[B,C,N,G,D]`，保留每组独立投影贡献。原始rho=1位置精确回到base。

- [ ] **Step 1:** 写`test_fusion_initializes_as_uniform_valid_residual_sum`：D64/G8，手工delta，断言可用组加none均匀、projection每块为I、输出为base+加权修正和；写`test_fusion_missing_groups_and_fully_observed_patches_are_exact`，全无证据/观测完整时精确等于base。

八组全有效且rho<1的核心断言：

```python
torch.testing.assert_close(out.group_weights, torch.full_like(out.group_weights, 1 / 9))
torch.testing.assert_close(out.latents, base + deltas.sum(dim=-2) / 9)
```

- [ ] **Step 2:** 写`test_single_group_is_exact_bypass_without_new_parameters`、`test_fusion_gradients_reach_every_valid_group_and_projection`、错误shape及非有限输入拒绝；运行文件保存RED。
- [ ] **Step 3:** 实现spec固定134/65维打分、门控及拼接投影；D较小时输入维数为2D+6/D+1，正式D64保持134/65。G1省略新增参数且直接使用单组修正。
- [ ] **Step 4:** 同一文件GREEN，权重合法、无效组权重为0、无证据有限；保存增量与证据。

