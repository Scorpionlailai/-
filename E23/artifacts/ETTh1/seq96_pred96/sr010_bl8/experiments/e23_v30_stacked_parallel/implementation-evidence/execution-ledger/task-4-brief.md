### Task 4: 独立层/组控制与全局specialist exposure

**Files:** Create `src/malr/models/e23/stacked.py`; Create `tests/unit/test_e23_stacked.py`。

**Interfaces:** 消费Task2的TVHGroup、grouped_tv_h及prepare_grouped_variable_sources，和Task3的StackedGroupFusion。

`StackLayerDiagnostics(layer_index:int,latents:Tensor,routes:Tensor,preview_gates:Tensor,group_weights:Tensor,group_deltas:Tensor,allowed:Tensor,update_rms:Tensor)`。`StackedRepairState(router:RouterOutput,evidence:Mapping[str,ExpertEvidence],layers:tuple[StackLayerDiagnostics,...],specialist_selected:Tensor|None)`。

`StackedTVHController(variable_count:int,patch_length:int,d_model:int,*,layers:int,groups:int,activation_checkpointing:bool=False,variable_topk:int=4,variable_offsets:tuple[int,...]=(-1,0,1),temporal_local_radius:int=2,temporal_boundaries:tuple[int,...]=(2,4,8,16)).forward(inputs:ExpertInputs,*,historical:HistoricalCandidateProvider,epoch:int=0,explore_variable:bool=False,specialist_exposure:float=0.,specialist_seed:int=0,disabled_experts:frozenset[str]=frozenset(),disabled_groups:frozenset[tuple[int,int]]=frozenset())->StackedRepairState`；关闭组的tuple索引从0起。

- [ ] **Step 1:** 写`test_stack_parameters_are_distinct_across_layers_and_groups`、`test_next_layer_reads_previous_fused_state_without_detach`、`test_stack_historical_lookup_occurs_once`；沿用v29真实库fixture，直接类小样例L2/G2/D8，正式配置另验4/8/64。
- [ ] **Step 2:** 写`test_stack_specialist_sample_occurs_once_across_all_groups_and_layers`，捕获原可用性并集及一次sample；写`test_singleton_variable_cannot_read_temporal_source_updates`，受控改变T对邻变量的状态更新，V-only目标不变。

跨组singleton的核心断言（selected为`[B,C,N]`）：

```python
for layer in out.layers:
    allowed = layer.allowed[out.specialist_selected]  # [selected,G,3]
    assert allowed.any(dim=1).sum(dim=-1).le(1).all()
```

- [ ] **Step 3:** 写`test_disabled_group_or_expert_cannot_change_any_downstream_layer`、`test_all_unavailable_returns_original_base_exactly`及fully observed/padding测试；写`test_stack_group_batch_matches_module_reference`比较批量单层与独立组reference。运行文件保存RED。
- [ ] **Step 4:** 实现真实控制：共享一次H cache，提前按各层各组独立V参数构建来源缓存，T原来源可用性来自可见掩码/方向band；按各组实际原可用性并集抽样，跨全部层/组应用同一允许类型。按层调用Task2、Task3，每层状态不断梯度。诊断保留组维，公开router.available取最后层并集，weights按门控聚合且注明描述用途；公开evidence按最后层门控汇总，完整组信息在layers中。
- [ ] **Step 5:** GREEN并验证八组完整参数独立。保存增量，不能把某组来源缓存给其他组，不增加跨forward缓存。

