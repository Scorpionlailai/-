### Task 2: 独立参数的组维批量专家计算

**Files:** Create `src/malr/models/e23/stacked_group.py`, `group_ops.py`, `stacked_group_kernels.py`; Create `tests/unit/test_e23_group_ops.py`, `test_e23_grouped_experts.py`。

**Interfaces:** `grouped_linear(x:Tensor,layers:Sequence[nn.Linear])->Tensor`、`grouped_layer_norm(x:Tensor,layers:Sequence[nn.LayerNorm])->Tensor`、`grouped_mlp(x:Tensor,layers:Sequence[nn.Sequential])->Tensor`，特征张量的倒数第二维为G、最后一维为feature；参数由原独立模块torch.stack构建，不能detach或重新包装成替代Parameter。

`TVHGroup(variable_count:int,patch_length:int,d_model:int,*,variable_topk:int=4,variable_offsets:tuple[int,...]=(-1,0,1),temporal_local_radius:int=2,temporal_boundaries:tuple[int,...]=(2,4,8,16))`在stacked_group.py中拥有historical:HistoricalMessageEncoder、variable:SparseVariableExpert、temporal:MultiScaleTemporalExpert、router:ExpertRouter及零初始化的`condition_embedding:nn.Embedding(3,D)`。各专家启用contextual，条件使用embedding第0项，整个3×D参数形状沿用v29便于数值映射；构造每组新网络，不复制随机初始化结果。

`GroupedExpertEvidence(message:Tensor,quality:Tensor,available:Tensor,provenance:Mapping[str,Tensor])`分别为`[B,C,N,G,D]`、同形、`[B,C,N,G]`。`GroupedRoundOutput(latents,weights,deltas,preview_gates,available)`形状分别为`[B,C,N,G,D]`、`[B,C,N,G,4]`、各专家修正、`[B,C,N,G,2]`、`[B,C,N,G,4]`。

`prepare_grouped_variable_sources(inputs:ExpertInputs,experts:Sequence[SparseVariableExpert],*,epoch:int,explore:bool)->tuple[VariableSourceCache,...]`保留每组独立选择与可微prior。`grouped_historical(inputs,states:Tensor,cache,encoders:Sequence[HistoricalMessageEncoder],embeddings:Tensor)->GroupedExpertEvidence`；`grouped_variable(inputs,states,experts:Sequence[SparseVariableExpert],caches:Sequence[VariableSourceCache],*,source_context_allowed:Tensor|None,embeddings:Tensor)->GroupedExpertEvidence`；`grouped_temporal(inputs,states,experts:Sequence[MultiScaleTemporalExpert],*,source_context_allowed:Tensor|None,embeddings:Tensor)->GroupedExpertEvidence`。states为`[B,C,N,G,D]`，embeddings为`[G,D]`；原H缓存共享，V来源每组独立选择，T每组独立编码。`grouped_tv_h(inputs,groups:Sequence[TVHGroup],h_cache,v_caches:Sequence[VariableSourceCache],allowed:Tensor,source_context_allowed:Tensor|None)->GroupedRoundOutput`实现一次T→V→H；allowed为`[B,C,N,G,3]`，source_context_allowed为`[B,C,N]`或None。

- [ ] **Step 1:** 写组linear/MLP/LayerNorm对照测试，用不同权重的两组，与逐组原PyTorch模块结果及输入/每组参数梯度比较。写`test_grouped_operations_do_not_alias_or_detach_parameters`，扰动一组不会改变另一组输出。

独立linear oracle（用float64测输出及相同loss下的梯度）：

```python
expected = torch.stack([layer(x[..., g, :]) for g, layer in enumerate(layers)], dim=-2)
actual = grouped_linear(x, layers)
torch.testing.assert_close(actual, expected, rtol=0, atol=1e-10)
assert layers[0].weight.data_ptr() != layers[1].weight.data_ptr()
```

- [ ] **Step 2:** 写`test_grouped_h_v_t_match_independent_reference`，用真实非空H库、V四来源及T方向band，对照现有from_cache；写`test_grouped_v_uses_each_groups_source_embeddings`，受控变量嵌入让两组选出不同来源，断言批量结果与各自独立选择一致。
- [ ] **Step 3:** 写`test_grouped_single_variable_and_empty_evidence_are_finite`及padding/隐藏输入测试；写受控T/V修正下的`test_grouped_tv_h_matches_one_round_reference`，比较预门控、来源状态、H条件与最终一次写回。运行两新文件并保存RED。
- [ ] **Step 4:** 实现组维数学，H/V/T候选维位于组维前，V将K×offset展开成候选维；保留原权重归一化、有界状态匹配、来源资格与singleton回退。结构/有限性检查在组维边界执行，不移除旧路径检查。
- [ ] **Step 5:** 运行两新文件及既有T/V/router/context测试；float64小样例对照误差≤1e-10、float32≤1e-6，有限非零梯度进入每组完整网络；保存证据与增量。

